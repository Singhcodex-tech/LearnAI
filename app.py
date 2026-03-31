import os
import json
import re
import time
import uuid
from io import BytesIO

import requests
from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production")

CORS(app, origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","))

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["30 per minute"],
    storage_uri="memory://",
)

# ---------------------------------------------------------------------------
# Groq API configuration
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY environment variable is not set. "
        "Export it before starting the server."
    )

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.3-70b-versatile"       # primary model — best quality
FALLBACK_MODEL = "llama-3.1-8b-instant"      # fallback if primary fails / rate-limited
MAX_TOPIC_LENGTH = 200

# ---------------------------------------------------------------------------
# XP / Gamification constants
# ---------------------------------------------------------------------------
XP_PER_QUIZ_CORRECT = 10      # per correct answer in a 3-question quiz
XP_PER_SLIDE_UNDERSTOOD = 5
XP_STREAK_BONUS = 3           # bonus XP for every 5-slide streak

# ---------------------------------------------------------------------------
# In-memory session store
# {
#   session_id: {
#     topic, slides, performance: [...], created_at,
#     quiz_cache:   { slide_index: [questions] },
#     visual_cache: { slide_index: visual_dict },
#     notes:        { slide_index: "text" },
#     xp: int,
#     streak_slides: int,   # consecutive slides understood + quiz >= 0.6
#   }
# }
# ---------------------------------------------------------------------------
sessions: dict = {}

SESSION_TTL = 7200   # 2 hours


def _prune_sessions():
    now = time.time()
    stale = [sid for sid, s in sessions.items() if now - s["created_at"] > SESSION_TTL]
    for sid in stale:
        del sessions[sid]


# ---------------------------------------------------------------------------
# Learner profile helpers
# ---------------------------------------------------------------------------

def compute_learner_profile(performance: list, xp: int = 0, streak: int = 0) -> dict:
    """Aggregate session-wide performance into a compact profile dict."""
    if not performance:
        return {
            "summary": "new learner – no data yet",
            "avg_quiz_score": None,
            "understood_rate": None,
            "avg_time_seconds": None,
            "difficulty_hint": "normal",
            "weak_slides": [],
            "needs_review": [],
            "xp": xp,
            "streak_slides": streak,
        }

    avg_quiz = sum(p["quiz_score"] for p in performance) / len(performance)
    understood_rate = sum(1 for p in performance if p["understood"]) / len(performance)
    avg_time = sum(p["time_spent"] for p in performance) / len(performance)

    weak_slides = [
        p["slide_title"]
        for p in performance
        if p["quiz_score"] < 0.5 or not p["understood"]
    ]

    # Spaced-repetition: flag slides that need a second look
    needs_review = [
        {"slide_title": p["slide_title"], "slide_index": p["slide_index"]}
        for p in performance
        if p["quiz_score"] < 0.6 or not p["understood"]
    ]

    if avg_quiz < 0.45 or understood_rate < 0.4:
        difficulty_hint = "simplify"
    elif avg_quiz > 0.85 and understood_rate > 0.85:
        difficulty_hint = "advanced"
    else:
        difficulty_hint = "normal"

    summary_parts = []
    if avg_quiz < 0.5:
        summary_parts.append("struggling with quizzes")
    elif avg_quiz > 0.8:
        summary_parts.append("excelling at quizzes")
    if understood_rate < 0.5:
        summary_parts.append("frequently marking slides as unclear")
    if avg_time > 120:
        summary_parts.append("spending a lot of time per slide")
    elif avg_time < 15:
        summary_parts.append("progressing quickly through slides")
    if weak_slides:
        summary_parts.append(f"weak areas: {', '.join(weak_slides[:3])}")

    return {
        "summary": "; ".join(summary_parts) or "performing adequately",
        "avg_quiz_score": round(avg_quiz, 2),
        "understood_rate": round(understood_rate, 2),
        "avg_time_seconds": round(avg_time, 1),
        "difficulty_hint": difficulty_hint,
        "weak_slides": weak_slides,
        "needs_review": needs_review,
        "xp": xp,
        "streak_slides": streak,
    }


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _call_groq(
    prompt: str,
    max_tokens: int = 2000,
    system: str | None = None,
    use_fallback: bool = False,
) -> str | None:
    """
    Generic Groq call with automatic fallback to a smaller model.
    - Tries MODEL_NAME (llama-3.3-70b-versatile) first.
    - On 429 (rate-limit) or timeout, retries once with FALLBACK_MODEL.
    Returns raw text or None on error.
    """
    model = FALLBACK_MODEL if use_fallback else MODEL_NAME
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": max_tokens,
            },
            timeout=45,   # increased from 30 — 70B is slower
        )

        # Rate-limit or server error → retry with fallback
        if response.status_code in (429, 503) and not use_fallback:
            print(f"Groq {response.status_code} on {model} — retrying with fallback model.")
            return _call_groq(prompt, max_tokens, system, use_fallback=True)

        if response.status_code == 401:
            print("ERROR: Groq API key is invalid or missing (401 Unauthorized). "
                  "Check your GROQ_API_KEY environment variable.")
            return None

        if response.status_code != 200:
            print(f"Groq API error {response.status_code}: {response.text[:500]}")
            return None

        content = response.json()["choices"][0]["message"]["content"]
        if not content or not content.strip():
            print(f"WARNING: Groq returned an empty response for model {model}.")
            return None
        return content

    except requests.exceptions.Timeout:
        if not use_fallback:
            print(f"Groq timeout on {model} — retrying with fallback model.")
            return _call_groq(prompt, max_tokens, system, use_fallback=True)
        print("ERROR: Groq fallback model also timed out.")
        return None
    except Exception as e:
        print(f"ERROR calling Groq: {e}")
        return None


def extract_json_array(text: str) -> str | None:
    """
    Extract the first top-level JSON array from text using bracket balancing.
    More robust than regex for deeply nested structures (slides with points arrays).
    """
    start = text.find('[')
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def extract_json_object(text: str) -> str | None:
    """
    Extract the first top-level JSON object from text using bracket balancing.
    """
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _extract_all_objects(text: str) -> list:
    """
    Scan through text and extract every complete top-level JSON object using
    bracket balancing. Handles deeply nested structures (points with sub_steps,
    worked_example, etc.) that single-level regex cannot match.
    Returns a list of successfully parsed dicts.
    """
    results = []
    pos = 0
    while pos < len(text):
        start = text.find('{', pos)
        if start == -1:
            break
        depth = 0
        in_string = False
        escape_next = False
        end = None
        for i, ch in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            # Unclosed object — truncated response, stop here
            break
        candidate = text[start:end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                results.append(obj)
        except json.JSONDecodeError:
            pass
        pos = end + 1
    return results


def _repair_slides_json(raw_text: str, math_topic: bool) -> list:
    """
    Ask the model to repair malformed slide JSON into a valid array.
    Returns parsed list or [].
    """
    repair_prompt = f"""
You are a strict JSON repair engine.
Convert the following malformed model output into a valid JSON array of slides.
Return ONLY JSON array. No prose.

Required shape:
[
  {{
    "title": "Slide title",
    "points": [
      {{
        "text": "point text",
        "source_title": "source title",
        "source_url": "url or empty"
      }}
    ]
  }}
]

{ "Math topics may include inline_latex, inline_label, sub_steps, worked_example fields." if math_topic else "" }

Malformed input:
{raw_text[:12000]}
"""
    repaired = _call_groq(repair_prompt, max_tokens=3000, system="Return valid JSON only.")
    if not repaired:
        return []
    repaired = strip_markdown_fences(repaired)
    try:
        data = json.loads(repaired)
    except json.JSONDecodeError:
        arr = extract_json_array(repaired)
        if not arr:
            return []
        try:
            data = json.loads(arr)
        except json.JSONDecodeError:
            return []
    if isinstance(data, dict):
        data = data.get("slides") or data.get("data") or []
    return data if isinstance(data, list) else []


def strip_markdown_fences(text: str) -> str:
    """
    Remove ```json ... ``` or ``` ... ``` code fences that the 70B model
    sometimes adds despite being told not to.
    """
    text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text.strip())
    return text.strip()


def validate_slides(slides: list) -> list:
    validated = []
    for idx, slide in enumerate(slides):
        if not isinstance(slide, dict):
            print(f"validate_slides: skipping slide {idx} — not a dict")
            continue
        title = str(slide.get("title", "")).strip()
        raw_points = slide.get("points", [])
        if not title:
            print(f"validate_slides: skipping slide {idx} — missing title")
            continue
        if not isinstance(raw_points, list):
            raw_points = []

        # Points can be either plain strings (non-math) or rich objects (math).
        # Normalise into a uniform list, preserving rich structure when present.
        points = []
        for p in raw_points:
            if isinstance(p, dict):
                text = str(p.get("text", "")).strip()
                if not text:
                    continue
                point_obj = {"text": text}
                # Per-point citation metadata
                source_title = str(p.get("source_title", "")).strip()
                source_url = str(p.get("source_url", "")).strip()
                if source_title:
                    point_obj["source_title"] = source_title
                if source_url:
                    point_obj["source_url"] = source_url
                # inline equation
                il = str(p.get("inline_latex", "")).strip()
                lbl = str(p.get("inline_label", "")).strip()
                if il:
                    point_obj["inline_latex"] = il
                    point_obj["inline_label"] = lbl
                # sub-steps
                ss = p.get("sub_steps", [])
                if isinstance(ss, list):
                    ss = [str(s).strip() for s in ss if str(s).strip()]
                    if ss:
                        point_obj["sub_steps"] = ss
                points.append(point_obj)
            elif isinstance(p, str):
                text = p.strip()
                if text:
                    points.append({"text": text})

        if len(points) < 1:
            print(f"validate_slides: skipping slide '{title}' — no valid points parsed")
            continue

        entry: dict = {"title": title, "points": points[:6]}

        # Preserve worked example (math slides)
        worked = slide.get("worked_example")
        if isinstance(worked, dict):
            problem = str(worked.get("problem", "")).strip()
            steps = worked.get("steps", [])
            answer = str(worked.get("answer", "")).strip()
            if isinstance(steps, list):
                steps = [str(s).strip() for s in steps if str(s).strip()]
            if problem and steps:
                entry["worked_example"] = {
                    "problem": problem,
                    "steps": steps,
                    "answer": answer,
                }

        validated.append(entry)
    return validated


def _expand_short_points_with_api(slides: list, min_words: int = 50) -> list:
    """
    API-only expansion pass: if in-depth points are too short, ask the model
    to rewrite only those points to at least `min_words`.
    """
    short_exists = False
    for s in slides:
        for p in s.get("points", []):
            if isinstance(p, dict) and len(str(p.get("text", "")).split()) < min_words:
                short_exists = True
                break
        if short_exists:
            break
    if not short_exists:
        return slides

    prompt = f"""
You are editing slide JSON.
Expand only short point texts so each point has at least {min_words} words.
Keep meaning intact. Keep title, source_title, source_url, and any math fields.
Return ONLY valid JSON array, no prose.

Input slides:
{json.dumps(slides, ensure_ascii=False)}
"""
    raw = _call_groq(prompt, max_tokens=3200, system="Return valid JSON only.")
    if not raw:
        return slides
    raw = strip_markdown_fences(raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        arr = extract_json_array(raw)
        if not arr:
            return slides
        try:
            data = json.loads(arr)
        except json.JSONDecodeError:
            return slides
    if isinstance(data, dict):
        data = data.get("slides") or data.get("data") or []
    if not isinstance(data, list):
        return slides
    fixed = validate_slides(data)
    return fixed or slides


def generate_slides_rescue(topic: str, explanation_mode: str = "in_depth") -> list:
    """
    API-only rescue generation with simpler constraints to maximize reliability.
    """
    min_words = 50 if explanation_mode == "in_depth" else 12
    prompt = f"""
You are an expert lecturer. Return ONLY valid JSON array.
Topic: {topic}

Generate exactly 5 slides.
Each slide must have exactly 3 points.
Each point must be an object with keys: text, source_title, source_url.
In in-depth mode, each text must be at least {min_words} words.
Use real credible sources.
No markdown fences. No text outside JSON.
"""
    raw = _call_groq(prompt, max_tokens=3600, use_fallback=True)
    if not raw:
        return []
    raw = strip_markdown_fences(raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        arr = extract_json_array(raw)
        if not arr:
            repaired = _repair_slides_json(raw, math_topic=is_math_topic(topic))
            if not repaired:
                return []
            data = repaired
        else:
            try:
                data = json.loads(arr)
            except json.JSONDecodeError:
                repaired = _repair_slides_json(raw, math_topic=is_math_topic(topic))
                if not repaired:
                    return []
                data = repaired
    if isinstance(data, dict):
        data = data.get("slides") or data.get("data") or []
    if not isinstance(data, list):
        return []
    return validate_slides(data)


# ---------------------------------------------------------------------------
# Math topic detection
# ---------------------------------------------------------------------------

MATH_KEYWORDS = {
    # Pure math branches
    "algebra", "calculus", "geometry", "trigonometry", "statistics",
    "probability", "number theory", "linear algebra", "differential equations",
    "integral", "derivative", "matrix", "vector", "polynomial", "logarithm",
    "exponential", "binomial", "permutation", "combination", "fourier",
    "laplace", "topology", "set theory", "graph theory", "arithmetic",
    "fraction", "equation", "inequality", "quadratic", "sequence", "series",
    "limit", "continuity", "complex number", "real analysis", "discrete math",
    # Applied math / physics math
    "optimization", "eigenvalue", "gradient", "divergence", "curl",
    "bayes", "hypothesis testing", "regression", "variance", "standard deviation",
    "normal distribution", "correlation", "integration", "differentiation",
}

def is_math_topic(topic: str) -> bool:
    """Return True if the topic is mathematical in nature."""
    lower = topic.lower()
    return any(kw in lower for kw in MATH_KEYWORDS)


# ---------------------------------------------------------------------------
# Slide generation (original + adaptive)
# ---------------------------------------------------------------------------

def generate_slides(
    topic: str,
    learner_profile: dict | None = None,
    explanation_mode: str = "in_depth",
    retry: bool = False,
) -> list:
    explanation_mode = str(explanation_mode or "in_depth").strip().lower()
    if explanation_mode not in {"brief", "in_depth"}:
        explanation_mode = "in_depth"

    if explanation_mode == "brief":
        depth_note = (
            "\nNOTE: Keep explanations brief and easy to scan. "
            "Use concise wording and only essential detail."
        )
        point_length_rule_math = "- EVERY point text must be exactly 1 short clear sentence (about 12-20 words)"
        point_length_rule_non_math = "- EVERY bullet point must be exactly 1 short clear sentence (about 12-20 words)"
    else:
        depth_note = (
            "\nNOTE: Explain in depth. Include richer context, clear reasoning, and concrete details."
        )
        point_length_rule_math = "- EVERY point text must be detailed and at least 50 words"
        point_length_rule_non_math = "- EVERY bullet point must be detailed and at least 50 words"

    difficulty_hint = ""
    if learner_profile:
        hint = learner_profile.get("difficulty_hint", "normal")
        profile_summary = learner_profile.get("summary", "")
        if hint == "simplify":
            difficulty_hint = (
                "\nNOTE: This learner has been struggling. Use simpler language, "
                "more analogies, and concrete step-by-step examples. Avoid jargon. "
                f"Learner profile: {profile_summary}"
            )
        elif hint == "advanced":
            difficulty_hint = (
                "\nNOTE: This learner is advanced. Include more technical depth, "
                "edge cases, and challenging concepts. "
                f"Learner profile: {profile_summary}"
            )

    math_topic = is_math_topic(topic)

    if math_topic:
        prompt = (
            "You are an expert mathematics professor and textbook author creating rigorous, deeply detailed, exam-quality slides.\n\n"
            "Topic: %s\n"
            "%s%s\n\n"
            "Return ONLY a valid JSON array. No extra text, no markdown fences.\n\n"
            "CRITICAL: Every bullet point must carry its OWN inline equation, detailed explanation, and sub-steps.\n\n"
            "Required JSON structure for EVERY slide:\n"
            "[\n"
            "  {\n"
            "    \"title\": \"Slide title\",\n"
            "    \"points\": [\n"
            "      {\n"
            "        \"text\": \"A clear 1-2 sentence explanation of the concept — define it and state when to use it.\",\n"
            "        \"source_title\": \"Credible source title for this exact bullet (e.g., Stewart Calculus, MIT OCW, Wikipedia)\",\n"
            "        \"source_url\": \"Direct URL/DOI for this bullet source, or empty string if unavailable\",\n"
            "        \"inline_latex\": \"\\\\frac{-b \\\\pm \\\\sqrt{b^2-4ac}}{2a}\",\n"
            "        \"inline_label\": \"Quadratic Formula\",\n"
            "        \"sub_steps\": [\n"
            "          \"Step 1 — Setup: identify values and formula to use\",\n"
            "          \"Step 2 — Apply: substitute with actual numbers\",\n"
            "          \"Step 3 — Simplify: compute the result\",\n"
            "          \"Step 4 — Verify: check the answer\"\n"
            "        ]\n"
            "      }\n"
            "    ],\n"
            "    \"worked_example\": {\n"
            "      \"problem\": \"State a concrete, non-trivial numeric problem with full context\",\n"
            "      \"steps\": [\n"
            "        \"Step 1 — Setup: identify values and applicable formula\",\n"
            "        \"Step 2 — Substitute: write out the substitution\",\n"
            "        \"Step 3 — Compute: show key arithmetic\",\n"
            "        \"Step 4 — Interpret: state the final answer and its meaning\"\n"
            "      ],\n"
            "      \"answer\": \"Answer with units\"\n"
            "    }\n"
            "  }\n"
            "]\n\n"
            "STRICT RULES:\n"
            "- Generate exactly 6 slides covering the topic from foundations to advanced applications\n"
            "- Each slide: exactly 4 points\n"
            "%s\n"
            "- EVERY point MUST be an OBJECT (not a string) with keys: text, source_title, source_url, inline_latex, inline_label, sub_steps\n"
            "- inline_latex: valid LaTeX, single backslashes (\\\\frac, \\\\sqrt, \\\\int, \\\\pm, \\\\alpha, \\\\theta)\n"
            "- sub_steps: exactly 4 steps, each a short clear string showing the key action and result\n"
            "- source_title and source_url must be specific to each bullet and from real, credible references\n"
            "- worked_example: a clear numeric problem with 4 concise steps\n"
            "- Slides must progress logically: definition → properties → techniques → applications → edge cases\n"
            "- Do NOT repeat the same formula or example across slides\n"
            "- Do NOT output anything outside the JSON array\n"
            "- Do NOT use plain strings for points — ALWAYS use the object format above\n"
        ) % (topic, difficulty_hint, depth_note, point_length_rule_math)
    else:
        prompt = f"""
You are an expert university professor and textbook author creating deeply detailed, lecture-quality academic slides.

Topic: {topic}
{difficulty_hint}{depth_note}

Return ONLY a valid JSON array. No extra text.

Format:
[
  {{
    "title": "Slide title",
    "points": [
      {{
        "text": "Detailed explanation sentence",
        "source_title": "Credible source title for this bullet",
        "source_url": "Direct URL/DOI for this bullet source, or empty string if unavailable"
      }},
      {{
        "text": "Another detailed explanation",
        "source_title": "Another credible source title",
        "source_url": "Direct URL/DOI for this bullet source, or empty string if unavailable"
      }}
    ]
  }}
]

STRICT RULES:
- Generate exactly 8 slides covering the topic thoroughly from fundamentals to advanced aspects
- Each slide must have exactly 5 bullet points
{point_length_rule_non_math}
- EVERY bullet point MUST be an object with keys: text, source_title, source_url
- EVERY point must explain the concept clearly, including the "what" and briefly the "why"
- NO vague lines like "It is important", "Has many applications", "This is used in many fields"
- EVERY point MUST include at least one of:
  • a precise definition or key term
  • a core mechanism or process briefly described
  • a real-world example with a specific name or number
  • a comparison with a related concept
  • a cause-and-effect relationship
  • a quantitative fact or formula
- Slides must flow logically: start with history/definition, build to core mechanisms, then real-world applications, then limitations or future directions
- source_title and source_url must be real, credible references that match the exact bullet content
- Do NOT repeat the same information across slides
- Do NOT output anything outside JSON
"""

    # Larger token budget is needed when points are long (>=50 words).
    max_tok = 5200 if not retry else 6500
    raw_text = _call_groq(prompt, max_tokens=max_tok)
    if not raw_text:
        print("generate_slides: _call_groq returned None — API call failed.")
        if not retry:
            return generate_slides(
                topic,
                learner_profile=learner_profile,
                explanation_mode=explanation_mode,
                retry=True,
            )
        return []

    raw_text = strip_markdown_fences(raw_text)
    print(f"generate_slides: raw API response preview: {raw_text[:300]!r}")

    try:
        slides = json.loads(raw_text)
    except json.JSONDecodeError as e:
        print(f"generate_slides: json.loads failed ({e}). Attempting extraction fallback.")
        # Try to salvage a truncated response by extracting whatever complete
        # slide objects exist before the truncation point
        json_part = extract_json_array(raw_text)
        if not json_part:
            # Last resort: extract all top-level {...} objects individually
            # using bracket balancing (handles nested arrays like points/sub_steps)
            slides = _extract_all_objects(raw_text)
            if slides:
                print(f"generate_slides: recovered {len(slides)} objects via bracket scan.")
            else:
                print("generate_slides: all extraction methods failed.")
                if not retry:
                    return generate_slides(
                        topic,
                        learner_profile=learner_profile,
                        explanation_mode=explanation_mode,
                        retry=True,
                    )
                return []
        else:
            try:
                slides = json.loads(json_part)
            except json.JSONDecodeError:
                if not retry:
                    return generate_slides(
                        topic,
                        learner_profile=learner_profile,
                        explanation_mode=explanation_mode,
                        retry=True,
                    )
                return []

    if isinstance(slides, dict):
        slides = slides.get("slides") or slides.get("data") or []

    if not isinstance(slides, list):
        return []

    result = validate_slides(slides)
    print(f"generate_slides: validated {len(result)} slides from {len(slides) if isinstance(slides, list) else '?'} raw slides.")
    if len(result) < 3 and not retry:
        print("generate_slides: too few validated slides, retrying once.")
        return generate_slides(
            topic,
            learner_profile=learner_profile,
            explanation_mode=explanation_mode,
            retry=True,
        )
    if len(result) < 2 and raw_text:
        print("generate_slides: trying JSON repair path.")
        repaired_slides = _repair_slides_json(raw_text, math_topic=math_topic)
        if repaired_slides:
            repaired_valid = validate_slides(repaired_slides)
            if repaired_valid:
                result = repaired_valid
    if explanation_mode == "in_depth" and result:
        result = _expand_short_points_with_api(result, min_words=50)
    return result


# ---------------------------------------------------------------------------
# Quiz generation
# ---------------------------------------------------------------------------

def generate_quiz_for_slide(slide: dict, learner_profile: dict | None = None, retry: bool = False) -> list:
    """Generate 3 adaptive MCQ questions for a slide."""
    difficulty_note = ""
    if learner_profile:
        hint = learner_profile.get("difficulty_hint", "normal")
        if hint == "simplify":
            difficulty_note = "Make questions straightforward and conceptual. Avoid trick questions."
        elif hint == "advanced":
            difficulty_note = "Make questions challenging. Include application-based and analytical questions."

    prompt = f"""
You are a university professor writing a quiz based on this slide.

Slide Title: {slide['title']}
Slide Content:
{json.dumps(slide['points'], indent=2)}

{difficulty_note}

Generate exactly 3 multiple-choice questions that test understanding of this slide's content.

Return ONLY a valid JSON array. No extra text, no markdown.

Format:
[
  {{
    "question": "Clear, specific question based on slide content?",
    "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
    "correct": 0,
    "explanation": "Concise explanation of why this answer is correct, referencing the slide."
  }}
]

RULES:
- All 4 options must be plausible (no obviously wrong options)
- "correct" is the 0-based index of the correct option
- Questions must be directly answerable from the slide content
- Vary question types: recall, application, conceptual
- Keep options concise (under 15 words each)
- Do NOT output anything outside JSON
"""

    raw_text = _call_groq(prompt, max_tokens=1800)  # increased for 70B verbosity
    if not raw_text:
        return []

    raw_text = strip_markdown_fences(raw_text)

    try:
        questions = json.loads(raw_text)
    except json.JSONDecodeError:
        json_part = extract_json_array(raw_text)
        if not json_part:
            if not retry:
                return generate_quiz_for_slide(slide, learner_profile, retry=True)
            return []
        try:
            questions = json.loads(json_part)
        except json.JSONDecodeError:
            return []

    if not isinstance(questions, list):
        return []

    validated = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        if not all(k in q for k in ("question", "options", "correct", "explanation")):
            continue
        if not isinstance(q["options"], list) or len(q["options"]) != 4:
            continue
        if not isinstance(q["correct"], int) or not (0 <= q["correct"] <= 3):
            continue
        validated.append({
            "question": str(q["question"]).strip(),
            "options": [str(o).strip() for o in q["options"]],
            "correct": q["correct"],
            "explanation": str(q["explanation"]).strip(),
        })

    return validated[:3]


# ---------------------------------------------------------------------------
# Re-teaching (simplified slide regeneration)
# ---------------------------------------------------------------------------

def reteach_slide(slide: dict, retry: bool = False) -> dict | None:
    """Regenerate a slide with simpler language, more examples, and analogies.
    For math slides (those with equations/worked_example), also regenerates
    a simpler worked example with more detailed steps."""
    is_math = "equations" in slide or "worked_example" in slide

    if is_math:
        points_block = json.dumps(slide.get("points", []), indent=2)
        worked_block = json.dumps(slide.get("worked_example", {}), indent=2)
        reteach_prompt = (
            "A student did not understand this mathematics slide.\n"
            "Rewrite it with SIMPLER language and a NEW, EASIER worked example.\n\n"
            "Original Slide Title: %s\n"
            "Original Points:\n%s\n\n"
            "Original Worked Example:\n%s\n\n"
            "Rules:\n"
            "- Keep the same title\n"
            "- Rewrite each point as a rich object with: text, source_title, source_url, inline_latex, inline_label, sub_steps\n"
            "- Use simpler language and everyday analogies\n"
            "- inline_latex: same formula but write a clearer inline_label and simpler sub_steps\n"
            "- sub_steps: 3-5 steps, each explaining WHY the step is done, with smaller numbers\n"
            "- worked_example: a NEW simpler numeric problem, 4-6 steps, more arithmetic detail\n"
            "- Keep 4 to 6 points\n\n"
            "Return ONLY a valid JSON object. No extra text.\n\n"
            "Format:\n"
            "{\n"
            "  \"title\": \"Same title\",\n"
            "  \"points\": [\n"
            "    {\n"
            "      \"text\": \"Simpler explanation sentence\",\n"
            "      \"source_title\": \"Credible source title for this bullet\",\n"
            "      \"source_url\": \"Direct URL/DOI for this bullet source, or empty string if unavailable\",\n"
            "      \"inline_latex\": \"\\\\frac{a}{b}\",\n"
            "      \"inline_label\": \"Formula name\",\n"
            "      \"sub_steps\": [\"Step 1 — reason: detail\", \"Step 2 — reason: detail\"]\n"
            "    }\n"
            "  ],\n"
            "  \"worked_example\": {\n"
            "    \"problem\": \"A simpler problem\",\n"
            "    \"steps\": [\"Step 1 — Setup: ...\", \"Step 2 — Compute: ...\"],\n"
            "    \"answer\": \"Final answer\"\n"
            "  },\n"
            "  \"reteach\": true\n"
            "}\n"
        ) % (slide["title"], points_block, worked_block)
        prompt = reteach_prompt
        max_tokens = 2000
    else:
        prompt = f"""
A student did not understand this slide. Rewrite it to be clearer and easier to understand.

Original Slide Title: {slide['title']}
Original Content:
{json.dumps(slide['points'], indent=2)}

Rewrite rules:
- Keep the same title
- Use simpler language — explain as if to a high-school student
- Add a real-world analogy or everyday example for each concept
- Break complex ideas into step-by-step explanations
- Replace technical jargon with plain English where possible
- Each point must still be a complete, informative sentence
- Keep 4 to 6 bullet points

Return ONLY a valid JSON object. No extra text.

Format:
{{
  "title": "Same title",
  "points": [
    "Simplified explanation with example",
    "Another clearer explanation"
  ],
  "reteach": true
}}
"""
        max_tokens = 1000

    raw_text = _call_groq(prompt, max_tokens=max_tokens)
    if not raw_text:
        return None

    raw_text = strip_markdown_fences(raw_text)

    try:
        slide_data = json.loads(raw_text)
    except json.JSONDecodeError:
        json_part = extract_json_object(raw_text)
        if not json_part:
            if not retry:
                return reteach_slide(slide, retry=True)
            return None
        try:
            slide_data = json.loads(json_part)
        except json.JSONDecodeError:
            return None

    if not isinstance(slide_data, dict):
        return None

    title = str(slide_data.get("title", slide["title"])).strip()
    points = slide_data.get("points", [])
    if not isinstance(points, list):
        return None
    points = [str(p).strip() for p in points if str(p).strip()]
    if len(points) < 2:
        return None

    # Re-parse points using the same normalisation as validate_slides
    raw_points = slide_data.get("points", [])
    if not isinstance(raw_points, list):
        raw_points = []
    norm_points = []
    for p in raw_points:
        if isinstance(p, dict):
            text = str(p.get("text", "")).strip()
            if not text:
                continue
            po = {"text": text}
            source_title = str(p.get("source_title", "")).strip()
            source_url = str(p.get("source_url", "")).strip()
            if source_title:
                po["source_title"] = source_title
            if source_url:
                po["source_url"] = source_url
            il = str(p.get("inline_latex", "")).strip()
            lbl = str(p.get("inline_label", "")).strip()
            if il:
                po["inline_latex"] = il
                po["inline_label"] = lbl
            ss = p.get("sub_steps", [])
            if isinstance(ss, list):
                ss = [str(s).strip() for s in ss if str(s).strip()]
                if ss:
                    po["sub_steps"] = ss
            norm_points.append(po)
        elif isinstance(p, str):
            t = p.strip()
            if t:
                norm_points.append({"text": t})

    if len(norm_points) < 2:
        return None

    result: dict = {"title": title, "points": norm_points[:6], "reteach": True}

    if is_math:
        worked = slide_data.get("worked_example")
        if isinstance(worked, dict):
            problem = str(worked.get("problem", "")).strip()
            steps = worked.get("steps", [])
            answer = str(worked.get("answer", "")).strip()
            if isinstance(steps, list):
                steps = [str(s).strip() for s in steps if str(s).strip()]
            if problem and steps:
                result["worked_example"] = {
                    "problem": problem,
                    "steps": steps,
                    "answer": answer,
                }

    return result


# ---------------------------------------------------------------------------
# NEW: Topic summary generation
# ---------------------------------------------------------------------------

def generate_topic_summary(topic: str, slides: list, performance: list) -> str | None:
    """
    Generate a concise end-of-session summary with key takeaways.
    Personalises the summary based on weak slides if performance data exists.
    """
    slide_titles = [s["title"] for s in slides]
    weak_titles = [p["slide_title"] for p in performance if p["quiz_score"] < 0.6 or not p["understood"]]

    weak_note = ""
    if weak_titles:
        weak_note = f"\nThe learner struggled with: {', '.join(weak_titles[:5])}. Briefly flag those areas for extra study."

    prompt = f"""
You are an expert tutor. A student just finished a learning session on "{topic}".

The session covered these slides:
{json.dumps(slide_titles, indent=2)}
{weak_note}

Write a clear, motivating end-of-session summary for the student. Include:
1. A 2-3 sentence overview of what was learned
2. 5-7 key takeaways as concise bullet points
3. A short "What to explore next" section with 2-3 related topics
4. An encouraging closing line

Keep the tone friendly and academic. Plain text — no JSON, no markdown headers.
"""

    return _call_groq(prompt, max_tokens=800)


# ---------------------------------------------------------------------------
# NEW: AI Tutor chat (contextual Q&A on current slide)
# ---------------------------------------------------------------------------

def answer_student_question(question: str, slide: dict, topic: str, profile: dict) -> str | None:
    """
    Answer a free-form student question in the context of the current slide.
    """
    difficulty = profile.get("difficulty_hint", "normal")
    tone_note = {
        "simplify": "Use simple, jargon-free language with everyday analogies.",
        "advanced": "Use precise technical language and include deeper details.",
        "normal": "Use clear, academic language suitable for an undergraduate student.",
    }.get(difficulty, "")

    system = (
        f"You are a friendly, knowledgeable university tutor teaching '{topic}'. "
        f"{tone_note} Answer only questions related to the course material. "
        "If the question is off-topic, gently redirect the student."
    )

    prompt = f"""Current slide the student is on:
Title: {slide['title']}
Content:
{json.dumps(slide['points'], indent=2)}

Student's question: {question}

Give a helpful, concise answer (3-6 sentences). If relevant, reference specific points from the slide.
"""

    return _call_groq(prompt, max_tokens=500, system=system)


# ---------------------------------------------------------------------------
# Visual generation
# ---------------------------------------------------------------------------

def generate_visual_for_slide(slide: dict, retry: bool = False) -> dict | None:
    """
    Decide the best visual type for a slide and return structured data
    that the frontend can render as an SVG/HTML diagram.
    """
    prompt = f"""You are a visual-learning engine. Given a slide, choose the BEST diagram type and return ONLY a valid JSON object — no prose, no markdown fences.

Slide title: {slide['title']}
Slide content:
{json.dumps(slide['points'], indent=2)}

Choose ONE type from: flowchart, cycle, comparison, timeline, pyramid, mindmap.

Guidelines:
- flowchart   → sequential steps / cause-effect / process with decisions
- cycle       → repeating loop (water cycle, cell cycle, feedback loop)
- comparison  → two or more things being contrasted
- timeline    → events ordered by date / historical sequence
- pyramid     → hierarchical importance (Maslow, OSI layers, taxonomies)
- mindmap     → broad topic with multiple independent sub-branches

Return this exact JSON shape (fill in for your chosen type, omit unused keys):

{{
  "type": "<chosen_type>",
  "title": "<short diagram title, max 8 words>",
  "data": {{

    // --- flowchart ---
    "nodes": [
      {{"id": "n1", "label": "max 4 words", "sub": "optional 1 line", "type": "start|step|decision|end"}}
    ],
    "edges": [
      {{"from": "n1", "to": "n2", "label": ""}}
    ],

    // --- cycle ---
    "steps": [
      {{"label": "Step name", "sub": "one-line detail"}}
    ],

    // --- comparison ---
    "headers": ["Name A", "Name B"],
    "items": [
      {{"label": "Criterion", "a": "value for A", "b": "value for B"}}
    ],

    // --- timeline ---
    "events": [
      {{"year": "1900", "label": "Event name", "sub": "brief detail"}}
    ],

    // --- pyramid ---
    "levels": [
      {{"label": "Top level", "sub": "detail", "width_pct": 30}},
      {{"label": "Mid level", "sub": "detail", "width_pct": 60}},
      {{"label": "Base level", "sub": "detail", "width_pct": 90}}
    ],

    // --- mindmap ---
    "center": "Central concept",
    "branches": [
      {{"label": "Branch", "items": ["item1", "item2"]}}
    ]
  }}
}}

STRICT RULES:
- Return ONLY the JSON object. Absolutely no text outside it.
- 5 to 8 nodes/steps/events/levels/branches maximum.
- Labels: write COMPLETE, MEANINGFUL phrases — do NOT truncate or abbreviate. Use 3 to 8 words per label.
- Sub-labels: write FULL explanatory phrases, 5 to 12 words. Every word matters — do not cut short.
- flowchart must have exactly one "start" and one "end" node.
- cycle must have 4 to 7 steps.
- comparison must have 2 headers and 4 to 6 items.
- mindmap: write complete branch labels (3-6 words) and complete sub-items (4-8 words each).
"""

    raw = _call_groq(prompt, max_tokens=1200)
    if not raw:
        return None

    raw = strip_markdown_fences(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        obj = extract_json_object(raw)
        if not obj:
            if not retry:
                return generate_visual_for_slide(slide, retry=True)
            return None
        try:
            data = json.loads(obj)
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict):
        return None
    if data.get("type") not in {"flowchart", "cycle", "comparison", "timeline", "pyramid", "mindmap"}:
        return None
    if not isinstance(data.get("data"), dict):
        return None

    return data


# ---------------------------------------------------------------------------
# Sources generation
# ---------------------------------------------------------------------------

def generate_sources_for_slide(slide: dict, topic: str, retry: bool = False) -> list:
    """
    Generate a list of credible reference sources for the content on a given slide.
    Returns a list of source dicts: { title, type, description, url_hint }
    """
    points_text = "\n".join(
        (p["text"] if isinstance(p, dict) else str(p))
        for p in slide.get("points", [])
    )

    prompt = f"""You are an academic librarian. For the slide below, list the 3 to 5 most relevant, credible reference sources a student could use to learn more.

Topic: {topic}
Slide Title: {slide['title']}
Slide Content:
{points_text}

Return ONLY a valid JSON array. No extra text.

Format:
[
  {{
    "title": "Full name of the source (book, paper, website, or course)",
    "type": "book" | "paper" | "website" | "course" | "encyclopedia",
    "authors": "Author(s) or organization (e.g. 'Goodfellow et al.' or 'Wikipedia')",
    "year": "Publication year or 'ongoing' for websites",
    "description": "One sentence explaining what this source covers and why it is relevant to this slide.",
    "url_hint": "A plausible URL or DOI (e.g. 'https://en.wikipedia.org/wiki/...' or 'https://arxiv.org/...'). Use an empty string if unknown."
  }}
]

RULES:
- Only cite real, well-known sources (textbooks, Wikipedia, Khan Academy, arXiv, MDN, official docs, etc.)
- Do NOT invent fictional papers or fake DOIs
- Match sources to the exact content of the slide (not just the general topic)
- Prefer freely accessible sources where possible
- Do NOT output anything outside the JSON array
"""

    raw = _call_groq(prompt, max_tokens=1200)
    if not raw:
        return []

    raw = strip_markdown_fences(raw)

    try:
        sources = json.loads(raw)
    except json.JSONDecodeError:
        arr = extract_json_array(raw)
        if not arr:
            if not retry:
                return generate_sources_for_slide(slide, topic, retry=True)
            return []
        try:
            sources = json.loads(arr)
        except json.JSONDecodeError:
            return []

    if not isinstance(sources, list):
        return []

    validated = []
    for s in sources:
        if not isinstance(s, dict):
            continue
        title = str(s.get("title", "")).strip()
        if not title:
            continue
        validated.append({
            "title": title,
            "type": str(s.get("type", "website")).strip(),
            "authors": str(s.get("authors", "")).strip(),
            "year": str(s.get("year", "")).strip(),
            "description": str(s.get("description", "")).strip(),
            "url_hint": str(s.get("url_hint", "")).strip(),
        })

    return validated[:5]


# ---------------------------------------------------------------------------
# XP / Streak helpers
# ---------------------------------------------------------------------------

def _award_xp(session: dict, quiz_score: float, understood: bool) -> dict:
    """
    Update XP and streak in-place and return a dict with the delta info.
    """
    correct_answers = round(quiz_score * 3)  # 3-question quiz
    earned = correct_answers * XP_PER_QUIZ_CORRECT
    if understood:
        earned += XP_PER_SLIDE_UNDERSTOOD

    # Streak logic
    if quiz_score >= 0.6 and understood:
        session["streak_slides"] = session.get("streak_slides", 0) + 1
    else:
        session["streak_slides"] = 0

    # Streak bonus every 5 slides
    streak = session["streak_slides"]
    if streak > 0 and streak % 5 == 0:
        earned += XP_STREAK_BONUS

    session["xp"] = session.get("xp", 0) + earned

    return {
        "xp_earned": earned,
        "total_xp": session["xp"],
        "streak_slides": session["streak_slides"],
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    """Simple health check for deployment monitoring."""
    return jsonify({"status": "ok", "model": MODEL_NAME, "sessions": len(sessions)})


@app.route("/generate", methods=["POST"])
@limiter.limit("10 per minute")
def generate():
    """
    Start a new learning session.
    Body: { "topic": "...", "explanation_mode": "brief" | "in_depth" }
    Returns: { "session_id": "...", "slides": [...] }
    """
    _prune_sessions()
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    topic = str(data.get("topic", "")).strip()
    explanation_mode = str(data.get("explanation_mode", "in_depth")).strip().lower()
    if not topic:
        return jsonify({"error": "No topic provided"}), 400
    if len(topic) > MAX_TOPIC_LENGTH:
        return jsonify({"error": f"Topic must be {MAX_TOPIC_LENGTH} characters or fewer"}), 400
    if explanation_mode not in {"brief", "in_depth"}:
        explanation_mode = "in_depth"

    slides = []
    for attempt in range(3):
        slides = generate_slides(topic, explanation_mode=explanation_mode)
        if slides:
            break
        print(f"[/generate] Attempt {attempt + 1}/3 failed for topic='{topic}'. Retrying...")
        time.sleep(0.8 * (attempt + 1))
    if not slides:
        print(f"[/generate] Standard generation failed for topic='{topic}'. Trying rescue generation.")
        slides = generate_slides_rescue(topic, explanation_mode=explanation_mode)
    if not slides:
        print(f"[/generate] No slides produced for topic='{topic}' after retries.")
        return jsonify({
            "error": "Slide generation failed after retries. Please try once more."
        }), 500

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "topic": topic,
        "slides": slides,
        "performance": [],
        "created_at": time.time(),
        "quiz_cache": {},      # { slide_index_str: [questions] }
        "visual_cache": {},    # { slide_index_str: visual_dict }
        "sources_cache": {},   # { slide_index_str: [sources] }
        "notes": {},           # { slide_index_str: "text" }
        "explanation_mode": explanation_mode,
        "xp": 0,
        "streak_slides": 0,
    }

    return jsonify({"session_id": session_id, "slides": slides, "explanation_mode": explanation_mode})


@app.route("/quiz", methods=["POST"])
@limiter.limit("20 per minute")
def quiz():
    """
    Generate adaptive quiz for a specific slide (cached per slide).
    Body: { "session_id": "...", "slide_index": 0 }
    Returns: { "questions": [...] }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    session_id = data.get("session_id", "")
    slide_index = data.get("slide_index", 0)

    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    slides = session["slides"]
    if not (0 <= slide_index < len(slides)):
        return jsonify({"error": "Invalid slide index"}), 400

    cache_key = str(slide_index)
    if cache_key in session["quiz_cache"]:
        return jsonify({"questions": session["quiz_cache"][cache_key], "cached": True})

    profile = compute_learner_profile(session["performance"], session.get("xp", 0), session.get("streak_slides", 0))
    questions = generate_quiz_for_slide(slides[slide_index], profile)

    if not questions:
        return jsonify({"error": "Failed to generate quiz. Please try again."}), 500

    session["quiz_cache"][cache_key] = questions
    return jsonify({"questions": questions})


@app.route("/feedback", methods=["POST"])
@limiter.limit("30 per minute")
def feedback():
    """
    Record learner performance for a slide and award XP.
    Body: {
        "session_id": "...",
        "slide_index": 0,
        "slide_title": "...",
        "time_spent": 45,
        "understood": true,
        "quiz_score": 0.67,
        "quiz_answers": [0, 2, 1]
    }
    Returns: { "profile": {...}, "message": "...", "xp": {...} }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    session_id = data.get("session_id", "")
    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    record = {
        "slide_index": int(data.get("slide_index", 0)),
        "slide_title": str(data.get("slide_title", "")).strip(),
        "time_spent": float(data.get("time_spent", 0)),
        "understood": bool(data.get("understood", True)),
        "quiz_score": float(data.get("quiz_score", 1.0)),
        "quiz_answers": data.get("quiz_answers", []),
        "timestamp": time.time(),
    }

    session["performance"] = [
        p for p in session["performance"]
        if p["slide_index"] != record["slide_index"]
    ]
    session["performance"].append(record)

    # Award XP
    xp_result = _award_xp(session, record["quiz_score"], record["understood"])

    profile = compute_learner_profile(session["performance"], session.get("xp", 0), session.get("streak_slides", 0))

    if record["quiz_score"] >= 0.8 and record["understood"]:
        message = "Excellent work! You have a strong grasp of this topic."
    elif record["quiz_score"] >= 0.5 or record["understood"]:
        message = "Good effort. Review any points you found confusing before moving on."
    else:
        message = "This topic needs more attention. Try the simplified explanation."

    return jsonify({"profile": profile, "message": message, "xp": xp_result})


@app.route("/reteach", methods=["POST"])
@limiter.limit("10 per minute")
def reteach():
    """
    Return a simplified version of a slide.
    Body: { "session_id": "...", "slide_index": 0 }
    Returns: { "slide": { "title": "...", "points": [...], "reteach": true } }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    session_id = data.get("session_id", "")
    slide_index = data.get("slide_index", 0)

    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    slides = session["slides"]
    if not (0 <= slide_index < len(slides)):
        return jsonify({"error": "Invalid slide index"}), 400

    simplified = reteach_slide(slides[slide_index])
    if not simplified:
        return jsonify({"error": "Failed to generate simplified slide."}), 500

    return jsonify({"slide": simplified})


@app.route("/stats/<session_id>", methods=["GET"])
def stats(session_id):
    """
    Return full session analytics.
    Returns: { "topic", "total_slides", "completed_slides", "profile", "performance" }
    """
    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    profile = compute_learner_profile(session["performance"], session.get("xp", 0), session.get("streak_slides", 0))

    return jsonify({
        "topic": session["topic"],
        "total_slides": len(session["slides"]),
        "completed_slides": len(session["performance"]),
        "profile": profile,
        "performance": session["performance"],
        "created_at": session["created_at"],
    })


@app.route("/visual", methods=["POST"])
@limiter.limit("15 per minute")
def visual():
    """
    Generate a visual diagram for a specific slide (cached per slide).
    Body:  { "session_id": "...", "slide_index": 0 }
    Returns: { "visual": { "type": "...", "title": "...", "data": {...} } }
    """
    req_data = request.get_json(silent=True)
    if not req_data:
        return jsonify({"error": "Invalid JSON"}), 400

    session_id = req_data.get("session_id", "")
    slide_index = req_data.get("slide_index", 0)

    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    slides = session["slides"]
    if not (0 <= slide_index < len(slides)):
        return jsonify({"error": "Invalid slide index"}), 400

    cache_key = str(slide_index)
    if cache_key in session["visual_cache"]:
        return jsonify({"visual": session["visual_cache"][cache_key], "cached": True})

    result = generate_visual_for_slide(slides[slide_index])
    if not result:
        return jsonify({"error": "Failed to generate visual. Please try again."}), 500

    session["visual_cache"][cache_key] = result
    return jsonify({"visual": result})


# ---------------------------------------------------------------------------
# NEW: AI Tutor chat endpoint
# ---------------------------------------------------------------------------

@app.route("/chat", methods=["POST"])
@limiter.limit("20 per minute")
def chat():
    """
    Ask the AI tutor a question about the current slide.
    Body: {
        "session_id": "...",
        "slide_index": 0,
        "question": "Why does X happen?"
    }
    Returns: { "answer": "..." }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    session_id = data.get("session_id", "")
    slide_index = data.get("slide_index", 0)
    question = str(data.get("question", "")).strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400
    if len(question) > 500:
        return jsonify({"error": "Question too long (max 500 characters)"}), 400

    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    slides = session["slides"]
    if not (0 <= slide_index < len(slides)):
        return jsonify({"error": "Invalid slide index"}), 400

    profile = compute_learner_profile(session["performance"], session.get("xp", 0), session.get("streak_slides", 0))
    answer = answer_student_question(question, slides[slide_index], session["topic"], profile)

    if not answer:
        return jsonify({"error": "Tutor is unavailable. Please try again."}), 500

    return jsonify({"answer": answer.strip()})


# ---------------------------------------------------------------------------
# NEW: End-of-session topic summary
# ---------------------------------------------------------------------------

@app.route("/summary", methods=["POST"])
@limiter.limit("5 per minute")
def summary():
    """
    Generate a personalised end-of-session summary.
    Body: { "session_id": "..." }
    Returns: { "summary": "..." }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    session_id = data.get("session_id", "")
    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    text = generate_topic_summary(session["topic"], session["slides"], session["performance"])
    if not text:
        return jsonify({"error": "Failed to generate summary. Please try again."}), 500

    return jsonify({"summary": text.strip()})


# ---------------------------------------------------------------------------
# NEW: Per-slide learner notes
# ---------------------------------------------------------------------------

@app.route("/note", methods=["POST"])
@limiter.limit("30 per minute")
def note():
    """
    Save or retrieve a learner's personal note for a slide.
    To SAVE:  { "session_id": "...", "slide_index": 0, "text": "My note..." }
    To GET:   { "session_id": "...", "slide_index": 0 }
    Returns: { "note": "..." }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    session_id = data.get("session_id", "")
    slide_index = data.get("slide_index", 0)

    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    slides = session["slides"]
    if not (0 <= slide_index < len(slides)):
        return jsonify({"error": "Invalid slide index"}), 400

    cache_key = str(slide_index)

    if "text" in data:
        text = str(data["text"]).strip()[:2000]   # cap at 2000 chars
        session["notes"][cache_key] = text
        return jsonify({"note": text, "saved": True})
    else:
        return jsonify({"note": session["notes"].get(cache_key, "")})


# ---------------------------------------------------------------------------
# NEW: Per-slide sources / references
# ---------------------------------------------------------------------------

@app.route("/sources", methods=["POST"])
@limiter.limit("15 per minute")
def sources():
    """
    Generate credible reference sources for a specific slide (cached per slide).
    Body:  { "session_id": "...", "slide_index": 0 }
    Returns: { "sources": [ { title, type, authors, year, description, url_hint } ] }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    session_id = data.get("session_id", "")
    slide_index = data.get("slide_index", 0)

    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    slides = session["slides"]
    if not (0 <= slide_index < len(slides)):
        return jsonify({"error": "Invalid slide index"}), 400

    cache_key = str(slide_index)
    if cache_key in session.get("sources_cache", {}):
        return jsonify({"sources": session["sources_cache"][cache_key], "cached": True})

    result = generate_sources_for_slide(slides[slide_index], session["topic"])
    if not result:
        return jsonify({"error": "The AI could not generate sources for this slide. Please try again."}), 500

    if "sources_cache" not in session:
        session["sources_cache"] = {}
    session["sources_cache"][cache_key] = result
    return jsonify({"sources": result})


# ---------------------------------------------------------------------------
# PPT Generation
# ---------------------------------------------------------------------------

def build_pptx(topic: str, slides_data: list, theme: str = "dark"):
    """
    Build a beautifully designed PPTX in either dark or light theme.
    Returns a python-pptx Presentation object, or None if pptx is unavailable.
    """
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt, Emu
        from pptx.dml.color import RGBColor
        from pptx.enum.text import PP_ALIGN
        from pptx.oxml.ns import qn
        from lxml import etree
    except ImportError:
        return None

    is_dark = (theme != "light")

    # ── Colour palettes ─────────────────────────────────────────
    if is_dark:
        C_BG      = RGBColor(0x08, 0x0b, 0x12)   # near-black
        C_SURF    = RGBColor(0x0f, 0x14, 0x20)   # surface
        C_CARD    = RGBColor(0x14, 0x19, 0x26)   # card
        C_BORDER  = RGBColor(0x2a, 0x33, 0x47)   # border
        C_BORDER2 = RGBColor(0x1e, 0x25, 0x35)
        C_ACCENT1 = RGBColor(0xe8, 0xa0, 0x20)   # gold
        C_ACCENT2 = RGBColor(0x20, 0xc5, 0xb0)   # teal
        C_TITLE   = RGBColor(0xea, 0xe6, 0xdf)   # near-white
        C_BODY    = RGBColor(0xb8, 0xc0, 0xd4)   # light grey body text
        C_MUTED   = RGBColor(0x8b, 0x93, 0xa8)
        C_DIM     = RGBColor(0x4a, 0x52, 0x68)
        C_PILL_BG = RGBColor(0x0b, 0x1f, 0x1c)
        BULLET_COLORS = [C_ACCENT1, C_ACCENT2, C_ACCENT1, C_ACCENT2, C_ACCENT1]
    else:
        C_BG      = RGBColor(0xf7, 0xf8, 0xfc)   # off-white
        C_SURF    = RGBColor(0xff, 0xff, 0xff)   # white
        C_CARD    = RGBColor(0xee, 0xf0, 0xf8)   # light card
        C_BORDER  = RGBColor(0xd0, 0xd6, 0xe8)
        C_BORDER2 = RGBColor(0xe0, 0xe5, 0xf2)
        C_ACCENT1 = RGBColor(0xc4, 0x7d, 0x0e)   # amber (darker gold for contrast)
        C_ACCENT2 = RGBColor(0x0d, 0x9c, 0x8a)   # dark teal
        C_TITLE   = RGBColor(0x12, 0x17, 0x2e)   # near-black
        C_BODY    = RGBColor(0x2c, 0x35, 0x52)   # dark navy body
        C_MUTED   = RGBColor(0x60, 0x6a, 0x88)
        C_DIM     = RGBColor(0x90, 0x9a, 0xb8)
        C_PILL_BG = RGBColor(0xe0, 0xf5, 0xf2)
        BULLET_COLORS = [C_ACCENT1, C_ACCENT2, C_ACCENT1, C_ACCENT2, C_ACCENT1]

    W = Inches(13.333)
    H = Inches(7.5)

    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H
    blank = prs.slide_layouts[6]

    # ── Helpers ─────────────────────────────────────────────────

    def set_bg(slide, color):
        bg = slide.background
        bg.fill.solid()
        bg.fill.fore_color.rgb = color

    def rect(slide, l, t, w, h, fill, line=None, line_w=Pt(0), rounding=None):
        shp = slide.shapes.add_shape(1, l, t, w, h)
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
        if line:
            shp.line.color.rgb = line
            shp.line.width = line_w
        else:
            shp.line.fill.background()
        return shp

    def textbox(slide, text, l, t, w, h,
                font='Calibri', size=12, bold=False, italic=False,
                color=None, align=PP_ALIGN.LEFT, wrap=True):
        tb = slide.shapes.add_textbox(l, t, w, h)
        tf = tb.text_frame
        tf.word_wrap = wrap
        p = tf.paragraphs[0]
        p.alignment = align
        run = p.add_run()
        run.text = text
        run.font.name  = font
        run.font.size  = Pt(size)
        run.font.bold  = bold
        run.font.italic = italic
        if color:
            run.font.color.rgb = color
        return tb

    # Expand long slides into continuation parts so full sentences fit in PPT.
    expanded_slides = []
    for s in slides_data:
        pts = s.get("points", []) if isinstance(s, dict) else []
        if not isinstance(pts, list):
            pts = []
        point_texts = [
            str(p.get("text", "") if isinstance(p, dict) else p).strip()
            for p in pts
            if str(p.get("text", "") if isinstance(p, dict) else p).strip()
        ]
        max_len = max((len(t) for t in point_texts), default=0)
        if max_len > 420:
            chunk_size = 2
        elif max_len > 260:
            chunk_size = 3
        elif max_len > 170:
            chunk_size = 4
        else:
            chunk_size = 5
        if not pts:
            expanded_slides.append({**s, "_ppt_points": [], "_ppt_part": 1, "_ppt_parts": 1})
            continue
        total_parts = (len(pts) + chunk_size - 1) // chunk_size
        for i in range(0, len(pts), chunk_size):
            expanded_slides.append({
                **s,
                "_ppt_points": pts[i:i + chunk_size],
                "_ppt_part": (i // chunk_size) + 1,
                "_ppt_parts": total_parts,
            })

    total = len(expanded_slides)

    # ════════════════════════════════════════════════════════════
    # SLIDE 0 — TITLE SLIDE
    # ════════════════════════════════════════════════════════════
    sl = prs.slides.add_slide(blank)
    set_bg(sl, C_BG)

    # ── Full-height left panel ──────────────────────────────────
    PANEL_W = Inches(5.2)
    rect(sl, 0, 0, PANEL_W, H, C_SURF)
    rect(sl, PANEL_W, 0, Inches(0.032), H, C_BORDER)

    # Top accent bar inside left panel
    rect(sl, 0, 0, PANEL_W, Inches(0.22), C_ACCENT1)

    # Decorative stacked rectangles (visual layering)
    rect(sl, Inches(0.35), Inches(0.55), Inches(0.9), Inches(0.9), C_CARD,
         line=C_BORDER, line_w=Pt(0.5))
    rect(sl, Inches(0.55), Inches(0.75), Inches(0.9), Inches(0.9), C_ACCENT1)
    textbox(sl, 'AI', Inches(0.6), Inches(0.78), Inches(0.8), Inches(0.85),
            font='Georgia', size=20, bold=True, color=C_BG,
            align=PP_ALIGN.CENTER)

    # Brand label
    textbox(sl, 'ADAPTIVE LEARN  ·  AI LECTURE ENGINE',
            Inches(0.35), Inches(1.85), PANEL_W - Inches(0.7), Inches(0.3),
            font='Courier New', size=7.5, bold=True, color=C_ACCENT2)

    # Thin rule
    rect(sl, Inches(0.35), Inches(2.22), Inches(0.55), Inches(0.038), C_ACCENT1)

    # Main title text
    t_short = topic if len(topic) <= 55 else topic[:53] + '…'
    tb = sl.shapes.add_textbox(Inches(0.35), Inches(2.35), PANEL_W - Inches(0.7), Inches(2.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = t_short
    run.font.name  = 'Georgia'
    run.font.size  = Pt(34)
    run.font.bold  = True
    run.font.color.rgb = C_TITLE

    # Slide count badge
    rect(sl, Inches(0.35), Inches(5.5), Inches(1.6), Inches(0.5), C_ACCENT1)
    textbox(sl, f'{total}  SLIDES', Inches(0.35), Inches(5.52),
            Inches(1.6), Inches(0.46),
            font='Courier New', size=11, bold=True, color=C_BG,
            align=PP_ALIGN.CENTER)

    textbox(sl, f'Generated by AdaptiveLearn AI',
            Inches(0.35), Inches(6.15), PANEL_W - Inches(0.7), Inches(0.3),
            font='Calibri', size=10, italic=True, color=C_MUTED)

    # ── Right decorative panel ──────────────────────────────────
    RX = PANEL_W + Inches(0.032)
    RW = W - RX

    # Large accent block top-right
    rect(sl, RX, 0, RW, Inches(2.5), C_ACCENT1)
    # Teal strip below it
    rect(sl, RX, Inches(2.5), RW, Inches(0.18), C_ACCENT2)

    # Big decorative number
    textbox(sl, str(total), RX, Inches(0.05), RW, Inches(2.4),
            font='Georgia', size=110, bold=True, color=C_BG,
            align=PP_ALIGN.CENTER)

    # Grid of decorative cards in right panel body
    card_y_start = Inches(2.9)
    card_h = Inches(1.1)
    card_gap = Inches(0.18)
    deco_labels = ['CONCEPTS', 'QUIZZES', 'INSIGHTS', 'REVIEW']
    for ci in range(min(4, total)):
        cy = card_y_start + ci * (card_h + card_gap)
        rect(sl, RX + Inches(0.3), cy, RW - Inches(0.6), card_h,
             C_CARD if is_dark else C_SURF,
             line=C_BORDER, line_w=Pt(0.6))
        rect(sl, RX + Inches(0.3), cy, Inches(0.06), card_h, C_ACCENT2)
        label = deco_labels[ci] if ci < len(deco_labels) else f'PART {ci+1}'
        textbox(sl, label, RX + Inches(0.5), cy + Inches(0.15),
                RW - Inches(1.0), Inches(0.3),
                font='Courier New', size=7, bold=True, color=C_ACCENT2)
        textbox(sl, f'Section {ci + 1}', RX + Inches(0.5), cy + Inches(0.45),
                RW - Inches(1.0), Inches(0.4),
                font='Calibri', size=12, bold=True, color=C_TITLE)

    # Bottom strip
    rect(sl, 0, H - Inches(0.12), W, Inches(0.12), C_ACCENT2)

    # ════════════════════════════════════════════════════════════
    # CONTENT SLIDES
    # ════════════════════════════════════════════════════════════
    for idx, sdata in enumerate(expanded_slides):
        sl = prs.slides.add_slide(blank)
        set_bg(sl, C_BG)

        title_txt = str(sdata.get('title', f'Slide {idx + 1}')).strip()
        part = int(sdata.get("_ppt_part", 1))
        parts = int(sdata.get("_ppt_parts", 1))
        if parts > 1:
            title_txt = f"{title_txt} (Part {part}/{parts})"
        points_raw = sdata.get('_ppt_points', sdata.get('points', []))
        n_pts = min(len(points_raw), 5)

        # ── Top header band ───────────────────────────────────────
        HEADER_H = Inches(1.45)
        rect(sl, 0, 0, W, HEADER_H, C_SURF)

        # Gold top stripe
        rect(sl, 0, 0, W, Inches(0.18), C_ACCENT1)

        # Teal left stripe (full height)
        rect(sl, 0, 0, Inches(0.1), H, C_ACCENT2)

        # Slide number box — top-left
        rect(sl, Inches(0.22), Inches(0.35), Inches(1.15), Inches(0.72), C_ACCENT1)
        textbox(sl, f'{idx + 1:02d}', Inches(0.22), Inches(0.35),
                Inches(1.15), Inches(0.72),
                font='Georgia', size=26, bold=True, color=C_BG,
                align=PP_ALIGN.CENTER)

        # Slide counter label
        textbox(sl, f'/ {total}', Inches(1.42), Inches(0.62),
                Inches(0.7), Inches(0.32),
                font='Courier New', size=9, color=C_MUTED)

        # Topic label right
        t_label = (topic[:52] + '…') if len(topic) > 52 else topic
        textbox(sl, t_label.upper(),
                W - Inches(5.5), Inches(0.55), Inches(5.2), Inches(0.28),
                font='Courier New', size=7.5, color=C_DIM,
                align=PP_ALIGN.RIGHT)

        # Separator line below header
        rect(sl, Inches(0.1), HEADER_H, W - Inches(0.1), Inches(0.025), C_BORDER)

        # ── Slide title ──────────────────────────────────────────
        title_disp = title_txt
        tb = sl.shapes.add_textbox(Inches(0.3), Inches(1.55), W - Inches(2.2), Inches(0.95))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        run = p.add_run()
        run.text = title_disp
        run.font.name  = 'Georgia'
        run.font.size  = Pt(26)
        run.font.bold  = True
        run.font.color.rgb = C_TITLE

        # Underline accent beneath title
        rect(sl, Inches(0.3), Inches(2.54), Inches(0.6), Inches(0.04), C_ACCENT1)
        rect(sl, Inches(0.95), Inches(2.54), Inches(0.22), Inches(0.04), C_ACCENT2)

        # ── Right sidebar ─────────────────────────────────────────
        SB_X = W - Inches(1.9)
        SB_W = Inches(1.75)
        rect(sl, SB_X, HEADER_H + Inches(0.08), SB_W, H - HEADER_H - Inches(0.35),
             C_CARD if is_dark else C_SURF,
             line=C_BORDER, line_w=Pt(0.5))
        rect(sl, SB_X, HEADER_H + Inches(0.08), Inches(0.06),
             H - HEADER_H - Inches(0.35), C_ACCENT2)

        # Sidebar: big slide number watermark
        textbox(sl, str(idx + 1),
                SB_X, Inches(3.0), SB_W, Inches(1.4),
                font='Georgia', size=64, bold=True, color=C_BORDER,
                align=PP_ALIGN.CENTER)

        # Sidebar: section label
        textbox(sl, 'SLIDE', SB_X, Inches(2.4), SB_W, Inches(0.35),
                font='Courier New', size=8, bold=True, color=C_DIM,
                align=PP_ALIGN.CENTER)

        # Sidebar: topic dots (decorative)
        for di in range(min(n_pts, 5)):
            dot_y = Inches(4.6) + di * Inches(0.38)
            rect(sl, SB_X + Inches(0.75), dot_y, Inches(0.12), Inches(0.12),
                 C_ACCENT1 if di % 2 == 0 else C_ACCENT2)

        # ── Bullet points area ────────────────────────────────────
        font_size = 13 if n_pts <= 3 else (12 if n_pts == 4 else 11)
        bullet_h  = Inches(0.78) if n_pts <= 3 else (Inches(0.7) if n_pts == 4 else Inches(0.62))
        start_y   = Inches(2.72)
        content_w = SB_X - Inches(0.45)

        for bi, point in enumerate(points_raw[:5]):
            pt_text = str(point.get('text', '') if isinstance(point, dict) else point).strip()
            if not pt_text:
                continue

            by = start_y + bi * bullet_h
            bc = BULLET_COLORS[bi % len(BULLET_COLORS)]

            # Card-style row background (alternating subtle tint)
            row_bg = (RGBColor(0x10, 0x16, 0x24) if is_dark else RGBColor(0xf0, 0xf3, 0xfc)) \
                     if bi % 2 == 0 else \
                     (RGBColor(0x0d, 0x12, 0x1e) if is_dark else RGBColor(0xfa, 0xfb, 0xff))
            rect(sl, Inches(0.22), by, content_w - Inches(0.22), bullet_h - Inches(0.06),
                 row_bg, line=C_BORDER2, line_w=Pt(0.4))

            # Left colour stripe on each row
            rect(sl, Inches(0.22), by, Inches(0.06), bullet_h - Inches(0.06), bc)

            # Bullet number circle (simulated with text)
            rect(sl, Inches(0.38), by + Inches(0.19), Inches(0.32), Inches(0.32), bc)
            textbox(sl, str(bi + 1),
                    Inches(0.38), by + Inches(0.18), Inches(0.32), Inches(0.34),
                    font='Courier New', size=9, bold=True, color=C_BG,
                    align=PP_ALIGN.CENTER)

            # Point text (full sentence; no truncation)
            pt_disp = pt_text
            tb2 = sl.shapes.add_textbox(Inches(0.82), by + Inches(0.08),
                                         content_w - Inches(1.1), bullet_h - Inches(0.14))
            tf2 = tb2.text_frame
            tf2.word_wrap = True
            p2 = tf2.paragraphs[0]
            run2 = p2.add_run()
            run2.text = pt_disp
            run2.font.name  = 'Calibri'
            text_len = len(pt_disp)
            adj_size = font_size
            if text_len > 420:
                adj_size = max(8, font_size - 3)
            elif text_len > 260:
                adj_size = max(9, font_size - 2)
            elif text_len > 170:
                adj_size = max(10, font_size - 1)
            run2.font.size = Pt(adj_size)
            run2.font.color.rgb = C_BODY

        # ── Bottom bar ────────────────────────────────────────────
        rect(sl, 0, H - Inches(0.14), W, Inches(0.14), C_ACCENT1)
        textbox(sl, 'ADAPTIVELEARN  ·  AI-POWERED EDUCATION',
                Inches(0.22), H - Inches(0.13), Inches(5.0), Inches(0.13),
                font='Courier New', size=6.5, bold=True, color=C_BG)
        textbox(sl, f'SLIDE {idx + 1} OF {total}',
                W - Inches(2.2), H - Inches(0.13), Inches(2.0), Inches(0.13),
                font='Courier New', size=6.5, bold=True, color=C_BG,
                align=PP_ALIGN.RIGHT)

    # ════════════════════════════════════════════════════════════
    # CLOSING SLIDE
    # ════════════════════════════════════════════════════════════
    sl = prs.slides.add_slide(blank)
    set_bg(sl, C_BG)

    # Bold split — left teal panel, right body
    rect(sl, 0, 0, Inches(4.8), H, C_ACCENT2)
    rect(sl, Inches(4.8), 0, Inches(0.06), H, C_ACCENT1)

    # Large checkmark-ish decoration on left panel
    textbox(sl, '✓', Inches(0.3), Inches(1.1), Inches(4.2), Inches(2.0),
            font='Georgia', size=100, bold=True, color=C_BG,
            align=PP_ALIGN.CENTER)
    textbox(sl, 'SESSION COMPLETE', Inches(0.3), Inches(3.3), Inches(4.2), Inches(0.55),
            font='Courier New', size=13, bold=True, color=C_BG,
            align=PP_ALIGN.CENTER)
    textbox(sl, f'{total} slides covered', Inches(0.3), Inches(3.95),
            Inches(4.2), Inches(0.4),
            font='Calibri', size=11, italic=True, color=RGBColor(0xd0,0xf5,0xef),
            align=PP_ALIGN.CENTER)

    # Right panel
    t_close = topic if len(topic) <= 58 else topic[:56] + '…'
    textbox(sl, 'YOU HAVE STUDIED', Inches(5.3), Inches(1.5),
            W - Inches(5.7), Inches(0.4),
            font='Courier New', size=9, bold=True, color=C_ACCENT2)
    rect(sl, Inches(5.3), Inches(1.97), Inches(0.55), Inches(0.038), C_ACCENT1)

    tb = sl.shapes.add_textbox(Inches(5.3), Inches(2.1), W - Inches(5.7), Inches(2.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = t_close
    run.font.name  = 'Georgia'
    run.font.size  = Pt(36)
    run.font.bold  = True
    run.font.color.rgb = C_TITLE

    textbox(sl, f'Powered by AdaptiveLearn AI  ·  {total} slides',
            Inches(5.3), Inches(5.0), W - Inches(5.7), Inches(0.4),
            font='Calibri', size=11, italic=True, color=C_MUTED)

    # Bottom bar
    rect(sl, 0, H - Inches(0.14), W, Inches(0.14), C_ACCENT1)

    return prs


@app.route("/generate_ppt", methods=["POST"])
@limiter.limit("5 per minute")
def generate_ppt():
    """
    Build and return a .pptx file for the current session.
    Body: { "session_id": "...", "theme": "dark"|"light" }
    Returns: binary .pptx file download
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    session_id = data.get("session_id", "")
    theme      = data.get("theme", "dark")
    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    prs = build_pptx(session["topic"], session["slides"], theme=theme)
    if prs is None:
        return jsonify({
            "error": "python-pptx is not installed on the server. "
                     "Run: pip install python-pptx"
        }), 500

    buf = BytesIO()
    prs.save(buf)
    buf.seek(0)

    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', session["topic"][:40]).strip('_')
    theme_tag  = "Dark" if theme != "light" else "Light"
    filename = f"{safe_name}_{theme_tag}_AdaptiveLearn.pptx"

    return send_file(
        buf,
        as_attachment=True,
        download_name=filename,
        mimetype=(
            "application/vnd.openxmlformats-officedocument"
            ".presentationml.presentation"
        ),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False)
#Just for the commit