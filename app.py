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
LAST_GROQ_CALL_AT = 0.0

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
    global LAST_GROQ_CALL_AT
    model = FALLBACK_MODEL if use_fallback else MODEL_NAME
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        # Throttle outbound LLM calls to reduce 429 bursts across endpoints.
        now = time.time()
        elapsed = now - LAST_GROQ_CALL_AT
        min_gap = 0.9
        if elapsed < min_gap:
            time.sleep(min_gap - elapsed)

        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": max_tokens,
            },
            timeout=45,   # increased from 30 — 70B is slower
        )
        LAST_GROQ_CALL_AT = time.time()

        # Rate-limit or server error → retry with fallback
        if response.status_code in (429, 503) and not use_fallback:
            print(f"Groq {response.status_code} on {model} — retrying with fallback model.")
            time.sleep(1.2)
            return _call_groq(prompt, max_tokens, system, use_fallback=True)
        if response.status_code in (429, 503) and use_fallback:
            print(f"Groq {response.status_code} on fallback model — backing off once.")
            time.sleep(2.0)
            return None

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


def _normalize_point_word_lengths(slides: list, explanation_mode: str) -> list:
    """
    Fix #3: Audit-only pass — never truncate point text.

    The old implementation truncated at max_w words, destroying the detail
    that was explicitly requested in the prompt (45-55 words in_depth).
    This version only logs short points so we can monitor quality without
    destroying any content.
    """
    if explanation_mode == "brief":
        min_w = 15
    else:
        min_w = 35

    for slide in slides:
        for p in slide.get("points", []):
            if not isinstance(p, dict):
                continue
            text = str(p.get("text", "")).strip()
            if not text:
                continue
            word_count = len(text.split())
            if word_count < min_w:
                print(
                    f"_normalize_point_word_lengths: short point ({word_count} words) "
                    f"in slide '{slide.get('title', '?')}' — keeping as-is."
                )
            # Never modify p["text"] — return slides untouched
    return slides


def generate_slides_rescue(topic: str, explanation_mode: str = "in_depth") -> list:
    """
    Fix #4: Rescue generation using the same 1-slide-per-call + role approach as
    generate_slides(), but forced onto the fallback model (llama-3.1-8b-instant).
    Iterates all 12 _DECK_ROLES so the rescue can always fill a complete deck,
    unlike the old version that was capped at 5 slides / 3 points.
    """
    point_length_rule = (
        "- Every point text should be around 50 words (target range 45-55 words)"
        if explanation_mode == "in_depth"
        else "- Every point text should be around 20 to 25 words"
    )
    depth_note = (
        "\nNOTE: Explain in depth with concrete details."
        if explanation_mode == "in_depth"
        else "\nNOTE: Keep explanations brief and easy to scan."
    )
    math_topic = is_math_topic(topic)

    rescued: list = []
    for slide_idx, (role_title, role_instruction) in enumerate(_DECK_ROLES):
        slide_number = slide_idx + 1
        slide = _generate_single_slide(
            topic=topic,
            slide_number=slide_number,
            role_title=role_title,
            role_instruction=role_instruction,
            math_topic=math_topic,
            difficulty_hint="",
            depth_note=depth_note,
            point_length_rule=point_length_rule,
            retry=True,   # force fallback model
        )
        if slide is not None:
            rescued.append(slide)
        print(f"generate_slides_rescue: {len(rescued)}/{slide_number} slides rescued.")

    return rescued


# ---------------------------------------------------------------------------
# Structured Learning Control — Subject / Mode / Depth prompt builder
# ---------------------------------------------------------------------------

VALID_MODES  = {"Learn", "Practice", "Test"}
VALID_DEPTHS = {"Beginner", "Exam", "Advanced"}

# Every subject gets its own mode × depth matrix.
# Keys must exactly match what the frontend sends (title-cased).
_SUBJECT_MODE_INSTRUCTIONS: dict[str, dict[str, str]] = {
    # ── STEM ────────────────────────────────────────────────────────────
    "Math": {
        "Learn":    "Explain the concept step-by-step with clear worked examples. "
                    "Show each calculation stage and define every symbol used.",
        "Practice": "Generate realistic practice problems (numeric, word-problem, proof) "
                    "with full solutions and reasoning shown at each step.",
        "Test":     "Generate exam-style questions ONLY. Do NOT include answers, "
                    "hints, or solutions anywhere in the slide content.",
    },
    "Physics": {
        "Learn":    "Explain the physical law or phenomenon with intuition first, "
                    "then derive the governing equations step-by-step with SI units.",
        "Practice": "Provide quantitative physics problems with full worked solutions, "
                    "including free-body diagrams or vector descriptions where relevant.",
        "Test":     "Pose physics problems with given data and ask for derivations or "
                    "numerical answers ONLY. Do NOT include solutions or hints.",
    },
    "Chemistry": {
        "Learn":    "Explain chemical concepts with reaction mechanisms, bonding diagrams, "
                    "and real-world examples of compounds or processes.",
        "Practice": "Provide stoichiometry, reaction-balancing, or mechanism exercises "
                    "with complete solutions and mole/unit tracking shown.",
        "Test":     "Present chemical problems or name-the-product tasks ONLY. "
                    "Do NOT include answers, structures, or explanations.",
    },
    "Biology": {
        "Learn":    "Explain biological processes and structures using clear diagrams "
                    "described in text, real organism examples, and function-structure links.",
        "Practice": "Provide labelling exercises, process-ordering tasks, or case-study "
                    "questions with model answers and common-mistake notes.",
        "Test":     "Generate structured biology questions and diagrams-to-label ONLY. "
                    "Do NOT include answers or model responses.",
    },
    "Coding": {
        "Learn":    "Explain the programming concept clearly, then show a concise working "
                    "code example. Break down each line and explain why it is written that way.",
        "Practice": "Provide a coding challenge with input/output examples and a partial "
                    "skeleton or hint. Include the full correct solution with explanation.",
        "Test":     "Present programming problems and specifications ONLY. Do NOT include "
                    "solutions, pseudocode answers, or implementation hints.",
    },
    "Computer Science": {
        "Learn":    "Explain CS theory (algorithms, data structures, complexity) with "
                    "step-by-step traces, pseudocode, and Big-O analysis.",
        "Practice": "Provide algorithm design or analysis exercises with full solutions, "
                    "including time/space complexity justification.",
        "Test":     "Pose CS theory or algorithm questions ONLY. Do NOT include "
                    "solutions, proofs, or complexity answers.",
    },
    "Statistics": {
        "Learn":    "Explain statistical concepts with formulas, real datasets, and "
                    "interpretation of results in plain language.",
        "Practice": "Provide calculation exercises (hypothesis tests, confidence intervals, "
                    "regression) with full worked solutions and interpretation.",
        "Test":     "Present statistical problems with data tables ONLY. "
                    "Do NOT include calculations, p-values, or conclusions.",
    },
    # ── HUMANITIES ──────────────────────────────────────────────────────
    "English": {
        "Learn":    "Explain grammar rules or literary concepts with clear examples "
                    "and counter-examples. Use real sentences to illustrate every rule.",
        "Practice": "Provide writing or grammar correction exercises. Include model "
                    "answers and explanations of common mistakes.",
        "Test":     "Generate comprehension tasks, essay prompts, or grammar exercises "
                    "ONLY. Do NOT include model answers or corrections.",
    },
    "History": {
        "Learn":    "Narrate historical events with causes, key figures, dates, and "
                    "long-term consequences. Connect events to broader patterns.",
        "Practice": "Provide source-analysis or essay-structure exercises based on "
                    "real historical documents, with model responses.",
        "Test":     "Pose source-based or essay questions ONLY. "
                    "Do NOT include model answers or mark schemes.",
    },
    "Geography": {
        "Learn":    "Explain physical or human geography concepts with real place examples, "
                    "statistics, and cause-effect chains.",
        "Practice": "Provide case-study analysis or map-interpretation exercises "
                    "with full model answers and data commentary.",
        "Test":     "Present geography questions or data-response tasks ONLY. "
                    "Do NOT include answers or explanations.",
    },
    "Philosophy": {
        "Learn":    "Introduce the philosophical argument, key thinkers, objections, "
                    "and counter-arguments with clear logical structure.",
        "Practice": "Provide argument-analysis or essay-plan tasks with model responses "
                    "demonstrating critical evaluation techniques.",
        "Test":     "Pose philosophical essay questions or argument-analysis tasks ONLY. "
                    "Do NOT include model answers or evaluation commentary.",
    },
    # ── APPLIED / PROFESSIONAL ──────────────────────────────────────────
    "Economics": {
        "Learn":    "Explain economic theory with real-world data, diagrams described "
                    "in text (supply/demand, curves), and policy implications.",
        "Practice": "Provide calculation or diagram-analysis exercises (elasticity, "
                    "GDP, market equilibrium) with full solutions.",
        "Test":     "Present economics data-response or essay questions ONLY. "
                    "Do NOT include answers, calculations, or diagram descriptions.",
    },
    "Business": {
        "Learn":    "Explain business concepts (strategy, finance, marketing, HR) with "
                    "real company examples, frameworks, and key metrics.",
        "Practice": "Provide case-study analysis tasks or financial calculation exercises "
                    "with model answers and evaluation criteria.",
        "Test":     "Pose business case-study or essay questions ONLY. "
                    "Do NOT include model answers or mark schemes.",
    },
    "Law": {
        "Learn":    "Explain legal principles, landmark cases, and statutes clearly. "
                    "Use the IRAC framework (Issue, Rule, Application, Conclusion).",
        "Practice": "Provide problem scenarios requiring legal analysis with model "
                    "IRAC answers and note on jurisdiction where relevant.",
        "Test":     "Present legal problem questions or essay prompts ONLY. "
                    "Do NOT include model answers, case citations, or analysis.",
    },
    "Medicine": {
        "Learn":    "Explain anatomy, physiology, or clinical concepts with mechanisms, "
                    "pathophysiology, and real clinical presentations.",
        "Practice": "Provide clinical vignette exercises with full diagnostic reasoning, "
                    "differential diagnoses, and management plans.",
        "Test":     "Present clinical vignettes or MCQ stems ONLY. "
                    "Do NOT include diagnoses, reasoning, or management answers.",
    },
    "Psychology": {
        "Learn":    "Explain psychological theories, key studies, and real-world "
                    "applications with evaluation of strengths and limitations.",
        "Practice": "Provide essay-plan or study-evaluation exercises with model "
                    "answers demonstrating PEEL structure.",
        "Test":     "Pose psychology essay or short-answer questions ONLY. "
                    "Do NOT include model answers or study details.",
    },
}

VALID_SUBJECTS = set(_SUBJECT_MODE_INSTRUCTIONS.keys())

# Depth modifiers apply to ALL subjects uniformly
_DEPTH_INSTRUCTIONS: dict[str, str] = {
    "Beginner":  "Use simple, jargon-free language. Assume no prior knowledge. "
                 "Include everyday analogies and small, accessible examples.",
    "Exam":      "Use structured, moderately detailed explanations suited to exam "
                 "preparation. Balance clarity with academic rigour and use precise terminology.",
    "Advanced":  "Use precise technical language and provide deep, challenging content. "
                 "Include edge cases, nuances, competing theories, and expert-level detail.",
}


def build_learning_context(subject: str, mode: str, depth: str) -> str:
    """
    Return an instruction block prepended to every slide prompt when the user
    picks Subject + Mode + Depth.  Returns "" if any value is missing/invalid
    so the default university-professor prompt is used unchanged.
    All three controls must be set for the optimised prompt to activate.
    """
    subject = (subject or "").strip()
    mode    = (mode    or "").strip().title()
    depth   = (depth   or "").strip().title()

    # Normalise subject capitalisation against known keys
    subject_match = next(
        (s for s in VALID_SUBJECTS if s.lower() == subject.lower()), ""
    )

    if not subject_match or mode not in VALID_MODES or depth not in VALID_DEPTHS:
        # Incomplete selection → fall through to default professor prompt (no change)
        return ""

    mode_text  = _SUBJECT_MODE_INSTRUCTIONS[subject_match][mode]
    depth_text = _DEPTH_INSTRUCTIONS[depth]

    return (
        f"LEARNING CONTEXT — Subject: {subject_match} | Mode: {mode} | Depth: {depth}\n"
        f"Mode instruction: {mode_text}\n"
        f"Depth instruction: {depth_text}\n"
    )


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
# Slide generation — 1 slide per API call (Fix #1 + #2)
# ---------------------------------------------------------------------------

TOTAL_SLIDES = 12   # target number of slides per session

# Fix #2 — explicit role for every slide position in the 12-slide deck.
# Each role tells the model exactly what job this slide must do, so every
# slide is distinct and purposeful (no more generic, repetitive output).
_DECK_ROLES: list[tuple[str, str]] = [
    ("Introduction & Definition",
     "Define the topic clearly. Explain what it is, where it comes from, and why it matters. "
     "Include the core vocabulary a learner must know before going further."),
    ("Historical Background & Origin",
     "Cover the history, origin, and key milestones of the topic. Who developed it, when, and why? "
     "Mention specific names, dates, or events."),
    ("Core Principles & Mechanisms",
     "Explain the fundamental principles, rules, or laws that govern the topic. "
     "Focus on HOW it works at a mechanistic level."),
    ("Key Components & Structure",
     "Break down the major parts, components, layers, or categories. "
     "Use precise terminology and explain what each component does."),
    ("Types, Variants & Classification",
     "Describe the main types, variants, or sub-categories of the topic. "
     "Explain the distinguishing features and when each variant is used."),
    ("Step-by-Step Process or Workflow",
     "Walk through the main process, algorithm, lifecycle, or workflow step by step. "
     "Be concrete — use numbered or ordered logic."),
    ("Real-World Applications & Use Cases",
     "Give specific, concrete real-world examples and applications. "
     "Mention industries, companies, products, or specific scenarios with names and numbers."),
    ("Advantages, Strengths & Benefits",
     "Detail the key advantages, strengths, and benefits. "
     "Support each claim with a reason or a quantitative fact where possible."),
    ("Limitations, Challenges & Criticisms",
     "Explain the key limitations, drawbacks, trade-offs, and criticisms. "
     "Be specific — vague negatives are not useful."),
    ("Comparison with Related Concepts",
     "Compare and contrast the topic with closely related concepts, alternatives, or competing approaches. "
     "Highlight what makes each distinct and when to prefer one over the other."),
    ("Current Trends, Research & Future Directions",
     "Cover the latest developments, active research areas, and where the field is heading. "
     "Mention specific technologies, papers, or events from recent years."),
    ("Summary, Key Takeaways & Next Steps",
     "Synthesise the entire topic into the most important insights a learner should remember. "
     "End with practical next steps or resources for deeper study."),
]


def _build_single_slide_prompt(
    topic: str,
    slide_number: int,          # 1-based position in the deck
    role_title: str,
    role_instruction: str,
    math_topic: bool,
    difficulty_hint: str,
    depth_note: str,
    point_length_rule: str,
    learning_context: str = "",
) -> tuple[str, int]:
    """
    Build a prompt that requests exactly ONE slide.
    Single-slide requests produce tiny JSON payloads (~300-500 tokens),
    making JSON truncation and corruption impossible.
    Returns (prompt_str, max_tokens).
    """
    position_ctx = (
        f"You are building slide {slide_number} of {TOTAL_SLIDES} "
        f"in a deck about: {topic}\n"
        f"This slide's role: {role_title}\n"
        f"What this slide must cover: {role_instruction}\n"
    )

    # Prepend structured learning context if provided
    ctx_block = f"{learning_context}\n" if learning_context else ""

    if math_topic:
        prompt = (
            "You are an expert mathematics professor creating one rigorous, exam-quality slide.\n\n"
            f"{ctx_block}"
            f"{position_ctx}"
            f"{difficulty_hint}{depth_note}\n\n"
            "Return ONLY a valid JSON object (a single slide). No markdown fences. No extra text.\n\n"
            "Required format:\n"
            "{\n"
            "  \"title\": \"Slide title that reflects the role\",\n"
            "  \"points\": [\n"
            "    {\n"
            "      \"text\": \"One sentence: state the rule or theorem precisely. No filler words.\",\n"
            "      \"source_title\": \"Real credible source (e.g. Stewart Calculus, MIT OCW, Wikipedia)\",\n"
            "      \"source_url\": \"Direct URL or empty string\",\n"
            "      \"inline_latex\": \"\\\\frac{-b \\\\pm \\\\sqrt{b^2-4ac}}{2a}\",\n"
            "      \"inline_label\": \"Formula name (e.g. Quadratic Formula)\",\n"
            "      \"sub_steps\": [\n"
            "        \"Step 1 — Identify: $$a = ...,\\\\; b = ...,\\\\; c = ...$$\",\n"
            "        \"Step 2 — Substitute: $$x = \\\\frac{-b \\\\pm \\\\sqrt{b^2-4ac}}{2a}$$\",\n"
            "        \"Step 3 — Simplify: $$x = ...$$\",\n"
            "        \"Step 4 — Verify: $$...$$\"\n"
            "      ]\n"
            "    }\n"
            "  ],\n"
            "  \"worked_example\": {\n"
            "    \"problem\": \"State the problem using LaTeX, e.g. \\\"Solve $$x^2 - 5x + 6 = 0$$\\\"\",\n"
            "    \"steps\": [\n"
            "      \"Step 1 — Setup: $$a=1,\\\\; b=-5,\\\\; c=6$$\",\n"
            "      \"Step 2 — Substitute: $$x = \\\\frac{5 \\\\pm \\\\sqrt{25-24}}{2}$$\",\n"
            "      \"Step 3 — Simplify: $$x = \\\\frac{5 \\\\pm 1}{2}$$\",\n"
            "      \"Step 4 — Solve: $$x = 3 \\\\text{ or } x = 2$$\"\n"
            "    ],\n"
            "    \"answer\": \"$$x = 2$$ or $$x = 3$$\"\n"
            "  }\n"
            "}\n\n"
            "STRICT RULES:\n"
            "- Exactly 4 points\n"
            f"{point_length_rule}\n"
            "- Every point MUST be an object with: text, source_title, source_url, inline_latex, inline_label, sub_steps\n"
            "- text: ONE sentence — state the mathematical rule, theorem, or definition precisely. "
            "  NO long explanations. May contain inline $...$ LaTeX.\n"
            "- inline_latex: the key formula as a display LaTeX expression (double backslashes: \\\\frac, \\\\sqrt, \\\\int, \\\\pm)\n"
            "- inline_label: short name for the formula (e.g. 'Chain Rule', 'Euler's Formula')\n"
            "- sub_steps: exactly 4 steps, EACH formatted as 'Step N — <Action>: $$<LaTeX expression>$$'. "
            "  ZERO prose sentences — every step is a concrete mathematical operation in LaTeX. "
            "  Valid actions: Identify, Set up, Substitute, Expand, Factor, Differentiate, Integrate, "
            "  Simplify, Solve, Apply, Verify, Evaluate, Rearrange.\n"
            "- worked_example.problem: state clearly with LaTeX math\n"
            "- worked_example.steps: 4–5 steps each 'Step N — <Action>: $$<LaTeX>$$' — "
            "  every single intermediate result MUST be in $$...$$\n"
            "- worked_example.answer: final result in LaTeX e.g. '$$x = 2$$ or $$x = 3$$'\n"
            "- Do NOT use bullet points, prose explanations, or long sentences in sub_steps\n"
            "- Do NOT repeat formulas or examples from other slides\n"
            "- Output ONLY the JSON object — no array wrapper, no prose\n"
        )
        max_tok = 1600
    else:
        prompt = (
            "You are an expert university professor creating one deeply detailed, lecture-quality slide.\n\n"
            f"{ctx_block}"
            f"{position_ctx}"
            f"{difficulty_hint}{depth_note}\n\n"
            "Return ONLY a valid JSON object (a single slide). No markdown fences. No extra text.\n\n"
            "Required format:\n"
            "{\n"
            "  \"title\": \"Slide title that reflects the role\",\n"
            "  \"points\": [\n"
            "    {\n"
            "      \"text\": \"Detailed, specific explanation sentence\",\n"
            "      \"source_title\": \"Real credible source title\",\n"
            "      \"source_url\": \"Direct URL or empty string\"\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "STRICT RULES:\n"
            "- Exactly 4 bullet points\n"
            f"{point_length_rule}\n"
            "- Every point MUST be an object with keys: text, source_title, source_url\n"
            "- Every point must explain the 'what' and briefly the 'why'\n"
            "- NO vague lines like 'It is important' or 'Has many applications'\n"
            "- Every point MUST include at least one of: a definition, a mechanism, a real example "
            "with a name or number, a comparison, a cause-effect relationship, or a formula\n"
            "- Do NOT repeat information from other slides\n"
            "- Output ONLY the JSON object — no array wrapper, no prose\n"
        )
        max_tok = 1200

    return prompt, max_tok


def _generate_single_slide(
    topic: str,
    slide_number: int,
    role_title: str,
    role_instruction: str,
    math_topic: bool,
    difficulty_hint: str,
    depth_note: str,
    point_length_rule: str,
    retry: bool = False,
    learning_context: str = "",
) -> dict | None:
    """
    Call the API for exactly one slide and return a validated slide dict (or None).
    """
    prompt, max_tok = _build_single_slide_prompt(
        topic, slide_number, role_title, role_instruction,
        math_topic, difficulty_hint, depth_note, point_length_rule,
        learning_context=learning_context,
    )

    raw = _call_groq(prompt, max_tokens=max_tok, use_fallback=retry)
    if not raw:
        if not retry:
            print(f"_generate_single_slide [{slide_number}]: API returned None, retrying with fallback.")
            return _generate_single_slide(
                topic, slide_number, role_title, role_instruction,
                math_topic, difficulty_hint, depth_note, point_length_rule,
                retry=True, learning_context=learning_context,
            )
        print(f"_generate_single_slide [{slide_number}]: giving up.")
        return None

    raw = strip_markdown_fences(raw)
    print(f"_generate_single_slide [{slide_number}]: preview {raw[:120]!r}")

    # Parse — expect a single JSON object
    obj = None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"_generate_single_slide [{slide_number}]: json.loads failed ({e}), trying extraction.")
        # Maybe model wrapped it in an array
        arr_str = extract_json_array(raw)
        if arr_str:
            try:
                arr = json.loads(arr_str)
                if isinstance(arr, list) and arr:
                    obj = arr[0]
            except json.JSONDecodeError:
                pass
        if obj is None:
            obj_str = extract_json_object(raw)
            if obj_str:
                try:
                    obj = json.loads(obj_str)
                except json.JSONDecodeError:
                    pass
        if obj is None:
            objs = _extract_all_objects(raw)
            if objs:
                obj = objs[0]

    if not isinstance(obj, dict):
        if not retry:
            return _generate_single_slide(
                topic, slide_number, role_title, role_instruction,
                math_topic, difficulty_hint, depth_note, point_length_rule,
                retry=True, learning_context=learning_context,
            )
        return None

    validated = validate_slides([obj])
    if not validated:
        if not retry:
            return _generate_single_slide(
                topic, slide_number, role_title, role_instruction,
                math_topic, difficulty_hint, depth_note, point_length_rule,
                retry=True, learning_context=learning_context,
            )
        return None

    print(f"_generate_single_slide [{slide_number}]: OK — '{validated[0]['title']}'")
    return validated[0]


def generate_slides(
    topic: str,
    learner_profile: dict | None = None,
    explanation_mode: str = "in_depth",
    compact_mode: bool = False,
    retry: bool = False,
    learning_context: str = "",
) -> list:
    explanation_mode = str(explanation_mode or "in_depth").strip().lower()
    if explanation_mode not in {"brief", "in_depth"}:
        explanation_mode = "in_depth"

    if explanation_mode == "brief":
        depth_note = (
            "\nNOTE: Keep explanations brief and easy to scan. "
            "Use concise wording and only essential detail."
        )
        point_length_rule = "- Every point text should be around 20 to 25 words"
    else:
        depth_note = (
            "\nNOTE: Explain in depth. Include richer context, clear reasoning, and concrete details."
        )
        point_length_rule = "- Every point text should be around 50 words (target range 45–55 words)"

    difficulty_hint = ""
    if learner_profile:
        hint = learner_profile.get("difficulty_hint", "normal")
        profile_summary = learner_profile.get("summary", "")
        if hint == "simplify":
            difficulty_hint = (
                "\nNOTE: This learner has been struggling. Use simpler language, "
                "more analogies, and concrete step-by-step examples. Avoid jargon. "
                f"Learner profile: {profile_summary}\n"
            )
        elif hint == "advanced":
            difficulty_hint = (
                "\nNOTE: This learner is advanced. Include more technical depth, "
                "edge cases, and challenging concepts. "
                f"Learner profile: {profile_summary}\n"
            )

    math_topic = is_math_topic(topic)

    # ── 1-slide-per-call generation ──────────────────────────────────────────
    # Each API call requests exactly 1 slide using an explicit role from
    # _DECK_ROLES. This guarantees:
    #   • JSON output is tiny (~300–500 tokens) — impossible to truncate/corrupt
    #   • Each slide has a distinct, purposeful job — no generic repetition
    all_slides: list = []
    for slide_idx, (role_title, role_instruction) in enumerate(_DECK_ROLES):
        slide_number = slide_idx + 1
        slide = _generate_single_slide(
            topic=topic,
            slide_number=slide_number,
            role_title=role_title,
            role_instruction=role_instruction,
            math_topic=math_topic,
            difficulty_hint=difficulty_hint,
            depth_note=depth_note,
            point_length_rule=point_length_rule,
            learning_context=learning_context,
        )
        if slide is not None:
            all_slides.append(slide)
        else:
            print(f"generate_slides: slide {slide_number} failed — skipping.")

        print(f"generate_slides: {len(all_slides)}/{slide_number} slides so far.")

    # ── Rescue: fill any missing slides via fallback model ───────────────────
    if len(all_slides) < TOTAL_SLIDES:
        missing = TOTAL_SLIDES - len(all_slides)
        print(f"generate_slides: {missing} slides missing — running rescue pass.")
        rescue = generate_slides_rescue(topic, explanation_mode)
        needed = TOTAL_SLIDES - len(all_slides)
        all_slides.extend(rescue[:needed])

    # Fix #3 — do NOT truncate points; audit only (normalize is now a no-op truncation-wise)
    if all_slides:
        all_slides = _normalize_point_word_lengths(all_slides, explanation_mode)

    print(f"generate_slides: final count = {len(all_slides)} slides.")
    return all_slides


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

    # Detect math slides by presence of inline_latex in any point
    is_math_slide = any(
        isinstance(p, dict) and p.get("inline_latex")
        for p in slide.get("points", [])
    )
    math_note = (
        "This is a mathematics slide. "
        "Questions and options may contain LaTeX expressions using $...$ for inline math "
        "and $$...$$ for display equations. "
        "Include at least one question that requires evaluating or identifying a formula. "
        "Options involving numbers or expressions must use LaTeX."
    ) if is_math_slide else ""

    prompt = f"""
You are a university professor writing a quiz based on this slide.

Slide Title: {slide['title']}
Slide Content:
{json.dumps(slide['points'], indent=2)}

{difficulty_note}
{math_note}

Generate exactly 3 multiple-choice questions that test understanding of this slide's content.

Return ONLY a valid JSON array. No extra text, no markdown.

Format:
[
  {{
    "question": "Clear, specific question? (use $LaTeX$ for any math expressions)",
    "options": ["Option A (use $LaTeX$ for math)", "Option B", "Option C", "Option D"],
    "correct": 0,
    "explanation": "Concise explanation referencing the slide. Use $LaTeX$ for any math."
  }}
]

RULES:
- All 4 options must be plausible (no obviously wrong options)
- "correct" is the 0-based index of the correct option
- Questions must be directly answerable from the slide content
- Vary question types: recall, application, conceptual
- Keep options concise (under 20 words each)
- For math slides: at least one question must test formula recall or application
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
            "- sub_steps: 3-5 steps, each following \'Step N — <Label>: <explanation with LaTeX>\' format; "
            "every number or expression must use $...$ or $$...$$ LaTeX delimiters\n"
            "- worked_example: a NEW simpler numeric problem, 4-6 steps with more arithmetic detail; "
            "every step must follow \'Step N — <Label>: <LaTeX working>\' and use $$...$$ for calculations\n"
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
    For math topics, includes key formulas in LaTeX.
    """
    slide_titles = [s["title"] for s in slides]
    weak_titles = [p["slide_title"] for p in performance if p["quiz_score"] < 0.6 or not p["understood"]]

    weak_note = ""
    if weak_titles:
        weak_note = f"\nThe learner struggled with: {', '.join(weak_titles[:5])}. Briefly flag those areas for extra study."

    math_summary_note = ""
    if is_math_topic(topic):
        # Collect any inline_latex from slides to give the LLM context
        formulas = []
        for s in slides:
            for p in s.get("points", []):
                if isinstance(p, dict) and p.get("inline_latex"):
                    lbl = p.get("inline_label", "")
                    lat = p["inline_latex"]
                    formulas.append(f"{lbl}: $${lat}$$" if lbl else f"$${lat}$$")
        if formulas:
            math_summary_note = (
                "\nKey formulas covered (include these in your summary using LaTeX $...$ notation):\n"
                + "\n".join(formulas[:8])
            )
        math_summary_note += (
            "\nUse LaTeX for ALL mathematical expressions in your summary: "
            "inline with $expression$ and display equations with $$expression$$."
        )

    prompt = f"""
You are an expert tutor. A student just finished a learning session on "{topic}".

The session covered these slides:
{json.dumps(slide_titles, indent=2)}
{weak_note}
{math_summary_note}

Write a clear, motivating end-of-session summary for the student. Include:
1. A 2-3 sentence overview of what was learned
2. 5-7 key takeaways as concise bullet points (include key formulas in LaTeX for math topics)
3. A short "What to explore next" section with 2-3 related topics
4. An encouraging closing line

Keep the tone friendly and academic. Plain text — no JSON, no markdown headers.
"""

    return _call_groq(prompt, max_tokens=900)


# ---------------------------------------------------------------------------
# NEW: AI Tutor chat (contextual Q&A on current slide)
# ---------------------------------------------------------------------------

# Extended keyword set for question-level math detection (broader than
# is_math_topic which targets topic names; this targets student questions).
_MATH_QUESTION_KEYWORDS = MATH_KEYWORDS | {
    "solve", "calculate", "compute", "prove", "simplify", "differentiate",
    "integrate", "find", "evaluate", "expand", "factor", "factorise",
    "determinant", "eigenvalue", "eigenvector", "sum", "product",
    "formula", "expression", "simplify", "roots", "zeros",
}


def is_mathematical_question(question: str) -> bool:
    """Return True if the student's question is asking for a mathematical solution."""
    lower = question.lower()
    return any(kw in lower for kw in _MATH_QUESTION_KEYWORDS)


def generate_math_solution(question: str, slide_points: list) -> dict | None:
    """
    Return a structured, step-by-step mathematical solution plus a plain-English explanation.
    Returns dict: { "steps": str, "explanation": str }
    """
    context = ""
    if slide_points:
        context_lines = [
            (p["text"] if isinstance(p, dict) else str(p))
            for p in slide_points[:3]
        ]
        context = "Relevant slide context:\n" + "\n".join(context_lines) + "\n\n"

    system = (
        "You are a precise mathematics tutor. "
        "Solve the problem completely, showing every intermediate calculation. "
        "Format your entire response as clearly numbered steps — no bullet points, "
        "no dashes, no markdown headers or symbols. "
        "Each step must be on its own line labelled exactly "
        "\'Step N — <action label>: <explanation>\' where N is the step number. "
        "Use LaTeX for ALL mathematical expressions: "
        "inline with $expression$ and display equations on their own line with $$expression$$. "
        "Every number, variable, formula, and equation must be wrapped in LaTeX. "
        "Explain what you are doing and why at each step. "
        "End with a line labelled \'Step N — Answer: $$result$$\' containing the final result "
        "with units where applicable. "
        "Never use bullet points, dashes, or plain text for mathematical expressions."
    )

    prompt = (
        f"{context}"
        f"Solve the following problem completely, showing every step:\n\n"
        f"{question}\n\n"
        "Use this exact format for every step:\n"
        "Step 1 — <action label>: <explanation with LaTeX e.g. Substituting $a=2$ into $$x = \\frac{-b}{2a}$$>\n"
        "Step 2 — <action label>: ...\n"
        "...\n"
        "Step N — Answer: $$<final result>$$"
    )

    raw = _call_groq(prompt, max_tokens=1200, system=system)
    if raw:
        raw = strip_markdown_fences(raw)
    else:
        return None

    # Now generate plain-English explanation
    explain_system = (
        "You are a friendly math teacher. "
        "Given a step-by-step mathematical solution, write a concise plain-English explanation "
        "of the overall approach and key ideas used. "
        "Do NOT repeat the steps. Do NOT use LaTeX or symbols. "
        "Write 3-5 sentences in simple, clear language that a student could easily understand. "
        "Focus on WHY the method works, not HOW each step was computed."
    )
    explain_prompt = (
        f"Problem: {question}\n\n"
        f"Solution steps:\n{raw}\n\n"
        "Now write a plain-English explanation of the overall approach and key ideas used to solve this problem."
    )
    explanation = _call_groq(explain_prompt, max_tokens=400, system=explain_system)
    if explanation:
        explanation = strip_markdown_fences(explanation)

    return {"steps": raw, "explanation": explanation or ""}


def answer_student_question(question: str, slide: dict, topic: str, profile: dict) -> str | None:
    # Mathematical questions bypass the slide-context prompt entirely and go
    # straight to a dedicated solver that returns structured step-by-step output.
    if is_mathematical_question(question):
        result = generate_math_solution(question, slide.get("points", []))
        if result is None:
            return None
        # Return combined string for the chat endpoint (steps + explanation)
        parts = [result["steps"]]
        if result.get("explanation"):
            parts.append("\n\n---EXPLANATION---\n" + result["explanation"])
        return "\n".join(parts)

    difficulty = profile.get("difficulty_hint", "normal")
    tone_note = {
        "simplify": "Use simple, jargon-free language with everyday analogies.",
        "advanced": "Use precise technical language and include deeper details.",
        "normal": "Use clear, academic language suitable for an undergraduate student.",
    }.get(difficulty, "")

    # Detect if the current slide is math-heavy so tutor uses LaTeX
    slide_is_math = any(
        isinstance(p, dict) and p.get("inline_latex")
        for p in slide.get("points", [])
    ) or is_math_topic(topic)
    latex_note = (
        "Use LaTeX for ALL mathematical expressions in your answer: "
        "inline with $expression$ and display equations with $$expression$$. "
        "Never write equations or formulas in plain text."
    ) if slide_is_math else ""

    system = (
        f"You are a friendly, knowledgeable university tutor teaching '{topic}'. "
        f"{tone_note} {latex_note} "
        "Answer only questions related to the course material. "
        "If the question is off-topic, gently redirect the student. "
        "Keep your answer focused and under 200 words."
    )

    prompt = f"""Current slide the student is on:
Title: {slide['title']}
Content:
{json.dumps(slide['points'], indent=2)}

Student's question: {question}
"""

    return _call_groq(prompt, max_tokens=600, system=system)


# ---------------------------------------------------------------------------
# Visual generation
# ---------------------------------------------------------------------------

def generate_visual_for_slide(slide: dict, retry: bool = False) -> dict | None:
    """
    Decide the best visual type for a slide and return structured data
    that the frontend can render as an SVG/HTML diagram.
    """
    prompt = f"""Return ONLY valid JSON object (no markdown).
Create one educational diagram for this slide.

Slide title: {slide['title']}
Slide points:
{json.dumps(slide.get('points', [])[:4], ensure_ascii=False)}

Use one type from: flowchart, cycle, comparison, timeline, pyramid, mindmap.
Keep structure compact (4 to 6 elements).

Output format:
{{
  "type": "mindmap",
  "title": "Short title",
  "data": {{
    "center": "Main idea",
    "branches": [
      {{"label": "Branch label", "items": ["item one", "item two"]}}
    ]
  }}
}}
If you choose another type, keep the same top-level keys and valid data for that type.
"""

    raw = _call_groq(prompt, max_tokens=650, use_fallback=True)
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


def build_fallback_visual(slide: dict) -> dict:
    """
    Deterministic fallback visual so frontend always receives a renderable diagram.
    """
    title = str(slide.get("title", "Slide Concept")).strip() or "Slide Concept"
    raw_points = slide.get("points", [])
    texts = []
    for p in raw_points:
        t = str(p.get("text", "") if isinstance(p, dict) else p).strip()
        if t:
            texts.append(t)
    if not texts:
        texts = [title]

    branches = []
    for i, t in enumerate(texts[:5]):
        words = t.split()
        label = " ".join(words[:4]) or f"Idea {i + 1}"
        items = []
        if len(words) > 4:
            items.append(" ".join(words[4:10]))
        if len(words) > 10:
            items.append(" ".join(words[10:16]))
        if not items:
            items = ["Key point from this slide", "Related concept to review"]
        branches.append({"label": label[:48], "items": [it[:72] for it in items[:2]]})

    return {
        "type": "mindmap",
        "title": title[:48],
        "data": {
            "center": title[:48],
            "branches": branches[:5],
        },
    }


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
@app.route("/health.php")
def health():
    """Simple health check for deployment monitoring."""
    return jsonify({"status": "ok", "model": MODEL_NAME, "sessions": len(sessions)})


@app.route("/generate", methods=["POST"])
@app.route("/generate.php", methods=["POST"])
@limiter.limit("10 per minute")
def generate():
    """
    Start a new learning session.
    Body: {
      "topic": "...",
      "explanation_mode": "brief" | "in_depth",
      "subject": "Math" | "English" | "Coding",   (optional)
      "mode":    "Learn" | "Practice" | "Test",    (optional)
      "depth":   "Beginner" | "Exam" | "Advanced"  (optional)
    }
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

    # Structured learning controls (optional)
    subject = str(data.get("subject", "")).strip()
    mode    = str(data.get("mode",    "")).strip()
    depth   = str(data.get("depth",   "")).strip()
    learning_context = build_learning_context(subject, mode, depth)

    slides = []
    for attempt in range(3):
        slides = generate_slides(topic, explanation_mode=explanation_mode,
                                 compact_mode=False, learning_context=learning_context)
        if slides:
            break
        print(f"[/generate] Attempt {attempt + 1}/3 failed for topic='{topic}'. Retrying...")
        time.sleep(1.2 * (attempt + 1))
    # If normal-size generation still fails, automatically switch to compact payload.
    if not slides:
        print(f"[/generate] Switching to compact generation for topic='{topic}'.")
        for attempt in range(3):
            slides = generate_slides(topic, explanation_mode=explanation_mode,
                                     compact_mode=True, learning_context=learning_context)
            if slides:
                break
            print(f"[/generate] Compact attempt {attempt + 1}/3 failed for topic='{topic}'. Retrying...")
            time.sleep(1.5 * (attempt + 1))
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
        "quiz_cache": {},
        "visual_cache": {},
        "sources_cache": {},
        "notes": {},
        "explanation_mode": explanation_mode,
        "subject": subject,
        "mode": mode,
        "depth": depth,
        "xp": 0,
        "streak_slides": 0,
    }

    return jsonify({"session_id": session_id, "slides": slides, "explanation_mode": explanation_mode})


@app.route("/quiz", methods=["POST"])
@app.route("/quiz.php", methods=["POST"])
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
@app.route("/feedback.php", methods=["POST"])
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
@app.route("/reteach.php", methods=["POST"])
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


def _stats_payload(session_id: str):
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


@app.route("/stats/<session_id>", methods=["GET"])
def stats(session_id):
    return _stats_payload(session_id)


@app.route("/stats.php", methods=["GET"])
def stats_php():
    session_id = str(request.args.get("session_id", "")).strip()
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400
    return _stats_payload(session_id)


@app.route("/visual", methods=["POST"])
@app.route("/visual.php", methods=["POST"])
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
        time.sleep(0.8)
        result = generate_visual_for_slide(slides[slide_index], retry=True)
    if not result:
        result = build_fallback_visual(slides[slide_index])

    session["visual_cache"][cache_key] = result
    return jsonify({"visual": result})


# ---------------------------------------------------------------------------
# NEW: AI Tutor chat endpoint
# ---------------------------------------------------------------------------

@app.route("/chat", methods=["POST"])
@app.route("/chat.php", methods=["POST"])
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

    answer = answer.strip()
    explanation = ""
    if "---EXPLANATION---" in answer:
        parts = answer.split("---EXPLANATION---", 1)
        answer = parts[0].strip()
        explanation = parts[1].strip()

    return jsonify({"answer": answer, "explanation": explanation})


# ---------------------------------------------------------------------------
# Math Tutor — helpers
# ---------------------------------------------------------------------------

DIRECT_SOLVE_TRIGGERS = {
    "solve", "simplify", "calculate", "compute", "evaluate", "integrate",
    "differentiate", "derive", "factor", "expand", "find", "prove",
    "what is", "what's", "how much", "how many",
}

def _is_direct_solve(question: str) -> bool:
    """Return True if the user wants a full solution right now."""
    q = question.lower()
    return any(kw in q for kw in DIRECT_SOLVE_TRIGGERS)


def _math_direct_solve(question: str, difficulty: str) -> str | None:
    diff_instruction = {
        "Beginner":  "Show every micro-step. Write each arithmetic operation on its own line.",
        "Exam":      "Show all working. Every intermediate result must appear as a display equation.",
        "Advanced":  "Use rigorous notation, state domain restrictions, and justify each transformation.",
    }.get(difficulty, "Show every intermediate result as a display equation.")

    system = (
        "You are a mathematics solution engine. "
        "Your ONLY job is to output a clean, structured, step-by-step mathematical solution. "
        "STRICT OUTPUT RULES — violating any rule is an error:\n"
        "1. Every step MUST be on its own line in EXACTLY this format: "
        "   'Step N — <Short Action Label>: $$<LaTeX expression>$$'\n"
        "2. The action label must be a short math verb: e.g. 'Expand', 'Factor', 'Substitute', "
        "   'Differentiate', 'Simplify', 'Integrate', 'Apply quadratic formula', 'Solve for x'.\n"
        "3. EVERY mathematical expression — equations, formulas, numbers, variables — MUST be "
        "   in LaTeX using $...$ (inline) or $$...$$ (display). NO plain-text math.\n"
        "4. The final step MUST be: 'Step N — Answer: $$<result>$$'\n"
        "5. ZERO prose, ZERO explanatory sentences, ZERO bullet points, ZERO markdown headers.\n"
        "6. If a brief clarification of what was done is needed, append it in parentheses "
        "   AFTER the LaTeX on the SAME line only — maximum 6 words.\n"
        f"7. {diff_instruction}\n"
        "Output ONLY the numbered steps. Nothing before Step 1. Nothing after the Answer step."
    )
    prompt = (
        f"Solve completely:\n\n{question}\n\n"
        "Format each line EXACTLY as:\n"
        "Step 1 — <Action>: $$<LaTeX>$$\n"
        "Step 2 — <Action>: $$<LaTeX>$$\n"
        "...\n"
        "Step N — Answer: $$<final result>$$\n\n"
        "Every equation must be in LaTeX. No prose. No bullets. No headers."
    )
    raw = _call_groq(prompt, max_tokens=1800, system=system)
    if raw:
        raw = strip_markdown_fences(raw)
    return raw


def _generate_math_problem(topic: str, difficulty: str) -> dict | None:
    """
    Ask the model to produce a structured math problem JSON:
    { problem, steps[], final_answer, similar_problems[] }
    """
    diff_note = {
        "Beginner": "appropriate for a beginner (arithmetic / basic algebra)",
        "Exam":     "exam-level difficulty (multi-step, real techniques required)",
        "Advanced": "advanced (proof-based or multi-concept)",
    }.get(difficulty, "intermediate difficulty")

    step_count = {"Beginner": "3–4", "Exam": "4–6", "Advanced": "5–7"}.get(difficulty, "4–5")

    system = (
        "You are a math tutor. Return ONLY a valid JSON object. "
        "No markdown fences. No extra text."
    )
    prompt = f"""Generate a math problem on the topic: "{topic}".
Difficulty: {diff_note}.

Return ONLY this JSON shape:
{{
  "problem": "Full problem statement. Use LaTeX for ALL math: inline $expression$ or display $$expression$$. Every number, variable, and formula must be in LaTeX.",
  "steps": [
    "Step 1 — Setup: identify given values $a = ...$, $b = ...$",
    "Step 2 — Apply: substitute into $$formula$$",
    "Step 3 — Compute: $$intermediate = result$$",
    "Step N — Answer: $$final = value$$"
  ],
  "final_answer": "$$answer$$ with units where applicable",
  "similar_problems": [
    "Similar problem 1 (use LaTeX for any math)",
    "Similar problem 2"
  ]
}}

Rules:
- steps[] must be {step_count} items, each a complete verifiable calculation step
- Every step must follow the format "Step N — <Label>: <working with LaTeX>"
- Every mathematical expression in steps, problem, and final_answer MUST use LaTeX ($...$ or $$...$$)
- final_answer must be precise, correct, and in LaTeX
- similar_problems must be 1–2 items at the same difficulty
- Use correct math — verify every calculation before writing
- Do NOT use plain text for any equation, variable, or number"""

    raw = _call_groq(prompt, max_tokens=1200, system=system)
    if not raw:
        return None
    raw = strip_markdown_fences(raw)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        s = extract_json_object(raw)
        if not s:
            return None
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return None

    if not isinstance(obj, dict):
        return None
    if not obj.get("problem") or not isinstance(obj.get("steps"), list):
        return None
    return obj


def _validate_math_step(user_answer: str, expected_step: str, problem: str, topic: str) -> dict:
    """
    Return { correct: bool, hint: str|None }
    Asks the model to judge correctness tolerantly (accept equivalent forms).
    Hint uses LaTeX so the frontend can render it properly.
    """
    system = "You are a strict but fair math evaluator. Return ONLY valid JSON. No prose outside JSON."
    prompt = f"""Problem: {problem}
Expected step: {expected_step}
Student's answer: {user_answer}

Is the student's answer mathematically equivalent to or correctly performing the expected step?
Be tolerant of notation differences (e.g. "x=2" vs "$x = 2$") but strict about mathematical correctness.

If incorrect, write a hint that:
1. Points out the specific error in one sentence
2. Shows the correct intermediate expression using LaTeX (e.g. "Try $$x = \\frac{{-b}}{{2a}}$$")
3. Never gives away the full answer

Return ONLY valid JSON:
{{"correct": true/false, "hint": "hint with LaTeX if incorrect, null if correct"}}"""

    raw = _call_groq(prompt, max_tokens=300, system=system)
    if not raw:
        return {"correct": False, "hint": "Could not validate. Please try again."}
    raw = strip_markdown_fences(raw)
    try:
        obj = json.loads(raw)
        return {
            "correct": bool(obj.get("correct", False)),
            "hint": obj.get("hint") or None,
        }
    except Exception:
        s = extract_json_object(raw)
        if s:
            try:
                obj = json.loads(s)
                return {"correct": bool(obj.get("correct", False)), "hint": obj.get("hint")}
            except Exception:
                pass
        return {"correct": False, "hint": "Could not validate. Please try again."}


# ---------------------------------------------------------------------------
# Math Tutor — /math_tutor endpoint
# ---------------------------------------------------------------------------

@app.route("/math_tutor", methods=["POST"])
@app.route("/math_tutor.php", methods=["POST"])
@limiter.limit("30 per minute")
def math_tutor():
    """
    Unified Math Tutor endpoint.

    Body:
      { "session_id": "...", "message": "...", "action": "start"|"answer"|"solve" }

    action="start"  → generate a new problem and return first step prompt
    action="answer" → validate user's step answer
    action="solve"  → direct solve mode (full solution)

    Session math_state shape:
      {
        current_problem: str,
        steps: [],
        current_step_index: int,
        final_answer: str,
        similar_problems: [],
        completed: bool
      }

    Returns:
      {
        type: "problem"|"step_feedback"|"complete"|"solution"|"error",
        problem: str,           (when type=problem)
        step_prompt: str,       (current step question)
        step_index: int,
        total_steps: int,
        correct: bool,          (on step_feedback)
        hint: str|None,
        solution_steps: [],     (on complete or direct solve)
        final_answer: str,
        similar_problems: [],
        full_solution: str      (on type=solution)
      }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    session_id = data.get("session_id", "")
    message    = str(data.get("message", "")).strip()
    action     = str(data.get("action", "start")).strip()

    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    # Only operates in Math subject sessions
    if session.get("subject", "").lower() != "math":
        return jsonify({"error": "Math tutor only available for Math subject sessions"}), 400

    topic      = session["topic"]
    difficulty = session.get("depth", "Exam")

    # ── DIRECT SOLVE MODE ─────────────────────────────────────
    if action == "solve" or (action == "start" and _is_direct_solve(message) and message):
        solution = _math_direct_solve(message or topic, difficulty)
        if not solution:
            return jsonify({"error": "Could not generate solution."}), 500
        # Generate plain-English explanation after step-by-step solution
        explain_system = (
            "You are a friendly math teacher. Given a step-by-step solution, write a concise "
            "3-5 sentence plain-English explanation of the overall approach and key ideas used. "
            "Do NOT repeat the steps. Do NOT use LaTeX or symbols. Use simple language."
        )
        explain_prompt = (
            f"Problem: {message or topic}\n\nSolution steps:\n{solution[:2000]}\n\n"
            "Summarise the overall approach and key ideas in plain English (3-5 sentences)."
        )
        explanation = _call_groq(explain_prompt, max_tokens=350, system=explain_system) or ""
        if explanation:
            explanation = strip_markdown_fences(explanation)
        return jsonify({"type": "solution", "full_solution": solution.strip(), "explanation": explanation})

    # ── START / GENERATE PROBLEM ──────────────────────────────
    if action == "start":
        prob = _generate_math_problem(topic, difficulty)
        if not prob:
            return jsonify({"error": "Could not generate problem. Please try again."}), 500

        session["math_state"] = {
            "current_problem": prob["problem"],
            "steps": prob["steps"],
            "current_step_index": 0,
            "final_answer": prob.get("final_answer", ""),
            "similar_problems": prob.get("similar_problems", []),
            "completed": False,
        }
        ms = session["math_state"]
        return jsonify({
            "type": "problem",
            "problem": ms["current_problem"],
            "step_prompt": f"Step 1: {ms['steps'][0]}",
            "step_index": 0,
            "total_steps": len(ms["steps"]),
            "final_answer": "",
            "similar_problems": [],
        })

    # ── VALIDATE STEP ANSWER ──────────────────────────────────
    if action == "answer":
        ms = session.get("math_state")
        if not ms or ms.get("completed"):
            return jsonify({"error": "No active problem. Use action=start."}), 400

        idx      = ms["current_step_index"]
        expected = ms["steps"][idx]
        result   = _validate_math_step(message, expected, ms["current_problem"], topic)

        if not result["correct"]:
            return jsonify({
                "type": "step_feedback",
                "correct": False,
                "hint": result["hint"],
                "step_index": idx,
                "total_steps": len(ms["steps"]),
                "step_prompt": f"Step {idx + 1}: Try again — {result['hint'] or 'Review your work.'}",
            })

        # Correct — advance
        idx += 1
        ms["current_step_index"] = idx

        if idx >= len(ms["steps"]):
            # Completed
            ms["completed"] = True
            return jsonify({
                "type": "complete",
                "correct": True,
                "problem": ms["current_problem"],
                "solution_steps": ms["steps"],
                "final_answer": ms["final_answer"],
                "similar_problems": ms["similar_problems"],
                "step_index": idx,
                "total_steps": len(ms["steps"]),
            })

        return jsonify({
            "type": "step_feedback",
            "correct": True,
            "hint": None,
            "step_index": idx,
            "total_steps": len(ms["steps"]),
            "step_prompt": f"Step {idx + 1}: {ms['steps'][idx]}",
        })

    return jsonify({"error": f"Unknown action: {action}"}), 400


# ---------------------------------------------------------------------------
# NEW: End-of-session topic summary
# ---------------------------------------------------------------------------

@app.route("/summary", methods=["POST"])
@app.route("/summary.php", methods=["POST"])
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
@app.route("/note.php", methods=["POST"])
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
@app.route("/sources.php", methods=["POST"])
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
@app.route("/generate_ppt.php", methods=["POST"])
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
# NEW: File / Image analysis endpoint
# ---------------------------------------------------------------------------

@app.route("/analyse", methods=["POST"])
@app.route("/analyse.php", methods=["POST"])
@limiter.limit("10 per minute")
def analyse():
    """
    Analyse an uploaded image or text file using the LLM.
    Accepts multipart/form-data with:
      - file: the uploaded file (image: png/jpg/webp/gif, or text: pdf/txt/csv)
      - question: optional user question about the file (default: "Analyse this")
      - session_id: optional, used to add math context

    For images: sends base64-encoded image to Groq vision model.
    For text files: extracts text and sends as prompt context.

    Returns: { "analysis": "...", "explanation": "..." }
    """
    import base64
    import mimetypes

    question = request.form.get("question", "").strip() or "Analyse this in detail."
    session_id = request.form.get("session_id", "").strip()

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded = request.files["file"]
    filename = uploaded.filename or ""
    mime_type = uploaded.mimetype or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    raw_bytes = uploaded.read()

    if not raw_bytes:
        return jsonify({"error": "Uploaded file is empty"}), 400

    MAX_SIZE = 10 * 1024 * 1024  # 10 MB
    if len(raw_bytes) > MAX_SIZE:
        return jsonify({"error": "File too large (max 10 MB)"}), 400

    IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}
    TEXT_TYPES  = {"text/plain", "text/csv", "application/csv",
                   "application/pdf", "text/markdown"}

    is_image = mime_type in IMAGE_TYPES or any(
        filename.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif")
    )
    is_text = mime_type in TEXT_TYPES or any(
        filename.lower().endswith(ext) for ext in (".txt", ".csv", ".md")
    )

    # ── IMAGE analysis ────────────────────────────────────────────────────
    if is_image:
        b64 = base64.b64encode(raw_bytes).decode("utf-8")
        # Groq supports vision via llama-4-scout (or meta-llama/llama-4-scout-17b-16e-instruct)
        VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            "You are an expert tutor analysing an image uploaded by a student. "
                            f"Student's question: {question}\n\n"
                            "If the image contains mathematics: show every step of the solution "
                            "clearly labelled 'Step N — <action>: <working>' with LaTeX for all math. "
                            "If it contains a graph or diagram: describe it in detail and explain what it shows. "
                            "If it contains text: read and explain it. "
                            "Be thorough and educational."
                        )
                    }
                ]
            }
        ]
        try:
            resp = requests.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": VISION_MODEL,
                    "messages": messages,
                    "max_tokens": 1800,
                    "temperature": 0.2,
                },
                timeout=60,
            )
            if resp.status_code == 404:
                # Vision model not available, fall back to text description
                return jsonify({"error": "Vision model unavailable on this Groq plan. Please use a text file."}), 503
            if resp.status_code != 200:
                print(f"Groq vision error {resp.status_code}: {resp.text[:300]}")
                return jsonify({"error": f"Vision model error: {resp.status_code}"}), 500
            analysis = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Vision analysis error: {e}")
            return jsonify({"error": "Vision analysis failed. Please try again."}), 500

    # ── TEXT / PDF analysis ───────────────────────────────────────────────
    elif is_text or filename.lower().endswith(".pdf"):
        if filename.lower().endswith(".pdf"):
            # Try basic text extraction from PDF bytes
            try:
                import re as _re
                text_content = raw_bytes.decode("latin-1", errors="replace")
                # Extract readable text chunks (very basic, no pdfminer dependency)
                chunks = _re.findall(r'[\x20-\x7E\n\r\t]{20,}', text_content)
                text_content = "\n".join(chunks[:200])[:8000]
            except Exception:
                text_content = raw_bytes.decode("utf-8", errors="replace")[:8000]
        else:
            text_content = raw_bytes.decode("utf-8", errors="replace")[:8000]

        if not text_content.strip():
            return jsonify({"error": "Could not extract readable text from file"}), 400

        system = (
            "You are an expert tutor analysing a document uploaded by a student. "
            "If the document contains mathematics, show every solution step labelled "
            "'Step N — <action>: <working>' with LaTeX for all expressions. "
            "Be thorough, accurate, and educational."
        )
        prompt = (
            f"Student's question: {question}\n\n"
            f"Document content:\n{text_content}\n\n"
            "Provide a detailed analysis or solution based on the student's question."
        )
        analysis = _call_groq(prompt, max_tokens=1800, system=system)
        if not analysis:
            return jsonify({"error": "Analysis failed. Please try again."}), 500
        analysis = strip_markdown_fences(analysis)

    else:
        return jsonify({
            "error": f"Unsupported file type '{mime_type}'. "
                     "Upload an image (PNG/JPG/WEBP) or text file (TXT/CSV/PDF)."
        }), 400

    # Generate plain-English explanation after analysis
    explain_system = (
        "You are a friendly tutor. Given an AI analysis of a student's uploaded file, "
        "write a concise 3-4 sentence plain-English summary of the key findings or conclusions. "
        "Do not use LaTeX or mathematical symbols. Use simple language."
    )
    explain_prompt = (
        f"Original question: {question}\n\n"
        f"Analysis:\n{analysis[:3000]}\n\n"
        "Summarise the key findings in 3-4 plain sentences."
    )
    explanation = _call_groq(explain_prompt, max_tokens=300, system=explain_system)
    if explanation:
        explanation = strip_markdown_fences(explanation)

    return jsonify({
        "analysis": analysis,
        "explanation": explanation or "",
        "filename": filename,
        "file_type": "image" if is_image else "text",
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False)