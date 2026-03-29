import os
import json
import re
import time
import uuid

import requests
from flask import Flask, jsonify, render_template, request
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

        if response.status_code != 200:
            print(f"Groq API error {response.status_code}: {response.text}")
            return None

        return response.json()["choices"][0]["message"]["content"]

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
    match = re.search(r'\[\s*{.*?}\s*\]', text, re.DOTALL)
    return match.group(0) if match else None


def extract_json_object(text: str) -> str | None:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else None


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
    for slide in slides:
        if not isinstance(slide, dict):
            continue
        title = str(slide.get("title", "")).strip()
        points = slide.get("points", [])
        if not title:
            continue
        if not isinstance(points, list):
            points = [str(points)]
        points = [str(p).strip() for p in points if str(p).strip()]
        if len(points) < 2:
            continue

        entry: dict = {"title": title, "points": points[:6]}

        # Preserve equations (math slides)
        equations = slide.get("equations")
        if isinstance(equations, list):
            valid_eqs = []
            for eq in equations:
                if not isinstance(eq, dict):
                    continue
                label = str(eq.get("label", "")).strip()
                latex = str(eq.get("latex", "")).strip()
                explanation = str(eq.get("explanation", "")).strip()
                if label and latex:
                    valid_eqs.append({
                        "label": label,
                        "latex": latex,
                        "explanation": explanation,
                    })
            if valid_eqs:
                entry["equations"] = valid_eqs[:3]

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

def generate_slides(topic: str, learner_profile: dict | None = None, retry: bool = False) -> list:
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
        prompt = f"""
You are an expert mathematics professor creating rigorous, exam-quality academic slides.

Topic: {topic}
{difficulty_hint}

Return ONLY a valid JSON array. No extra text, no markdown fences.

Format:
[
  {{
    "title": "Slide title",
    "points": [
      "Detailed conceptual explanation sentence.",
      "Another detailed explanation with context."
    ],
    "equations": [
      {{
        "label": "Short name for this equation (e.g. Quadratic Formula)",
        "latex": "x = \\\\frac{{-b \\\\pm \\\\sqrt{{b^2 - 4ac}}}}{{2a}}",
        "explanation": "One sentence explaining what each symbol means and when this is used."
      }}
    ],
    "worked_example": {{
      "problem": "State a specific numeric problem clearly. E.g. Solve: 2x^2 - 4x - 6 = 0",
      "steps": [
        "Step 1: Identify a=2, b=-4, c=-6 from the standard form ax^2 + bx + c = 0.",
        "Step 2: Substitute into the quadratic formula: x = (4 ± sqrt(16 + 48)) / 4",
        "Step 3: Simplify inside the square root: sqrt(64) = 8",
        "Step 4: Compute both roots: x = (4+8)/4 = 3  and  x = (4-8)/4 = -1"
      ],
      "answer": "x = 3 or x = -1"
    }}
  }}
]

STRICT RULES:
- Generate 10 to 12 slides
- Each slide must have 4 to 6 bullet points in "points"
- EVERY point must be a COMPLETE, INFORMATIVE sentence (definition, proof idea, property, or real-world use)
- EVERY slide MUST include:
  • "equations": at least 1 and up to 3 key equations/formulas for that slide, each with a label, LaTeX string, and explanation
  • "worked_example": exactly 1 fully solved numeric example with clear step-by-step solution
- LaTeX must be valid and use double-escaped backslashes (e.g. \\\\frac, \\\\sqrt, \\\\int)
- Steps in worked_example must be numbered, show all arithmetic, and explain each operation
- Do NOT output anything outside JSON
"""
    else:
        prompt = f"""
You are an expert university professor creating high-quality academic slides.

Topic: {topic}
{difficulty_hint}

Return ONLY a valid JSON array. No extra text.

Format:
[
  {{
    "title": "Slide title",
    "points": [
      "Detailed explanation sentence",
      "Another detailed explanation"
    ]
  }}
]

STRICT RULES:
- Generate 10 to 12 slides
- Each slide must have 4 to 6 bullet points
- EVERY point must be a COMPLETE, INFORMATIVE sentence
- NO generic lines like "It is important" or "Has many applications"
- EVERY point MUST include at least one of:
  • definition
  • step-by-step process
  • real-world example
  • numerical value or formula
  • cause and effect explanation
- Include scientific explanations, real examples, and formulas where relevant
- Make content feel like a textbook, not a summary
- Do NOT output anything outside JSON
"""

    raw_text = _call_groq(prompt, max_tokens=4000)  # 70B generates more verbose JSON — needs higher limit
    if not raw_text:
        return []

    raw_text = strip_markdown_fences(raw_text)

    try:
        slides = json.loads(raw_text)
    except json.JSONDecodeError:
        json_part = extract_json_array(raw_text)
        if not json_part:
            if not retry:
                return generate_slides(topic, learner_profile, retry=True)
            return []
        try:
            slides = json.loads(json_part)
        except json.JSONDecodeError:
            return []

    if isinstance(slides, dict):
        slides = slides.get("slides") or slides.get("data") or []

    if not isinstance(slides, list):
        return []

    return validate_slides(slides)


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
        equations_block = json.dumps(slide.get("equations", []), indent=2)
        worked_block = json.dumps(slide.get("worked_example", {}), indent=2)
        prompt = f"""
A student did not understand this mathematics slide. Rewrite it with simpler language and a new, easier worked example.

Original Slide Title: {slide['title']}
Original Content:
{json.dumps(slide['points'], indent=2)}

Original Equations:
{equations_block}

Original Worked Example:
{worked_block}

Rewrite rules:
- Keep the same title
- Use simpler language — explain as if to a high-school student
- Keep ALL original equations in "equations" (same labels/latex) but rewrite their explanations more clearly
- In "worked_example": create a NEW, simpler numeric problem (smaller/rounder numbers) for the same concept
- Show MORE steps — break each arithmetic step into its own line
- Explain WHY each step is done, not just what to do
- Keep 4 to 6 bullet points in "points"

Return ONLY a valid JSON object. No extra text.

Format:
{{
  "title": "Same title",
  "points": [
    "Simplified explanation with analogy",
    "Another clearer explanation"
  ],
  "equations": [
    {{
      "label": "Same label as before",
      "latex": "same LaTeX string",
      "explanation": "Clearer, simpler explanation of what each symbol means."
    }}
  ],
  "worked_example": {{
    "problem": "A simpler numeric problem testing the same concept",
    "steps": [
      "Step 1: ...",
      "Step 2: ..."
    ],
    "answer": "Final numeric answer"
  }},
  "reteach": true
}}
"""
        max_tokens = 1600
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

    result: dict = {"title": title, "points": points[:6], "reteach": True}

    # Carry through equations for math slides
    if is_math:
        equations = slide_data.get("equations") or slide.get("equations")
        if isinstance(equations, list):
            valid_eqs = []
            for eq in equations:
                if isinstance(eq, dict) and eq.get("label") and eq.get("latex"):
                    valid_eqs.append({
                        "label": str(eq["label"]).strip(),
                        "latex": str(eq["latex"]).strip(),
                        "explanation": str(eq.get("explanation", "")).strip(),
                    })
            if valid_eqs:
                result["equations"] = valid_eqs[:3]

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
    Body: { "topic": "..." }
    Returns: { "session_id": "...", "slides": [...] }
    """
    _prune_sessions()
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    topic = str(data.get("topic", "")).strip()
    if not topic:
        return jsonify({"error": "No topic provided"}), 400
    if len(topic) > MAX_TOPIC_LENGTH:
        return jsonify({"error": f"Topic must be {MAX_TOPIC_LENGTH} characters or fewer"}), 400

    slides = generate_slides(topic)
    if not slides:
        return jsonify({"error": "AI failed to generate slides. Please try again."}), 500

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "topic": topic,
        "slides": slides,
        "performance": [],
        "created_at": time.time(),
        "quiz_cache": {},      # { slide_index_str: [questions] }
        "visual_cache": {},    # { slide_index_str: visual_dict }
        "notes": {},           # { slide_index_str: "text" }
        "xp": 0,
        "streak_slides": 0,
    }

    return jsonify({"session_id": session_id, "slides": slides})


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
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False)