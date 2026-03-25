import os
import json
import re

import requests
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Secret key – required for sessions / CSRF; must be set in the environment
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production")

# CORS – restrict to your own domain in production, e.g. "https://yourdomain.com"
CORS(app, origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","))

# Rate limiting – 10 requests per minute per IP
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["10 per minute"],
    storage_uri="memory://",
)

# ---------------------------------------------------------------------------
# Groq API configuration  –  set GROQ_API_KEY in your environment, never here
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY environment variable is not set. "
        "Export it before starting the server."
    )

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.1-8b-instant"

# Maximum characters allowed for the topic input
MAX_TOPIC_LENGTH = 200


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

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

        validated.append({
            "title": title,
            "points": points[:6],
        })

    return validated


def extract_json_array(text: str) -> str | None:
    match = re.search(r'\[\s*{.*?}\s*\]', text, re.DOTALL)
    return match.group(0) if match else None


def generate_slides(topic: str, retry: bool = False) -> list:
    prompt = f"""
You are an expert university professor creating high-quality academic slides.

Topic: {topic}

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

    try:
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 2000,
            },
            timeout=30,  # prevent hanging requests
        )

        if response.status_code != 200:
            print(f"Groq API error {response.status_code}: {response.text}")
            return []

        data = response.json()
        raw_text = data["choices"][0]["message"]["content"]

    except requests.exceptions.Timeout:
        print("ERROR: Groq API request timed out.")
        return []
    except Exception as e:
        print(f"ERROR calling Groq: {e}")
        return []

    # Try parsing JSON directly
    try:
        slides = json.loads(raw_text)
    except json.JSONDecodeError:
        print("Direct JSON parse failed – attempting extraction...")

        json_part = extract_json_array(raw_text)

        if not json_part:
            if not retry:
                print("Retrying once...")
                return generate_slides(topic, retry=True)
            return []

        try:
            slides = json.loads(json_part)
        except json.JSONDecodeError:
            return []

    # Normalise: handle {"slides": [...]} wrapper
    if isinstance(slides, dict):
        slides = slides.get("slides") or slides.get("data") or []

    if not isinstance(slides, list):
        return []

    return validate_slides(slides)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
@limiter.limit("10 per minute")
def generate():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    topic = str(data.get("topic", "")).strip()

    if not topic:
        return jsonify({"error": "No topic provided"}), 400

    if len(topic) > MAX_TOPIC_LENGTH:
        return jsonify({
            "error": f"Topic must be {MAX_TOPIC_LENGTH} characters or fewer"
        }), 400

    slides = generate_slides(topic)

    if not slides:
        return jsonify({
            "error": "AI failed to generate slides. Please try again."
        }), 500

    return jsonify(slides)


# ---------------------------------------------------------------------------
# Entry point  (development only – use Gunicorn in production)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # debug=False is mandatory for production
    app.run(debug=False)