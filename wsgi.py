# wsgi.py — Vercel serverless entrypoint
# Vercel looks for a variable named `app` in this file.
from app import app  # noqa: F401  — re-exports the Flask app object
