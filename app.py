"""A deliberately small, local-first Flask application."""

import logging
import os
import secrets
from datetime import timedelta

from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, session, url_for
from flask_wtf.csrf import CSRFError, CSRFProtect
from werkzeug.exceptions import RequestEntityTooLarge

from models import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_OPENAI_MODEL,
    SoftwareEngineerAgent,
    create_model,
    execution_enabled,
    validate_model_config,
)
from verification import LANGUAGES

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32),
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=30),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("SESSION_COOKIE_SECURE", "").lower() == "true",
    MAX_CONTENT_LENGTH=64 * 1024,
)
CSRFProtect(app)


@app.context_processor
def template_settings():
    return {
        "languages": LANGUAGES,
        "execution_enabled": execution_enabled(),
        "provider_labels": {
            "OpenAI": f"OpenAI ({os.getenv('OPENAI_MODEL') or DEFAULT_OPENAI_MODEL})",
            "Claude": f"Claude ({os.getenv('CLAUDE_MODEL') or DEFAULT_CLAUDE_MODEL})",
            "Deepseek": "DeepSeek / local GGUF",
            "Llama": "Llama / local GGUF",
        },
    }


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        return configure_model()
    return render_template("index.html")


@app.post("/configure-model")
def configure_model():
    model_type = request.form.get("model_type", "")
    try:
        validate_model_config(model_type)
    except ValueError as exc:
        return render_template("index.html", error=str(exc)), 400
    # Store only the provider name, never clients, agents, prompts, or API keys.
    session["model"] = model_type
    session.permanent = True
    return redirect(url_for("index"), code=303)


@app.post("/solve")
def solve():
    problem = request.form.get("problem", "").strip()
    language = request.form.get("language", "python")
    if "model" not in session:
        return render_template(
            "index.html", error="Select and save a model first."
        ), 400
    if not problem or len(problem) > 20000:
        return render_template(
            "index.html",
            error="Enter a problem between 1 and 20,000 characters.",
            problem=problem[:20000],
            selected_language=language,
        ), 400
    if language not in LANGUAGES:
        return render_template(
            "index.html", error="Select a supported language.", problem=problem
        ), 400

    model = None
    try:
        model = create_model(session["model"])
        result = SoftwareEngineerAgent(model).process_task(problem, language)
        return render_template("results.html", result=result), 502 if result.get(
            "error"
        ) else 200
    except ValueError as exc:
        return render_template(
            "index.html",
            error=str(exc),
            problem=problem,
            selected_language=language,
        ), 400
    except Exception:
        logger.exception("Could not process task")
        return render_template(
            "error.html",
            error="Could not process the task. Check your configuration and the server log.",
        ), 500
    finally:
        if model is not None:
            model.close()


@app.errorhandler(CSRFError)
def csrf_error(error):
    return render_template(
        "error.html",
        error="This form expired or is invalid. Reload the main page and try again.",
    ), 400


@app.errorhandler(RequestEntityTooLarge)
def too_large(error):
    return render_template("error.html", error="The submitted form is too large."), 413


@app.after_request
def response_headers(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
