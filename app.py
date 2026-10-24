from math import isfinite
from pathlib import Path

from flask import Flask, render_template, request
import joblib
import pandas as pd

from part1_rag.rag_engine import GroundedAssistant

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
assistant = GroundedAssistant(ROOT / "data")
model_path = ROOT / "part3_ml" / "lead_scoring_model.joblib"
model = joblib.load(model_path) if model_path.exists() else None


def _required_text(form, name):
    value = (form.get(name) or "").strip()
    if not value:
        raise ValueError(f"{name.replace('_', ' ').capitalize()} is required.")
    return value


def _optional_number(form, name, *, integer=False, minimum=0, maximum=None):
    raw = (form.get(name) or "").strip()
    if not raw:
        return None
    try:
        value = int(raw) if integer else float(raw)
    except ValueError as exc:
        raise ValueError(f"{name.replace('_', ' ').capitalize()} must be a number.") from exc
    if not isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        upper = f" and no greater than {maximum}" if maximum is not None else ""
        raise ValueError(f"{name.replace('_', ' ').capitalize()} must be at least {minimum}{upper}.")
    return value


def _binary_choice(form, name):
    value = (form.get(name) or "").strip()
    if value not in {"0", "1"}:
        raise ValueError(f"{name.replace('_', ' ').capitalize()} must be 0 or 1.")
    return int(value)


def parse_lead_form(form):
    """Validate a single lead-scoring form submission before model inference."""
    return {
        "source": _required_text(form, "source"),
        "city": _required_text(form, "city"),
        "area": (form.get("area") or "").strip() or None,
        "property_type": _required_text(form, "property_type"),
        "budget_pkr_lac": _optional_number(form, "budget_pkr_lac"),
        "bedrooms": _optional_number(form, "bedrooms", integer=True, minimum=1, maximum=10),
        "is_overseas": _binary_choice(form, "is_overseas"),
        "referred_by_existing_client": _binary_choice(form, "referred_by_existing_client"),
        "has_financing_approved": _binary_choice(form, "has_financing_approved"),
    }


@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    sources = []
    score = None
    error = None
    status = 200

    if request.method == "POST":
        action = request.form.get("action")
        if action == "ask":
            question = (request.form.get("question") or "").strip()
            if not question:
                error = "Enter a question to search the supplied MGC documents."
                status = 400
            else:
                result = assistant.answer(question)
                answer, sources = result["answer"], result["sources"]
        elif action == "score":
            if model is None:
                error = "The lead-scoring model has not been trained. Run part3_ml/train.py first."
                status = 503
            else:
                try:
                    row = parse_lead_form(request.form)
                    score = float(model.predict_proba(pd.DataFrame([row]))[:, 1][0])
                except ValueError as exc:
                    error = str(exc)
                    status = 400
        else:
            error = "Choose a supported action."
            status = 400

    response = render_template(
        "index.html", answer=answer, sources=sources, score=score, error=error
    )
    return response, status


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

