from flask import Flask, render_template, request
from pathlib import Path
import pandas as pd
import joblib

from part1_rag.rag_engine import GroundedAssistant

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
assistant = GroundedAssistant(ROOT / "data")
model_path = ROOT / "part3_ml" / "lead_scoring_model.joblib"
model = joblib.load(model_path) if model_path.exists() else None

@app.route("/", methods=["GET","POST"])
def home():
    answer = None
    sources = []
    score = None
    if request.method == "POST":
        action = request.form.get("action")
        if action == "ask":
            result = assistant.answer(request.form.get("question",""))
            answer, sources = result["answer"], result["sources"]
        elif action == "score" and model is not None:
            row = {
                "source": request.form.get("source"),
                "city": request.form.get("city"),
                "area": request.form.get("area") or None,
                "property_type": request.form.get("property_type"),
                "budget_pkr_lac": float(request.form.get("budget_pkr_lac") or 0),
                "bedrooms": float(request.form.get("bedrooms") or 0) if request.form.get("bedrooms") else None,
                "is_overseas": int(request.form.get("is_overseas") or 0),
                "referred_by_existing_client": int(request.form.get("referred_by_existing_client") or 0),
                "has_financing_approved": int(request.form.get("has_financing_approved") or 0)
            }
            score = float(model.predict_proba(pd.DataFrame([row]))[:,1][0])
    return render_template("index.html", answer=answer, sources=sources, score=score)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
