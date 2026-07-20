
from pathlib import Path
import json
import pandas as pd
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "sample_input.json"

def main():
    model_path = ROOT / "artifacts" / "best_model_v5_by_r2.joblib"
    model_obj = joblib.load(model_path)
    data = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    X = pd.DataFrame([data])

    # Normal sklearn model
    if hasattr(model_obj, "predict"):
        pred = float(model_obj.predict(X)[0])
    # Manual CatBoost artifact
    elif isinstance(model_obj, dict) and "model" in model_obj:
        steps = model_obj["preprocess_steps"]
        cols = model_obj["columns"]
        num = model_obj["numeric_cols"]
        cat = model_obj["categorical_cols"]
        model = model_obj["model"]
        Xp = steps.transform(X)
        Xp = Xp[cols].copy()
        for c in num:
            Xp[c] = pd.to_numeric(Xp[c], errors="coerce").fillna(pd.to_numeric(Xp[c], errors="coerce").median())
        for c in cat:
            Xp[c] = Xp[c].fillna("missing").astype(str)
        pred = float(np.expm1(model.predict(Xp))[0])
    else:
        raise ValueError("Model format tanınmadı.")

    gross = float(data.get("gross_m2", 0) or 0)
    total = pred * gross if gross > 0 else None
    print(json.dumps({
        "predicted_unit_price_gross": round(pred, 2),
        "gross_m2": gross,
        "predicted_total_price": round(total, 2) if total is not None else None
    }, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
