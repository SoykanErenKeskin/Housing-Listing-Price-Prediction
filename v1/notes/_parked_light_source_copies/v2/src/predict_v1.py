
from pathlib import Path
import json
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "best_model.joblib"
SAMPLE_PATH = ROOT / "sample_input.json"

def main():
    model = joblib.load(MODEL_PATH)
    data = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    X = pd.DataFrame([data])
    unit_price = float(model.predict(X)[0])
    gross_m2 = float(data.get("gross_m2", 0) or 0)
    total_price = unit_price * gross_m2 if gross_m2 > 0 else None
    print(json.dumps({
        "predicted_unit_price_gross": round(unit_price, 2),
        "gross_m2": gross_m2,
        "predicted_total_price": round(total_price, 2) if total_price is not None else None
    }, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
