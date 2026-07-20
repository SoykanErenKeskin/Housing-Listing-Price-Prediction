from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "listing_dataset_cleaned.csv"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def clean_str(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    return np.nan if s == "" or s.lower() in {"nan", "none", "null"} else s


def to_num(s):
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float, np.number)):
        return float(s)
    txt = str(s).strip()
    txt = txt.replace("TL", "").replace("₺", "").replace("m²", "").replace("m2", "")
    txt = txt.replace(".", "").replace(",", ".")
    txt = "".join(ch for ch in txt if ch.isdigit() or ch in ".-")
    try:
        return float(txt)
    except Exception:
        return np.nan


def floor_to_num(v):
    if pd.isna(v):
        return np.nan
    raw = str(v).lower()
    if "bodrum" in raw:
        return -1
    if "zemin" in raw or "giriş" in raw or "bahçe" in raw:
        return 0
    if "çatı" in raw:
        return np.nan
    nums = "".join(ch if ch.isdigit() or ch == "-" else " " for ch in raw).split()
    return float(nums[0]) if nums else np.nan


def prepare_data(df):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].map(clean_str)

    numeric_candidates = [
        "price", "unit_price_gross", "gross_m2", "net_m2", "open_area_m2",
        "building_age", "total_floors", "bathroom_count"
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = df[c].map(to_num)

    if "unit_price_gross" not in df.columns or df["unit_price_gross"].isna().all():
        df["unit_price_gross"] = df["price"] / df["gross_m2"]

    if "gross_m2" in df.columns and "net_m2" in df.columns:
        df["net_gross_ratio"] = df["net_m2"] / df["gross_m2"]
    else:
        df["net_gross_ratio"] = np.nan

    df["has_open_area"] = df.get("open_area_m2", pd.Series(index=df.index)).fillna(0).gt(0).astype(int)
    df["floor_num"] = df["floor"].map(floor_to_num) if "floor" in df.columns else np.nan

    mask = (
        df["unit_price_gross"].notna()
        & df["gross_m2"].notna()
        & (df["gross_m2"] > 20)
        & (df["gross_m2"] < 1000)
        & (df["unit_price_gross"] > 1000)
        & (df["unit_price_gross"] < 1000000)
    )
    return df.loc[mask].copy()


def make_models(numeric_features, categorical_features):
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", ohe)
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features)
    ])

    ridge = TransformedTargetRegressor(
        regressor=Pipeline([
            ("preprocess", preprocess),
            ("model", RidgeCV(alphas=np.logspace(-3, 4, 30)))
        ]),
        func=np.log1p,
        inverse_func=np.expm1
    )

    gb = TransformedTargetRegressor(
        regressor=Pipeline([
            ("preprocess", preprocess),
            ("model", GradientBoostingRegressor(
                random_state=42,
                learning_rate=0.05,
                n_estimators=300,
                max_depth=2,
                min_samples_leaf=3
            ))
        ]),
        func=np.log1p,
        inverse_func=np.expm1
    )

    return ridge, gb


def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def metrics(y_true, y_pred):
    return {
        "mape": mape(y_true, y_pred),
        "mae_tl_per_m2": float(mean_absolute_error(y_true, y_pred)),
        "median_ae_tl_per_m2": float(median_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred))
    }


def main():
    df = pd.read_csv(DATA_PATH)
    prepared = prepare_data(df)

    numeric_features = [
        "gross_m2", "net_m2", "building_age", "floor_num", "total_floors",
        "bathroom_count", "open_area_m2", "net_gross_ratio", "has_open_area"
    ]
    categorical_features = [
        "real_estate_type", "room_count", "floor_segment", "heating", "kitchen",
        "balcony", "elevator", "parking", "furnished", "usage_status", "site_inside",
        "credit_eligible", "energy_certificate", "deed_status", "seller_type", "barter",
        "city", "county", "district"
    ]
    numeric_features = [c for c in numeric_features if c in prepared.columns]
    categorical_features = [c for c in categorical_features if c in prepared.columns]

    X = prepared[numeric_features + categorical_features].copy()
    y = prepared["unit_price_gross"].astype(float)

    ridge, gb = make_models(numeric_features, categorical_features)

    n_splits = min(5, len(prepared))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    ridge_pred = cross_val_predict(ridge, X, y, cv=cv)
    gb_pred = cross_val_predict(gb, X, y, cv=cv)

    ridge.fit(X, y)
    gb.fit(X, y)

    joblib.dump(ridge, ARTIFACTS / "pure_listing_ridge_log_unit_price_v0.joblib")
    joblib.dump(gb, ARTIFACTS / "pure_listing_gradient_boosting_log_unit_price_v0.joblib")

    payload = {
        "target": "unit_price_gross",
        "rows_raw": int(len(df)),
        "rows_used": int(len(prepared)),
        "features": {"numeric": numeric_features, "categorical": categorical_features},
        "models": {
            "ridge_log_target": metrics(y, ridge_pred),
            "gradient_boosting_log_target": metrics(y, gb_pred)
        },
        "note": "Pure listing-data baseline. external demographics provider/trend/floor external references are not used."
    }
    (ARTIFACTS / "model_metrics_v0.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    preds = prepared[["source_url", "classified_id", "title", "price", "unit_price_gross", "gross_m2", "city", "county", "district"]].copy()
    preds["ridge_cv_pred_unit_price"] = ridge_pred
    preds["gb_cv_pred_unit_price"] = gb_pred
    preds["ridge_abs_pct_error"] = np.abs(preds["unit_price_gross"] - preds["ridge_cv_pred_unit_price"]) / preds["unit_price_gross"]
    preds["gb_abs_pct_error"] = np.abs(preds["unit_price_gross"] - preds["gb_cv_pred_unit_price"]) / preds["unit_price_gross"]
    preds.to_csv(ARTIFACTS / "cv_predictions_v0.csv", index=False, encoding="utf-8-sig")

    # Ridge coefficients
    inner = ridge.regressor_
    names = inner.named_steps["preprocess"].get_feature_names_out()
    coefs = inner.named_steps["model"].coef_
    coef_df = pd.DataFrame({"feature": names, "coefficient_log_unit_price": coefs})
    coef_df["abs_coefficient"] = coef_df["coefficient_log_unit_price"].abs()
    coef_df.sort_values("abs_coefficient", ascending=False).to_csv(ARTIFACTS / "ridge_coefficients_v0.csv", index=False, encoding="utf-8-sig")

    # Gradient boosting importance
    inner = gb.regressor_
    names = inner.named_steps["preprocess"].get_feature_names_out()
    importance = inner.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"feature": names, "importance": importance})
    imp_df.sort_values("importance", ascending=False).to_csv(ARTIFACTS / "gb_feature_importance_v0.csv", index=False, encoding="utf-8-sig")

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
