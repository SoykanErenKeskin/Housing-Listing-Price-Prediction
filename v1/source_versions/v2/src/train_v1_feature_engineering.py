
from pathlib import Path
import json, re, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, make_scorer

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "input" / "listing_dataset_cleaned.csv"
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ROOT / "data" / "output"
REPORTS = ROOT / "reports"
for p in [ARTIFACTS, OUTPUT, REPORTS]:
    p.mkdir(parents=True, exist_ok=True)

TARGET = "unit_price_gross"
RANDOM_STATE = 42

BASE_NUMERIC = ["gross_m2","net_m2","building_age","floor_num","total_floors","bathroom_count","open_area_m2","net_gross_ratio","has_open_area"]
BASE_CATEGORICAL = ["real_estate_type","room_count","floor_segment","heating","kitchen","balcony","elevator","parking","furnished","usage_status","site_inside","credit_eligible","energy_certificate","deed_status","seller_type","barter","city","county","district"]
ENGINEERED_NUMERIC = ["floor_ratio","remaining_floors","is_ground_floor","is_basement","is_top_floor","is_middle_floor","rooms","living_rooms","total_room_score","is_new_building","is_old_building","is_small_flat","is_large_flat","quality_score","district_target_encoded","county_target_encoded"]
ENGINEERED_CATEGORICAL = ["building_age_group","m2_group"]

def clean_str(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    return np.nan if s == "" or s.lower() in {"nan","none","null"} else s

def to_num(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.number)): return float(x)
    s = str(x).replace("TL","").replace("₺","").replace("m²","").replace("m2","")
    s = s.replace(".","").replace(",",".")
    s = "".join(ch for ch in s if ch.isdigit() or ch in ".-")
    try: return float(s)
    except Exception: return np.nan

def floor_to_num(v):
    if pd.isna(v): return np.nan
    s = str(v).lower()
    if "bodrum" in s: return -1.0
    if "zemin" in s or "giriş" in s or "bahçe" in s: return 0.0
    if "çatı" in s: return np.nan
    nums = "".join(ch if ch.isdigit() or ch == "-" else " " for ch in s).split()
    return float(nums[0]) if nums else np.nan

def parse_room(v):
    if pd.isna(v): return np.nan, np.nan, np.nan
    s = str(v).replace(" ","").lower()
    m = re.search(r"(\d+)\+(\d+)", s)
    if m:
        r, l = float(m.group(1)), float(m.group(2))
        return r, l, r + l
    m = re.search(r"(\d+)", s)
    return (float(m.group(1)), np.nan, float(m.group(1))) if m else (np.nan, np.nan, np.nan)

def prepare_raw(df):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object": df[c] = df[c].map(clean_str)
    for c in ["price","unit_price_gross","gross_m2","net_m2","open_area_m2","building_age","total_floors","bathroom_count"]:
        if c in df.columns: df[c] = df[c].map(to_num)
    if "unit_price_gross" not in df.columns: df["unit_price_gross"] = np.nan
    if "price" in df.columns and "gross_m2" in df.columns:
        m = df[TARGET].isna() & df["price"].notna() & df["gross_m2"].notna() & (df["gross_m2"] > 0)
        df.loc[m, TARGET] = df.loc[m, "price"] / df.loc[m, "gross_m2"]
    df["net_gross_ratio"] = df["net_m2"] / df["gross_m2"] if {"net_m2","gross_m2"}.issubset(df.columns) else np.nan
    df["open_area_m2"] = df["open_area_m2"] if "open_area_m2" in df.columns else np.nan
    df["has_open_area"] = df["open_area_m2"].fillna(0).gt(0).astype(int)
    df["floor_num"] = df["floor"].map(floor_to_num) if "floor" in df.columns else np.nan
    valid = df[TARGET].notna() & df["gross_m2"].notna() & (df["gross_m2"] > 20) & (df["gross_m2"] < 1000) & (df[TARGET] > 1000) & (df[TARGET] < 1000000)
    return df.loc[valid].copy()

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        df = X.copy()
        for c in ["gross_m2","net_m2","building_age","floor_num","total_floors","bathroom_count"]:
            if c not in df.columns: df[c] = np.nan
        denom = df["total_floors"].replace(0, np.nan)
        df["floor_ratio"] = df["floor_num"] / denom
        df["remaining_floors"] = df["total_floors"] - df["floor_num"]
        df["is_ground_floor"] = (df["floor_num"] == 0).astype(int)
        df["is_basement"] = (df["floor_num"] < 0).astype(int)
        df["is_top_floor"] = (df["floor_num"].notna() & df["total_floors"].notna() & (df["total_floors"] > 0) & (df["floor_num"] >= df["total_floors"])).astype(int)
        df["is_middle_floor"] = (df["floor_num"].notna() & df["total_floors"].notna() & (df["floor_num"] > 0) & (df["floor_num"] < df["total_floors"])).astype(int)
        if "room_count" in df.columns:
            parsed = df["room_count"].apply(parse_room)
            df["rooms"] = parsed.apply(lambda x: x[0])
            df["living_rooms"] = parsed.apply(lambda x: x[1])
            df["total_room_score"] = parsed.apply(lambda x: x[2])
        else:
            df["rooms"] = df["living_rooms"] = df["total_room_score"] = np.nan
        age = df["building_age"]
        df["is_new_building"] = age.fillna(999).le(2).astype(int)
        df["is_old_building"] = age.fillna(0).ge(25).astype(int)
        df["building_age_group"] = pd.cut(age, [-1,0,5,10,20,30,200], labels=["0","1-5","6-10","11-20","21-30","30+"]).astype(object)
        gross = df["gross_m2"]
        df["is_small_flat"] = gross.fillna(9999).le(75).astype(int)
        df["is_large_flat"] = gross.fillna(0).ge(160).astype(int)
        df["m2_group"] = pd.cut(gross, [0,75,100,125,150,200,1000], labels=["0-75","76-100","101-125","126-150","151-200","200+"]).astype(object)
        q = pd.Series(0.0, index=df.index)
        def yes_like(s): return s.fillna("").astype(str).str.lower().str.contains("var|evet|açık|kapalı|kapali|site", regex=True)
        if "elevator" in df.columns: q += yes_like(df["elevator"]).astype(int)
        if "parking" in df.columns: q += yes_like(df["parking"]).astype(int)
        if "site_inside" in df.columns: q += yes_like(df["site_inside"]).astype(int)
        if "bathroom_count" in df.columns: q += df["bathroom_count"].fillna(1).ge(2).astype(int)
        if "heating" in df.columns: q += df["heating"].fillna("").astype(str).str.lower().str.contains("merkezi|kombi|yerden", regex=True).astype(int)
        df["quality_score"] = q
        return df

class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, smoothing=20):
        self.columns = columns or []
        self.smoothing = smoothing
    def fit(self, X, y):
        X = X.copy()
        y = pd.Series(y).astype(float)
        self.global_mean_ = float(y.mean())
        self.maps_ = {}
        for col in self.columns:
            if col not in X.columns: continue
            tmp = pd.DataFrame({"key": X[col].fillna("missing").astype(str), "target": y.values})
            stats = tmp.groupby("key")["target"].agg(["mean","count"])
            smooth = (stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing) / (stats["count"] + self.smoothing)
            self.maps_[col] = smooth.to_dict()
        return self
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            out = f"{col}_target_encoded"
            if col not in X.columns or col not in self.maps_:
                X[out] = self.global_mean_
            else:
                X[out] = X[col].fillna("missing").astype(str).map(self.maps_[col]).fillna(self.global_mean_).astype(float)
        return X

def remove_useless_features(df, num, cat):
    keep_num, keep_cat, removed = [], [], {}
    for c in num:
        if c not in df.columns: removed[c] = "missing"
        elif df[c].notna().sum() == 0: removed[c] = "all_missing"
        elif df[c].nunique(dropna=True) <= 1: removed[c] = "constant"
        else: keep_num.append(c)
    for c in cat:
        if c not in df.columns: removed[c] = "missing"
        elif df[c].notna().sum() == 0: removed[c] = "all_missing"
        elif df[c].nunique(dropna=True) <= 1: removed[c] = "constant"
        else: keep_cat.append(c)
    return keep_num, keep_cat, removed

def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)))

def metric_dict(y, p):
    return {"mape": mape(y,p), "mae_tl_per_m2": float(mean_absolute_error(y,p)), "median_ae_tl_per_m2": float(median_absolute_error(y,p)), "r2": float(r2_score(y,p))}

def make_preprocessor(num, cat):
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=3, sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=3, sparse=False)
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="missing")), ("onehot", ohe)]), cat),
    ])

def make_model(estimator, num, cat):
    return TransformedTargetRegressor(
        regressor=Pipeline([
            ("feature_engineering", FeatureEngineer()),
            ("target_encoding", TargetEncoder(columns=["district", "county"], smoothing=20)),
            ("preprocess", make_preprocessor(num, cat)),
            ("model", estimator)
        ]),
        func=np.log1p,
        inverse_func=np.expm1
    )

def save_explainability(model, name):
    inner = model.regressor_
    pre = inner.named_steps["preprocess"]
    est = inner.named_steps["model"]
    try: names = pre.get_feature_names_out()
    except Exception: return
    if hasattr(est, "coef_"):
        out = pd.DataFrame({"feature": names, "coefficient": est.coef_})
        out["abs_coefficient"] = out["coefficient"].abs()
        out.sort_values("abs_coefficient", ascending=False).to_csv(ARTIFACTS / f"{name}_coefficients.csv", index=False, encoding="utf-8-sig")
    if hasattr(est, "feature_importances_"):
        out = pd.DataFrame({"feature": names, "importance": est.feature_importances_})
        out.sort_values("importance", ascending=False).to_csv(ARTIFACTS / f"{name}_feature_importance.csv", index=False, encoding="utf-8-sig")

def write_error_reports(df, pred, name):
    out = df.copy()
    out[f"{name}_pred_unit_price"] = pred
    out[f"{name}_abs_pct_error"] = np.abs(out[TARGET] - pred) / out[TARGET]
    out[f"{name}_abs_error"] = np.abs(out[TARGET] - pred)
    out.to_csv(OUTPUT / f"{name}_cv_predictions.csv", index=False, encoding="utf-8-sig")
    err, abs_err = f"{name}_abs_pct_error", f"{name}_abs_error"
    for col in ["district","room_count","floor_segment","building_age_group","m2_group","heating","site_inside"]:
        if col in out.columns:
            rep = out.groupby(col, dropna=False).agg(
                n=(TARGET,"size"), mape=(err,"mean"), median_ape=(err,"median"),
                mae_tl_per_m2=(abs_err,"mean"), median_ae_tl_per_m2=(abs_err,"median"),
                mean_unit_price=(TARGET,"mean")
            ).reset_index().sort_values("mape", ascending=False)
            rep.to_csv(OUTPUT / f"{name}_error_by_{col}.csv", index=False, encoding="utf-8-sig")
    out.sort_values(err, ascending=False).head(50).to_csv(OUTPUT / f"{name}_top_50_errors.csv", index=False, encoding="utf-8-sig")

def main():
    warnings.filterwarnings("ignore")
    raw = pd.read_csv(INPUT_PATH)
    df = prepare_raw(raw)

    fe = FeatureEngineer().fit_transform(df.copy())
    te = TargetEncoder(columns=["district","county"], smoothing=20).fit(fe.copy(), df[TARGET]).transform(fe.copy())
    num, cat, removed = remove_useless_features(te, BASE_NUMERIC + ENGINEERED_NUMERIC, BASE_CATEGORICAL + ENGINEERED_CATEGORICAL)

    X = df.drop(columns=[TARGET], errors="ignore").copy()
    y = df[TARGET].astype(float)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "ridge_fe": make_model(RidgeCV(alphas=np.logspace(-3,4,40)), num, cat),
        "elasticnet_fe": make_model(ElasticNetCV(l1_ratio=[.05,.1,.3,.5,.7], alphas=np.logspace(-4,2,40), max_iter=20000, random_state=RANDOM_STATE), num, cat),
        "gradient_boosting_fe": make_model(GradientBoostingRegressor(random_state=RANDOM_STATE, learning_rate=.04, n_estimators=500, max_depth=2, min_samples_leaf=5, subsample=.85), num, cat),
        "hist_gradient_boosting_fe": make_model(HistGradientBoostingRegressor(random_state=RANDOM_STATE, max_iter=300, learning_rate=.04, max_leaf_nodes=31, l2_regularization=.05), num, cat),
        "extra_trees_fe": make_model(ExtraTreesRegressor(random_state=RANDOM_STATE, n_estimators=350, min_samples_leaf=2, max_features=.75, n_jobs=-1), num, cat),
    }

    results, preds = {}, {}
    for name, model in models.items():
        print(f"CV: {name}")
        p = cross_val_predict(model, X, y, cv=cv)
        preds[name] = p
        results[name] = metric_dict(y, p)
        model.fit(X, y)
        joblib.dump(model, ARTIFACTS / f"{name}.joblib")
        save_explainability(model, name)

    print("Tuning: gradient_boosting_tuned_fe")
    base = make_model(GradientBoostingRegressor(random_state=RANDOM_STATE), num, cat)
    param_dist = {
        "regressor__model__n_estimators": [300,500,700],
        "regressor__model__learning_rate": [.03,.04,.05,.07],
        "regressor__model__max_depth": [2,3],
        "regressor__model__min_samples_leaf": [3,5,8,12],
        "regressor__model__subsample": [.75,.85,1.0],
    }
    scorer = make_scorer(lambda yt, yp: -mape(yt, yp), greater_is_better=True)
    search = RandomizedSearchCV(base, param_dist, n_iter=14, cv=cv, scoring=scorer, random_state=RANDOM_STATE, n_jobs=-1)
    search.fit(X, y)
    tuned = search.best_estimator_
    tuned_pred = cross_val_predict(tuned, X, y, cv=cv)
    tuned.fit(X, y)
    results["gradient_boosting_tuned_fe"] = metric_dict(y, tuned_pred)
    results["gradient_boosting_tuned_fe"]["best_params"] = search.best_params_
    preds["gradient_boosting_tuned_fe"] = tuned_pred
    joblib.dump(tuned, ARTIFACTS / "gradient_boosting_tuned_fe.joblib")
    save_explainability(tuned, "gradient_boosting_tuned_fe")

    best = min(results, key=lambda k: results[k]["mape"])
    joblib.dump(joblib.load(ARTIFACTS / f"{best}.joblib"), ARTIFACTS / "best_model.joblib")

    feature_view = TargetEncoder(columns=["district","county"], smoothing=20).fit(FeatureEngineer().fit_transform(df.copy()), df[TARGET]).transform(FeatureEngineer().fit_transform(df.copy()))
    feature_view.to_csv(OUTPUT / "feature_engineered_dataset_preview.csv", index=False, encoding="utf-8-sig")
    write_error_reports(feature_view, preds[best], best)

    pd.DataFrame([{ "model": k, **{kk: vv for kk, vv in v.items() if kk != "best_params"} } for k,v in results.items()]).sort_values("mape").to_csv(REPORTS / "model_comparison_v1.csv", index=False, encoding="utf-8-sig")

    report = {
        "target": TARGET,
        "rows_raw": int(len(raw)),
        "rows_used": int(len(df)),
        "features": {"numeric_used": num, "categorical_used": cat, "removed_features": removed},
        "models": results,
        "best_model": best,
        "note": "V1 includes feature engineering, KFold-safe target encoding, useless-column removal, model comparison, light tuning, and error analysis."
    }
    (REPORTS / "model_metrics_v1.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
