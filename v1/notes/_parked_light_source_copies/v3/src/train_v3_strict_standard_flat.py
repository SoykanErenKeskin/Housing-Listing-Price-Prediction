
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
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
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

TITLE_EXCLUDE_PATTERNS = [
    r"dublex", r"dubleks", r"düblex", r"dübleks", r"duplex",
    r"tripleks", r"triplex",
    r"lüks", r"lux", r"luxury",
    r"villa", r"müstakil", r"mustakil",
    r"bahçe", r"bahceli", r"bahçeli", r"bahce"
]
ALLOWED_ROOM_COUNTS = {"1+1", "2+1", "3+1", "4+1"}
MIN_GROSS_M2 = 45
MAX_GROSS_M2 = 220

BASE_NUMERIC = [
    "gross_m2","net_m2","building_age","floor_num","total_floors","bathroom_count",
    "net_gross_ratio","floor_ratio","remaining_floors","is_ground_floor","is_basement",
    "is_top_floor","is_middle_floor","rooms","living_rooms","total_room_score",
    "is_new_building","is_old_building","is_small_flat","is_large_flat","quality_score",
    "district_target_encoded","district_baseline_unit_price"
]
BASE_CATEGORICAL = [
    "real_estate_type","room_count","floor_segment","heating","kitchen","balcony",
    "elevator","parking","furnished","usage_status","site_inside","credit_eligible",
    "energy_certificate","deed_status","seller_type","barter","district",
    "building_age_group","m2_group","district_age_group","district_m2_group","district_room_count"
]

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

def normalize_tr(text):
    if pd.isna(text): return ""
    s = str(text).lower()
    table = str.maketrans({"ı":"i","ğ":"g","ü":"u","ş":"s","ö":"o","ç":"c","İ":"i","Ğ":"g","Ü":"u","Ş":"s","Ö":"o","Ç":"c"})
    return s + " " + s.translate(table)

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

def normalized_room_count(v):
    if pd.isna(v): return np.nan
    s = str(v).replace(" ", "").lower()
    m = re.search(r"(\d+)\+(\d+)", s)
    if m:
        return f"{int(m.group(1))}+{int(m.group(2))}"
    return s

def prepare_raw(df):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].map(clean_str)
    for c in ["price","unit_price_gross","gross_m2","net_m2","building_age","total_floors","bathroom_count"]:
        if c in df.columns:
            df[c] = df[c].map(to_num)
    if TARGET not in df.columns:
        df[TARGET] = np.nan
    if "price" in df.columns and "gross_m2" in df.columns:
        m = df[TARGET].isna() & df["price"].notna() & df["gross_m2"].notna() & (df["gross_m2"] > 0)
        df.loc[m, TARGET] = df.loc[m, "price"] / df.loc[m, "gross_m2"]

    if {"net_m2","gross_m2"}.issubset(df.columns):
        df["net_gross_ratio"] = df["net_m2"] / df["gross_m2"]
    else:
        df["net_gross_ratio"] = np.nan

    df["floor_num"] = df["floor"].map(floor_to_num) if "floor" in df.columns else np.nan

    basic_valid = df[TARGET].notna() & df["gross_m2"].notna() & (df["gross_m2"] > 20) & (df["gross_m2"] < 1000) & (df[TARGET] > 1000) & (df[TARGET] < 1000000)
    df = df.loc[basic_valid].copy()

    reason = pd.Series("", index=df.index, dtype="object")
    def add_reason(mask, label):
        reason.loc[mask] = reason.loc[mask].apply(lambda old: label if not old else old + "; " + label)

    if "title" in df.columns:
        norm = df["title"].map(normalize_tr)
        title_mask = norm.str.contains("|".join(TITLE_EXCLUDE_PATTERNS), case=False, regex=True, na=False)
        add_reason(title_mask, "title_special_segment")

    if "real_estate_type" in df.columns:
        rt = df["real_estate_type"].fillna("").map(normalize_tr)
        add_reason(rt.str.contains("villa|mustakil|müstakil", regex=True, na=False), "real_estate_type_special_segment")

    room_norm = df["room_count"].map(normalized_room_count) if "room_count" in df.columns else pd.Series(np.nan, index=df.index)
    add_reason(~room_norm.isin(ALLOWED_ROOM_COUNTS), "room_count_not_standard_1to4_plus_1")

    add_reason(df["gross_m2"] < MIN_GROSS_M2, "gross_m2_below_45")
    add_reason(df["gross_m2"] > MAX_GROSS_M2, "gross_m2_above_220")

    marked = df.copy()
    marked["strict_filter_reason"] = reason
    removed = marked[marked["strict_filter_reason"].ne("")].copy()
    kept = marked[marked["strict_filter_reason"].eq("")].drop(columns=["strict_filter_reason"], errors="ignore").copy()

    marked.to_csv(OUTPUT / "dataset_marked_with_strict_filter_reason.csv", index=False, encoding="utf-8-sig")
    removed.to_csv(OUTPUT / "removed_by_strict_standard_flat_filter.csv", index=False, encoding="utf-8-sig")
    kept.to_csv(OUTPUT / "strict_standard_flat_dataset.csv", index=False, encoding="utf-8-sig")

    return kept, removed, marked

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
        df["m2_group"] = pd.cut(gross, [0,75,100,125,150,180,220], labels=["0-75","76-100","101-125","126-150","151-180","181-220"]).astype(object)

        q = pd.Series(0.0, index=df.index)
        def yes_like(s): return s.fillna("").astype(str).str.lower().str.contains("var|evet|açık|kapalı|kapali|site", regex=True)
        if "elevator" in df.columns: q += yes_like(df["elevator"]).astype(int)
        if "parking" in df.columns: q += yes_like(df["parking"]).astype(int)
        if "site_inside" in df.columns: q += yes_like(df["site_inside"]).astype(int)
        if "bathroom_count" in df.columns: q += df["bathroom_count"].fillna(1).ge(2).astype(int)
        if "heating" in df.columns: q += df["heating"].fillna("").astype(str).str.lower().str.contains("merkezi|kombi|yerden", regex=True).astype(int)
        df["quality_score"] = q

        def comb(a,b):
            aa = df[a].fillna("missing").astype(str) if a in df.columns else pd.Series("missing", index=df.index)
            bb = df[b].fillna("missing").astype(str) if b in df.columns else pd.Series("missing", index=df.index)
            combo = aa + "__" + bb
            return combo
        df["district_age_group"] = comb("district", "building_age_group")
        df["district_m2_group"] = comb("district", "m2_group")
        df["district_room_count"] = comb("district", "room_count")
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

class BaselineEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, smoothing=10):
        self.columns = columns or []
        self.smoothing = smoothing
    def fit(self, X, y):
        X = X.copy()
        y = pd.Series(y).astype(float)
        self.global_median_ = float(y.median())
        self.maps_ = {}
        for col in self.columns:
            if col not in X.columns: continue
            tmp = pd.DataFrame({"key": X[col].fillna("missing").astype(str), "target": y.values})
            stats = tmp.groupby("key")["target"].agg(["median","count"])
            smooth = (stats["median"] * stats["count"] + self.global_median_ * self.smoothing) / (stats["count"] + self.smoothing)
            self.maps_[col] = smooth.to_dict()
        return self
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            out = f"{col}_baseline_unit_price"
            if col not in X.columns or col not in self.maps_:
                X[out] = self.global_median_
            else:
                X[out] = X[col].fillna("missing").astype(str).map(self.maps_[col]).fillna(self.global_median_).astype(float)
        return X

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, min_count=8):
        self.columns = columns or []
        self.min_count = min_count
    def fit(self, X, y=None):
        X = X.copy()
        self.valid_values_ = {}
        for col in self.columns:
            if col not in X.columns: continue
            counts = X[col].fillna("missing").astype(str).value_counts()
            self.valid_values_[col] = set(counts[counts >= self.min_count].index)
        return self
    def transform(self, X):
        X = X.copy()
        for col, valid in self.valid_values_.items():
            if col not in X.columns: continue
            vals = X[col].fillna("missing").astype(str)
            X[col] = vals.where(vals.isin(valid), "other")
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
    log_r2 = float(r2_score(np.log1p(y), np.log1p(np.maximum(p, 0))))
    return {
        "mape": mape(y,p),
        "mae_tl_per_m2": float(mean_absolute_error(y,p)),
        "median_ae_tl_per_m2": float(median_absolute_error(y,p)),
        "r2": float(r2_score(y,p)),
        "log_r2": log_r2
    }

def make_preprocessor(num, cat):
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=4, sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=4, sparse=False)
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="missing")), ("onehot", ohe)]), cat),
    ])

def make_model(estimator, num, cat, rare_min_count=8):
    return TransformedTargetRegressor(
        regressor=Pipeline([
            ("feature_engineering", FeatureEngineer()),
            ("rare_category_grouper", RareCategoryGrouper(columns=cat, min_count=rare_min_count)),
            ("baseline_encoding", BaselineEncoder(columns=["district"], smoothing=10)),
            ("target_encoding", TargetEncoder(columns=["district"], smoothing=20)),
            ("preprocess", make_preprocessor(num, cat)),
            ("model", estimator)
        ]),
        func=np.log1p,
        inverse_func=np.expm1
    )

def catboost_available():
    try:
        import catboost
        return True
    except Exception:
        return False

class CatBoostNativePrepare(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_cols=None, categorical_cols=None, rare_min_count=8):
        self.numeric_cols = numeric_cols or []
        self.categorical_cols = categorical_cols or []
        self.rare_min_count = rare_min_count
    def fit(self, X, y=None):
        X = X.copy()
        self.numeric_keep_ = [c for c in self.numeric_cols if c in X.columns and X[c].notna().sum() > 0 and X[c].nunique(dropna=True) > 1]
        self.categorical_keep_ = [c for c in self.categorical_cols if c in X.columns and X[c].notna().sum() > 0 and X[c].nunique(dropna=True) > 1]
        self.columns_ = self.numeric_keep_ + self.categorical_keep_
        self.num_medians_ = {c: float(pd.to_numeric(X[c], errors="coerce").median()) for c in self.numeric_keep_}
        self.valid_values_ = {}
        for c in self.categorical_keep_:
            counts = X[c].fillna("missing").astype(str).value_counts()
            self.valid_values_[c] = set(counts[counts >= self.rare_min_count].index)
        return self
    def transform(self, X):
        X = X.copy()
        out = pd.DataFrame(index=X.index)
        for c in self.numeric_keep_:
            out[c] = pd.to_numeric(X[c], errors="coerce").fillna(self.num_medians_.get(c, 0.0))
        for c in self.categorical_keep_:
            vals = X[c].fillna("missing").astype(str)
            vals = vals.where(vals.isin(self.valid_values_.get(c, set())), "other")
            out[c] = vals
        return out

def make_native_catboost_model(num, cat):
    from catboost import CatBoostRegressor
    # This uses pandas DataFrame categorical columns natively through cat_features names.
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="R2",
        iterations=1400,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=8,
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        cat_features=cat
    )
    return TransformedTargetRegressor(
        regressor=Pipeline([
            ("feature_engineering", FeatureEngineer()),
            ("baseline_encoding", BaselineEncoder(columns=["district"], smoothing=10)),
            ("target_encoding", TargetEncoder(columns=["district"], smoothing=20)),
            ("catboost_prepare", CatBoostNativePrepare(numeric_cols=num, categorical_cols=cat, rare_min_count=8)),
            ("model", model)
        ]),
        func=np.log1p,
        inverse_func=np.expm1
    )

def save_explainability(model, name):
    try:
        inner = model.regressor_
        if hasattr(inner, "named_steps"):
            if "preprocess" in inner.named_steps:
                names = inner.named_steps["preprocess"].get_feature_names_out()
                est = inner.named_steps["model"]
                if hasattr(est, "coef_"):
                    out = pd.DataFrame({"feature": names, "coefficient": est.coef_})
                    out["abs_coefficient"] = out["coefficient"].abs()
                    out.sort_values("abs_coefficient", ascending=False).to_csv(ARTIFACTS / f"{name}_coefficients.csv", index=False, encoding="utf-8-sig")
                if hasattr(est, "feature_importances_"):
                    out = pd.DataFrame({"feature": names, "importance": est.feature_importances_})
                    out.sort_values("importance", ascending=False).to_csv(ARTIFACTS / f"{name}_feature_importance.csv", index=False, encoding="utf-8-sig")
    except Exception:
        pass

def write_error_reports(df, pred, name):
    view = FeatureEngineer().fit_transform(df.copy())
    out = view.copy()
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
    df, removed, marked = prepare_raw(raw)

    fe = FeatureEngineer().fit_transform(df.copy())
    be = BaselineEncoder(columns=["district"], smoothing=10).fit(fe.copy(), df[TARGET]).transform(fe.copy())
    te = TargetEncoder(columns=["district"], smoothing=20).fit(be.copy(), df[TARGET]).transform(be.copy())
    num, cat, removed_features = remove_useless_features(te, BASE_NUMERIC, BASE_CATEGORICAL)

    X = df.drop(columns=[TARGET], errors="ignore").copy()
    y = df[TARGET].astype(float)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "ridge_v3_strict": make_model(RidgeCV(alphas=np.logspace(-3,4,60)), num, cat),
        "elasticnet_v3_strict": make_model(ElasticNetCV(l1_ratio=[.05,.1,.2,.3,.5,.7,.9], alphas=np.logspace(-4,2,60), max_iter=40000, random_state=RANDOM_STATE), num, cat),
        "gradient_boosting_v3_strict": make_model(GradientBoostingRegressor(random_state=RANDOM_STATE, learning_rate=.035, n_estimators=800, max_depth=2, min_samples_leaf=5, subsample=.85), num, cat),
        "hist_gradient_boosting_v3_strict": make_model(HistGradientBoostingRegressor(random_state=RANDOM_STATE, max_iter=600, learning_rate=.035, max_leaf_nodes=31, l2_regularization=.05), num, cat),
        "extra_trees_v3_strict": make_model(ExtraTreesRegressor(random_state=RANDOM_STATE, n_estimators=600, min_samples_leaf=2, max_features=.8, n_jobs=-1), num, cat),
        "random_forest_v3_strict": make_model(RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=500, min_samples_leaf=3, max_features=.8, n_jobs=-1), num, cat),
    }

    catboost_status = "not_installed"
    if catboost_available():
        try:
            catboost_status = "available"
            models["catboost_native_v3_strict"] = make_native_catboost_model(num, cat)
        except Exception as e:
            catboost_status = f"available_but_model_create_failed: {e}"

    results, preds = {}, {}
    for name, model in models.items():
        print(f"CV: {name}")
        try:
            p = cross_val_predict(model, X, y, cv=cv)
            preds[name] = p
            results[name] = metric_dict(y, p)
            model.fit(X, y)
            joblib.dump(model, ARTIFACTS / f"{name}.joblib")
            save_explainability(model, name)
        except Exception as e:
            results[name] = {"error": str(e)}

    print("Tuning: ridge/gb/hgb simple")
    # Keep tuning limited so runtime does not explode.
    gb_base = make_model(GradientBoostingRegressor(random_state=RANDOM_STATE), num, cat)
    param_dist = {
        "regressor__model__n_estimators": [500,700,900,1100],
        "regressor__model__learning_rate": [.025,.03,.035,.04,.05],
        "regressor__model__max_depth": [2,3],
        "regressor__model__min_samples_leaf": [3,5,8,12,16],
        "regressor__model__subsample": [.75,.85,1.0],
    }
    scorer = make_scorer(lambda yt, yp: r2_score(yt, yp), greater_is_better=True)
    search = RandomizedSearchCV(gb_base, param_dist, n_iter=18, cv=cv, scoring=scorer, random_state=RANDOM_STATE, n_jobs=-1)
    try:
        search.fit(X, y)
        tuned = search.best_estimator_
        tuned_pred = cross_val_predict(tuned, X, y, cv=cv)
        tuned.fit(X, y)
        results["gradient_boosting_tuned_r2_v3_strict"] = metric_dict(y, tuned_pred)
        results["gradient_boosting_tuned_r2_v3_strict"]["best_params"] = search.best_params_
        preds["gradient_boosting_tuned_r2_v3_strict"] = tuned_pred
        joblib.dump(tuned, ARTIFACTS / "gradient_boosting_tuned_r2_v3_strict.joblib")
        save_explainability(tuned, "gradient_boosting_tuned_r2_v3_strict")
    except Exception as e:
        results["gradient_boosting_tuned_r2_v3_strict"] = {"error": str(e)}

    valid_models = {k:v for k,v in results.items() if "r2" in v}
    best_by_r2 = max(valid_models, key=lambda k: valid_models[k]["r2"]) if valid_models else None
    best_by_mape = min(valid_models, key=lambda k: valid_models[k]["mape"]) if valid_models else None

    if best_by_r2:
        joblib.dump(joblib.load(ARTIFACTS / f"{best_by_r2}.joblib"), ARTIFACTS / "best_model_v3_by_r2.joblib")
        write_error_reports(df, preds[best_by_r2], best_by_r2)
    if best_by_mape and best_by_mape != best_by_r2:
        write_error_reports(df, preds[best_by_mape], best_by_mape)

    comparison = pd.DataFrame([
        {"model": k, **{kk: vv for kk, vv in v.items() if kk != "best_params"}}
        for k,v in results.items()
    ])
    if "r2" in comparison.columns:
        comparison = comparison.sort_values("r2", ascending=False, na_position="last")
    comparison.to_csv(REPORTS / "model_comparison_v3.csv", index=False, encoding="utf-8-sig")

    filter_counts = {}
    if len(removed):
        filter_counts = removed["strict_filter_reason"].str.split("; ").explode().value_counts().to_dict()

    report = {
        "target": TARGET,
        "rows_raw": int(len(raw)),
        "rows_used_after_strict_standard_flat_filter": int(len(df)),
        "rows_removed_by_strict_filter": int(len(removed)),
        "filter_counts": filter_counts,
        "strict_filters": {
            "title_exclude_patterns": TITLE_EXCLUDE_PATTERNS,
            "allowed_room_counts": sorted(ALLOWED_ROOM_COUNTS),
            "min_gross_m2": MIN_GROSS_M2,
            "max_gross_m2": MAX_GROSS_M2
        },
        "catboost_status": catboost_status,
        "features": {
            "numeric_used": num,
            "categorical_used": cat,
            "removed_features": removed_features
        },
        "models": results,
        "best_model_by_r2": best_by_r2,
        "best_model_by_mape": best_by_mape,
        "note": "V3 uses strict standard-flat filtering, district interaction features, leakage-safe target/baseline encoding, rare category grouping, and optional native CatBoost."
    }
    (REPORTS / "model_metrics_v3.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
