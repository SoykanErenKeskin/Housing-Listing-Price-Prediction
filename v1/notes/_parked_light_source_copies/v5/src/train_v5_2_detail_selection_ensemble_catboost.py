
from pathlib import Path
import json, re, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV, ElasticNetCV, Ridge
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, make_scorer

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data/input/listing_dataset_cleaned.csv"
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ROOT / "data/output"
REPORTS = ROOT / "reports"

for p in [ARTIFACTS, OUTPUT, REPORTS]:
    p.mkdir(parents=True, exist_ok=True)

TARGET = "unit_price_gross"
RANDOM_STATE = 42
DETAIL_MIN_ONES = 20

DETAIL_PREFIXES = ("front_", "view_", "transport_", "near_", "out_", "in_", "subtype_")

BASE_NUMERIC = [
    "gross_m2", "net_m2", "building_age", "floor_num", "total_floors", "bathroom_count",
    "open_area_m2", "net_gross_ratio", "has_open_area",
    "floor_ratio", "remaining_floors", "is_ground_floor", "is_basement", "is_top_floor", "is_middle_floor",
    "rooms", "living_rooms", "total_room_score",
    "is_new_building", "is_old_building", "is_small_flat", "is_large_flat",
    "quality_score", "district_target_encoded", "county_target_encoded",
    "district_baseline_unit_price", "county_baseline_unit_price",
    "detail_selected_count", "detail_quality_score",
    "detail_front_count", "detail_view_count", "detail_transport_count", "detail_near_count",
    "detail_inside_count", "detail_outside_count", "detail_subtype_count",
    "front_score", "view_score", "transport_score", "nearby_score",
    "inside_quality_score", "outside_quality_score", "premium_detail_score",
    "site_security_score", "accessibility_score"
]

BASE_CATEGORICAL = [
    "real_estate_type", "room_count", "floor_segment", "heating", "kitchen",
    "balcony", "elevator", "parking", "furnished", "usage_status", "site_inside",
    "credit_eligible", "energy_certificate", "deed_status", "seller_type", "barter",
    "city", "county", "district", "building_age_group", "m2_group",
    "district_age_group", "district_m2_group", "district_room_count",
    "district_view_group", "district_transport_group", "district_quality_group",
    "district_site_inside", "detail_cephe", "detail_manzara", "detail_konut_tipi"
]

DETAIL_RAW_COLUMNS = {
    "detail_cephe": "detail_front_count",
    "detail_manzara": "detail_view_count",
    "detail_ulasim": "detail_transport_count",
    "detail_muhit": "detail_near_count",
    "detail_ic_ozellikler": "detail_inside_count",
    "detail_dis_ozellikler": "detail_outside_count",
    "detail_konut_tipi": "detail_subtype_count",
}

SCORE_GROUPS = {
    "front_score": ["front_west", "front_east", "front_south", "front_north"],
    "view_score": ["view_bosphorus", "view_sea", "view_nature", "view_lake", "view_pool", "view_river", "view_park_green", "view_city"],
    "transport_score": ["transport_main_road", "transport_e5", "transport_tem", "transport_tram", "transport_train", "transport_bus_stop", "transport_minibus", "transport_metro", "transport_airport"],
    "nearby_score": ["near_mall", "near_mosque", "near_pharmacy", "near_hospital", "near_school", "near_university", "near_market", "near_park", "near_city_center", "near_sea_zero", "near_lake_zero"],
    "outside_quality_score": ["out_security", "out_camera", "out_pool", "out_open_pool", "out_closed_pool", "out_heat_insulation", "out_sound_insulation", "out_generator", "out_hydrofor", "out_children_playground", "out_sports_area", "out_sauna_hamam"],
    "inside_quality_score": ["in_builtin_kitchen", "in_parent_bathroom", "in_glass_balcony", "in_terrace", "in_air_conditioner", "in_smart_home", "in_steel_door", "in_fiber", "in_intercom", "in_dressing_room", "in_pantry", "in_laminate_floor", "in_pvc", "in_heat_glass"],
    "premium_detail_score": ["view_sea", "view_lake", "view_nature", "view_park_green", "out_pool", "out_security", "out_camera", "in_parent_bathroom", "in_builtin_kitchen", "in_smart_home", "subtype_middle_floor"],
    "site_security_score": ["out_security", "out_camera", "out_generator", "out_hydrofor", "out_heat_insulation", "out_sound_insulation"],
    "accessibility_score": ["transport_tram", "transport_bus_stop", "transport_minibus", "transport_main_road", "transport_tem", "transport_e5", "near_market", "near_hospital", "near_school"]
}

SCORE_COLUMNS = set(SCORE_GROUPS.keys())

def unique_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out

def is_detail_binary_col(col):
    # Score columns like front_score/view_score/inside_quality_score also match prefixes,
    # but they are aggregate numeric features, not raw binary detail flags.
    return col.startswith(DETAIL_PREFIXES) and col not in SCORE_COLUMNS and not col.endswith("_score")

def dedupe_dataframe_columns(df):
    # Pandas returns a DataFrame instead of Series when duplicate column labels exist.
    # This defensive cleanup prevents CatBoost/manual preprocessing crashes.
    return df.loc[:, ~df.columns.duplicated()].copy()


def clean_str(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    return np.nan if s == "" or s.lower() in {"nan", "none", "null"} else s

def to_num(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).replace("TL", "").replace("₺", "").replace("m²", "").replace("m2", "")
    s = s.replace(".", "").replace(",", ".")
    s = "".join(ch for ch in s if ch.isdigit() or ch in ".-")
    try:
        return float(s)
    except Exception:
        return np.nan

def floor_to_num(v):
    if pd.isna(v):
        return np.nan
    s = str(v).lower()
    if "bodrum" in s:
        return -1.0
    if "zemin" in s or "giriş" in s or "giris" in s or "bahçe" in s or "bahce" in s:
        return 0.0
    if "çatı" in s or "cati" in s:
        return np.nan
    nums = "".join(ch if ch.isdigit() or ch == "-" else " " for ch in s).split()
    return float(nums[0]) if nums else np.nan

def parse_room(v):
    if pd.isna(v):
        return np.nan, np.nan, np.nan
    s = str(v).replace(" ", "").lower()
    m = re.search(r"(\d+)\+(\d+)", s)
    if m:
        r, l = float(m.group(1)), float(m.group(2))
        return r, l, r + l
    m = re.search(r"(\d+)", s)
    return (float(m.group(1)), np.nan, float(m.group(1))) if m else (np.nan, np.nan, np.nan)

def count_pipe_values(x):
    if pd.isna(x):
        return 0
    s = str(x).strip()
    return 0 if not s else len([p for p in s.split("|") if p.strip()])

def prepare_raw(df):
    df = df.copy()

    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].map(clean_str)

    for c in [
        "price", "unit_price_gross", "gross_m2", "net_m2", "open_area_m2",
        "building_age", "total_floors", "bathroom_count",
        "detail_selected_count", "detail_quality_score"
    ]:
        if c in df.columns:
            df[c] = df[c].map(to_num)

    if TARGET not in df.columns:
        df[TARGET] = np.nan

    if "price" in df.columns and "gross_m2" in df.columns:
        m = df[TARGET].isna() & df["price"].notna() & df["gross_m2"].notna() & (df["gross_m2"] > 0)
        df.loc[m, TARGET] = df.loc[m, "price"] / df.loc[m, "gross_m2"]

    if {"net_m2", "gross_m2"}.issubset(df.columns):
        df["net_gross_ratio"] = df["net_m2"] / df["gross_m2"]
    else:
        df["net_gross_ratio"] = np.nan

    if "open_area_m2" in df.columns:
        df["has_open_area"] = df["open_area_m2"].fillna(0).gt(0).astype(int)
    else:
        df["open_area_m2"] = np.nan
        df["has_open_area"] = 0

    df["floor_num"] = df["floor"].map(floor_to_num) if "floor" in df.columns else np.nan

    for c in df.columns:
        if c.startswith(DETAIL_PREFIXES):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(0, 1).astype(int)

    for raw_col, count_col in DETAIL_RAW_COLUMNS.items():
        df[count_col] = df[raw_col].map(count_pipe_values).astype(int) if raw_col in df.columns else 0

    if "detail_selected_count" not in df.columns:
        df["detail_selected_count"] = df[list(DETAIL_RAW_COLUMNS.values())].sum(axis=1)

    if "detail_quality_score" not in df.columns:
        cols = [c for c in df.columns if c.startswith(("out_", "in_"))]
        df["detail_quality_score"] = df[cols].sum(axis=1) if cols else 0

    for score_col, cols in SCORE_GROUPS.items():
        available = [c for c in cols if c in df.columns]
        df[score_col] = df[available].sum(axis=1) if available else 0

    valid = (
        df[TARGET].notna()
        & df["gross_m2"].notna()
        & (df["gross_m2"] > 20)
        & (df["gross_m2"] < 1000)
        & (df[TARGET] > 1000)
        & (df[TARGET] < 1000000)
    )
    return df.loc[valid].copy()

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        for c in ["gross_m2", "net_m2", "building_age", "floor_num", "total_floors", "bathroom_count"]:
            if c not in df.columns:
                df[c] = np.nan

        denom = df["total_floors"].replace(0, np.nan)
        df["floor_ratio"] = df["floor_num"] / denom
        df["remaining_floors"] = df["total_floors"] - df["floor_num"]
        df["is_ground_floor"] = (df["floor_num"] == 0).astype(int)
        df["is_basement"] = (df["floor_num"] < 0).astype(int)
        df["is_top_floor"] = (
            df["floor_num"].notna() & df["total_floors"].notna()
            & (df["total_floors"] > 0)
            & (df["floor_num"] >= df["total_floors"])
        ).astype(int)
        df["is_middle_floor"] = (
            df["floor_num"].notna() & df["total_floors"].notna()
            & (df["floor_num"] > 0)
            & (df["floor_num"] < df["total_floors"])
        ).astype(int)

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
        df["building_age_group"] = pd.cut(
            age, [-1, 0, 5, 10, 20, 30, 200],
            labels=["0", "1-5", "6-10", "11-20", "21-30", "30+"]
        ).astype(object)

        gross = df["gross_m2"]
        df["is_small_flat"] = gross.fillna(9999).le(75).astype(int)
        df["is_large_flat"] = gross.fillna(0).ge(160).astype(int)
        df["m2_group"] = pd.cut(
            gross, [0, 75, 100, 125, 150, 200, 1000],
            labels=["0-75", "76-100", "101-125", "126-150", "151-200", "200+"]
        ).astype(object)

        q = pd.Series(0.0, index=df.index)

        def yes_like(s):
            return s.fillna("").astype(str).str.lower().str.contains("var|evet|açık|kapalı|kapali|site", regex=True)

        if "elevator" in df.columns:
            q += yes_like(df["elevator"]).astype(int)
        if "parking" in df.columns:
            q += yes_like(df["parking"]).astype(int)
        if "site_inside" in df.columns:
            q += yes_like(df["site_inside"]).astype(int)
        if "bathroom_count" in df.columns:
            q += df["bathroom_count"].fillna(1).ge(2).astype(int)
        if "heating" in df.columns:
            q += df["heating"].fillna("").astype(str).str.lower().str.contains("merkezi|kombi|yerden", regex=True).astype(int)
        if "detail_quality_score" in df.columns:
            q += pd.to_numeric(df["detail_quality_score"], errors="coerce").fillna(0) * 0.35

        df["quality_score"] = q

        for score_col, cols in SCORE_GROUPS.items():
            available = [c for c in cols if c in df.columns]
            df[score_col] = df[available].sum(axis=1) if available else 0

        def comb(a, b):
            aa = df[a].fillna("missing").astype(str) if a in df.columns else pd.Series("missing", index=df.index)
            bb = df[b].fillna("missing").astype(str) if b in df.columns else pd.Series("missing", index=df.index)
            return aa + "__" + bb

        df["district_age_group"] = comb("district", "building_age_group")
        df["district_m2_group"] = comb("district", "m2_group")
        df["district_room_count"] = comb("district", "room_count")
        df["district_view_group"] = comb("district", "view_score")
        df["district_transport_group"] = comb("district", "transport_score")
        df["district_quality_group"] = comb("district", "premium_detail_score")
        df["district_site_inside"] = comb("district", "site_inside")

        return df

class RareBinaryDropper(BaseEstimator, TransformerMixin):
    def __init__(self, detail_cols=None, min_ones=20):
        self.detail_cols = detail_cols or []
        self.min_ones = min_ones

    def fit(self, X, y=None):
        X = X.copy()
        self.keep_ = []
        self.removed_ = {}
        for c in self.detail_cols:
            if c not in X.columns:
                self.removed_[c] = "missing"
                continue
            ones = pd.to_numeric(X[c], errors="coerce").fillna(0).sum()
            if ones < self.min_ones:
                self.removed_[c] = f"rare_ones_lt_{self.min_ones}"
            else:
                self.keep_.append(c)
        return self

    def transform(self, X):
        X = X.copy()
        # Important: do NOT drop columns here. ColumnTransformer is configured
        # with the feature list decided before CV. If a rare binary column is
        # physically dropped inside a fold, sklearn raises:
        # "A given column is not a column of the dataframe".
        # We keep the column and neutralize it by setting it to 0.
        for c in self.detail_cols:
            if c not in X.columns:
                X[c] = 0
            elif c not in self.keep_:
                X[c] = 0
        return X

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
            if col not in X.columns:
                continue
            tmp = pd.DataFrame({"key": X[col].fillna("missing").astype(str), "target": y.values})
            stats = tmp.groupby("key")["target"].agg(["mean", "count"])
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
            if col not in X.columns:
                continue
            tmp = pd.DataFrame({"key": X[col].fillna("missing").astype(str), "target": y.values})
            stats = tmp.groupby("key")["target"].agg(["median", "count"])
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
    def __init__(self, columns=None, min_count=5):
        self.columns = columns or []
        self.min_count = min_count

    def fit(self, X, y=None):
        X = X.copy()
        self.valid_values_ = {}
        for col in self.columns:
            if col not in X.columns:
                continue
            counts = X[col].fillna("missing").astype(str).value_counts()
            self.valid_values_[col] = set(counts[counts >= self.min_count].index)
        return self

    def transform(self, X):
        X = X.copy()
        for col, valid in self.valid_values_.items():
            if col not in X.columns:
                continue
            vals = X[col].fillna("missing").astype(str)
            X[col] = vals.where(vals.isin(valid), "other")
        return X

def get_detail_cols(df):
    return [c for c in df.columns if is_detail_binary_col(c)]

def remove_useless_features(df, num, cat):
    num = unique_preserve_order(num)
    cat = unique_preserve_order(cat)
    keep_num, keep_cat, removed = [], [], {}
    for c in num:
        if c not in df.columns:
            removed[c] = "missing"
        elif df[c].notna().sum() == 0:
            removed[c] = "all_missing"
        elif df[c].nunique(dropna=True) <= 1:
            removed[c] = "constant"
        else:
            keep_num.append(c)
    for c in cat:
        if c not in df.columns:
            removed[c] = "missing"
        elif df[c].notna().sum() == 0:
            removed[c] = "all_missing"
        elif df[c].nunique(dropna=True) <= 1:
            removed[c] = "constant"
        else:
            keep_cat.append(c)
    return keep_num, keep_cat, removed

def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)))

def metric_dict(y, p):
    p = np.asarray(p, dtype=float)
    return {
        "mape": mape(y, p),
        "mae_tl_per_m2": float(mean_absolute_error(y, p)),
        "median_ae_tl_per_m2": float(median_absolute_error(y, p)),
        "r2": float(r2_score(y, p)),
        "log_r2": float(r2_score(np.log1p(y), np.log1p(np.maximum(p, 0))))
    }

def make_preprocessor(num, cat):
    num = unique_preserve_order(num)
    cat = unique_preserve_order(cat)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=3, sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=3, sparse=False)

    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", ohe)
        ]), cat)
    ])

def make_model(estimator, num, cat, detail_cols):
    num = unique_preserve_order(num)
    cat = unique_preserve_order(cat)
    detail_cols = unique_preserve_order(detail_cols)
    return TransformedTargetRegressor(
        regressor=Pipeline([
            ("feature_engineering", FeatureEngineer()),
            ("rare_binary_dropper", RareBinaryDropper(detail_cols=detail_cols, min_ones=DETAIL_MIN_ONES)),
            ("rare_category_grouper", RareCategoryGrouper(columns=cat, min_count=5)),
            ("baseline_encoding", BaselineEncoder(columns=["district", "county"], smoothing=10)),
            ("target_encoding", TargetEncoder(columns=["district", "county"], smoothing=20)),
            ("preprocess", make_preprocessor(num, cat)),
            ("model", estimator)
        ]),
        func=np.log1p,
        inverse_func=np.expm1
    )

def save_explainability(model, name):
    try:
        inner = model.regressor_
        pre = inner.named_steps["preprocess"]
        est = inner.named_steps["model"]
        names = pre.get_feature_names_out()
        if hasattr(est, "coef_"):
            out = pd.DataFrame({"feature": names, "coefficient": est.coef_})
            out["abs_coefficient"] = out["coefficient"].abs()
            out.sort_values("abs_coefficient", ascending=False).to_csv(ARTIFACTS / f"{name}_coefficients.csv", index=False, encoding="utf-8-sig")
        if hasattr(est, "feature_importances_"):
            out = pd.DataFrame({"feature": names, "importance": est.feature_importances_})
            out.sort_values("importance", ascending=False).to_csv(ARTIFACTS / f"{name}_feature_importance.csv", index=False, encoding="utf-8-sig")
    except Exception:
        pass

def fit_predict_cv(model, X, y, cv, name):
    print(f"CV: {name}")
    pred = cross_val_predict(model, X, y, cv=cv)
    fitted = clone(model)
    fitted.fit(X, y)
    joblib.dump(fitted, ARTIFACTS / f"{name}.joblib")
    save_explainability(fitted, name)
    return pred, fitted

def weighted_ensemble_candidates(preds, y):
    keys = [k for k in preds.keys() if any(s in k for s in ["hist", "gradient_boosting", "extra_trees", "random_forest"])]
    ens_results = {}
    if len(keys) < 2:
        return {}, {}

    # Predefined robust blends
    blends = {
        "ensemble_hgb_gb_et": {
            "hist_gradient_boosting_v5": 0.45,
            "gradient_boosting_tuned_r2_v5": 0.35,
            "extra_trees_v5": 0.20
        },
        "ensemble_gb_hgb": {
            "gradient_boosting_tuned_r2_v5": 0.50,
            "hist_gradient_boosting_v5": 0.50
        },
        "ensemble_hgb_et_rf": {
            "hist_gradient_boosting_v5": 0.45,
            "extra_trees_v5": 0.35,
            "random_forest_v5": 0.20
        }
    }

    pred_out = {}
    for name, weights in blends.items():
        usable = {k: w for k, w in weights.items() if k in preds}
        if len(usable) < 2:
            continue
        total = sum(usable.values())
        p = sum(preds[k] * (w / total) for k, w in usable.items())
        pred_out[name] = p
        ens_results[name] = metric_dict(y, p)

    # Small grid for top 3 by R2
    top3 = sorted(keys, key=lambda k: r2_score(y, preds[k]), reverse=True)[:3]
    if len(top3) >= 2:
        grid_best = None
        grid_best_metric = None
        grid_best_weights = None
        weights_grid = np.arange(0, 1.01, 0.1)
        if len(top3) == 2:
            a, b = top3
            for wa in weights_grid:
                wb = 1 - wa
                p = preds[a] * wa + preds[b] * wb
                m = metric_dict(y, p)
                if grid_best_metric is None or m["r2"] > grid_best_metric["r2"]:
                    grid_best, grid_best_metric, grid_best_weights = p, m, {a: float(wa), b: float(wb)}
        else:
            a, b, c = top3
            for wa in weights_grid:
                for wb in weights_grid:
                    wc = 1 - wa - wb
                    if wc < -1e-9:
                        continue
                    p = preds[a] * wa + preds[b] * wb + preds[c] * wc
                    m = metric_dict(y, p)
                    if grid_best_metric is None or m["r2"] > grid_best_metric["r2"]:
                        grid_best, grid_best_metric, grid_best_weights = p, m, {a: float(wa), b: float(wb), c: float(wc)}
        if grid_best is not None:
            grid_best_metric["weights"] = grid_best_weights
            pred_out["ensemble_grid_best_r2_v5"] = grid_best
            ens_results["ensemble_grid_best_r2_v5"] = grid_best_metric

    return ens_results, pred_out

def manual_catboost_cv(X, y, cv, num, cat):
    try:
        from catboost import CatBoostRegressor
    except Exception as e:
        return None, {"error": f"catboost_not_available: {e}"}

    # Prepare features once, then do manual leakage-safe encodings per fold.
    base_fe = FeatureEngineer().fit_transform(X.copy())
    detail_cols = get_detail_cols(base_fe)
    used_num = [c for c in num if c in base_fe.columns]
    used_cat = [c for c in cat if c in base_fe.columns]
    all_cols = used_num + used_cat

    preds = np.zeros(len(X), dtype=float)
    fold_metrics = []

    for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
        X_tr_raw = X.iloc[tr].copy()
        X_va_raw = X.iloc[va].copy()
        y_tr = y.iloc[tr].copy()
        y_va = y.iloc[va].copy()

        steps = Pipeline([
            ("feature_engineering", FeatureEngineer()),
            ("rare_binary_dropper", RareBinaryDropper(detail_cols=detail_cols, min_ones=DETAIL_MIN_ONES)),
            ("rare_category_grouper", RareCategoryGrouper(columns=cat, min_count=5)),
            ("baseline_encoding", BaselineEncoder(columns=["district", "county"], smoothing=10)),
            ("target_encoding", TargetEncoder(columns=["district", "county"], smoothing=20)),
        ])

        X_tr = steps.fit_transform(X_tr_raw, y_tr)
        X_va = steps.transform(X_va_raw)

        keep_num, keep_cat, _ = remove_useless_features(X_tr, num, cat)
        keep_num = unique_preserve_order(keep_num)
        keep_cat = unique_preserve_order([c for c in keep_cat if c not in keep_num])
        cols = unique_preserve_order(keep_num + keep_cat)

        X_tr = dedupe_dataframe_columns(X_tr)
        X_va = dedupe_dataframe_columns(X_va)
        X_tr = X_tr[cols].copy()
        X_va = X_va[cols].copy()

        for c in keep_num:
            med = pd.to_numeric(X_tr[c], errors="coerce").median()
            X_tr[c] = pd.to_numeric(X_tr[c], errors="coerce").fillna(med)
            X_va[c] = pd.to_numeric(X_va[c], errors="coerce").fillna(med)

        for c in keep_cat:
            X_tr[c] = X_tr[c].fillna("missing").astype(str)
            X_va[c] = X_va[c].fillna("missing").astype(str)

        cat_features = [c for c in keep_cat if c in X_tr.columns]

        model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=1600,
            learning_rate=0.025,
            depth=6,
            l2_leaf_reg=8,
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False
        )

        model.fit(X_tr, np.log1p(y_tr), cat_features=cat_features)
        pred = np.expm1(model.predict(X_va))
        preds[va] = pred
        fold_metrics.append(metric_dict(y_va, pred))

    # Fit final model artifacts
    steps = Pipeline([
        ("feature_engineering", FeatureEngineer()),
        ("rare_binary_dropper", RareBinaryDropper(detail_cols=get_detail_cols(base_fe), min_ones=DETAIL_MIN_ONES)),
        ("rare_category_grouper", RareCategoryGrouper(columns=cat, min_count=5)),
        ("baseline_encoding", BaselineEncoder(columns=["district", "county"], smoothing=10)),
        ("target_encoding", TargetEncoder(columns=["district", "county"], smoothing=20)),
    ])
    X_all = steps.fit_transform(X.copy(), y)
    keep_num, keep_cat, _ = remove_useless_features(X_all, num, cat)
    keep_num = unique_preserve_order(keep_num)
    keep_cat = unique_preserve_order([c for c in keep_cat if c not in keep_num])
    cols = unique_preserve_order(keep_num + keep_cat)

    X_all = dedupe_dataframe_columns(X_all)
    X_all = X_all[cols].copy()
    for c in keep_num:
        med = pd.to_numeric(X_all[c], errors="coerce").median()
        X_all[c] = pd.to_numeric(X_all[c], errors="coerce").fillna(med)
    for c in keep_cat:
        X_all[c] = X_all[c].fillna("missing").astype(str)

    final_model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=1600,
        learning_rate=0.025,
        depth=6,
        l2_leaf_reg=8,
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False
    )
    final_model.fit(X_all, np.log1p(y), cat_features=keep_cat)

    joblib.dump({
        "preprocess_steps": steps,
        "columns": cols,
        "numeric_cols": keep_num,
        "categorical_cols": keep_cat,
        "model": final_model,
        "target_transform": "log1p"
    }, ARTIFACTS / "catboost_manual_v5.joblib")

    feature_importance = pd.DataFrame({
        "feature": cols,
        "importance": final_model.get_feature_importance()
    }).sort_values("importance", ascending=False)
    feature_importance.to_csv(ARTIFACTS / "catboost_manual_v5_feature_importance.csv", index=False, encoding="utf-8-sig")

    result = metric_dict(y, preds)
    result["fold_metrics"] = fold_metrics
    return preds, result

def write_error_reports(df, pred, name):
    view = FeatureEngineer().fit_transform(df.copy())
    out = view.copy()
    out[f"{name}_pred_unit_price"] = pred
    out[f"{name}_abs_pct_error"] = np.abs(out[TARGET] - pred) / out[TARGET]
    out[f"{name}_abs_error"] = np.abs(out[TARGET] - pred)
    out.to_csv(OUTPUT / f"{name}_cv_predictions.csv", index=False, encoding="utf-8-sig")

    err, abs_err = f"{name}_abs_pct_error", f"{name}_abs_error"
    for col in [
        "district", "room_count", "floor_segment", "building_age_group", "m2_group",
        "heating", "site_inside", "detail_cephe", "detail_manzara", "detail_konut_tipi"
    ]:
        if col in out.columns:
            rep = out.groupby(col, dropna=False).agg(
                n=(TARGET, "size"),
                mape=(err, "mean"),
                median_ape=(err, "median"),
                mae_tl_per_m2=(abs_err, "mean"),
                median_ae_tl_per_m2=(abs_err, "median"),
                mean_unit_price=(TARGET, "mean")
            ).reset_index().sort_values("mape", ascending=False)
            rep.to_csv(OUTPUT / f"{name}_error_by_{col}.csv", index=False, encoding="utf-8-sig")

    out.sort_values(err, ascending=False).head(50).to_csv(OUTPUT / f"{name}_top_50_errors.csv", index=False, encoding="utf-8-sig")

def main():
    warnings.filterwarnings("ignore")
    raw = pd.read_csv(INPUT_PATH)
    df = prepare_raw(raw)

    detail_cols_all = get_detail_cols(df)
    detail_coverage_rows = []
    selected_detail_cols = []
    for c in detail_cols_all:
        vals = pd.to_numeric(df[c], errors="coerce").fillna(0)
        ones = int(vals.sum())
        detail_coverage_rows.append({"feature": c, "ones": ones, "ratio": float(vals.mean()), "selected_min_ones": ones >= DETAIL_MIN_ONES})
        if ones >= DETAIL_MIN_ONES:
            if c not in selected_detail_cols:
                selected_detail_cols.append(c)

    pd.DataFrame(detail_coverage_rows).sort_values("ones", ascending=False).to_csv(
        REPORTS / "detail_feature_coverage_v5_2.csv",
        index=False,
        encoding="utf-8-sig"
    )

    candidate_numeric = unique_preserve_order(BASE_NUMERIC + selected_detail_cols)
    candidate_categorical = unique_preserve_order(BASE_CATEGORICAL)

    fe = FeatureEngineer().fit_transform(df.copy())
    be = BaselineEncoder(columns=["district", "county"], smoothing=10).fit(fe.copy(), df[TARGET]).transform(fe.copy())
    te = TargetEncoder(columns=["district", "county"], smoothing=20).fit(be.copy(), df[TARGET]).transform(be.copy())
    num, cat, removed_features = remove_useless_features(te, candidate_numeric, candidate_categorical)

    X = df.drop(columns=[TARGET], errors="ignore").copy()
    y = df[TARGET].astype(float)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "ridge_v5": make_model(RidgeCV(alphas=np.logspace(-3, 4, 60)), num, cat, selected_detail_cols),
        "elasticnet_v5": make_model(ElasticNetCV(
            l1_ratio=[.05, .1, .2, .3, .5, .7, .9],
            alphas=np.logspace(-4, 2, 60),
            max_iter=40000,
            random_state=RANDOM_STATE
        ), num, cat, selected_detail_cols),
        "gradient_boosting_v5": make_model(GradientBoostingRegressor(
            random_state=RANDOM_STATE,
            learning_rate=.035,
            n_estimators=800,
            max_depth=2,
            min_samples_leaf=5,
            subsample=.85
        ), num, cat, selected_detail_cols),
        "hist_gradient_boosting_v5": make_model(HistGradientBoostingRegressor(
            random_state=RANDOM_STATE,
            max_iter=700,
            learning_rate=.03,
            max_leaf_nodes=31,
            l2_regularization=.05
        ), num, cat, selected_detail_cols),
        "extra_trees_v5": make_model(ExtraTreesRegressor(
            random_state=RANDOM_STATE,
            n_estimators=700,
            min_samples_leaf=2,
            max_features=.8,
            n_jobs=-1
        ), num, cat, selected_detail_cols),
        "random_forest_v5": make_model(RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_estimators=600,
            min_samples_leaf=3,
            max_features=.8,
            n_jobs=-1
        ), num, cat, selected_detail_cols),
    }

    results, preds = {}, {}
    for name, model in models.items():
        try:
            pred, fitted = fit_predict_cv(model, X, y, cv, name)
            preds[name] = pred
            results[name] = metric_dict(y, pred)
        except Exception as e:
            results[name] = {"error": str(e)}

    print("Tuning: gradient_boosting_tuned_r2_v5")
    gb_base = make_model(GradientBoostingRegressor(random_state=RANDOM_STATE), num, cat, selected_detail_cols)
    param_dist = {
        "regressor__model__n_estimators": [700, 900, 1100, 1300],
        "regressor__model__learning_rate": [.02, .025, .03, .035, .04],
        "regressor__model__max_depth": [2, 3],
        "regressor__model__min_samples_leaf": [5, 8, 12, 16, 20],
        "regressor__model__subsample": [.75, .85, 1.0],
    }
    scorer = make_scorer(lambda yt, yp: r2_score(yt, yp), greater_is_better=True)
    search = RandomizedSearchCV(gb_base, param_dist, n_iter=20, cv=cv, scoring=scorer, random_state=RANDOM_STATE, n_jobs=-1)
    try:
        search.fit(X, y)
        tuned = search.best_estimator_
        tuned_pred = cross_val_predict(tuned, X, y, cv=cv)
        tuned.fit(X, y)
        results["gradient_boosting_tuned_r2_v5"] = metric_dict(y, tuned_pred)
        results["gradient_boosting_tuned_r2_v5"]["best_params"] = search.best_params_
        preds["gradient_boosting_tuned_r2_v5"] = tuned_pred
        joblib.dump(tuned, ARTIFACTS / "gradient_boosting_tuned_r2_v5.joblib")
        save_explainability(tuned, "gradient_boosting_tuned_r2_v5")
    except Exception as e:
        results["gradient_boosting_tuned_r2_v5"] = {"error": str(e)}

    print("Manual CatBoost CV")
    cb_pred, cb_result = manual_catboost_cv(X, y, cv, num, cat)
    if cb_pred is not None:
        preds["catboost_manual_v5"] = cb_pred
    results["catboost_manual_v5"] = cb_result

    ens_results, ens_preds = weighted_ensemble_candidates(preds, y)
    for name, result in ens_results.items():
        results[name] = result
        preds[name] = ens_preds[name]

    valid = {k: v for k, v in results.items() if "r2" in v}
    best_by_r2 = max(valid, key=lambda k: valid[k]["r2"]) if valid else None
    best_by_mape = min(valid, key=lambda k: valid[k]["mape"]) if valid else None

    if best_by_r2:
        write_error_reports(df, preds[best_by_r2], best_by_r2)
        if (ARTIFACTS / f"{best_by_r2}.joblib").exists():
            joblib.dump(joblib.load(ARTIFACTS / f"{best_by_r2}.joblib"), ARTIFACTS / "best_model_v5_by_r2.joblib")
        elif best_by_r2 == "catboost_manual_v5":
            shutil_path = ARTIFACTS / "catboost_manual_v5.joblib"
            if shutil_path.exists():
                joblib.dump(joblib.load(shutil_path), ARTIFACTS / "best_model_v5_by_r2.joblib")
    if best_by_mape and best_by_mape != best_by_r2:
        write_error_reports(df, preds[best_by_mape], best_by_mape)

    comparison = pd.DataFrame([
        {"model": k, **{kk: vv for kk, vv in v.items() if kk not in {"best_params", "fold_metrics", "weights"}}}
        for k, v in results.items()
    ])
    if "r2" in comparison.columns:
        comparison = comparison.sort_values("r2", ascending=False, na_position="last")
    comparison.to_csv(REPORTS / "model_comparison_v5_2.csv", index=False, encoding="utf-8-sig")

    # Save ensemble weights and tuning info
    aux = {
        "ensemble_weights": {k: v.get("weights") for k, v in results.items() if "weights" in v},
        "tuned_params": {k: v.get("best_params") for k, v in results.items() if "best_params" in v},
    }
    (REPORTS / "model_aux_v5_2.json").write_text(json.dumps(aux, indent=2, ensure_ascii=False), encoding="utf-8")

    report = {
        "target": TARGET,
        "rows_raw": int(len(raw)),
        "rows_used": int(len(df)),
        "detail_selection": {
            "min_ones": DETAIL_MIN_ONES,
            "detail_columns_found": detail_cols_all,
            "detail_columns_selected": selected_detail_cols,
            "detail_columns_removed_as_rare": [c for c in detail_cols_all if c not in selected_detail_cols],
            "rows_with_any_detail_binary": int((df[detail_cols_all].sum(axis=1) > 0).sum()) if detail_cols_all else 0
        },
        "features": {
            "numeric_used": num,
            "categorical_used": cat,
            "removed_features": removed_features
        },
        "models": results,
        "best_model_by_r2": best_by_r2,
        "best_model_by_mape": best_by_mape,
        "note": "V5.2 includes rare detail feature selection, grouped detail scores, district-detail interactions, tuned models, weighted ensembles, and manual CatBoost CV."
    }
    (REPORTS / "model_metrics_v5_2.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
