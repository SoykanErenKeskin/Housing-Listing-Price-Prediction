
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
INPUT_PATH = ROOT/'data'/'input'/'listing_dataset_cleaned.csv'
ARTIFACTS = ROOT/'artifacts'; OUTPUT = ROOT/'data'/'output'; REPORTS = ROOT/'reports'
for p in [ARTIFACTS, OUTPUT, REPORTS]: p.mkdir(parents=True, exist_ok=True)
TARGET='unit_price_gross'; RANDOM_STATE=42
DETAIL_BINARY_PREFIXES=('front_','view_','transport_','near_','out_','in_','subtype_')
DETAIL_RAW_COLUMNS={'detail_cephe':'detail_front_count','detail_manzara':'detail_view_count','detail_ulasim':'detail_transport_count','detail_muhit':'detail_near_count','detail_ic_ozellikler':'detail_inside_count','detail_dis_ozellikler':'detail_outside_count','detail_konut_tipi':'detail_subtype_count'}
BASE_NUMERIC=['gross_m2','net_m2','building_age','floor_num','total_floors','bathroom_count','open_area_m2','net_gross_ratio','has_open_area','floor_ratio','remaining_floors','is_ground_floor','is_basement','is_top_floor','is_middle_floor','rooms','living_rooms','total_room_score','is_new_building','is_old_building','is_small_flat','is_large_flat','quality_score','district_target_encoded','county_target_encoded','district_baseline_unit_price','county_baseline_unit_price','detail_selected_count','detail_quality_score','detail_front_count','detail_view_count','detail_transport_count','detail_near_count','detail_inside_count','detail_outside_count','detail_subtype_count']
BASE_CATEGORICAL=['real_estate_type','room_count','floor_segment','heating','kitchen','balcony','elevator','parking','furnished','usage_status','site_inside','credit_eligible','energy_certificate','deed_status','seller_type','barter','city','county','district','building_age_group','m2_group','district_age_group','district_m2_group','district_room_count','detail_cephe','detail_manzara','detail_konut_tipi']

def clean_str(x):
    if pd.isna(x): return np.nan
    s=str(x).strip(); return np.nan if s=='' or s.lower() in {'nan','none','null'} else s

def to_num(x):
    if pd.isna(x): return np.nan
    if isinstance(x,(int,float,np.number)): return float(x)
    s=str(x).replace('TL','').replace('₺','').replace('m²','').replace('m2','').replace('.','').replace(',','.')
    s=''.join(ch for ch in s if ch.isdigit() or ch in '.-')
    try: return float(s)
    except Exception: return np.nan

def floor_to_num(v):
    if pd.isna(v): return np.nan
    s=str(v).lower()
    if 'bodrum' in s: return -1.0
    if any(x in s for x in ['zemin','giriş','giris','bahçe','bahce']): return 0.0
    if 'çatı' in s or 'cati' in s: return np.nan
    nums=''.join(ch if ch.isdigit() or ch=='-' else ' ' for ch in s).split()
    return float(nums[0]) if nums else np.nan

def parse_room(v):
    if pd.isna(v): return np.nan,np.nan,np.nan
    s=str(v).replace(' ','').lower(); m=re.search(r'(\d+)\+(\d+)',s)
    if m:
        r,l=float(m.group(1)),float(m.group(2)); return r,l,r+l
    m=re.search(r'(\d+)',s); return (float(m.group(1)),np.nan,float(m.group(1))) if m else (np.nan,np.nan,np.nan)

def count_pipe_values(x):
    if pd.isna(x): return 0
    s=str(x).strip(); return len([p for p in s.split('|') if p.strip()]) if s else 0

def prepare_raw(df):
    df=df.copy()
    for c in df.columns:
        if df[c].dtype=='object': df[c]=df[c].map(clean_str)
    for c in ['price','unit_price_gross','gross_m2','net_m2','open_area_m2','building_age','total_floors','bathroom_count','detail_selected_count','detail_quality_score']:
        if c in df.columns: df[c]=df[c].map(to_num)
    if TARGET not in df.columns: df[TARGET]=np.nan
    if 'price' in df.columns and 'gross_m2' in df.columns:
        m=df[TARGET].isna() & df['price'].notna() & df['gross_m2'].notna() & (df['gross_m2']>0)
        df.loc[m,TARGET]=df.loc[m,'price']/df.loc[m,'gross_m2']
    df['net_gross_ratio']=df['net_m2']/df['gross_m2'] if {'net_m2','gross_m2'}.issubset(df.columns) else np.nan
    df['open_area_m2']=df['open_area_m2'] if 'open_area_m2' in df.columns else np.nan
    df['has_open_area']=df['open_area_m2'].fillna(0).gt(0).astype(int)
    df['floor_num']=df['floor'].map(floor_to_num) if 'floor' in df.columns else np.nan
    for c in df.columns:
        if c.startswith(DETAIL_BINARY_PREFIXES): df[c]=pd.to_numeric(df[c], errors='coerce').fillna(0).clip(0,1).astype(int)
    for raw_col,count_col in DETAIL_RAW_COLUMNS.items():
        df[count_col]=df[raw_col].map(count_pipe_values).astype(int) if raw_col in df.columns else 0
    if 'detail_selected_count' not in df.columns: df['detail_selected_count']=df[list(DETAIL_RAW_COLUMNS.values())].sum(axis=1)
    if 'detail_quality_score' not in df.columns:
        cols=[c for c in df.columns if c.startswith(('out_','in_'))]
        df['detail_quality_score']=df[cols].sum(axis=1) if cols else 0
    valid=df[TARGET].notna() & df['gross_m2'].notna() & (df['gross_m2']>20) & (df['gross_m2']<1000) & (df[TARGET]>1000) & (df[TARGET]<1000000)
    return df.loc[valid].copy()

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self,X,y=None): return self
    def transform(self,X):
        df=X.copy()
        for c in ['gross_m2','net_m2','building_age','floor_num','total_floors','bathroom_count']:
            if c not in df.columns: df[c]=np.nan
        denom=df['total_floors'].replace(0,np.nan)
        df['floor_ratio']=df['floor_num']/denom; df['remaining_floors']=df['total_floors']-df['floor_num']
        df['is_ground_floor']=(df['floor_num']==0).astype(int); df['is_basement']=(df['floor_num']<0).astype(int)
        df['is_top_floor']=(df['floor_num'].notna() & df['total_floors'].notna() & (df['total_floors']>0) & (df['floor_num']>=df['total_floors'])).astype(int)
        df['is_middle_floor']=(df['floor_num'].notna() & df['total_floors'].notna() & (df['floor_num']>0) & (df['floor_num']<df['total_floors'])).astype(int)
        if 'room_count' in df.columns:
            parsed=df['room_count'].apply(parse_room); df['rooms']=parsed.apply(lambda x:x[0]); df['living_rooms']=parsed.apply(lambda x:x[1]); df['total_room_score']=parsed.apply(lambda x:x[2])
        else: df['rooms']=df['living_rooms']=df['total_room_score']=np.nan
        age=df['building_age']; df['is_new_building']=age.fillna(999).le(2).astype(int); df['is_old_building']=age.fillna(0).ge(25).astype(int)
        df['building_age_group']=pd.cut(age,[-1,0,5,10,20,30,200],labels=['0','1-5','6-10','11-20','21-30','30+']).astype(object)
        gross=df['gross_m2']; df['is_small_flat']=gross.fillna(9999).le(75).astype(int); df['is_large_flat']=gross.fillna(0).ge(160).astype(int)
        df['m2_group']=pd.cut(gross,[0,75,100,125,150,200,1000],labels=['0-75','76-100','101-125','126-150','151-200','200+']).astype(object)
        q=pd.Series(0.0,index=df.index)
        def yes_like(s): return s.fillna('').astype(str).str.lower().str.contains('var|evet|açık|kapalı|kapali|site', regex=True)
        if 'elevator' in df.columns: q+=yes_like(df['elevator']).astype(int)
        if 'parking' in df.columns: q+=yes_like(df['parking']).astype(int)
        if 'site_inside' in df.columns: q+=yes_like(df['site_inside']).astype(int)
        if 'bathroom_count' in df.columns: q+=df['bathroom_count'].fillna(1).ge(2).astype(int)
        if 'heating' in df.columns: q+=df['heating'].fillna('').astype(str).str.lower().str.contains('merkezi|kombi|yerden', regex=True).astype(int)
        if 'detail_quality_score' in df.columns: q+=pd.to_numeric(df['detail_quality_score'], errors='coerce').fillna(0)*0.35
        df['quality_score']=q
        def comb(a,b):
            aa=df[a].fillna('missing').astype(str) if a in df.columns else pd.Series('missing',index=df.index)
            bb=df[b].fillna('missing').astype(str) if b in df.columns else pd.Series('missing',index=df.index)
            return aa+'__'+bb
        df['district_age_group']=comb('district','building_age_group'); df['district_m2_group']=comb('district','m2_group'); df['district_room_count']=comb('district','room_count')
        return df

class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, smoothing=20): self.columns=columns or []; self.smoothing=smoothing
    def fit(self,X,y):
        X=X.copy(); y=pd.Series(y).astype(float); self.global_mean_=float(y.mean()); self.maps_={}
        for col in self.columns:
            if col not in X.columns: continue
            tmp=pd.DataFrame({'key':X[col].fillna('missing').astype(str),'target':y.values}); stats=tmp.groupby('key')['target'].agg(['mean','count'])
            smooth=(stats['mean']*stats['count']+self.global_mean_*self.smoothing)/(stats['count']+self.smoothing); self.maps_[col]=smooth.to_dict()
        return self
    def transform(self,X):
        X=X.copy()
        for col in self.columns:
            out=f'{col}_target_encoded'
            X[out]=X[col].fillna('missing').astype(str).map(self.maps_.get(col,{})).fillna(self.global_mean_).astype(float) if col in X.columns else self.global_mean_
        return X

class BaselineEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, smoothing=10): self.columns=columns or []; self.smoothing=smoothing
    def fit(self,X,y):
        X=X.copy(); y=pd.Series(y).astype(float); self.global_median_=float(y.median()); self.maps_={}
        for col in self.columns:
            if col not in X.columns: continue
            tmp=pd.DataFrame({'key':X[col].fillna('missing').astype(str),'target':y.values}); stats=tmp.groupby('key')['target'].agg(['median','count'])
            smooth=(stats['median']*stats['count']+self.global_median_*self.smoothing)/(stats['count']+self.smoothing); self.maps_[col]=smooth.to_dict()
        return self
    def transform(self,X):
        X=X.copy()
        for col in self.columns:
            out=f'{col}_baseline_unit_price'
            X[out]=X[col].fillna('missing').astype(str).map(self.maps_.get(col,{})).fillna(self.global_median_).astype(float) if col in X.columns else self.global_median_
        return X

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, min_count=5): self.columns=columns or []; self.min_count=min_count
    def fit(self,X,y=None):
        X=X.copy(); self.valid_values_={}
        for col in self.columns:
            if col not in X.columns: continue
            counts=X[col].fillna('missing').astype(str).value_counts(); self.valid_values_[col]=set(counts[counts>=self.min_count].index)
        return self
    def transform(self,X):
        X=X.copy()
        for col,valid in self.valid_values_.items():
            if col not in X.columns: continue
            vals=X[col].fillna('missing').astype(str); X[col]=vals.where(vals.isin(valid),'other')
        return X

def get_detail_binary_columns(df): return [c for c in df.columns if c.startswith(DETAIL_BINARY_PREFIXES)]

def remove_useless_features(df,num,cat):
    keep_num=[]; keep_cat=[]; removed={}
    for c in num:
        if c not in df.columns: removed[c]='missing'
        elif df[c].notna().sum()==0: removed[c]='all_missing'
        elif df[c].nunique(dropna=True)<=1: removed[c]='constant'
        else: keep_num.append(c)
    for c in cat:
        if c not in df.columns: removed[c]='missing'
        elif df[c].notna().sum()==0: removed[c]='all_missing'
        elif df[c].nunique(dropna=True)<=1: removed[c]='constant'
        else: keep_cat.append(c)
    return keep_num,keep_cat,removed

def mape(y_true,y_pred):
    y_true,y_pred=np.asarray(y_true,float),np.asarray(y_pred,float); return float(np.mean(np.abs((y_true-y_pred)/y_true)))

def metric_dict(y,p):
    p=np.asarray(p,float)
    return {'mape':mape(y,p),'mae_tl_per_m2':float(mean_absolute_error(y,p)),'median_ae_tl_per_m2':float(median_absolute_error(y,p)),'r2':float(r2_score(y,p)),'log_r2':float(r2_score(np.log1p(y),np.log1p(np.maximum(p,0))))}

def make_preprocessor(num,cat):
    try: ohe=OneHotEncoder(handle_unknown='ignore',min_frequency=3,sparse_output=False)
    except TypeError: ohe=OneHotEncoder(handle_unknown='ignore',min_frequency=3,sparse=False)
    return ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),num),('cat',Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value='missing')),('onehot',ohe)]),cat)])

def make_model(estimator,num,cat):
    return TransformedTargetRegressor(regressor=Pipeline([('feature_engineering',FeatureEngineer()),('rare_category_grouper',RareCategoryGrouper(columns=cat,min_count=5)),('baseline_encoding',BaselineEncoder(columns=['district','county'],smoothing=10)),('target_encoding',TargetEncoder(columns=['district','county'],smoothing=20)),('preprocess',make_preprocessor(num,cat)),('model',estimator)]),func=np.log1p,inverse_func=np.expm1)

def save_explainability(model,name):
    try:
        inner=model.regressor_; pre=inner.named_steps['preprocess']; est=inner.named_steps['model']; names=pre.get_feature_names_out()
        if hasattr(est,'coef_'):
            out=pd.DataFrame({'feature':names,'coefficient':est.coef_}); out['abs_coefficient']=out['coefficient'].abs(); out.sort_values('abs_coefficient',ascending=False).to_csv(ARTIFACTS/f'{name}_coefficients.csv',index=False,encoding='utf-8-sig')
        if hasattr(est,'feature_importances_'):
            out=pd.DataFrame({'feature':names,'importance':est.feature_importances_}); out.sort_values('importance',ascending=False).to_csv(ARTIFACTS/f'{name}_feature_importance.csv',index=False,encoding='utf-8-sig')
    except Exception: pass

def write_error_reports(df,pred,name):
    view=FeatureEngineer().fit_transform(df.copy()); out=view.copy(); out[f'{name}_pred_unit_price']=pred; out[f'{name}_abs_pct_error']=np.abs(out[TARGET]-pred)/out[TARGET]; out[f'{name}_abs_error']=np.abs(out[TARGET]-pred)
    out.to_csv(OUTPUT/f'{name}_cv_predictions.csv',index=False,encoding='utf-8-sig')
    err=f'{name}_abs_pct_error'; abs_err=f'{name}_abs_error'
    for col in ['district','room_count','floor_segment','building_age_group','m2_group','heating','site_inside','detail_cephe','detail_manzara','detail_konut_tipi']:
        if col in out.columns:
            rep=out.groupby(col,dropna=False).agg(n=(TARGET,'size'),mape=(err,'mean'),median_ape=(err,'median'),mae_tl_per_m2=(abs_err,'mean'),median_ae_tl_per_m2=(abs_err,'median'),mean_unit_price=(TARGET,'mean')).reset_index().sort_values('mape',ascending=False)
            rep.to_csv(OUTPUT/f'{name}_error_by_{col}.csv',index=False,encoding='utf-8-sig')
    out.sort_values(err,ascending=False).head(50).to_csv(OUTPUT/f'{name}_top_50_errors.csv',index=False,encoding='utf-8-sig')

def main():
    warnings.filterwarnings('ignore')
    raw=pd.read_csv(INPUT_PATH); df=prepare_raw(raw)
    detail_binary_cols=get_detail_binary_columns(df)
    detail_coverage={'detail_binary_columns_found':detail_binary_cols,'detail_binary_columns_count':int(len(detail_binary_cols)),'rows_with_any_detail_binary':int((df[detail_binary_cols].sum(axis=1)>0).sum()) if detail_binary_cols else 0,'rows_used':int(len(df))}
    candidate_numeric=BASE_NUMERIC+detail_binary_cols; candidate_categorical=BASE_CATEGORICAL
    fe=FeatureEngineer().fit_transform(df.copy()); be=BaselineEncoder(columns=['district','county'],smoothing=10).fit(fe.copy(),df[TARGET]).transform(fe.copy()); te=TargetEncoder(columns=['district','county'],smoothing=20).fit(be.copy(),df[TARGET]).transform(be.copy())
    num,cat,removed_features=remove_useless_features(te,candidate_numeric,candidate_categorical)
    X=df.drop(columns=[TARGET],errors='ignore').copy(); y=df[TARGET].astype(float); cv=KFold(n_splits=5,shuffle=True,random_state=RANDOM_STATE)
    models={
        'ridge_v4_detail':make_model(RidgeCV(alphas=np.logspace(-3,4,60)),num,cat),
        'elasticnet_v4_detail':make_model(ElasticNetCV(l1_ratio=[.05,.1,.2,.3,.5,.7,.9],alphas=np.logspace(-4,2,60),max_iter=40000,random_state=RANDOM_STATE),num,cat),
        'gradient_boosting_v4_detail':make_model(GradientBoostingRegressor(random_state=RANDOM_STATE,learning_rate=.035,n_estimators=700,max_depth=2,min_samples_leaf=5,subsample=.85),num,cat),
        'hist_gradient_boosting_v4_detail':make_model(HistGradientBoostingRegressor(random_state=RANDOM_STATE,max_iter=500,learning_rate=.035,max_leaf_nodes=31,l2_regularization=.05),num,cat),
        'extra_trees_v4_detail':make_model(ExtraTreesRegressor(random_state=RANDOM_STATE,n_estimators=600,min_samples_leaf=2,max_features=.8,n_jobs=-1),num,cat),
        'random_forest_v4_detail':make_model(RandomForestRegressor(random_state=RANDOM_STATE,n_estimators=500,min_samples_leaf=3,max_features=.8,n_jobs=-1),num,cat),
    }
    results={}; preds={}
    for name,model in models.items():
        print(f'CV: {name}')
        try:
            p=cross_val_predict(model,X,y,cv=cv); preds[name]=p; results[name]=metric_dict(y,p); model.fit(X,y); joblib.dump(model,ARTIFACTS/f'{name}.joblib'); save_explainability(model,name)
        except Exception as e: results[name]={'error':str(e)}
    print('Tuning: gradient_boosting_tuned_r2_v4_detail')
    gb_base=make_model(GradientBoostingRegressor(random_state=RANDOM_STATE),num,cat)
    param_dist={'regressor__model__n_estimators':[500,700,900,1100],'regressor__model__learning_rate':[.025,.03,.035,.04,.05],'regressor__model__max_depth':[2,3],'regressor__model__min_samples_leaf':[3,5,8,12,16],'regressor__model__subsample':[.75,.85,1.0]}
    scorer=make_scorer(lambda yt,yp:r2_score(yt,yp),greater_is_better=True)
    search=RandomizedSearchCV(gb_base,param_dist,n_iter=18,cv=cv,scoring=scorer,random_state=RANDOM_STATE,n_jobs=-1)
    try:
        search.fit(X,y); tuned=search.best_estimator_; tuned_pred=cross_val_predict(tuned,X,y,cv=cv); tuned.fit(X,y)
        results['gradient_boosting_tuned_r2_v4_detail']=metric_dict(y,tuned_pred); results['gradient_boosting_tuned_r2_v4_detail']['best_params']=search.best_params_; preds['gradient_boosting_tuned_r2_v4_detail']=tuned_pred; joblib.dump(tuned,ARTIFACTS/'gradient_boosting_tuned_r2_v4_detail.joblib'); save_explainability(tuned,'gradient_boosting_tuned_r2_v4_detail')
    except Exception as e: results['gradient_boosting_tuned_r2_v4_detail']={'error':str(e)}
    valid_models={k:v for k,v in results.items() if 'r2' in v}
    best_by_r2=max(valid_models,key=lambda k:valid_models[k]['r2']) if valid_models else None; best_by_mape=min(valid_models,key=lambda k:valid_models[k]['mape']) if valid_models else None
    if best_by_r2:
        joblib.dump(joblib.load(ARTIFACTS/f'{best_by_r2}.joblib'),ARTIFACTS/'best_model_v4_by_r2.joblib'); write_error_reports(df,preds[best_by_r2],best_by_r2)
    if best_by_mape and best_by_mape!=best_by_r2: write_error_reports(df,preds[best_by_mape],best_by_mape)
    comparison=pd.DataFrame([{'model':k,**{kk:vv for kk,vv in v.items() if kk!='best_params'}} for k,v in results.items()])
    if 'r2' in comparison.columns: comparison=comparison.sort_values('r2',ascending=False,na_position='last')
    comparison.to_csv(REPORTS/'model_comparison_v4.csv',index=False,encoding='utf-8-sig')
    detail_summary=[{'feature':c,'ones':int(pd.to_numeric(df[c],errors='coerce').fillna(0).sum()),'ratio':float(pd.to_numeric(df[c],errors='coerce').fillna(0).mean())} for c in detail_binary_cols]
    pd.DataFrame(detail_summary).sort_values('ones',ascending=False).to_csv(REPORTS/'detail_feature_coverage_v4.csv',index=False,encoding='utf-8-sig')
    report={'target':TARGET,'rows_raw':int(len(raw)),'rows_used':int(len(df)),'detail_coverage':detail_coverage,'features':{'numeric_used':num,'categorical_used':cat,'removed_features':removed_features},'models':results,'best_model_by_r2':best_by_r2,'best_model_by_mape':best_by_mape,'note':'V4 adds helper-enriched detail binary features such as front/view/transport/near/out/in/subtype plus detail counts and quality score. No strict segment filter is applied by default.'}
    (REPORTS/'model_metrics_v4.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
