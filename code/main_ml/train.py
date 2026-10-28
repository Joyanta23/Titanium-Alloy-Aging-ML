"""Train seven baseline models for resistivity and hardness.
Run: python train.py --input path/to/dataset.xlsx
Run complete_xgb_shap.py next to add XGBoost and SHAP analysis.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse, json, hashlib, platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn, joblib
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
SEED=42
ELEMENTS=['Al','Cr','Mn','Fe','Mo','Nb','Ta','Zr']
def save(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=str))
def scores(y,p):
    return dict(R2=float(r2_score(y,p)),RMSE=float(mean_squared_error(y,p)**.5),MAE=float(mean_absolute_error(y,p)))
def models():
    return {
      'RandomForest':(RandomForestRegressor(n_estimators=200,random_state=SEED,n_jobs=1),dict(max_depth=[None,6,12],min_samples_leaf=[1,2,4],max_features=[.7,1.])),
      'ExtraTrees':(ExtraTreesRegressor(n_estimators=200,random_state=SEED,n_jobs=1),dict(max_depth=[None,6,12],min_samples_leaf=[1,2,4],max_features=[.7,1.])),
      'GradientBoosting':(GradientBoostingRegressor(random_state=SEED),dict(n_estimators=[150,300,500],max_depth=[2,3,4],learning_rate=[.03,.08,.15],min_samples_leaf=[2,5])),
      'SVR':(SVR(),dict(C=[1,10,100,1000],gamma=['scale',.03,.1,.3],epsilon=[.01,.05,.1])),
      'KNN':(KNeighborsRegressor(),dict(n_neighbors=[2,3,5,7,11],weights=['uniform','distance'],p=[1,2])),
      'Ridge':(Ridge(),dict(alpha=[.01,.1,1,10,100])),
      'MeanBaseline':(DummyRegressor(),{})}

def run(input_path):
    save(ROOT/'protocol.json',dict(seed=SEED,split='80:20 shuffled rows, fixed seed 42',tuning='Training-only shuffled 5-fold CV, 12 configurations/model maximum',selection='Minimum training CV RMSE',claim='Within represented alloy/processing domain; not unseen-alloy performance',missing='Only unlisted major alloying elements mapped to design-zero; unknown within-system concentration is an error. Raw workbook unchanged.',excluded_features='DOI, source ID, curve ID, alloy label, target-derived features, sparse O/H/N; solution_time constant 3.6 ks',input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest()))
    all_results=[]
    for sheet,target,name in [('RT_Resistivity','resistivity_uOhm_m','resistivity'),('Vickers_Hardness','hardness_HV','hardness')]:
        out=ROOT/name;out.mkdir(exist_ok=True)
        d=pd.read_excel(input_path,sheet_name=sheet).sort_values('record_id').reset_index(drop=True)
        assert len(d)==(836 if name=='resistivity' else 568)
        assert (d.aging_time_ks>0).all() and d[target].notna().all()
        X=pd.DataFrame(index=d.index);changes=[]
        for e in ELEMENTS:
            c=e+'_wt_pct';X[c]=pd.to_numeric(d[c])
            missing=X[c].isna()
            system_contains=d.alloy_system.str.split('-').apply(lambda a:e in a)
            assert not (missing & system_contains).any(),f'Unknown within-system {e}'
            for i in d.index[missing]:changes.append(dict(record_id=d.at[i,'record_id'],feature=c,model_value=0,reason='Not listed in designated alloy system; design-zero, not analytical purity claim'))
            X.loc[missing,c]=0.
        X['aging_temperature_K']=d.aging_temperature_K
        X['log10_aging_time_ks']=np.log10(d.aging_time_ks)
        X['solution_temperature_K']=d.solution_temperature_K
        assert np.isfinite(X).all().all()
        y=d[target]
        tr,te=train_test_split(np.arange(len(d)),test_size=.2,random_state=SEED)
        split=np.where(np.isin(np.arange(len(d)),tr),'train','test')
        d.assign(split=split).to_csv(out/'source_records_and_split.csv',index=False)
        X.assign(record_id=d.record_id,target=y,split=split).to_csv(out/'model_ready.csv',index=False)
        pd.DataFrame(changes).to_csv(out/'design_zero_log.csv',index=False)
        cv=KFold(5,shuffle=True,random_state=SEED)
        folds=np.full(len(d),-1)
        for n,(_,v) in enumerate(cv.split(tr)):folds[tr[v]]=n
        pd.DataFrame(dict(record_id=d.record_id,fold=folds)).to_csv(out/'training_folds.csv',index=False)
        rows=[];fitted={}
        # Use the same scaling pipeline for all models in this comparison.
        for label,(model,grid) in models().items():
            pipe=Pipeline([('scale',StandardScaler()),('regressor',TransformedTargetRegressor(regressor=model,transformer=StandardScaler()))])
            space={'regressor__regressor__'+k:v for k,v in grid.items()}
            count=int(np.prod([len(v) for v in grid.values()])) if grid else 1
            search=RandomizedSearchCV(pipe,space,n_iter=min(12,count),scoring='neg_mean_squared_error',cv=cv,random_state=SEED,n_jobs=2,error_score='raise')
            search.fit(X.iloc[tr],y.iloc[tr]);fitted[label]=search.best_estimator_
            rows.append(dict(model=label,CV_RMSE=float((-search.best_score_)**.5),parameters=search.best_params_))
            pd.DataFrame(search.cv_results_).to_csv(out/(label+'_search.csv'),index=False)
            print(name,label,'CV RMSE',rows[-1]['CV_RMSE'],flush=True)
        chosen=min(rows,key=lambda a:a['CV_RMSE'])['model']
        save(out/'selection_before_test.json',dict(selected=chosen,criterion='Minimum training CV RMSE'))
        for row in rows:
            model=fitted[row['model']]
            row.update({'train_'+k:v for k,v in scores(y.iloc[tr],model.predict(X.iloc[tr])).items()})
            row.update({'test_'+k:v for k,v in scores(y.iloc[te],model.predict(X.iloc[te])).items()})
            row['selected']=row['model']==chosen
        pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
        save(out/'comparison.json',rows)
        best=fitted[chosen];pred=best.predict(X)
        joblib.dump(dict(model=best,features=X.columns.tolist(),train_indices=tr,test_indices=te),out/'selected_model.joblib')
        pd.DataFrame(dict(record_id=d.record_id,observed=y,predicted=pred,split=split)).to_csv(out/'predictions.csv',index=False)
        selected=next(r for r in rows if r['selected'])
        unit='µΩ·m' if name=='resistivity' else 'HV'
        fig,ax=plt.subplots(figsize=(6,5))
        ax.scatter(y.iloc[tr],pred[tr],alpha=.3,s=16,label='Training fit',color='gray')
        ax.scatter(y.iloc[te],pred[te],alpha=.85,s=24,label='Test rows',color='#007f86')
        lo=min(y.min(),pred.min());hi=max(y.max(),pred.max());ax.plot([lo,hi],[lo,hi],'k--',lw=1)
        ax.set(xlabel=f'Measured ({unit})',ylabel=f'Predicted ({unit})',title=f'{name.title()}: {chosen}\nTest R² = {selected["test_R2"]:.4f}')
        ax.legend();fig.tight_layout();fig.savefig(out/'prediction.png',dpi=180);plt.close(fig)
        summary=dict(target=name,rows=len(d),train_rows=len(tr),test_rows=len(te),selected_model=chosen,results=selected,features=X.columns.tolist(),status='Seven-model comparison complete; run complete_xgb_shap.py for XGBoost and SHAP',dropped_rows=0)
        save(out/'summary.json',summary);all_results.append(summary)
        print('RESULT',json.dumps(summary),flush=True)
    save(ROOT/'summary.json',all_results)
    save(ROOT/'environment.json',dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    run(parser.parse_args().input)
