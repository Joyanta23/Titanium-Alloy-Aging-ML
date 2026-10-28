"""Add XGBoost and SHAP using the data split and folds saved by train.py.
Run from any directory: python complete_xgb_shap.py
Writes ../ml_completed_v1, leaving the initial run untouched.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[k]='1'
import json, hashlib, platform, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn, joblib, xgboost, shap
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INITIAL=Path(__file__).resolve().parent
OUT=INITIAL.parent/'ml_completed_v1'
SEED=42
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
def save(p,obj):
    Path(p).write_text(json.dumps(obj,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)))
def metrics(y,p):
    return dict(R2=float(r2_score(y,p)),MAE=float(mean_absolute_error(y,p)),RMSE=float(mean_squared_error(y,p)**.5))
def friendly(c):
    return c.replace('_wt_pct',' (wt.%)').replace('aging_temperature_K','Aging temperature (K)').replace('log10_aging_time_ks','log10(aging time / ks)').replace('solution_temperature_K','Solution temperature (K)')

def run():
    OUT.mkdir(exist_ok=True)
    # Fixed 12-trial XGBoost search on the saved training folds.
    space=dict(n_estimators=[150,300,500],max_depth=[2,3,5],learning_rate=[.03,.08,.15],min_child_weight=[1,3,6],reg_lambda=[1,5,15],subsample=[.8,1.],colsample_bytree=[.8,1.])
    save(OUT/'continuation_protocol.json',dict(seed=SEED,n_iter=12,search_space=space,selection='Minimum training CV RMSE across all eight models',prior_test_visibility='XGBoost was added after the seven-model comparison; it uses the same split and folds, with tuning based on training CV.',SHAP='Interventional TreeSHAP for selected tree models; training-only background, explanations on test rows. Contributions converted to original target units.',descriptors='Same eleven input features as train.py.'))
    summaries=[]
    for name in ['resistivity','hardness']:
        src=INITIAL/name;out=OUT/name;out.mkdir(exist_ok=True)
        d=pd.read_csv(src/'model_ready.csv')
        old=json.loads((src/'summary.json').read_text());features=old['features']
        X=d[features];y=d.target
        tr=np.flatnonzero(d.split.eq('train'));te=np.flatnonzero(d.split.eq('test'))
        assert len(tr)==old['train_rows'] and len(te)==old['test_rows']
        # Original CV fold IDs are mapped onto this training-array order.
        f=pd.read_csv(src/'training_folds.csv').set_index('record_id')['fold']
        f=f.loc[d.record_id.iloc[tr]].to_numpy()
        cv=[(np.flatnonzero(f!=k),np.flatnonzero(f==k)) for k in range(5)]
        assert all(len(v)>0 and not set(a)&set(v) for a,v in cv)
        for fn in ['model_ready.csv','training_folds.csv','source_records_and_split.csv','design_zero_log.csv']:
            shutil.copy2(src/fn,out/fn)
        model=XGBRegressor(random_state=SEED,n_jobs=1,tree_method='hist',objective='reg:squarederror')
        pipe=Pipeline([('scale',StandardScaler()),('regressor',TransformedTargetRegressor(regressor=model,transformer=StandardScaler()))])
        search=RandomizedSearchCV(pipe,{'regressor__regressor__'+k:v for k,v in space.items()},n_iter=12,scoring='neg_mean_squared_error',cv=cv,random_state=SEED,n_jobs=2,error_score='raise')
        search.fit(X.iloc[tr],y.iloc[tr])
        pd.DataFrame(search.cv_results_).to_csv(out/'XGBoost_search.csv',index=False)
        row=dict(model='XGBoost',CV_RMSE=float((-search.best_score_)**.5),parameters=search.best_params_)
        rows=json.loads((src/'comparison.json').read_text())
        chosen=min(rows+[row],key=lambda r:r['CV_RMSE'])['model']
        save(out/'selection_before_xgb_test.json',dict(model=chosen,XGBoost_CV_RMSE=row['CV_RMSE']))
        row.update({'train_'+k:v for k,v in metrics(y.iloc[tr],search.predict(X.iloc[tr])).items()})
        row.update({'test_'+k:v for k,v in metrics(y.iloc[te],search.predict(X.iloc[te])).items()})
        rows.append(row)
        for r in rows:r['selected']=r['model']==chosen
        save(out/'comparison.json',rows);pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
        joblib.dump(search.best_estimator_,out/'XGBoost_model.joblib')
        if chosen=='XGBoost':best=search.best_estimator_
        else:
            assert chosen==old['selected_model']
            best=joblib.load(src/'selected_model.joblib')['model']
        selected=next(r for r in rows if r['selected'])
        pred=best.predict(X)
        joblib.dump(dict(model=best,features=features,train_indices=tr,test_indices=te),out/'selected_model.joblib')
        pd.DataFrame(dict(record_id=d.record_id,observed=y,predicted=pred,split=d.split)).to_csv(out/'predictions.csv',index=False)
        unit='µΩ·m' if name=='resistivity' else 'HV'
        fig,ax=plt.subplots(figsize=(6,5))
        ax.scatter(y.iloc[tr],pred[tr],s=14,alpha=.25,c='gray',label='Training fit')
        ax.scatter(y.iloc[te],pred[te],s=25,alpha=.85,c='#007f86',label='Test rows')
        lo=min(y.min(),pred.min());hi=max(y.max(),pred.max());ax.plot([lo,hi],[lo,hi],'k--',lw=1)
        ax.set(xlabel=f'Measured ({unit})',ylabel=f'Predicted ({unit})',title=f'{name.title()}: {chosen}\nTest R² = {selected["test_R2"]:.4f}; RMSE = {selected["test_RMSE"]:.4f} {unit}')
        ax.legend(frameon=False);fig.tight_layout()
        for ext in ['png','svg']:fig.savefig(out/f'prediction.{ext}')
        plt.close(fig)
        comparison=pd.DataFrame(rows).sort_values('CV_RMSE')
        fig,ax=plt.subplots(figsize=(8,5))
        labels=comparison.model.replace({'RandomForest':'Random forest','GradientBoosting':'Gradient boosting','MeanBaseline':'Mean baseline'})
        bars=ax.barh(labels,comparison.test_RMSE,color=['#007f86' if a else '#8496a8' for a in comparison.selected])
        ax.bar_label(bars,labels=[f'{v:.4f}' if name=='resistivity' else f'{v:.2f}' for v in comparison.test_RMSE],padding=4)
        ax.set(xlabel=f'Test RMSE ({unit})',title=f'{name.title()}: model comparison (ordered by training CV RMSE)')
        ax.margins(x=.18)
        ax.invert_yaxis();fig.tight_layout();fig.savefig(out/'model_comparison.png');plt.close(fig)
        print('COMPARISON',name,json.dumps(rows),flush=True)
        # Explain the selected fitted model, using only training rows as background.
        scaler=best.named_steps['scale'];wrapper=best.named_steps['regressor'];core=wrapper.regressor_
        scale=float(wrapper.transformer_.scale_[0]);offset=float(wrapper.transformer_.mean_[0])
        A=scaler.transform(X.iloc[tr]);B=scaler.transform(X.iloc[te])
        rng=np.random.default_rng(SEED);bgix=rng.choice(len(tr),size=min(64,len(tr)),replace=False)
        explainer=shap.TreeExplainer(core,data=A[bgix],feature_perturbation='interventional',model_output='raw')
        explanation=explainer(B,check_additivity=True)
        values=np.asarray(explanation.values)*scale
        base=np.asarray(explanation.base_values)*scale+offset
        error=float(np.max(np.abs(base+values.sum(axis=1)-pred[te])))
        assert np.allclose(base+values.sum(axis=1),pred[te],rtol=1e-5,atol=1e-5),error
        E=shap.Explanation(values=values,base_values=base,data=X.iloc[te].to_numpy(),feature_names=[friendly(c) for c in features])
        shap.plots.beeswarm(E,max_display=len(features),show=False,plot_size=(9,6))
        plt.title(f'{name.title()} | {chosen} | test rows');plt.xlabel(f'SHAP contribution ({unit})');plt.tight_layout()
        for ext in ['png','svg']:plt.savefig(out/f'shap_beeswarm.{ext}',bbox_inches='tight')
        plt.close('all')
        importance=pd.DataFrame(dict(feature=features,mean_abs_SHAP=np.abs(values).mean(0))).sort_values('mean_abs_SHAP',ascending=False)
        importance.to_csv(out/'shap_importance.csv',index=False)
        for label,subset in [('shap_importance',importance),('element_importance',importance[importance.feature.str.endswith('_wt_pct')])]:
            tab=subset.sort_values('mean_abs_SHAP')
            fig,ax=plt.subplots(figsize=(8,5));ax.barh([friendly(c) for c in tab.feature],tab.mean_abs_SHAP,color='#007f86')
            ax.set(xlabel=f'Mean absolute SHAP ({unit})',title=f'{name.title()}: '+('element contributions' if label=='element_importance' else 'feature contributions'))
            fig.tight_layout()
            for ext in ['png','svg']:fig.savefig(out/f'{label}.{ext}')
            plt.close(fig)
        tab=pd.DataFrame(values,columns=features);tab.insert(0,'record_id',d.record_id.iloc[te].to_numpy());tab.insert(1,'base_value',base);tab.to_csv(out/'shap_values.csv',index=False)
        fig,axes=plt.subplots(2,2,figsize=(10,7))
        for ax,c in zip(axes.flat,importance.head(4).feature):
            i=features.index(c);ax.scatter(X.iloc[te][c],values[:,i],s=22,alpha=.75,c='#007f86');ax.axhline(0,lw=.7,c='gray')
            ax.set(xlabel=friendly(c),ylabel=f'SHAP ({unit})')
        fig.suptitle(f'{name.title()}: SHAP dependence');fig.tight_layout();fig.savefig(out/'shap_dependence.png');plt.close(fig)
        save(out/'shap_method.json',dict(method='Interventional TreeSHAP',background_record_ids=d.record_id.iloc[tr[bgix]].tolist(),explained_rows=len(te),target_unit=unit,additivity_max_absolute_error=error,caveat='SHAP describes model associations; correlated composition and processing inputs affect attribution.'))
        s=dict(target=name,selected_model=chosen,rows=len(d),train_rows=len(tr),test_rows=len(te),metrics=selected,xgboost=row,shap_top_features=importance.head(5).to_dict('records'),dropped_rows=0)
        save(out/'summary.json',s);summaries.append(s)
        print('DONE',name,chosen,selected['test_R2'],'SHAP ERROR',error,flush=True)
    save(OUT/'summary.json',summaries)
    save(OUT/'environment.json',dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,xgboost=xgboost.__version__,shap=shap.__version__))
if __name__=='__main__':run()
