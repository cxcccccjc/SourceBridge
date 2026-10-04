"""Separate implementation audit of the declared static Triangles comparison.

Reconstruct inputs from CSV without the generator/pandas; recompute all saved
metrics. Reimplement core equations without importing the compared predictors.
Extend selected nonconverged MLNI fits to 2000 iterations, without retuning or
choosing by test error. This is a team internal audit, not external replication.
"""
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
import sys,json,csv,hashlib,time
from collections import defaultdict
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'deps'))
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readj(p):return json.loads((B/p).read_text('utf-8'))
def close(a,b,tol=1e-7):
    d=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    assert d<=tol,(d,tol)
    return d

def fit_map(x,y):
    x=np.asarray(x);y=np.asarray(y);d=x-x.mean(axis=0)
    divisor=np.sum(d*d,axis=0);cross=np.sum(d*(y-y.mean()).reshape((-1,)+(1,)*(x.ndim-1)),axis=0)
    slope=np.ones_like(divisor)
    mask=divisor>1e-14
    # Equivalent penalized centered normal equation; no shared implementation.
    np.divide(cross+divisor/100,divisor*101/100,out=slope,where=mask)
    return slope,y.mean()-slope*x.mean(axis=0)

def transform(x,refs,q,variant):
    if variant in ['raw','post_affine']:return x.copy()
    if variant=='offset':return x-(x[refs].mean(axis=0)-q.mean())
    if variant=='individual_affine':
        slope,intercept=fit_map(x[refs],q)
        return x*slope+intercept
    raise ValueError(variant)

def mlni(x,c,q,cfg,cap=150,trace=False):
    n,g=x.shape;strength=cfg['prior_strength']
    mu=float(np.mean(q));truthvar=max(float(np.var(q))*cfg['truth_prior'],1e-5)
    error=c-q[:,None];h0=np.mean(error,axis=0);v0=np.maximum(np.var(error,axis=0),1e-5)
    hvar=v0/strength;prior_a=np.full(g,strength);prior_b=v0*(prior_a+1)
    h=h0.copy();v=v0.copy();z=np.mean(x,axis=1);sumx=np.sum(x,axis=0)
    def objective():
        residual=x-z[:,None]-h[None,:]
        return float(.5*np.sum((z-mu)**2)/truthvar+.5*np.sum((h-h0)**2/hvar)
            +np.sum((prior_a+1+n/2)*np.log(v)+(prior_b+.5*np.sum(residual**2,axis=0))/v))
    original=objective();before=original;maxrise=0.;at150=None
    for step in range(1,cap+1):
        oldz=z.copy();oldh=h.copy();oldv=v.copy();precision=1/v
        z=(x@precision-np.dot(h,precision)+mu/truthvar)/(np.sum(precision)+1/truthvar)
        h=(h0/hvar+(sumx-np.sum(z))*precision)/(1/hvar+n*precision)
        residual=x-z[:,None]-h[None,:]
        v=np.maximum((prior_b+.5*np.sum(residual*residual,axis=0))/(prior_a+1+n/2),1e-10)
        delta=max(float(np.max(abs(z-oldz))),float(np.max(abs(h-oldh))),float(np.max(abs(v-oldv))))
        if trace:
            now=objective();maxrise=max(maxrise,now-before);before=now
        if step==150:at150=dict(objective=objective(),max_delta=delta)
        if delta<1e-6:break
    return z,dict(iterations=step,converged=delta<1e-6,max_delta=delta,
                  objective=objective(),initial_objective=original,max_objective_increase=maxrise,at150=at150)

def core(name,x,c,q,cfg=None,cap=150,trace=False):
    if name=='MLNI_JSAC2022':return mlni(x,c,q,cfg,cap,trace)
    n,g=x.shape
    if name.startswith('FETD_'):
        # Explicit pairwise differences avoids the original Gram subtraction.
        disparity=np.mean((x[:,:,None]-x[:,None,:])**2,axis=0).sum(axis=1)/(g-1)
        if '_D_' in name:w=disparity.max()-disparity
        else:w=1/np.maximum(disparity-g*disparity.mean()/(2*(g-1)),1e-8)
        if np.sum(w)<=1e-12:w=np.ones(g)
        return np.sum(x*w[None,:],axis=1)/np.sum(w),{}
    z=x.mean(axis=1)
    if name=='EPTD_TIFS2022_squared_CRH':
        for it in range(100):
            d=np.maximum(np.sum((x-z[:,None])**2,axis=0),1e-10)
            w=np.maximum(-np.log(d/np.sum(d)),1e-10)
            nz=np.sum(x*w,axis=1)/np.sum(w);delta=float(np.max(abs(nz-z)));z=nz
            if delta<1e-6:break
        return z,dict(iterations=it+1,converged=delta<1e-6)
    if name=='CRH_TKDE2016':
        scale=np.maximum(np.std(x,axis=1),1e-10);order=np.argsort(x,axis=1,kind='stable')
        for it in range(100):
            loss=np.maximum(np.sum(abs(x-z[:,None])/scale[:,None],axis=0),1e-10)
            w=np.log(np.max(loss))-np.log(loss)
            if np.sum(w)<=1e-12:w=np.ones(g)
            nz=np.empty(n)
            for j in range(n):
                index=np.searchsorted(np.cumsum(w[order[j]]),np.sum(w)/2,side='left')
                nz[j]=x[j,order[j,min(index,g-1)]]
            delta=float(np.max(abs(nz-z)));z=nz
            if delta<1e-6:break
        return z,dict(iterations=it+1,converged=delta<1e-6)
    raise ValueError(name)

def pipeline(name,variant,x,refs,q,cfg=None,cap=150,trace=False):
    tx=transform(x,refs,q,variant)
    z,meta=core(name,tx,tx[refs],q,cfg,cap,trace)
    if variant=='post_affine':
        slope,intercept=fit_map(z[refs],q);z=z*slope+intercept
    return z,meta

def controls(x,refs,q):
    avg=x.mean(axis=1);med=np.median(x,axis=1)
    a,b=fit_map(x[refs],q);cx=x*a+b
    out={'raw_mean':avg,'raw_median':med,'individual_affine_mean':cx.mean(axis=1),'individual_affine_median':np.median(cx,axis=1)}
    for tag,z in [('mean',avg),('median',med)]:
        a,b=fit_map(z[refs],q);out['post_affine_'+tag]=z*a+b
    v=float(np.sum((q-q.mean())**2));a=float(np.dot(q-q.mean(),avg[refs]-avg[refs].mean())/v) if v>1e-14 else 0.
    b=avg[refs].mean()-a*q.mean()
    out['forward_affine_mean']=(avg-b)/a if abs(a)>1e-8 else np.full(len(avg),q.mean())
    return {'Control_'+k:v for k,v in out.items()}

def main():
    tick=time.perf_counter();P=readj('triangles_calibration_comparison_protocol.json');IP=readj('triangles_calibration_protocol.json')
    M=readj('results/triangles_calibration_input_manifest.json');R=readj('results/triangles_calibration_comparison.json')
    inputs=np.load(B/P['input']);predictions=np.load(B/'results/triangles_calibration_predictions.npz')
    assert sha(B/'triangles_calibration_comparison_protocol.json')==R['protocol_sha256']
    assert sha(B/'results/triangles_calibration_predictions.npz')==R['predictions_sha256']
    assert sha(B/P['input'])==M['raw_sha256']
    for f,h in P['source_hashes'].items():assert verify_identity(B/f,h),f
    original={};repeats=[]
    for es,(rf,tf) in IP['fields'].items():
        e=int(es);path=B/'data/triangles'/f'Exp{e}data.csv';assert sha(path)==IP['raw_hashes'][path.name]
        cells=defaultdict(list);subject_tasks=defaultdict(set)
        with path.open('r',encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                s=int(r['SubjectID']);object_key=(float(r['Angle']),float(r['Base factor']))
                cells[(s,object_key)].append((float(r[rf]),float(r[tf])));subject_tasks[s].add(object_key)
        taskkeys=sorted({t for _,t in cells});ids={t:k for k,t in enumerate(taskkeys)}
        cohorts=sorted({tuple(sorted(ids[t] for t in tt)) for tt in subject_tasks.values()})
        for c,tasks in enumerate(cohorts):
            sources=sorted(s for s,tt in subject_tasks.items() if tuple(sorted(ids[t] for t in tt))==tasks)
            report=np.empty((15,len(sources)));truth=[]
            for j,task in enumerate(tasks):
                labels=[]
                for i,s in enumerate(sources):
                    vals=np.asarray(cells[(s,taskkeys[task])]);report[j,i]=sum(vals[:,0])/len(vals)
                    assert np.ptp(vals[:,1])<1e-8;labels.append(float(vals[0,1]));repeats.append(len(vals))
                assert np.ptp(labels)<1e-8;truth.append(labels[0])
            original[(e,c)]=(tasks,sources,report,np.array(truth))
    input_max=0.;case_data={}
    for case in M['cases']:
        k=case['key'];K=case['budget'];tasks,sources,x,t=original[(case['experiment'],case['cohort'])]
        order=np.random.default_rng(case['seed']+100*case['experiment']+case['cohort']).permutation(15);refs=order[:K];test=order[K:]
        assert not set(refs)&set(test);assert len(refs)==K and len(test)==15-K
        checks={'cal_reports':x[refs],'test_reports':x[test],'cal_truth':t[refs],'test_truth':t[test],
                'reference_ids':np.array(tasks)[refs],'test_ids':np.array(tasks)[test],'source_ids':sources}
        for name,v in checks.items():input_max=max(input_max,close(inputs[k+'__'+name],v,1e-8))
        q=inputs[k+'__cal_truth'];center=float(q.mean());scale=float(q.std()) if q.std()>1e-8 else 1.
        whole=np.vstack([inputs[k+'__cal_reports'],inputs[k+'__test_reports']]);xn=(whole-center)/scale;qn=(q-center)/scale
        case_data[k]=(xn,np.arange(K),qn,center,scale,inputs[k+'__test_truth'])
    rows={(r['key'],r['method']):r for r in R['rows']};metric_max=0.
    assert len(rows)==len(R['rows'])==len(M['cases'])*27
    for r in R['rows']:
        key=r['key'];truth=case_data[key][-1];pred=predictions[key+'__'+r['method']];close(predictions[key+'__truth'],truth)
        assert not r['failed'] and np.isfinite(pred).all();err=pred-truth
        for name,value in [('mse',np.mean(err**2)),('mae',np.mean(abs(err))),('p95',np.quantile(abs(err),.95))]:
            metric_max=max(metric_max,close(value,r[name],1e-8))
    summary_max=0.
    for summary in R['summary']:
        rr=[r for r in R['rows'] if (r['experiment'],r['budget'],r['method'])==(summary['experiment'],summary['budget'],summary['method'])]
        assert len(rr)==summary['conditions'] and summary['failed']==0
        summary_max=max(summary_max,close(np.sqrt(np.mean([r['mse'] for r in rr])),summary['rmse'],1e-9))
        summary_max=max(summary_max,close(np.mean([r['mae'] for r in rr]),summary['mae'],1e-9))
    tuning={(r['key'],r['variant']):r for r in R['tuning']}
    assert len(tuning)==2304
    for r in R['tuning']:
        assert len(r['scores'])==9 and r['selected']==int(np.argmin(r['scores']))
        assert P['mlni_grid'][r['selected']]==r['parameters']
        assert rows[(r['key'],'MLNI_JSAC2022__'+r['variant'])]['parameters']==r['parameters']
    reproduced=[];extended=[];core_max=defaultdict(float);cvrows=[];permrows=[];metadata_disagreements=[]
    for ci,case in enumerate(M['cases']):
        key=case['key'];x,refs,q,center,scale,truth=case_data[key];K=len(refs)
        for method,pred in controls(x,refs,q).items():
            d=close(pred[K:]*scale+center,predictions[key+'__'+method],1e-6);core_max[method]=max(core_max[method],d)
        for name in P['cores']:
            for variant in P['calibration_variants']:
                method=name+'__'+variant;r=rows[(key,method)];cfg=r['parameters']
                pred,meta=pipeline(name,variant,x,refs,q,cfg)
                d=close(pred[K:]*scale+center,predictions[key+'__'+method],1e-5);core_max[method]=max(core_max[method],d)
                reproduced.append(dict(key=key,method=method,max_prediction_difference=d))
                if name=='MLNI_JSAC2022':
                    if meta['converged']!=r['converged']:
                        metadata_disagreements.append(dict(key=key,method=method,reported=r['converged'],audited=meta['converged'],iterations=meta['iterations']))
                    if not r['converged']:
                        ep,em=pipeline(name,variant,x,refs,q,cfg,cap=2000,trace=True)
                        pp=ep[K:]*scale+center;oldp=predictions[key+'__'+method]
                        extended.append(dict(key=key,experiment=case['experiment'],budget=K,method=method,
                            parameters=cfg,old_iterations=r['iterations'],new_iterations=em['iterations'],new_converged=em['converged'],
                            old_max_delta=meta['max_delta'],new_max_delta=em['max_delta'],
                            old_objective=meta['objective'],new_objective=em['objective'],max_objective_increase=em['max_objective_increase'],
                            max_prediction_change=float(np.max(abs(pp-oldp))),rms_prediction_change=float(np.sqrt(np.mean((pp-oldp)**2))),
                            old_mse=r['mse'],extended_mse=float(np.mean((pp-truth)**2)),extended_predictions=pp.tolist()))
        # One complete grid per field/budget, for the first natural cohort/seed.
        if case['cohort']==0 and case['seed']==min(IP['seeds']):
            for variant in P['calibration_variants']:
                scores=[]
                for cfg in P['mlni_grid']:
                    e=[]
                    for held in refs:
                        train=refs[refs!=held]
                        pp,_=pipeline('MLNI_JSAC2022',variant,x,train,q[train],cfg)
                        e.append((pp[held]-q[held])**2)
                    scores.append(float(np.mean(e)))
                tr=tuning[(key,variant)];diff=close(scores,tr['scores'],1e-7)
                assert int(np.argmin(scores))==tr['selected']
                cvrows.append(dict(key=key,variant=variant,max_score_difference=diff,selected=tr['selected']))
        # Selected one-per-field/budget task permutation test; no temporal use.
        if case['cohort']==0 and case['seed']==min(IP['seeds']):
            order=np.random.default_rng(20260929+case['experiment']+K).permutation(len(x));inverse=np.argsort(order);newrefs=inverse[refs]
            for name in P['cores']:
                for variant in P['calibration_variants']:
                    cfg=rows[(key,name+'__'+variant)]['parameters'];base=predictions[key+'__'+name+'__'+variant]
                    pp,_=pipeline(name,variant,x[order],newrefs,q,cfg);pp=pp[inverse][K:]*scale+center
                    diff=close(pp,base,1e-5);permrows.append(dict(key=key,method=name+'__'+variant,max_difference=diff))
        if ci%48==0:print(json.dumps(dict(stage='audit',cases=ci+1,of=len(M['cases']),elapsed=time.perf_counter()-tick)),flush=True)
    extlookup={(r['key'],r['method']):r for r in extended};sensitivity=[]
    for summary in R['summary']:
        if not summary['method'].startswith('MLNI_'):continue
        rr=[r for r in R['rows'] if (r['experiment'],r['budget'],r['method'])==(summary['experiment'],summary['budget'],summary['method'])]
        errors=[extlookup.get((r['key'],r['method']),{}).get('extended_mse',r['mse']) for r in rr]
        sensitivity.append(dict(experiment=summary['experiment'],budget=summary['budget'],method=summary['method'],
            old_rmse=summary['rmse'],fixed_selected_config_2000_rmse=float(np.sqrt(np.mean(errors))),
            changed_fits=sum((r['key'],r['method']) in extlookup for r in rr)))
    result=dict(completed=True,scope='Independent team implementation and source/metric audit; not independent external replication',
        cases=len(M['cases']),metric_rows=len(R['rows']),summary_rows=len(R['summary']),input_max_difference=input_max,
        metric_max_difference=metric_max,summary_max_difference=summary_max,reconstructed_cohorts=len(original),
        repeat_count_range=[min(repeats),max(repeats)],reproduced_paper_rows=len(reproduced),max_prediction_difference_by_method=dict(core_max),
        tuning_selection_rows_checked=len(tuning),independent_full_grid_cases=len(cvrows),full_grid_checks=cvrows,
        permutation_cases=len(permrows),max_task_permutation_difference=max(r['max_difference'] for r in permrows),
        convergence_metadata_disagreements=metadata_disagreements,extended_fits=len(extended),
        extended_still_unconverged=sum(not r['new_converged'] for r in extended),
        maximum_extended_prediction_change=max((r['max_prediction_change'] for r in extended),default=0),
        extended_mse_improved=sum(r['extended_mse']<r['old_mse'] for r in extended),
        extended_objective_max_increase=max((r['max_objective_increase'] for r in extended),default=0),
        extended_rows=extended,fixed_config_sensitivity=sensitivity,
        source_and_protocol_hashes={f:sha(B/f) for f in ['mlni_calibration.py','triangles_calibration_protocol.json',
            'triangles_calibration_comparison_protocol.json','results/triangles_calibration_comparison.json','results/triangles_calibration_predictions.npz']},
        caveats=['No tests of encryption, privacy or task-allocation systems.',
            'Reference normalization is full paid-label normalization, not strict fold-pure normalization; floors/tolerances are scale dependent.',
            'Reported control runtimes share one elapsed block; not separate comparable training/inference measurements.',
            'MLNI reported runtimes include grid selection, whereas other methods have no corresponding search; raw runtime ranking is not pure inference.',
            'Extension holds selected prior and calibration variant fixed; it does not check whether 2000-step grid selection changes.',
            'Development data repeated object splits are not independent collections.'],elapsed=time.perf_counter()-tick)
    (B/'results/triangles_calibration_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['extended_rows','fixed_config_sensitivity','full_grid_checks']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
