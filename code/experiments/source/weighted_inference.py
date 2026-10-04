"""One post-main development control: common precision-weighted forward affine."""
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,hashlib,argparse,csv,math
from collections import defaultdict,Counter
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'deps'))
import numpy as np
import acquisition_evaluation as old
from mlni_calibration import mlni
R=B/'results';NAME='weighted_inference';PF=B/(NAME+'_protocol.json')
CACHE=B/'reading'/NAME;CACHE.mkdir(parents=True,exist_ok=True)
def path(s):return R/(NAME+s)
def read(p):return json.loads(Path(p).read_text('utf-8'))
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def oh(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()

def freeze():
    old.verify();p=dict(status='FROZEN_BEFORE_WEIGHTED_CONTROL_EXECUTION',scope='Post-main development adapter control only; no new data/purchase or baseline paper',
        original_protocol_sha256=sha(old.PF),purchased_hash=sha(old.path('_purchased_inputs.npz')),original_runs_hash=sha(old.path('_runs.json')),
        method_names=old.PAPERS,grid=old.GRID,cap=20000,weights='3/(epsilon_j**2+h_j**2), normalized to sum K; declared nominal gain1 proxy',
        mlni='Same reference normalizer and prior construction as original; only the common affine front end changes; train-fold h only',
        K1=dict(prior_strength=2.,truth_prior=10.),prediction_repeats=3,warmup=1,primary='All core outputs projected to the same public T=[0,500]',
        source_hashes={n:sha(B/n) for n in [Path(__file__).name,NAME+'_protocol.md','acquisition_evaluation.py','mlni_calibration.py','air_quality_baselines.py','eptd_inference.py','batch_baselines.py']})
    if PF.exists():assert read(PF)==p
    else:dump(PF,p)
    print(json.dumps(dict(frozen=True,sha256=sha(PF))))

def verify():
    p=read(PF);old.verify()
    for n,h in p['source_hashes'].items():assert verify_identity(B/n,h),n
    assert sha(old.PF)==p['original_protocol_sha256']
    assert sha(old.path('_purchased_inputs.npz'))==p['purchased_hash']
    assert sha(old.path('_runs.json'))==p['original_runs_hash']
    return p

def weighted_adapt(X,q,ref,h,eps):
    K=len(q)
    if K==0:return old.adapt(X,q,ref,'bounded_source_affine')
    w=3/(eps*eps+h*h)
    if K==1 or np.all(w==w[0]):
        tx,meta=old.adapt(X,q,ref,'bounded_source_affine');return tx,dict(**meta,weights=np.ones(K).tolist(),exact_original_adapter_branch=True)
    w=w*K/w.sum();wm=float(w.sum());C=X[ref];qm=float(q@w/wm);ym=w@C/wm
    v=float(np.sum(w*(q-qm)**2));a=np.ones(X.shape[1]) if v<=1e-14 else np.clip((w*(q-qm))@(C-ym[None,:])/v,.7,1.3)
    bias=(w@(C-a[None,:]*q[:,None]))/wm
    return (X-bias[None,:])/a[None,:],dict(gain=a.tolist(),bias=bias.tolist(),weights=w.tolist(),exact_original_adapter_branch=False)

def mlni_fit(X,q,ref,h,eps,cfg):
    tx,am=weighted_adapt(X,q,ref,h,eps);center=float(q.mean());scale=float(q.std());scale=scale if scale>1e-8 else 1.
    tx=(tx-center)/scale;qq=(q-center)/scale;z,meta=mlni(tx,tx[ref],qq,cfg,cap=20000)
    return z*scale+center,dict(**meta,normalizer_center=center,normalizer_scale=scale,adapter=am)

def choose(X,q,h,eps):
    K=len(q);start=time.perf_counter()
    if K==0:return None,dict(status='not_applicable_no_reference',seconds=0.,candidates=[])
    if K==1:return dict(prior_strength=2.,truth_prior=10.),dict(status='single_reference_fixed_no_cv',seconds=0.,candidates=[])
    ref=np.arange(K);candidates=[]
    for index,cfg in enumerate(old.GRID):
        folds=[]
        for held in ref:
            train=ref[ref!=held];z,meta=mlni_fit(X,q[train],train,h[train],eps[train],cfg);pred=float(z[held]);loss=(pred-float(q[held]))**2
            folds.append(dict(held=int(held),training_indices=train.tolist(),prediction=pred,loss=loss,training_h=h[train].tolist(),**meta))
        candidates.append(dict(index=index,config=cfg,score=float(np.mean([f['loss'] for f in folds])),folds=folds))
    selected=int(np.argmin([c['score'] for c in candidates]));return old.GRID[selected],dict(status='reference_only_cv',selected_index=selected,candidates=candidates,seconds=time.perf_counter()-start)

def core(name,X,q,h,eps,cfg):
    ref=np.arange(len(q))
    if name=='MLNI_JSAC2022':return mlni_fit(X,q,ref,h,eps,cfg)
    tx,am=weighted_adapt(X,q,ref,h,eps)
    if name=='CRH_TKDE2016':z,meta=old.crh_trace(tx)
    elif name=='EPTD_TIFS2022':z,meta=old.eptd_crh(tx,max_iterations=500,tolerance=1e-6);meta['status']='converged' if meta['converged'] else 'max_iterations'
    elif name.startswith('FETD_'):z=old.fetd_batch(tx,name.split('_')[1]);meta=dict(status='closed_form',iterations=0,converged=True)
    else:raise KeyError(name)
    return z,dict(**meta,adapter=am)

def bundle(Y,A,q,timing=True):
    X=Y.T;h=(A[:,1]-A[:,0])/2;eps=A[:,2];cfg,grid=choose(X,q,h,eps);out=[]
    for name in old.PAPERS:
        if name=='MLNI_JSAC2022' and cfg is None:out.append(dict(method=name,status='not_applicable_no_reference',prediction=None));continue
        if timing:core(name,X,q,h,eps,cfg)
        times=[]
        for rep in range(3 if timing else 1):
            tic=time.perf_counter();z,meta=core(name,X,q,h,eps,cfg);times.append(time.perf_counter()-tic);assert np.isfinite(z).all()
        out.append(dict(method=name,label=name+'__weighted_source_affine',status=meta.get('status','converged' if meta.get('converged') else 'max_iterations'),
            prediction=float(z[-1]),projected_prediction=float(np.clip(z[-1],0,500)),whole_prediction=z.tolist(),metadata=meta,
            selected_config=cfg if name=='MLNI_JSAC2022' else None,inference_times=times,inference_seconds=float(np.median(times)),selection_seconds=grid['seconds'] if name=='MLNI_JSAC2022' else 0.))
    return dict(records=out,grid=grid)

def strip(x):
    if isinstance(x,dict):return {k:strip(v) for k,v in x.items() if 'seconds' not in k and k!='inference_times'}
    if isinstance(x,list):return [strip(v) for v in x]
    return x

def predict(shadow=False):
    verify();M=read(old.path('_purchased_manifest.json'));D=np.load(old.path('_purchased_inputs.npz'),allow_pickle=False);bundles={};rows=[];tic=time.perf_counter()
    for ci,c in enumerate(M['cases']):
        key=c['inference_key'];Y=D[key+'__reports'];A=D[key+'__anchors'];q=D[key+'__centers'];tag=oh(dict(Y=Y.tolist(),A=A.tolist(),q=q.tolist(),protocol_hash=sha(PF)))
        if tag not in bundles:
            cache=CACHE/(tag+'.json')
            if shadow:
                item=bundle(Y,A,q,timing=False);assert strip(item)==strip(read(cache)),tag
            elif cache.exists():item=read(cache)
            else:item=bundle(Y,A,q);dump(cache,item)
            bundles[tag]=item
        compact={k:c[k] for k in ['key','day','target_station','seed','profile','plan_key','policy','budget','selected_count','total_cost','inference_key','scenario']}
        rows.extend([{**compact,'fit_key':tag,**r} for r in bundles[tag]['records']])
        if ci%96==0:print(json.dumps(dict(conditions=ci+1,total=864,unique=len(bundles),elapsed=time.perf_counter()-tic,shadow=shadow)),flush=True)
    if shadow:dump(path('_counterfactual.json'),dict(unique_bundles=len(bundles),all_predictions_and_grids_equal=True,compared_result_rows=len(rows)));return
    dump(path('_runs.json'),dict(protocol_sha256=sha(PF),rows=rows,unique_bundles=len(bundles),elapsed=time.perf_counter()-tic))
    dump(path('_grids.json'),{k:v['grid'] for k,v in bundles.items()})

def counterfactual():
    orig=np.load;attempts=[]
    def guarded(file,*args,**kwargs):
        name=str(file)
        if 'evaluator' in name:
            attempts.append(name);d=orig(file,*args,**kwargs);return {k:np.flip(d[k])+10000 for k in d.files}
        if 'simulator_secret' in name or 'beijing_first28days' in name:raise AssertionError(name)
        return orig(file,*args,**kwargs)
    np.load=guarded
    try:predict(shadow=True)
    finally:np.load=orig
    result=read(path('_counterfactual.json'));result['evaluator_numeric_load_attempts']=len(attempts);dump(path('_counterfactual.json'),result);assert not attempts

def analyze():
    verify();runs=read(path('_runs.json'));original=read(old.path('_runs.json'));T=np.load(old.path('_evaluator_truth.npz'),allow_pickle=False);D=np.load(old.path('_purchased_inputs.npz'),allow_pickle=False)
    oldrows={(r['inference_key'],r['method']):r for r in original['rows'] if r['adapter']=='bounded_source_affine'}
    cells=[];groups=defaultdict(list);single_diffs=[];constant_diffs=[]
    for r in runs['rows']:
        q=float(T[r['key']+'__target'][0]);oldrow=oldrows[(r['inference_key'],r['method'])];p=r['projected_prediction'];op=oldrow['projected_prediction'];A=D[r['inference_key']+'__anchors'];h=(A[:,1]-A[:,0])/2;w=3/(A[:,2]**2+h*h)
        diff=abs(r['prediction']-oldrow['prediction'])
        if r['selected_count']==1:single_diffs.append(diff)
        if np.all(w==w[0]):constant_diffs.append(diff)
        c={k:r[k] for k in ['inference_key','key','day','target_station','seed','profile','policy','budget','scenario','method','label','selected_count','total_cost','status','fit_key']}
        c.update(truth=q,prediction=r['prediction'],projected_prediction=p,original_projected_prediction=op,squared_error=(p-q)**2,absolute_error=abs(p-q),
            raw_squared_error=(r['prediction']-q)**2,original_squared_error=(op-q)**2,original_absolute_error=abs(op-q),inference_seconds=r['inference_seconds'],selection_seconds=r['selection_seconds'])
        cells.append(c);groups[(r['profile'],r['budget'],r['scenario'],r['policy'],r['method'])].append(c)
    summary=[]
    for key,rr in sorted(groups.items()):
        summary.append(dict(profile=key[0],budget=key[1],scenario=key[2],policy=key[3],method=key[4],n=len(rr),
            weighted_rmse=math.sqrt(sum(r['squared_error'] for r in rr)/len(rr)),original_rmse=math.sqrt(sum(r['original_squared_error'] for r in rr)/len(rr)),
            weighted_mae=sum(r['absolute_error'] for r in rr)/len(rr),original_mae=sum(r['original_absolute_error'] for r in rr)/len(rr),
            weighted_max=max(r['absolute_error'] for r in rr),original_max=max(r['original_absolute_error'] for r in rr),
            weighted_raw_rmse=math.sqrt(sum(r['raw_squared_error'] for r in rr)/len(rr)),
            point_wins=sum(r['squared_error']<r['original_squared_error']-1e-12 for r in rr),point_ties=sum(abs(r['squared_error']-r['original_squared_error'])<=1e-12 for r in rr)))
    grids=read(path('_grids.json'));cases={r['fit_key']:r for r in runs['rows']};folds=0;nonconv=0;maxiter=0;scorediff=0.;scores=0;weightdiff=0.
    for tag,g in grids.items():
        key=cases[tag]['inference_key'];q=D[key+'__centers'];A=D[key+'__anchors'];h=(A[:,1]-A[:,0])/2;eps=A[:,2]
        for cand in g['candidates']:
            losses=[]
            for f in cand['folds']:
                tr=f['training_indices'];held=f['held'];assert held not in tr and f['training_h']==h[tr].tolist()
                mu=float(q[tr].mean());sd=float(q[tr].std());sd=sd if sd>1e-8 else 1.;assert mu==f['normalizer_center'] and sd==f['normalizer_scale']
                ww=3/(eps[tr]**2+h[tr]**2);ww=ww*len(tr)/ww.sum();weightdiff=max(weightdiff,float(np.max(abs(np.array(f['adapter']['weights'])-ww))))
                loss=(f['prediction']-float(q[held]))**2;losses.append(loss);scorediff=max(scorediff,abs(loss-f['loss']));folds+=1;nonconv+=not f['converged'];maxiter=max(maxiter,f['iterations'])
            scorediff=max(scorediff,abs(sum(losses)/len(losses)-cand['score']));scores+=1
        if g['candidates']:assert g['selected_index']==int(np.argmin([c['score'] for c in g['candidates']]))
    audit=dict(rows=len(cells),unique_bundles=runs['unique_bundles'],single_anchor_rows=len(single_diffs),single_anchor_max_change=max(single_diffs,default=0.),
        equal_weight_rows=len(constant_diffs),equal_weight_max_change=max(constant_diffs,default=0.),reference_scores=scores,reference_folds=folds,
        reference_nonconverged=nonconv,reference_max_iterations=maxiter,score_recomputation_max_diff=scorediff,training_weight_max_diff=weightdiff,
        final_nonconverged=sum(r['status']=='max_iterations' for r in cells),statuses=dict(Counter(r['status'] for r in cells)))
    dump(path('_summary.json'),dict(protocol_sha256=sha(PF),runs_hash=sha(path('_runs.json')),summary=summary,integrity=audit))
    for suffix,rr in [('_cells.csv',cells),('_summary.csv',summary)]:
        with path(suffix).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rr[0]));writer.writeheader();writer.writerows(rr)
    print(json.dumps(audit,indent=2))

def smoke():
    q=np.array([10.,50.,100.]);X=np.column_stack([q*1.2+7,q*.8-3]);ref=np.arange(3)
    for hh in [np.zeros(3),np.ones(3)*10]:
        a,_=weighted_adapt(X,q,ref,hh,np.ones(3)*5);b,_=old.adapt(X,q,ref,'bounded_source_affine');assert np.array_equal(a,b)
    a,_=weighted_adapt(X,q[:1],np.array([0]),np.array([80.]),np.array([5.]));b,_=old.adapt(X,q[:1],np.array([0]),'bounded_source_affine');assert np.array_equal(a,b)
    # Independent weighted normal equations with gain interior recover exact response.
    a,meta=weighted_adapt(X,q,ref,np.array([2.,10.,80.]),np.ones(3)*5);assert np.max(abs(a-q[:,None]))<1e-12
    print(json.dumps(dict(constant_and_single_anchor_equivalence=True,interior_gain_weighted_regression=True)))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','freeze','predict','counterfactual','analyze']);args=ap.parse_args()
    globals()[args.stage]()
