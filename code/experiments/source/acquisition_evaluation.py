"""Frozen development comparison: purchase, fulfillment, inference, evaluation separated."""
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
import os
for _k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[_k]='1'
import sys,json,hashlib,time,argparse,platform,itertools,math
from fractions import Fraction as F
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'deps'))
import numpy as np
from catalog_selection import design_anchors
from reference_geometry import worst_width,jsonable
from packet_geometry import packet_fusion
from certified_geometry import exact_design,exact_packet_fusion
from mlni_calibration import mlni
from air_quality_baselines import crh_trace
from eptd_inference import eptd_crh
from batch_baselines import fetd_batch
R=B/'results';R.mkdir(exist_ok=True)
NAME='acquisition_evaluation';PF=B/(NAME+'_protocol.json')
CACHE=B/'reading'/'source_bridge_acquisition_cache';CACHE.mkdir(parents=True,exist_ok=True)
PAPERS=['CRH_TKDE2016','EPTD_TIFS2022','FETD_AK_AAAI2023','FETD_D_AAAI2023','MLNI_JSAC2022']
ADAPTERS=['native','bounded_source_affine']
GRID=[dict(prior_strength=s,truth_prior=t) for s in [.2,2.,20.] for t in [1.,10.,100.]]

def path(s):return R/(NAME+s)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def read(p):return json.loads(Path(p).read_text('utf-8'))
def objhash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def arrayhash(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def common():return dict(target=[0.,500.],gain_bounds=[.7,1.3],epsilon=5.,m=9,f=2)

def freeze():
    files=[Path(__file__).name,'acquisition_evaluation_analyze.py','source_bridge_acquisition_published_core_protocol_DRAFT.md',
        'catalog_selection.py','reference_geometry.py','packet_geometry.py',
        'certified_geometry.py','mlni_calibration.py','air_quality_baselines.py',
        'eptd_inference.py','batch_baselines.py','group_methods.py']
    P=dict(status='FROZEN_BEFORE_EXECUTION',scope='Training-only MSRA development; simulated sources; no independent confirmation',
        dataset='data/msra_urban_air/interface/beijing_first28days_all_pollutants.npz',
        dataset_sha256='110e37b15b171cbbe596316f9340950d549248ea8da4bf73cf8ca8c9a8d51b40',
        days=['2014-05-01','2014-05-05','2014-05-09'],target_stations=['001001','001002'],target_hour=12,
        anchor_stations=['001003','001004','001005'],anchor_hours=[11,13],seeds=[419031,419032],
        profiles=dict(exact=[0,0,0,0,0,0],equal=[10,10,10,10,10,10],heterogeneous=[2,5,10,20,40,80]),
        costs=[1,2,1,2,1,2],budgets=[4,7],scenarios=['homogeneous_honest','affine_honest','affine_two_source_attack'],
        policies=['MinimaxPairTable','MaxSpan','NominalCOptimal','FixedIdBudget'],spec=common(),
        paper_cores=PAPERS,adapters=ADAPTERS,mlni_grid=GRID,mlni_cap=20000,mlni_K1=dict(prior_strength=2.,truth_prior=10.),
        cv='Fold-local training-reference mean/std with <=1e-8 fallback1; transform whole unlabeled batch; fold-local affine/priors; original-unit held-reference squared error.',
        coptimal_weight='3/(epsilon**2 + h**2)',timing=dict(warmup=1,repetitions=3,selection_repetitions=1,threads=1),
        counterfactual='Same fixed purchased inputs; evaluator numeric access forbidden; perturb evaluator only, not generation.',
        source_hashes={f:sha(B/f) for f in files})
    if PF.exists():assert read(PF)==P,'frozen protocol differs'
    else:dump(PF,P)
    print(json.dumps(dict(frozen=True,protocol_sha256=sha(PF)),indent=2),flush=True)

def verify():
    P=read(PF);assert P['status']=='FROZEN_BEFORE_EXECUTION'
    for name,dig in P['source_hashes'].items():assert verify_identity(B/name,dig),name
    return P

def prepare():
    P=verify();assert sha(B/P['dataset'])==P['dataset_sha256'];D=np.load(B/P['dataset'],allow_pickle=False)
    pi=int(np.where(D['pollutants']=='PM25_Concentration')[0][0]);pub={};secret={};truth={};cases=[];hidden=[]
    def value(station,day,hour):
        si=int(np.where(D['station_ids']==station)[0][0]);ti=int(np.where(D['timestamps']==np.datetime64(f'{day}T{hour:02}:00:00','s'))[0][0])
        assert D['station_split'][si]=='station_train' and D['temporal_split'][ti]=='train'
        return float(D['reference_values'][si,ti,pi]) if D['reference_observed'][si,ti,pi] else None
    for di,day in enumerate(P['days']):
        candidates=[(s,h) for s in P['anchor_stations'] for h in P['anchor_hours']]
        aq=[value(s,day,h) for s,h in candidates];available=[i for i,q in enumerate(aq) if q is not None]
        for seed in P['seeds']:
            # Centers are identical for both target stations of this day/seed/profile.
            measurement_u=np.random.default_rng(seed+1000*di+99991).uniform(-.9,.9,6)
            for tsi,station in enumerate(P['target_stations']):
                qt=value(station,day,P['target_hour']);rng=np.random.default_rng(seed+1000*di+100*tsi)
                gains=rng.uniform(.7,1.3,9);bias=rng.uniform(-20,20,9);noise=rng.uniform(-4.5,4.5,(9,7))
                fullq=np.array([np.nan if v is None else v for v in aq]+[np.nan if qt is None else qt])
                raw=dict(homogeneous_honest=fullq[None,:]+noise,affine_honest=gains[:,None]*fullq[None,:]+bias[:,None]+noise)
                raw['affine_two_source_attack']=raw['affine_honest'].copy();raw['affine_two_source_attack'][-2:,-1]+=80
                for profile,hh in P['profiles'].items():
                    key=f'd{di}_t{tsi}_s{seed}_{profile}';h=np.array(hh,dtype=float)[available];q=np.array([aq[i] for i in available]);centers=q+h*measurement_u[available]
                    lo=centers-h;hi=centers+h;anchors=np.column_stack([lo,hi,np.full(len(q),5.)]);costs=np.array(P['costs'])[available]
                    coarse_ok=all(F(float(l))<=F(float(v))<=F(float(u)) for l,v,u in zip(lo,q,hi))
                    pub[key+'__anchors']=anchors;pub[key+'__centers']=centers;pub[key+'__radii']=h;pub[key+'__costs']=costs
                    pub[key+'__anchor_original_ids']=np.array(available,dtype=int)
                    case=dict(key=key,day=day,day_index=di,target_station=station,seed=seed,profile=profile,
                        available_anchor_ids=available,missing_anchor_ids=[i for i in range(6) if i not in available],target_observed=qt is not None,
                        coarse_measurement_cost=sum(64 if v==0 else max(1,math.ceil(20/v)) for v in h),objects_available=len(q)+int(qt is not None))
                    cases.append(case);truth[key+'__target']=np.array([np.nan if qt is None else qt]);truth[key+'__anchor_truth']=q
                    for scenario,Y in raw.items():
                        sk=key+'__'+scenario;rows=range(9) if scenario!='affine_two_source_attack' else range(7)
                        aa=np.ones(9) if scenario=='homogeneous_honest' else gains;bb=np.zeros(9) if scenario=='homogeneous_honest' else bias
                        observed=available+([6] if qt is not None else [])
                        residuals=[abs(F(float(Y[i,j]))-F(float(aa[i]))*F(float(fullq[j]))-F(float(bb[i]))) for i in rows for j in observed]
                        honest_ok=all(r<=F(5) for r in residuals) and all(F(.7)<=F(float(a))<=F(1.3) for a in aa)
                        secret[sk+'__potential_reports']=Y[:,available+[6]]
                        hidden.append(dict(key=key,scenario=scenario,coarse_interval_valid=coarse_ok,honest_report_spec_valid=honest_ok,
                            target_domain_valid=qt is not None and 0<=qt<=500,maximum_exact_honest_residual=float(max(residuals,default=F(0))),
                            gains=aa.tolist(),biases=bb.tolist(),controlled_source_ids=[] if scenario!='affine_two_source_attack' else [7,8]))
    np.savez_compressed(path('_public_inputs.npz'),**pub);np.savez_compressed(path('_simulator_secret.npz'),**secret);np.savez_compressed(path('_evaluator_truth.npz'),**truth)
    dump(path('_evaluator_contracts.json'),hidden)
    dump(path('_manifest.json'),dict(cases=cases,protocol_sha256=sha(PF),public_hash=sha(path('_public_inputs.npz')),secret_hash=sha(path('_simulator_secret.npz')),
        evaluator_hash=sha(path('_evaluator_truth.npz')),contracts_hash=sha(path('_evaluator_contracts.json')),source_dataset_hash=sha(B/P['dataset']),
        environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),threads=1)))
    print(json.dumps(dict(prepared=len(cases),missing_targets=sum(not c['target_observed'] for c in cases),coarse_failures=sum(not c['coarse_interval_valid'] for c in hidden),honest_failures=sum(not c['honest_report_spec_valid'] for c in hidden))),flush=True)

def select(policy,A,centers,radii,costs,budget,context):
    n=len(A);S=common();meta={}
    feasible=[ids for k in range(n+1) for ids in itertools.combinations(range(n),k) if sum(costs[list(ids)])<=budget]
    if policy=='MinimaxPairTable':
        r=design_anchors(A.tolist(),S['target'],S['gain_bounds'],5.,costs.tolist(),budget);ids=tuple(r['anchor_ids']);meta=r
    elif policy=='MaxSpan':
        pairs=[ids for ids in feasible if len(ids)==2]
        if pairs:ids=min(pairs,key=lambda t:(-abs(centers[t[0]]-centers[t[1]]),sum(costs[list(t)]),t))
        else:ids=min(feasible,key=lambda t:(-len(t),sum(costs[list(t)]),t))
    elif policy=='NominalCOptimal':
        records=[];best=None
        for ids in feasible:
            score=math.inf;cond=None;status='rank_deficient'
            if len(ids)>=2:
                c=centers[list(ids)];h=radii[list(ids)];X=np.column_stack([np.ones(len(ids)),c]);M=X.T@((3/(25+h*h))[:,None]*X)
                if np.linalg.matrix_rank(M)==2:
                    cond=float(np.linalg.cond(M))
                    try:
                        inv=np.linalg.inv(M);score=max(float(np.array([1.,t])@inv@np.array([1.,t])) for t in [0.,500.]);status='finite' if np.isfinite(score) else 'nonfinite'
                    except np.linalg.LinAlgError:status='inverse_failed'
            score=score if np.isfinite(score) else math.inf;cost=int(sum(costs[list(ids)]));key=(score,cost,len(ids),ids)
            records.append(dict(ids=list(ids),score=None if math.isinf(score) else score,condition_number=cond,status=status))
            if best is None or key<best[0]:best=(key,ids)
        ids=best[1];meta=dict(candidate_states=records,selected_score=None if math.isinf(best[0][0]) else best[0][0])
    elif policy=='FixedIdBudget':
        order=sorted(range(n),key=lambda j:hashlib.sha256((context['day']+'|'+context['target_station']+'|'+str(context['original_ids'][j])+'|419031').encode()).hexdigest())
        chosen=[];spent=0
        for j in order:
            if spent+costs[j]<=budget:chosen.append(j);spent+=costs[j]
        ids=tuple(sorted(chosen));meta=dict(order=order)
    else:raise KeyError(policy)
    return list(ids),meta

def plans():
    P=verify();M=read(path('_manifest.json'));assert sha(path('_public_inputs.npz'))==M['public_hash'];D=np.load(path('_public_inputs.npz'),allow_pickle=False)
    plans=[];cache={};S=common();start=time.perf_counter()
    for ci,case in enumerate(M['cases']):
        key=case['key'];A=D[key+'__anchors'];centers=D[key+'__centers'];radii=D[key+'__radii'];costs=D[key+'__costs'];original=D[key+'__anchor_original_ids'].tolist()
        for budget in P['budgets']:
            for policy in P['policies']:
                context=dict(day=case['day'],target_station=case['target_station'],original_ids=original)
                tag=objhash(dict(policy=policy,anchors=A.tolist(),centers=centers.tolist(),radii=radii.tolist(),costs=costs.tolist(),budget=budget,context=context if policy=='FixedIdBudget' else None))
                if tag not in cache:
                    select(policy,A,centers,radii,costs,budget,context);times=[]
                    for repeat in range(3):
                        tick=time.perf_counter();ids,meta=select(policy,A,centers,radii,costs,budget,context);times.append(time.perf_counter()-tick)
                    width=worst_width(A[ids].tolist(),S['target'],S['gain_bounds'],5.)['width']
                    cache[tag]=dict(local_anchor_ids=ids,prequery_width=width,selection_times=times,selection_seconds=float(np.median(times)),selection_metadata=meta)
                r=cache[tag];ids=r['local_anchor_ids'];pk=f'{key}__B{budget}__{policy}'
                plans.append(dict(**case,plan_key=pk,policy=policy,budget=budget,cache_key=tag,original_selected_ids=[original[j] for j in ids],
                    selected_count=len(ids),report_anchor_cost_per_source=int(sum(costs[ids])),total_cost=case['coarse_measurement_cost']+9*(1+int(sum(costs[ids]))),**r))
        if ci%6==0:print(json.dumps(dict(planned_cases=ci+1,total=len(M['cases']),elapsed=time.perf_counter()-start)),flush=True)
    dump(path('_plans.json'),dict(protocol_sha256=sha(PF),public_hash=M['public_hash'],unique_selections=len(cache),plans=plans))

def fulfill():
    P=verify();M=read(path('_manifest.json'));plans=read(path('_plans.json'));D=np.load(path('_public_inputs.npz'),allow_pickle=False)
    assert sha(path('_simulator_secret.npz'))==M['secret_hash'];secret=np.load(path('_simulator_secret.npz'),allow_pickle=False);out={};cases=[]
    for plan in plans['plans']:
        key=plan['key'];ids=plan['local_anchor_ids'];n=len(D[key+'__anchors'])
        for scenario in P['scenarios']:
            ck=plan['plan_key']+'__'+scenario
            out[ck+'__reports']=secret[key+'__'+scenario+'__potential_reports'][:,ids+[n]]
            out[ck+'__anchors']=D[key+'__anchors'][ids];out[ck+'__centers']=D[key+'__centers'][ids]
            cases.append(dict(**plan,inference_key=ck,scenario=scenario))
    np.savez_compressed(path('_purchased_inputs.npz'),**out)
    dump(path('_purchased_manifest.json'),dict(cases=cases,protocol_sha256=sha(PF),plans_hash=sha(path('_plans.json')),purchased_hash=sha(path('_purchased_inputs.npz'))))
    print(json.dumps(dict(fulfilled=len(cases))),flush=True)

def adapt(X,q,ref,kind):
    if kind=='native' or len(ref)==0:return X.copy(),dict(gain=np.ones(X.shape[1]).tolist(),bias=np.zeros(X.shape[1]).tolist())
    C=X[ref];meanq=float(q.mean());v=float(np.sum((q-meanq)**2))
    a=np.ones(X.shape[1]) if v<=1e-14 else np.clip(np.sum((q-meanq)[:,None]*(C-C.mean(axis=0)),axis=0)/v,.7,1.3)
    b=np.mean(C-a[None,:]*q[:,None],axis=0)
    return (X-b[None,:])/a[None,:],dict(gain=a.tolist(),bias=b.tolist())

def mlni_fit(X,q,ref,adapter,cfg):
    center=float(q.mean());scale=float(q.std());scale=scale if scale>1e-8 else 1.
    # Apply the common physical-unit variance threshold before normalization.
    # Only this fold's training centers/ref rows enter the source adapter.
    tx_raw,am=adapt(X,q,ref,adapter);tx=(tx_raw-center)/scale;qq=(q-center)/scale
    z,meta=mlni(tx,tx[ref],qq,cfg,cap=20000)
    return z*scale+center,dict(**meta,normalizer_center=center,normalizer_scale=scale,adapter=am)

def core(name,X,q,adapter,cfg=None):
    ref=np.arange(len(q))
    if name=='MLNI_JSAC2022':return mlni_fit(X,q,ref,adapter,cfg)
    tx,am=adapt(X,q,ref,adapter)
    if name=='CRH_TKDE2016':z,meta=crh_trace(tx)
    elif name=='EPTD_TIFS2022':z,meta=eptd_crh(tx,max_iterations=500,tolerance=1e-6);meta['status']='converged' if meta['converged'] else 'max_iterations'
    elif name.startswith('FETD_'):z=fetd_batch(tx,name.split('_')[1]);meta=dict(status='closed_form',iterations=0,converged=True)
    else:raise KeyError(name)
    return z,dict(**meta,adapter=am)

def choose_mlni(X,q,adapter):
    K=len(q);tick=time.perf_counter()
    if K==0:return None,dict(status='not_applicable_no_reference',seconds=0.,candidates=[])
    if K==1:return dict(prior_strength=2.,truth_prior=10.),dict(status='single_reference_fixed_no_cv',seconds=0.,candidates=[])
    ref=np.arange(K);candidates=[]
    for gi,cfg in enumerate(GRID):
        folds=[]
        for held in ref:
            train=ref[ref!=held];pred,meta=mlni_fit(X,q[train],train,adapter,cfg);pv=float(pred[held]);loss=(pv-float(q[held]))**2
            folds.append(dict(held=int(held),prediction=pv,loss=loss,training_indices=train.tolist(),**meta))
        candidates.append(dict(index=gi,config=cfg,score=float(np.mean([f['loss'] for f in folds])),folds=folds))
    winner=int(np.argmin([c['score'] for c in candidates]));return GRID[winner],dict(status='reference_only_cv',selected_index=winner,candidates=candidates,seconds=time.perf_counter()-tick)

def fit_bundle(reports,anchors,q,timing=True):
    X=reports.T;records=[];selected={};S=common()
    for adapter in ADAPTERS:
        cfg,grid=choose_mlni(X,q,adapter);selected[adapter]=grid
        for name in PAPERS:
            label=name+'__'+adapter
            if name=='MLNI_JSAC2022' and cfg is None:
                records.append(dict(label=label,method=name,adapter=adapter,status='not_applicable_no_reference',prediction=None));continue
            reps=3 if timing else 1
            if timing:core(name,X,q,adapter,cfg)
            times=[]
            for repeat in range(reps):
                tick=time.perf_counter();z,meta=core(name,X,q,adapter,cfg);times.append(time.perf_counter()-tick)
                if not np.isfinite(z).all():raise RuntimeError('nonfinite '+label)
            status=meta.get('status','converged' if meta.get('converged',False) else 'max_iterations')
            records.append(dict(label=label,method=name,adapter=adapter,status=status,prediction=float(z[-1]),projected_prediction=float(np.clip(z[-1],0,500)),
                whole_prediction=z.tolist(),metadata=meta,selected_config=cfg if name=='MLNI_JSAC2022' else None,
                inference_times=times,inference_seconds=float(np.median(times)),selection_seconds=grid['seconds'] if name=='MLNI_JSAC2022' else 0.))
    if timing:packet_fusion(reports,anchors.tolist(),S['target'],S['gain_bounds'],5.,2)
    times=[]
    for repeat in range(3 if timing else 1):
        tick=time.perf_counter();r=packet_fusion(reports,anchors.tolist(),S['target'],S['gain_bounds'],5.,2);times.append(time.perf_counter()-tick)
    records.append(dict(label='SourceBridge_packet',method='SourceBridge_packet',adapter='bounded_interval_depth',status='feasible' if r['feasible'] else 'infeasible',prediction=r['point'],
        projected_prediction=r['point'],inference_times=times,inference_seconds=float(np.median(times)),selection_seconds=0.,metadata=r))
    return dict(records=records,mlni_grids=selected)

def predict(shadow=False):
    P=verify();M=read(path('_purchased_manifest.json'));assert sha(path('_purchased_inputs.npz'))==M['purchased_hash'];D=np.load(path('_purchased_inputs.npz'),allow_pickle=False)
    rows=[];bundles={};start=time.perf_counter();fitted=0;shadow_checks=[]
    for ci,case in enumerate(M['cases']):
        ck=case['inference_key'];reports=D[ck+'__reports'];anchors=D[ck+'__anchors'];q=D[ck+'__centers']
        tag=objhash(dict(reports=reports.tolist() if case['target_observed'] else 'missing_target_report',anchors=anchors.tolist(),centers=q.tolist(),protocol_hash=sha(PF)))
        if tag not in bundles:
            cp=CACHE/(tag+'.json')
            if not case['target_observed']:
                bundle=dict(records=[dict(label=name+'__'+a,method=name,adapter=a,status='missing_target_report',prediction=None) for a in ADAPTERS for name in PAPERS]+[dict(label='SourceBridge_packet',method='SourceBridge_packet',adapter='bounded_interval_depth',status='missing_target_report',prediction=None)],mlni_grids={})
            elif shadow:
                bundle=fit_bundle(reports,anchors,q,timing=False);original=read(cp)
                def strip(x):
                    if isinstance(x,dict):return {k:strip(v) for k,v in x.items() if 'second' not in k and k not in ['inference_times']}
                    if isinstance(x,list):return [strip(v) for v in x]
                    return x
                assert strip(bundle)==strip(original),tag;shadow_checks.append(tag)
            elif cp.exists():bundle=read(cp)
            else:
                bundle=fit_bundle(reports,anchors,q);dump(cp,bundle);fitted+=1
            bundles[tag]=bundle
        compact={k:case[k] for k in ['key','day','day_index','target_station','seed','profile','target_observed','plan_key','policy','budget','original_selected_ids',
            'selected_count','report_anchor_cost_per_source','total_cost','local_anchor_ids','prequery_width','inference_key','scenario']}
        compact['acquisition_seconds']=case['selection_seconds']
        for rec in bundles[tag]['records']:rows.append({**compact,'fit_key':tag,**rec})
        if ci%12==0:print(json.dumps(dict(predicted_cases=ci+1,total=len(M['cases']),unique_bundles=len(bundles),new_fits=fitted,elapsed=time.perf_counter()-start,shadow=shadow)),flush=True)
    if shadow:
        dump(path('_counterfactual_checks.json'),dict(unique_bundles=len(bundles),compared_bundles=len(shadow_checks),all_predictions_and_grids_identical=True));return
    dump(path('_runs.json'),dict(protocol_sha256=sha(PF),purchased_hash=M['purchased_hash'],unique_bundles=len(bundles),rows=rows,elapsed=time.perf_counter()-start))
    dump(path('_mlni_grids.json'),{k:v['mlni_grids'] for k,v in bundles.items()})

def exact_checks():
    P=verify();M=read(path('_purchased_manifest.json'));D=np.load(path('_purchased_inputs.npz'),allow_pickle=False);pub=np.load(path('_public_inputs.npz'),allow_pickle=False)
    plansdata=read(path('_plans.json'));S=common();designs={};fits={};records=[];tic=time.perf_counter()
    for i,plan in enumerate(plansdata['plans']):
        key=plan['key'];A=pub[key+'__anchors'];costs=pub[key+'__costs'];tag=objhash(dict(A=A.tolist(),costs=costs.tolist(),budget=plan['budget'],protocol_hash=sha(PF)))
        if tag not in designs:
            cache=CACHE/('exact_design_'+tag+'.json')
            if cache.exists():de=read(cache)
            else:
                er=exact_design(A.tolist(),S['target'],S['gain_bounds'],5.,costs.tolist(),plan['budget'])
                # Independent exhaustive feasible-subset value check, untimed.
                allsub=[]
                for k in range(len(A)+1):
                    for ids in itertools.combinations(range(len(A)),k):
                        if sum(costs[list(ids)])<=plan['budget']:
                            w=worst_width(A[list(ids)].tolist(),S['target'],S['gain_bounds'],5.,exact=True)['width'];allsub.append((w,int(sum(costs[list(ids)])),len(ids),ids))
                fullbest=min(allsub);assert er['width']==fullbest[0]
                de=dict(exact_design=jsonable(er),full_enumeration_value=str(fullbest[0]),full_enumeration_best_ids=list(fullbest[3]),full_enumeration_subset_count=len(allsub),value=float(er['width']))
                dump(cache,de)
            designs[tag]=de
        ids=plan['local_anchor_ids'];w=worst_width(A[ids].tolist(),S['target'],S['gain_bounds'],5.,exact=True)['width']
        records.append(dict(plan_key=plan['plan_key'],design_key=tag,selected_exact_width=str(w),selected_exact_width_float=float(w),float_width_difference=plan['prequery_width']-float(w),
            suboptimality=float(w)-designs[tag]['value'],same_exact_optimum=F(str(w))==F(designs[tag]['full_enumeration_value']),
            selected_cost=int(sum(costs[ids])),exact_optimal_cost=float(F(designs[tag]['exact_design']['cost'])),
            same_exact_cost_and_count=int(sum(costs[ids]))==F(designs[tag]['exact_design']['cost']) and len(ids)==len(designs[tag]['exact_design']['anchor_ids'])))
        if i%24==0:print(json.dumps(dict(exact_plans=i+1,total=len(plansdata['plans']),elapsed=time.perf_counter()-tic)),flush=True)
    fitrows=[]
    for ci,case in enumerate(M['cases']):
        if not case['target_observed']:continue
        ck=case['inference_key'];A=D[ck+'__anchors'];reports=D[ck+'__reports'];tag=objhash(dict(A=A.tolist(),reports=reports.tolist(),protocol_hash=sha(PF)))
        if tag not in fits:
            cache=CACHE/('exact_fit_'+tag+'.json')
            if cache.exists():er=read(cache)
            else:er=jsonable(exact_packet_fusion(reports.tolist(),A.tolist(),S['target'],S['gain_bounds'],5.,2));dump(cache,er)
            fits[tag]=er
        fitrows.append(dict(inference_key=ck,exact_fit_key=tag))
        if ci%48==0:print(json.dumps(dict(exact_fits=ci+1,total=len(M['cases']),unique=len(fits),elapsed=time.perf_counter()-tic)),flush=True)
    dump(path('_exact_checks.json'),dict(protocol_sha256=sha(PF),designs=designs,plan_checks=records,fit_references=fits,fit_cases=fitrows,elapsed=time.perf_counter()-tic))

def counterfactual():
    original=np.load;accesses=[]
    def guarded(file,*args,**kwargs):
        name=str(file)
        if 'evaluator' in name:
            accesses.append(name);data=original(file,*args,**kwargs)
            return {k:np.flip(np.array(data[k]))+10000 for k in data.files}
        if 'simulator_secret' in name or 'beijing_first28days' in name:raise AssertionError('prediction accessed unpurchased/private data: '+name)
        return original(file,*args,**kwargs)
    np.load=guarded
    try:predict(shadow=True)
    finally:np.load=original
    r=read(path('_counterfactual_checks.json'));r['evaluator_numeric_load_attempts']=len(accesses);r['evaluator_access_paths']=accesses;dump(path('_counterfactual_checks.json'),r)
    assert not accesses

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','prepare','plans','fulfill','predict','exact','counterfactual']);a=ap.parse_args()
    {'freeze':freeze,'prepare':prepare,'plans':plans,'fulfill':fulfill,'predict':predict,'exact':exact_checks,'counterfactual':counterfactual}[a.stage]()
