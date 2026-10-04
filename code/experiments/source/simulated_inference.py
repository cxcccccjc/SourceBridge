"""Outcome-driven pure-simulation mechanism test; prior results stay immutable."""
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,hashlib,time,argparse,itertools,math,contextlib,io
from fractions import Fraction as F
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'deps'))
import numpy as np
import acquisition_evaluation as old
import weighted_inference as weighted
from reference_geometry import worst_width,jsonable
from certified_geometry import exact_design,exact_packet_fusion
from packet_geometry import packet_fusion
from safe_projection import project
R=B/'results';NAME='simulated_inference';PF=B/(NAME+'_protocol.json');CACHE=B/'reading'/NAME;CACHE.mkdir(parents=True,exist_ok=True)
def path(s):return R/(NAME+s)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def oh(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()

def freeze():
    old.verify();weighted.verify()
    names=[Path(__file__).name,NAME+'_protocol.md','analyze_simulated_inference.py','acquisition_evaluation.py','weighted_inference.py',
        'reference_geometry.py','catalog_selection.py','certified_geometry.py','packet_geometry.py',
        'safe_projection.py','mlni_calibration.py','air_quality_baselines.py','eptd_inference.py','batch_baselines.py']
    p=dict(status='FROZEN_BEFORE_SPAN_DIAGNOSTIC_EXECUTION',scope='Outcome-driven pure simulation, not independent confirmation or new public data',
        spans=[20,150,450],anchor_positions=[-.5,-.3,-.1,.1,.3,.5],anchor_center=250.,target_values=[40.,250.,460.],seeds=[419131,419132],
        profiles=dict(exact=[0.,0.,0.,0.,0.,0.],heterogeneous=[2.,5.,10.,20.,40.,80.]),costs=[1,2,1,2,1,2],budgets=[4,7],
        scenarios=['affine_honest','affine_two_source_attack'],policies=['MinimaxPairTable','MaxSpan','NominalCOptimal','FixedIdBudget'],
        spec=old.common(),methods=old.PAPERS,adapters=['native','weighted_source_affine'],median_control=True,precision_requirement=20.,
        projection_core='EPTD_TIFS2022__weighted_source_affine',projection='Exact hull and exact purchased W/2, separate component table',
        mlni_grid=old.GRID,mlni_cap=20000,K1=dict(prior_strength=2.,truth_prior=10.),timing=dict(warmup=1,repetitions=3,selection_repetitions=1),
        source_hashes={n:sha(B/n) for n in names})
    if PF.exists():assert read(PF)==p
    else:dump(PF,p)
    print(json.dumps(dict(frozen=True,sha256=sha(PF))))

def verify():
    p=read(PF)
    for n,h in p['source_hashes'].items():assert verify_identity(B/n,h),n
    return p

def prepare():
    p=verify();public={};secret={};truth={};cases=[];contracts=[]
    for seed in p['seeds']:
        rng=np.random.default_rng(seed);a=rng.uniform(.7,1.3,9);b=rng.uniform(-20,20,9);e=rng.uniform(-4.5,4.5,(9,7));u=rng.uniform(-.9,.9,6)
        for span in p['spans']:
            aq=250+span*np.array(p['anchor_positions'])
            for profile,hh in p['profiles'].items():
                h=np.array(hh);c=aq+u*h;A=np.column_stack([c-h,c+h,np.ones(6)*5]);valid=all(F(float(lo))<=F(float(q))<=F(float(hi)) for q,(lo,hi,_) in zip(aq,A))
                for ti,qt in enumerate(p['target_values']):
                    key=f's{seed}_span{span}_{profile}_t{ti}';public[key+'__anchors']=A;public[key+'__centers']=c;public[key+'__radii']=h;public[key+'__costs']=np.array(p['costs'])
                    case=dict(key=key,span=span,profile=profile,target_slot=ti,seed=seed,target_observed=True,
                        coarse_measurement_cost=sum(64 if x==0 else max(1,math.ceil(20/x)) for x in h));cases.append(case)
                    q=np.r_[aq,qt];Y=a[:,None]*q[None,:]+b[:,None]+e;attacked=Y.copy();attacked[-2:,-1]+=80
                    truth[key+'__target']=np.array([qt]);truth[key+'__anchor_truth']=aq
                    for scenario,reports in [('affine_honest',Y),('affine_two_source_attack',attacked)]:
                        secret[key+'__'+scenario+'__potential_reports']=reports
                        honest=range(9) if scenario=='affine_honest' else range(7)
                        residual=[abs(F(float(reports[i,j]))-F(float(a[i]))*F(float(q[j]))-F(float(b[i]))) for i in honest for j in range(7)]
                        contracts.append(dict(key=key,scenario=scenario,coarse_valid=valid,report_valid=max(residual)<=F(5),target_valid=0<=qt<=500,maximum_residual=float(max(residual))))
    np.savez_compressed(path('_public_inputs.npz'),**public);np.savez_compressed(path('_simulator_secret.npz'),**secret);np.savez_compressed(path('_evaluator_truth.npz'),**truth)
    dump(path('_contracts.json'),contracts);dump(path('_manifest.json'),dict(cases=cases,protocol_sha256=sha(PF),public_hash=sha(path('_public_inputs.npz')),secret_hash=sha(path('_simulator_secret.npz')),
        evaluator_hash=sha(path('_evaluator_truth.npz')),contracts_hash=sha(path('_contracts.json'))))
    print(json.dumps(dict(public_cases=len(cases),invalid_generated=sum(not (r['coarse_valid'] and r['report_valid'] and r['target_valid']) for r in contracts))))

def plans():
    p=verify();M=read(path('_manifest.json'));assert sha(path('_public_inputs.npz'))==M['public_hash'];D=np.load(path('_public_inputs.npz'),allow_pickle=False);cache={};out=[];tic=time.perf_counter()
    for ci,c in enumerate(M['cases']):
        k=c['key'];A=D[k+'__anchors'];centers=D[k+'__centers'];radii=D[k+'__radii'];costs=D[k+'__costs']
        # Target slot is evaluator grouping only: even the hash procurement
        # control receives one fixed anonymous ID for all target locations.
        ctx=dict(day='span_diagnostic',target_station='anonymous_fixed_target',original_ids=list(range(6)))
        for budget in p['budgets']:
            for policy in p['policies']:
                tag=oh(dict(A=A.tolist(),c=centers.tolist(),h=radii.tolist(),costs=costs.tolist(),budget=budget,policy=policy,context=ctx if policy=='FixedIdBudget' else None,protocol_hash=sha(PF)))
                if tag not in cache:
                    old.select(policy,A,centers,radii,costs,budget,ctx);times=[]
                    for rep in range(3):
                        t=time.perf_counter();ids,meta=old.select(policy,A,centers,radii,costs,budget,ctx);times.append(time.perf_counter()-t)
                    w=worst_width(A[ids].tolist(),[0.,500.],[.7,1.3],5.)['width'];cache[tag]=dict(local_anchor_ids=ids,prequery_width=w,selection_metadata=meta,selection_seconds=float(np.median(times)),selection_times=times)
                r=cache[tag];ids=r['local_anchor_ids'];out.append(dict(**c,plan_key=f'{k}__B{budget}__{policy}',policy=policy,budget=budget,cache_key=tag,selected_count=len(ids),
                    report_anchor_cost_per_source=int(sum(costs[ids])),total_cost=c['coarse_measurement_cost']+9*(1+int(sum(costs[ids]))),**r))
    dump(path('_plans.json'),dict(protocol_sha256=sha(PF),public_hash=M['public_hash'],plans=out,unique_selections=len(cache),seconds=time.perf_counter()-tic));print(json.dumps(dict(plans=len(out),unique=len(cache))))

def fulfill():
    p=verify();M=read(path('_manifest.json'));assert sha(path('_simulator_secret.npz'))==M['secret_hash'];D=np.load(path('_public_inputs.npz'),allow_pickle=False);secret=np.load(path('_simulator_secret.npz'),allow_pickle=False);out={};cases=[]
    for plan in read(path('_plans.json'))['plans']:
        k=plan['key'];ids=plan['local_anchor_ids']
        for scenario in p['scenarios']:
            key=plan['plan_key']+'__'+scenario;out[key+'__reports']=secret[k+'__'+scenario+'__potential_reports'][:,ids+[6]]
            out[key+'__anchors']=D[k+'__anchors'][ids];out[key+'__centers']=D[k+'__centers'][ids];cases.append(dict(**plan,scenario=scenario,inference_key=key))
    np.savez_compressed(path('_purchased_inputs.npz'),**out);dump(path('_purchased_manifest.json'),dict(cases=cases,protocol_sha256=sha(PF),plans_hash=sha(path('_plans.json')),purchased_hash=sha(path('_purchased_inputs.npz'))));print(json.dumps(dict(conditions=len(cases))))

def fit(Y,A,q,timing=True):
    X=Y.T;h=(A[:,1]-A[:,0])/2;eps=A[:,2];grids={};out=[]
    for adapter in ['native','weighted_source_affine']:
        if adapter=='native':cfg,grid=old.choose_mlni(X,q,'native')
        else:cfg,grid=weighted.choose(X,q,h,eps)
        grids[adapter]=grid
        for name in old.PAPERS:
            label=name+'__'+adapter
            if name=='MLNI_JSAC2022' and cfg is None:out.append(dict(label=label,method=name,adapter=adapter,status='not_applicable_no_reference',prediction=None,projected_prediction=None));continue
            def evaluate():return old.core(name,X,q,'native',cfg) if adapter=='native' else weighted.core(name,X,q,h,eps,cfg)
            if timing:evaluate()
            times=[]
            for rep in range(3 if timing else 1):
                t=time.perf_counter();z,meta=evaluate();times.append(time.perf_counter()-t);assert np.isfinite(z).all()
            out.append(dict(label=label,method=name,adapter=adapter,prediction=float(z[-1]),projected_prediction=float(np.clip(z[-1],0,500)),whole_prediction=z.tolist(),
                status=meta.get('status','converged' if meta.get('converged') else 'max_iterations'),metadata=meta,selected_config=cfg if name=='MLNI_JSAC2022' else None,
                inference_times=times,inference_seconds=float(np.median(times)),selection_seconds=grid['seconds'] if name=='MLNI_JSAC2022' else 0.))
    def median():
        tx,am=weighted.weighted_adapt(X,q,np.arange(len(q)),h,eps);return np.median(tx,axis=1),am
    if timing:median()
    times=[]
    for rep in range(3 if timing else 1):
        t=time.perf_counter();z,am=median();times.append(time.perf_counter()-t)
    out.append(dict(label='weighted_source_affine_median',method='median_control',adapter='weighted_source_affine',prediction=float(z[-1]),projected_prediction=float(np.clip(z[-1],0,500)),whole_prediction=z.tolist(),
        status='closed_form',metadata=am,inference_times=times,inference_seconds=float(np.median(times)),selection_seconds=0.))
    if timing:packet_fusion(Y,A.tolist(),[0.,500.],[.7,1.3],5.,2)
    times=[]
    for rep in range(3 if timing else 1):
        t=time.perf_counter();r=packet_fusion(Y,A.tolist(),[0.,500.],[.7,1.3],5.,2);times.append(time.perf_counter()-t)
    out.append(dict(label='SourceBridge_packet',method='SourceBridge_packet',adapter='bounded_interval_depth',prediction=r['point'],projected_prediction=r['point'],status='feasible' if r['feasible'] else 'infeasible',
        metadata=r,inference_times=times,inference_seconds=float(np.median(times)),selection_seconds=0.))
    return dict(records=out,grids=grids)

def strip(x):
    if isinstance(x,dict):return {k:strip(v) for k,v in x.items() if 'seconds' not in k and k not in ['inference_times','selection_times']}
    if isinstance(x,list):return [strip(v) for v in x]
    return x

def predict(shadow=False):
    verify();M=read(path('_purchased_manifest.json'));assert sha(path('_purchased_inputs.npz'))==M['purchased_hash'];D=np.load(path('_purchased_inputs.npz'),allow_pickle=False);bundles={};rows=[];tic=time.perf_counter()
    for ci,c in enumerate(M['cases']):
        k=c['inference_key'];Y=D[k+'__reports'];A=D[k+'__anchors'];q=D[k+'__centers'];tag=oh(dict(Y=Y.tolist(),A=A.tolist(),q=q.tolist(),protocol_hash=sha(PF)))
        if tag not in bundles:
            cp=CACHE/('prediction_'+tag+'.json')
            if shadow:item=fit(Y,A,q,False);assert strip(item)==strip(read(cp)),tag
            elif cp.exists():item=read(cp)
            else:item=fit(Y,A,q);dump(cp,item)
            bundles[tag]=item
        small={a:c[a] for a in ['key','span','profile','target_slot','seed','plan_key','policy','budget','selected_count','total_cost','prequery_width','scenario','inference_key']};small['acquisition_seconds']=c['selection_seconds']
        rows.extend([{**small,'fit_key':tag,**r} for r in bundles[tag]['records']])
        if ci%48==0:print(json.dumps(dict(cases=ci+1,total=len(M['cases']),unique=len(bundles),seconds=time.perf_counter()-tic,shadow=shadow)),flush=True)
    if shadow:dump(path('_prediction_counterfactual.json'),dict(compared_unique_inputs=len(bundles),compared_rows=len(rows),predictions_and_grids_equal=True));return
    dump(path('_runs.json'),dict(protocol_sha256=sha(PF),rows=rows,unique_inputs=len(bundles),seconds=time.perf_counter()-tic));dump(path('_grids.json'),{tag:r['grids'] for tag,r in bundles.items()})

def exact():
    verify();pub=np.load(path('_public_inputs.npz'),allow_pickle=False);D=np.load(path('_purchased_inputs.npz'),allow_pickle=False);M=read(path('_purchased_manifest.json'));designs={};pc=[];fits={};fc=[]
    for plan in read(path('_plans.json'))['plans']:
        A=pub[plan['key']+'__anchors'];costs=pub[plan['key']+'__costs'];tag=oh(dict(A=A.tolist(),costs=costs.tolist(),budget=plan['budget'],protocol_hash=sha(PF)))
        if tag not in designs:
            cp=CACHE/('exact_design_'+tag+'.json')
            if cp.exists():r=read(cp)
            else:
                r=exact_design(A.tolist(),[0.,500.],[.7,1.3],5.,costs.tolist(),plan['budget']);values=[]
                for kk in range(7):
                    for ids in itertools.combinations(range(6),kk):
                        if sum(costs[list(ids)])<=plan['budget']:
                            w=worst_width(A[list(ids)].tolist(),[0.,500.],[.7,1.3],5.,exact=True)['width'];values.append((w,int(sum(costs[list(ids)])),len(ids),ids))
                best=min(values);assert r['width']==best[0];r=dict(exact_design=jsonable(r),full_subset_best_width=str(best[0]),full_subset_best_ids=list(best[3]),subsets=len(values));dump(cp,r)
            designs[tag]=r
        w=worst_width(A[plan['local_anchor_ids']].tolist(),[0.,500.],[.7,1.3],5.,exact=True)['width'];pc.append(dict(plan_key=plan['plan_key'],design_key=tag,selected_exact_width=str(w),
            selected_exact_width_float=float(w),exact_optimum=w==F(designs[tag]['full_subset_best_width']),prequery_r20_reachable=w<=40))
    for ci,c in enumerate(M['cases']):
        k=c['inference_key'];Y=D[k+'__reports'];A=D[k+'__anchors'];tag=oh(dict(Y=Y.tolist(),A=A.tolist(),protocol_hash=sha(PF)))
        if tag not in fits:
            cp=CACHE/('exact_fit_'+tag+'.json')
            if cp.exists():r=read(cp)
            else:r=jsonable(exact_packet_fusion(Y.tolist(),A.tolist(),[0.,500.],[.7,1.3],5.,2));dump(cp,r)
            fits[tag]=r
        fc.append(dict(inference_key=k,exact_fit_key=tag))
        if ci%96==0:print(json.dumps(dict(exact_cases=ci+1,total=len(M['cases']),unique=len(fits))),flush=True)
    dump(path('_exact_checks.json'),dict(protocol_sha256=sha(PF),designs=designs,plan_checks=pc,fit_references=fits,fit_cases=fc))

def projection():
    verify();EX=read(path('_exact_checks.json'));RUN=read(path('_runs.json'));pc={r['plan_key']:r for r in EX['plan_checks']};ec={r['inference_key']:EX['fit_references'][r['exact_fit_key']] for r in EX['fit_cases']};out=[]
    for r in RUN['rows']:
        if r['label']!='EPTD_TIFS2022__weighted_source_affine':continue
        e=ec[r['inference_key']];p=project(r['projected_prediction'],*e['interval'],pc[r['plan_key']]['selected_exact_width'],[0.,500.]) if e['feasible'] else dict(status='no_certificate',prediction=None)
        out.append({**{k:r[k] for k in ['inference_key','key','plan_key','span','target_slot','profile','budget','policy','scenario']},**p})
    dump(path('_projection_predictions.json'),dict(protocol_sha256=sha(PF),default='EPTD_WLS_only',rows=out))

def counterfactual():
    original=np.load;attempts=[]
    def guard(file,*args,**kwargs):
        s=str(file)
        if '_evaluator_' in s or '_simulator_secret' in s or '_contracts.json' in s:
            attempts.append(s);raise AssertionError('hidden numeric read '+s)
        return original(file,*args,**kwargs)
    # Each hidden-label access would be intercepted; zero access is required.
    np.load=guard
    try:predict(shadow=True)
    finally:np.load=original
    # Replay public-only purchase into a separate output path, leaving originals intact.
    original_path=path;shadow=CACHE/'purchase_replay.json'
    def redirected(s):return shadow if s=='_plans.json' else original_path(s)
    globals()['path']=redirected;np.load=guard
    try:
        with contextlib.redirect_stdout(io.StringIO()):plans()
    finally:globals()['path']=original_path;np.load=original
    a=read(path('_plans.json'));z=read(shadow);assert len(a['plans'])==len(z['plans']) and strip(a)==strip(z)
    r=read(path('_prediction_counterfactual.json'));r.update(purchase_plans=len(a['plans']),purchase_choices_equal=True,hidden_numeric_load_attempts=len(attempts));dump(path('_information_audit.json'),r)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','prepare','plans','fulfill','predict','exact','projection','counterfactual']);a=ap.parse_args();globals()[a.stage]()
