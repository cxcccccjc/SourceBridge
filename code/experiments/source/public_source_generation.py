"""Frozen station-and-time-held-out semi-synthetic full composition check."""
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,hashlib,argparse,itertools,time,math,csv
from fractions import Fraction as F
from collections import defaultdict,Counter
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'deps'))
import numpy as np
import simulated_inference as engine
import additive_selection as fast
from reference_geometry import worst_width,jsonable
from certified_geometry import exact_packet_fusion
from safe_projection import project
N='public_source_generation';R=B/'results';PF=B/(N+'_protocol.json')
def path(s):return R/(N+s)
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(jsonable(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def seed(key):return int.from_bytes(hashlib.sha256(('SourceBridge-public-confirmation-v1:'+key).encode()).digest()[:8],'big')
def public_only(fn):
    def wrapped():
        original=np.load
        def guard(file,*a,**kw):
            assert Path(file).resolve()==path('_public_inputs.npz').resolve(),'Only public numeric arrays allowed in procurement'
            return original(file,*a,**kw)
        np.load=guard
        try:return fn()
        finally:np.load=original
    return wrapped
def freeze():
    assert not PF.exists(),'Preserve protocol'
    engine.verify()
    files=[Path(__file__).name,N+'_protocol.md','additive_selection.py','simulated_inference_protocol.json',
           'data/msra_urban_air/subset_protocol.json']
    files+=list(read(engine.PF)['source_hashes'])
    p=dict(status='FROZEN_BEFORE_HOLDOUT_VALUE_READ_AND_EXECUTION',scope='Strict public station/time holdout; sources and trusted intervals semi-synthetic',
        dataset='data/msra_urban_air/interface/beijing_first28days_all_pollutants.npz',dataset_sha='110e37b15b171cbbe596316f9340950d549248ea8da4bf73cf8ca8c9a8d51b40',
        days=[f'2014-05-{d}' for d in range(22,29)],target_stations=['001006','001007','001008','001015','001017','001018','001022','001029'],target_hour=12,
        anchor_stations=['001003','001004','001005'],anchor_hours=[11,13],pollutant='PM25_Concentration',
        profiles=dict(exact=[0]*6,heterogeneous=[2,5,10,20,40,80]),costs=[1,2,1,2,1,2],budget=4,radii=[20,50,100,150],
        target=[0.,500.],gain_bounds=[.7,1.3],epsilon=5.,m=9,f=2,nominal='MLNI_JSAC2022__weighted_source_affine',
        target_shifts=[-80,-20,-5,5,20,80],spoof_gains=[.7,1.3],spoof_targets=[0,250,500],translation_controls=[-200,200],
        source_seed='first8bytes sha256(SourceBridge-public-confirmation-v1:day|station), big endian',interval_seed='same hash of interval|day',
        source_hashes={f:sha(B/f) for f in sorted(set(files))})
    dump(PF,p);print(json.dumps(dict(frozen=True,protocol_sha=sha(PF))),flush=True)
def verify():
    p=read(PF)
    for f,h in p['source_hashes'].items():assert verify_identity(B/f,h),f
    return p
def prepare():
    p=verify();assert not path('_manifest.json').exists();assert sha(resolve_resource(B/p['dataset']))==p['dataset_sha']
    d=np.load(resolve_resource(B/p['dataset']),allow_pickle=False);pi=int(np.where(d['pollutants']==p['pollutant'])[0][0]);public={};secret={};truth={};cases=[];pools=[];contracts=[];observations=[]
    def value(station,day,hour,role):
        si=int(np.where(d['station_ids']==station)[0][0]);ti=int(np.where(d['timestamps']==np.datetime64(f'{day}T{hour:02}:00:00','s'))[0][0])
        assert d['temporal_split'][ti]=='test'
        assert d['station_split'][si]==('station_test' if role=='target' else 'station_train')
        observed=bool(d['reference_observed'][si,ti,pi]);v=float(d['reference_values'][si,ti,pi]) if observed else None
        observations.append(dict(role=role,station=station,day=day,hour=hour,observed=observed,value=v,within_target_domain=None if v is None else 0<=v<=500))
        return v
    for day in p['days']:
        aq=[value(s,day,h,'anchor') for s in p['anchor_stations'] for h in p['anchor_hours']];available=[j for j,v in enumerate(aq) if v is not None]
        u=np.random.default_rng(seed('interval|'+day)).uniform(-.9,.9,6)
        for profile,hh in p['profiles'].items():
            h=np.array(hh,dtype=float)[available];q=np.array([aq[j] for j in available]);c=q+u[available]*h;A=np.column_stack([c-h,c+h,np.ones(len(q))*5]);pool=day+'_'+profile
            public[pool+'__anchors']=A;public[pool+'__centers']=c;public[pool+'__radii']=h;public[pool+'__costs']=np.array(p['costs'],dtype=int)[available]
            pools.append(dict(pool=pool,day=day,profile=profile,available_ids=available,missing_ids=[j for j in range(6) if j not in available],
                              coarse_cost=sum(64 if x==0 else max(1,math.ceil(20/x)) for x in h)))
        for station in p['target_stations']:
            qt=value(station,day,12,'target');event=day+'_'+station;rng=np.random.default_rng(seed(day+'|'+station));a=rng.uniform(.7,1.3,9);b=rng.uniform(-20,20,9);e=rng.uniform(-4.5,4.5,(9,7))
            Q=np.array([np.nan if v is None else v for v in aq]+[np.nan if qt is None else qt]);Y=a[:,None]*Q[None,:]+b[:,None]+e
            truth[event]=qt;secret[event+'__reports']=Y[:,available+[6]]
            for profile in p['profiles']:
                pool=day+'_'+profile;A=public[pool+'__anchors'];ok=all(F(float(lo))<=F(float(aq[j]))<=F(float(hi)) for j,(lo,hi,_) in zip(available,A))
                residual=[abs(F(float(Y[i,j]))-F(float(a[i]))*F(float(Q[j]))-F(float(b[i]))) for i in range(9) for j in available+([6] if qt is not None else [])]
                contracts.append(dict(event=event,profile=profile,coarse_valid=ok,honest_report_valid=max(residual,default=F(0))<=5,target_observed=qt is not None,
                    target_valid=qt is not None and 0<=qt<=500,maximum_honest_residual=float(max(residual,default=F(0))),source_gains=a.tolist(),source_biases=b.tolist()))
                cases.append(dict(event=event,base_key=event+'_'+profile,pool=pool,day=day,station=station,profile=profile,target_observed=qt is not None))
    np.savez_compressed(path('_public_inputs.npz'),**public);np.savez_compressed(path('_simulator_secret.npz'),**secret);dump(path('_evaluator_truth.json'),truth);dump(path('_contracts.json'),contracts);dump(path('_observations.json'),observations)
    dump(path('_manifest.json'),dict(protocol_sha=sha(PF),pools=pools,cases=cases,public_sha=sha(path('_public_inputs.npz')),secret_sha=sha(path('_simulator_secret.npz')),
        truth_sha=sha(path('_evaluator_truth.json')),contracts_sha=sha(path('_contracts.json')),observations_sha=sha(path('_observations.json'))))
    print(json.dumps(dict(events=len(truth),pools=len(pools),cases=len(cases),missing_targets=sum(v is None for v in truth.values()),
        out_of_domain_targets=sum(v is not None and not 0<=v<=500 for v in truth.values()),missing_anchor_observations=sum(not v['observed'] for v in observations if v['role']=='anchor'))),flush=True)
@public_only
def plans():
    p=verify();assert not path('_plans.json').exists();m=read(path('_manifest.json'));assert sha(path('_public_inputs.npz'))==m['public_sha'];d=np.load(path('_public_inputs.npz'),allow_pickle=False);plans=[];quotes=[];failures=[];catalogs=[]
    for pi,pool in enumerate(m['pools']):
        k=pool['pool'];A=d[k+'__anchors'];costs=d[k+'__costs'].tolist();n=len(A);tic=time.perf_counter()
        fast_result=fast.design_anchors_additive(A.tolist(),p['target'],p['gain_bounds'],5.,costs,p['budget']);fast_seconds=time.perf_counter()-tic
        rows=[];tic=time.perf_counter()
        for kk in range(n+1):
            for ids in itertools.combinations(range(n),kk):
                r=worst_width(A[list(ids)].tolist(),p['target'],p['gain_bounds'],5.,exact=True)
                rows.append(dict(ids=list(ids),cost=sum(costs[j] for j in ids),width=r['width'],positive=r['branches'][0]['width'],negative=r['branches'][1]['width'],exact_result=r))
        ids=list(fast_result['anchor_ids']);selected=next(r for r in rows if r['ids']==ids);best=min((r for r in rows if r['cost']<=p['budget']),key=lambda r:(r['width'],r['cost'],len(r['ids']),tuple(r['ids'])))
        rank_equal=(selected['width'],selected['cost'],len(ids),ids)==(best['width'],best['cost'],len(best['ids']),best['ids'])
        if not rank_equal:failures.append(dict(pool=k,type='budget_rank',selected=ids,oracle=best['ids'],selected_width=selected['width'],oracle_width=best['width']))
        cp=path('_catalog_'+k+'.json');dump(cp,dict(pool=k,rows=rows));catalogs.append(dict(pool=k,path=cp.name,sha=sha(cp),subsets=len(rows)))
        plans.append(dict(**pool,selected_ids=ids,original_selected_ids=[pool['available_ids'][j] for j in ids],selected_count=len(ids),width=selected['width'],rank_equal=rank_equal,
            anchor_cost_per_source=selected['cost'],total_cost=pool['coarse_cost']+9*(1+selected['cost']),fast_seconds=fast_seconds,exact_catalog_seconds=time.perf_counter()-tic,fast_result=fast_result))
        entries=[dict(ids=tuple(r['ids']),cost=r['cost'],positive=r['positive'],negative=r['negative']) for r in rows if len(r['ids'])<=2]
        for radius in p['radii']:
            query=fast.threshold_minimum(entries,costs,F(2*radius));feas=[r for r in rows if r['width']<=2*radius];oracle=min(feas,key=lambda r:(r['cost'],len(r['ids']),tuple(r['ids']))) if feas else None
            qr=None if query is None else (query['cost'],len(query['anchor_ids']),tuple(query['anchor_ids']));gr=None if oracle is None else (oracle['cost'],len(oracle['ids']),tuple(oracle['ids']))
            if qr!=gr:failures.append(dict(pool=k,r=radius,type='quote_rank'))
            quotes.append(dict(pool=k,day=pool['day'],profile=pool['profile'],r=radius,feasible=query is not None,rank_equal=qr==gr,
                selected_ids=None if query is None else list(query['anchor_ids']),selected_width=None if oracle is None else oracle['width'],
                anchor_cost=None if query is None else query['cost'],total_cost=None if query is None else pool['coarse_cost']+9*(1+query['cost'])))
        print(json.dumps(dict(planned=pi+1,total=len(m['pools']),n=n,selected_ids=ids,rank_equal=rank_equal)),flush=True)
    dump(path('_plans.json'),dict(protocol_sha=sha(PF),public_sha=m['public_sha'],numeric_loads=[str(path('_public_inputs.npz'))],plans=plans,quotes=quotes,catalogs=catalogs,failures=failures))
def library(Y,q,p):
    out=[('clean','clean',Y.copy())]
    for shift in p['target_shifts']:
        y=Y.copy();y[-2:,-1]+=shift;out.append((f'target_{shift:+}','target_shift',y))
    for gain in p['spoof_gains']:
        for t in p['spoof_targets']:
            y=Y.copy();bias=y[-2:,-1]-gain*t;y[-2:,:-1]=gain*q[None,:]+bias[:,None];out.append((f'anchor_g{gain}_q{t}','anchor_spoof',y))
    y=Y.copy();y[-2:,:-1]=y[-2:,:-1][:,::-1];out.append(('reverse_anchors','inconsistency_control',y))
    for shift in p['translation_controls']:
        y=Y.copy();y[-2:,:]+=shift;out.append((f'translate_{shift:+}','translation_control',y))
    assert len(out)==16 and all(np.array_equal(v[2][:7],Y[:7]) for v in out)
    return out
def fulfill():
    p=verify();assert not path('_purchased_manifest.json').exists();m=read(path('_manifest.json'));plans=read(path('_plans.json'));look={v['pool']:v for v in plans['plans']}
    assert sha(path('_simulator_secret.npz'))==m['secret_sha'];assert sha(path('_public_inputs.npz'))==m['public_sha']
    d=np.load(path('_public_inputs.npz'),allow_pickle=False);s=np.load(path('_simulator_secret.npz'),allow_pickle=False);out={};cases=[];skips=[]
    for case in m['cases']:
        plan=look[case['pool']];pool=case['pool'];ids=plan['selected_ids'];n=len(d[pool+'__anchors'])
        if not case['target_observed']:skips.append(dict(**case,status='missing_target_report'));continue
        if not ids:skips.append(dict(**case,status='no_reference_for_fixed_nominal'));continue
        Y=np.ascontiguousarray(s[case['event']+'__reports'][:,ids+[n]]);A=d[pool+'__anchors'][ids];q=d[pool+'__centers'][ids]
        for attack,family,y in library(Y,q,p):
            key=case['base_key']+'__'+attack;out[key+'__reports']=np.ascontiguousarray(y);out[key+'__anchors']=A;out[key+'__centers']=q
            cases.append(dict(**case,key=key,attack=attack,family=family,main_library=family!='translation_control',total_cost=plan['total_cost'],selected_count=len(ids),width=plan['width']))
    np.savez_compressed(path('_purchased_inputs.npz'),**out);dump(path('_purchased_manifest.json'),dict(protocol_sha=sha(PF),plans_sha=sha(path('_plans.json')),input_sha=sha(path('_purchased_inputs.npz')),cases=cases,skips=skips))
    print(json.dumps(dict(conditions=len(cases),skipped_base_conditions=len(skips))),flush=True)
def predict():
    p=verify();assert not path('_runs.json').exists();m=read(path('_purchased_manifest.json'));assert sha(path('_purchased_inputs.npz'))==m['input_sha'];loaded=[];original=np.load
    def guard(file,*a,**kw):
        assert Path(file).resolve()==path('_purchased_inputs.npz').resolve(),'Only purchased numeric arrays allowed in inference';loaded.append(str(Path(file).resolve()));return original(file,*a,**kw)
    np.load=guard
    try:
        d=np.load(path('_purchased_inputs.npz'),allow_pickle=False);bundles={};rows=[];cache=B/'reading'/N;cache.mkdir(exist_ok=True);tic=time.perf_counter()
        for ci,c in enumerate(m['cases']):
            key=c['key'];Y=d[key+'__reports'];A=d[key+'__anchors'];q=d[key+'__centers'];tag=hashlib.sha256(Y.tobytes()+A.tobytes()+q.tobytes()+sha(PF).encode()).hexdigest();cp=cache/(tag+'.json')
            if tag not in bundles:
                if cp.exists():item=read(cp)
                else:
                    fit=engine.fit(Y,A,q,timing=False);t=time.perf_counter();exact=exact_packet_fusion(Y,A.tolist(),p['target'],p['gain_bounds'],5.,2);certificate_seconds=time.perf_counter()-t
                    nominal=next(r for r in fit['records'] if r['label']==p['nominal']);z=nominal.get('projected_prediction');t=time.perf_counter()
                    pr=project(z,*exact['interval'],F(c['width']),p['target']) if exact['feasible'] and z is not None else dict(status='no_certificate_or_nominal',prediction=None)
                    item=dict(fit=fit,exact=exact,projection=pr,extra_certificate_seconds=certificate_seconds,projection_seconds=time.perf_counter()-t);dump(cp,item);item=read(cp)
                bundles[tag]=item
            item=bundles[tag]
            rows.extend([dict(**c,fit_key=tag,**r) for r in item['fit']['records']])
            pr=item['projection'];rows.append(dict(**c,fit_key=tag,label='SourceBridge_MLNI_WLS',method='complete_composition',adapter='fixed_published_nominal_plus_certificate',status=pr['status'],prediction=pr.get('prediction'),projected_prediction=pr.get('prediction'),metadata=pr))
            if (ci+1)%64==0:print(json.dumps(dict(predicted=ci+1,total=len(m['cases']),unique=len(bundles),seconds=time.perf_counter()-tic)),flush=True)
        dump(path('_runs.json'),dict(protocol_sha=sha(PF),input_sha=m['input_sha'],numeric_loads=loaded,rows=rows,bundles=bundles,total_seconds=time.perf_counter()-tic))
    finally:np.load=original
def aggregate(rows,group_fields):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in group_fields)].append(r)
    out=[]
    for keys,rr in sorted(groups.items()):
        for stratum in ['all','model_valid']:
            ss=rr if stratum=='all' else [r for r in rr if r['condition_valid']];clean=[r for r in ss if r['family']=='clean'];main=defaultdict(list)
            for r in ss:
                if r['main_library']:main[r['event']].append(r)
            finite=[r for r in ss if r['squared_error'] is not None];cc=[r for r in clean if r['squared_error'] is not None];complete=[v for v in main.values() if len(v)==14 and all(r['squared_error'] is not None for r in v)]
            out.append(dict(zip(group_fields,keys),stratum=stratum,conditions=len(ss),finite=len(finite),clean_events=len(clean),finite_clean=len(cc),worst_complete_events=len(complete),
                clean_rmse=None if not cc else math.sqrt(sum(r['squared_error'] for r in cc)/len(cc)),clean_mae=None if not cc else sum(r['absolute_error'] for r in cc)/len(cc),
                pooled_rmse=None if not finite else math.sqrt(sum(r['squared_error'] for r in finite)/len(finite)),maximum_error=None if not finite else max(r['absolute_error'] for r in finite),
                finite_library_worst_rmse=None if not complete else math.sqrt(sum(max(r['squared_error'] for r in v) for v in complete)/len(complete)),
                raw_clean_rmse=None if not cc else math.sqrt(sum(r['raw_squared_error'] for r in cc)/len(cc)),statuses=dict(Counter(r['status'] for r in ss))))
    return out
def evaluate():
    p=verify();m=read(path('_manifest.json'));pm=read(path('_purchased_manifest.json'));run=read(path('_runs.json'));plans=read(path('_plans.json'))
    assert run['input_sha']==sha(path('_purchased_inputs.npz'));assert sha(path('_evaluator_truth.json'))==m['truth_sha'];assert sha(path('_contracts.json'))==m['contracts_sha']
    truth=read(path('_evaluator_truth.json'));contracts={(r['event'],r['profile']):r for r in read(path('_contracts.json'))};D=np.load(path('_purchased_inputs.npz'),allow_pickle=False);cells=[];checks=[];worlds=0
    for c in pm['cases']:
        rows=[r for r in run['rows'] if r['key']==c['key']];tag=rows[0]['fit_key'];item=run['bundles'][tag];ex=item['exact'];pr=item['projection'];q=F(truth[c['event']]);w=F(c['width']);contract=contracts[(c['event'],c['profile'])]
        valid=all(contract[k] for k in ['coarse_valid','honest_report_valid','target_valid']);Y=D[c['key']+'__reports'];A=D[c['key']+'__anchors'];intervals=[]
        for i,s in enumerate(ex['source_results']):
            if s is None:continue
            iv=list(map(F,s['interval']));intervals.append(iv)
            for endpoint,side in zip(iv,['lower_world','upper_world']):
                u,v,t=map(F,s[side]);assert t==endpoint and 1/F(1.3)<=u<=1/F(.7) and 0<=t<=500
                for j,(lo,hi,eps) in enumerate(A):
                    y=F(float(Y[i,j]));ee=F(float(eps));assert (y-ee)*u+v<=F(float(hi)) and (y+ee)*u+v>=F(float(lo))
                assert abs(F(float(Y[i,-1]))*u+v-t)<=5*u;worlds+=1
        endpoints=sorted({t for iv in intervals for t in iv});accepted=[t for t in endpoints if sum(lo<=t<=hi for lo,hi in intervals)>=7];assert bool(accepted)==ex['feasible']
        if accepted:assert [accepted[0],accepted[-1]]==list(map(F,ex['interval']))
        safety=dict(**c,condition_valid=valid,feasible=ex['feasible'],projection_defined=pr['status']=='defined',changed=pr.get('changed_exact',False),movement=pr.get('movement'),extra_certificate_seconds=item['extra_certificate_seconds'],projection_seconds=item['projection_seconds'])
        if ex['feasible']:
            lo,hi=map(F,ex['interval']);safety.update(contains_truth=lo<=q<=hi,width_bound=hi-lo<=w,posterior_width=float(hi-lo))
        if pr['status']=='defined':safety.update(exact_point_bound=abs(F(pr['exact_point'])-q)<=w/2,float_radius_covers_truth=abs(F(pr['prediction'])-q)<=F(pr['float_radius_upper']),exported_radius=pr['float_radius_upper'])
        safety['conditional_failure']=valid and not all(safety.get(k,False) for k in ['feasible','projection_defined','contains_truth','width_bound','exact_point_bound','float_radius_covers_truth']);checks.append(safety)
        for r in rows:
            z=r.get('projected_prediction');raw=r.get('prediction');cells.append(dict(**{k:r[k] for k in ['event','base_key','pool','day','station','profile','key','attack','family','main_library','label','status','total_cost','selected_count']},truth=float(q),prediction=z,raw_prediction=raw,condition_valid=valid,
                squared_error=None if z is None else (z-float(q))**2,absolute_error=None if z is None else abs(z-float(q)),raw_squared_error=None if raw is None else (raw-float(q))**2))
    summary=aggregate(cells,['profile','label']);byday=aggregate(cells,['profile','day','label']);bystation=aggregate(cells,['profile','station','label']);lookup={(r['profile'],r['label'],r['stratum']):r for r in summary};gates=[]
    for profile in p['profiles']:
        for stratum in ['all','model_valid']:
            o=lookup.get((profile,'SourceBridge_MLNI_WLS',stratum));z=lookup.get((profile,p['nominal'],stratum));ratios={metric:None if not o or o[metric] is None or not z[metric] else o[metric]/z[metric] for metric in ['clean_rmse','finite_library_worst_rmse']}
            gates.append(dict(profile=profile,stratum=stratum,ratios=ratios,descriptive_performance_gate=all(v is not None and v<=1.1 for v in ratios.values())))
    folds=[f for item in run['bundles'].values() for grid in item['fit']['grids'].values() for cand in grid['candidates'] for f in cand['folds']]
    res=dict(protocol_sha=sha(PF),scope=p['scope'],public_events=len(truth),pools=len(m['pools']),base_conditions=len(m['cases']),inference_conditions=len(pm['cases']),skips=pm['skips'],
        rows=len(cells),unique_numeric_fits=len(run['bundles']),conditional_safety_failures=[c for c in checks if c['conditional_failure']],model_valid_conditions=sum(c['condition_valid'] for c in checks),projection_changed=sum(c['changed'] for c in checks),
        exact_endpoint_worlds_checked=worlds,depth_checks=len(checks),procurement_failures=plans['failures'],mlni_fold_count=len(folds),mlni_fold_nonconverged=sum(not f['converged'] for f in folds),
        method_statuses=dict(Counter(r['status'] for r in cells)),summary=summary,by_day=byday,by_station=bystation,gates=gates,safety=checks,
        hashes={str(v.relative_to(B)):sha(v) for v in [PF,path('_runs.json'),path('_plans.json'),path('_purchased_inputs.npz'),path('_evaluator_truth.json'),path('_contracts.json'),path('_observations.json')]})
    dump(path('_summary.json'),res)
    for suffix,data in [('_cells.csv',cells),('_overall.csv',summary),('_by_day.csv',byday),('_by_station.csv',bystation)]:
        with path(suffix).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print(json.dumps({k:res[k] for k in ['public_events','inference_conditions','rows','projection_changed','procurement_failures','mlni_fold_count','mlni_fold_nonconverged','gates']},ensure_ascii=False),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','prepare','plans','fulfill','predict','evaluate']);args=parser.parse_args();globals()[args.action]()
