"""Frozen finite complete-packet attack library; no new scientific method."""
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,hashlib,argparse,math,csv
from fractions import Fraction as F
from collections import defaultdict,Counter
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'deps'))
import numpy as np
import simulated_inference as span
from reference_geometry import worst_width,jsonable
from certified_geometry import exact_packet_fusion
from safe_projection import project
N='packet_attacks';R=B/'results';PF=B/(N+'_protocol.json')
def path(s):return R/(N+s)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def dump(p,x):Path(p).write_text(json.dumps(jsonable(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def freeze():
    span.verify()
    names=[Path(__file__).name,N+'_protocol.md','simulated_inference.py','safe_projection.py','certified_geometry.py','packet_geometry.py']
    sources={n:sha(B/n) for n in names}
    for suffix in ['_purchased_manifest.json','_purchased_inputs.npz','_evaluator_truth.npz']:
        sources['results/'+span.NAME+suffix]=sha(span.path(suffix))
    p=dict(status='FROZEN_BEFORE_ATTACK_EXECUTION',selection=dict(span=450,budget=4,policy='MinimaxPairTable',scenario='affine_honest'),
        target_shifts=[-80,-20,-5,5,20,80],spoof_gains=[.7,1.3],spoof_targets=[0,250,500],translation_controls=[-200,200],
        actual_attackers=[7,8],spec=dict(target=[0.,500.],gain_bounds=[.7,1.3],epsilon=5.,f=2),
        base_protocol_hash=sha(span.PF),source_hashes=sources)
    if PF.exists():assert read(PF)==p
    else:dump(PF,p)
    print(json.dumps(dict(frozen=True,sha256=sha(PF))))

def verify():
    p=read(PF);span.verify()
    for n,h in p['source_hashes'].items():assert verify_identity(B/n,h),n
    return p

def prepare():
    p=verify();M=read(span.path('_purchased_manifest.json'));D=np.load(span.path('_purchased_inputs.npz'),allow_pickle=False);T=np.load(span.path('_evaluator_truth.npz'),allow_pickle=False)
    cases=[];inputs={};truth={}
    for c in M['cases']:
        if any(c[k]!=v for k,v in p['selection'].items()):continue
        k=c['inference_key'];Y=D[k+'__reports'];A=D[k+'__anchors'];q=D[k+'__centers'];base=c['key'];truth[base]=float(T[base+'__target'][0])
        library=[('clean','clean',Y.copy())]
        for shift in p['target_shifts']:
            y=Y.copy();y[-2:,-1]+=shift;library.append((f'target_{shift:+}','target_shift',y))
        for gain in p['spoof_gains']:
            for qt in p['spoof_targets']:
                y=Y.copy();bias=y[-2:,-1]-gain*qt;y[-2:,:-1]=gain*q[None,:]+bias[:,None]
                library.append((f'anchor_g{gain}_q{qt}','anchor_spoof',y))
        y=Y.copy();y[-2:,:-1]=y[-2:,:-1][:,::-1];library.append(('reverse_anchors','inconsistency_control',y))
        for shift in p['translation_controls']:
            y=Y.copy();y[-2:,:]+=shift;library.append((f'translate_{shift:+}','translation_control',y))
        assert len(library)==16
        for attack,family,y in library:
            assert np.array_equal(y[:7],Y[:7]);ik=base+'__'+attack
            inputs[ik+'__reports']=y;inputs[ik+'__anchors']=A;inputs[ik+'__centers']=q
            cases.append(dict(base_key=base,key=ik,profile=c['profile'],target_slot=c['target_slot'],seed=c['seed'],attack=attack,family=family,
                selected_count=len(q),total_cost=c['total_cost'],main_library=family!='translation_control'))
    assert len(cases)==192 and len(truth)==12
    np.savez_compressed(path('_inputs.npz'),**inputs);dump(path('_evaluator_truth.json'),truth)
    dump(path('_manifest.json'),dict(cases=cases,input_sha=sha(path('_inputs.npz')),truth_sha=sha(path('_evaluator_truth.json')),protocol_sha=sha(PF)))
    print(json.dumps(dict(base_inputs=len(truth),attack_inputs=len(cases))))

def predict():
    verify();M=read(path('_manifest.json'));assert sha(path('_inputs.npz'))==M['input_sha'];loaded=[];base_load=np.load
    def guard(file,*args,**kwargs):
        resolved=Path(file).resolve();loaded.append(str(resolved))
        assert resolved==path('_inputs.npz').resolve(),'predictor numeric input outside purchased attack file'
        return base_load(file,*args,**kwargs)
    np.load=guard
    try:
        D=np.load(path('_inputs.npz'),allow_pickle=False);bundles={};rows=[];cert={};cache=B/'reading'/N;cache.mkdir(exist_ok=True)
        for ci,c in enumerate(M['cases']):
            k=c['key'];Y=D[k+'__reports'];A=D[k+'__anchors'];q=D[k+'__centers']
            tag=hashlib.sha256(Y.tobytes()+A.tobytes()+q.tobytes()+sha(PF).encode()).hexdigest();cp=cache/(tag+'.json')
            if tag not in bundles:
                if cp.exists():item=read(cp)
                else:
                    fit=span.fit(Y,A,q,timing=False);exact=exact_packet_fusion(Y,A,[0.,500.],[.7,1.3],5.,2)
                    w=worst_width(A.tolist(),[0.,500.],[.7,1.3],5.,exact=True)['width']
                    primary=next(r for r in fit['records'] if r['label']=='EPTD_TIFS2022__weighted_source_affine')
                    projection=project(primary['projected_prediction'],*exact['interval'],w,[0.,500.]) if exact['feasible'] else dict(status='no_feasible_certificate')
                    item=dict(fit=fit,exact=exact,prequery_width=w,projection=projection);dump(cp,item);item=read(cp)
                bundles[tag]=item
            item=bundles[tag];rows.extend([dict(**c,fit_key=tag,**r) for r in item['fit']['records']]);cert[k]=dict(fit_key=tag)
            if (ci+1)%16==0:print(json.dumps(dict(predicted=ci+1,total=len(M['cases']))),flush=True)
        dump(path('_runs.json'),dict(protocol_sha=sha(PF),input_sha=M['input_sha'],numeric_loads=loaded,rows=rows,bundles=bundles,cert=cert))
        print(json.dumps(dict(rows=len(rows),unique=len(bundles))))
    finally:np.load=base_load

def boundary_cases():
    out=[]
    configs=[
      ('excess_f_wrong',[[0,.5]]*6+[[0,1]]*3,[[0,0,.25]],.25,2,0,[.5,1]),
      ('excess_f_empty',[[0,.5]]*6+[[0,2]]*3,[[0,0,.25]],.25,2,0,None),
      ('honest_target_error_mismatch',[[0,1]]*9,[[0,0,.25]],.25,2,0,[.5,1.5]),
      ('trusted_anchor_invalid',[[0,0]]*9,[[1,1,0]],0,2,0,[1,1]),
      ('m_equals_2f',[[0,0]]*2+[[0,2]]*2,[[0,0,0]],0,2,0,[0,2]),
      ('m_equals_2f_plus1',[[0,0]]*3+[[0,2]]*2,[[0,0,0]],0,2,0,[0,0])]
    for name,Y,A,eps,f,truth,expected in configs:
        z=exact_packet_fusion(Y,A,[0,2],[1,1],eps,f);w=worst_width(A,[0,2],[1,1],eps,exact=True)['width']
        assert (list(z['interval']) if z['feasible'] else None)==expected
        point=z.get('exact_midpoint');out.append(dict(name=name,raw_reports=Y,anchors=A,epsilon=eps,declared_f=f,true_target=truth,expected_interval=expected,
            result=z,single_source_width=w,effective_prequery_width=w if len(Y)>2*f else F(2),
            absolute_error=None if point is None else abs(point-truth),contains_truth=z['feasible'] and z['interval'][0]<=truth<=z['interval'][1]))
    return out

def evaluate():
    verify();M=read(path('_manifest.json'));RUN=read(path('_runs.json'));assert RUN['input_sha']==sha(path('_inputs.npz')) and M['truth_sha']==sha(path('_evaluator_truth.json'))
    T=read(path('_evaluator_truth.json'));D=np.load(path('_inputs.npz'),allow_pickle=False);cells=[];safety=[];projection=[];worlds=0;depths=0
    for c in M['cases']:
        k=c['key'];item=RUN['bundles'][RUN['cert'][k]['fit_key']];er=item['exact'];w=F(item['prequery_width']);qq=F(T[c['base_key']]);Y=D[k+'__reports'];A=D[k+'__anchors']
        fl=next(r for r in item['fit']['records'] if r['label']=='SourceBridge_packet')['metadata'];intervals=[]
        for i,source in enumerate(er['source_results']):
            if source is None:continue
            iv=list(map(F,source['interval']));intervals.append(iv)
            for endpoint,side in zip(iv,['lower_world','upper_world']):
                u,v,qt=map(F,source[side]);assert qt==endpoint and F(1)/F(1.3)<=u<=F(1)/F(.7) and 0<=qt<=500
                for j,(lo,hi,eps) in enumerate(A):
                    yy=F(float(Y[i,j]));ee=F(float(eps));assert (yy-ee)*u+v<=F(float(hi)) and (yy+ee)*u+v>=F(float(lo))
                assert abs(F(float(Y[i,-1]))*u+v-qt)<=5*u;worlds+=1
        ends=sorted({x for iv in intervals for x in iv});accepted=[x for x in ends if sum(lo<=x<=hi for lo,hi in intervals)>=7]
        assert bool(accepted)==er['feasible'];depths+=1
        if accepted:assert [accepted[0],accepted[-1]]==list(map(F,er['interval']))
        check=dict(**c,exact_feasible=er['feasible'],float_state_match=fl['feasible']==er['feasible'],infeasible_sources=sum(s is None for s in er['source_results']))
        if er['feasible']:
            lo,hi=map(F,er['interval']);check.update(contains_truth=lo<=qq<=hi,width_bound=hi-lo<=w,midpoint_bound=abs(F(er['exact_midpoint'])-qq)<=w/2,
                posterior_width=float(hi-lo),prequery_width=float(w),max_endpoint_difference=max(abs(fl['interval'][i]-float(x)) for i,x in enumerate([lo,hi])) if fl['feasible'] else None)
            actual=F(er['point_float']);exported=F(er['float_point_error_radius_upper'])
            check.update(certified_float_radius_covers_truth=abs(actual-qq)<=exported,
                certified_float_radius_covers_hull=max(abs(actual-lo),abs(actual-hi))<=exported,
                certified_float_export_bound_excess=float(max(F(0),exported-w/2)),
                float_LP_point_actual_radius=None if not fl['feasible'] else float(max(abs(F(fl['point'])-lo),abs(F(fl['point'])-hi))))
        safety.append(check)
        pr=dict(**c,**item['projection']);pr['truth']=float(qq)
        if pr['status']=='defined':pr.update(squared_error=(pr['prediction']-float(qq))**2,absolute_error=abs(pr['prediction']-float(qq)),exact_bound=abs(F(pr['exact_point'])-qq)<=w/2,
            exported_radius_covers_truth=abs(F(pr['prediction'])-qq)<=F(pr['float_radius_upper']),exported_radius_excess=float(max(F(0),F(pr['float_radius_upper'])-w/2)))
        projection.append(pr)
    for row in RUN['rows']:
        truth=T[row['base_key']];pred=row['projected_prediction'];r={k:row[k] for k in ['base_key','key','profile','target_slot','seed','attack','family','main_library','label','status']}
        r.update(truth=truth,prediction=pred,raw_prediction=row['prediction'],absolute_error=None if pred is None else abs(pred-truth),squared_error=None if pred is None else (pred-truth)**2);cells.append(r)
    groups=defaultdict(list);clean=defaultdict(list);perattack=defaultdict(list)
    for r in cells:
        if r['main_library']:groups[(r['profile'],r['label'],r['base_key'])].append(r)
        if r['family']=='clean':clean[(r['profile'],r['label'])].append(r)
        perattack[(r['profile'],r['label'],r['attack'])].append(r)
    maximums=[]
    for (profile,label,base),rr in groups.items():
        finite=[r for r in rr if r['absolute_error'] is not None];best=max(finite,key=lambda r:(r['absolute_error'],r['attack'])) if finite else None
        maximums.append(dict(profile=profile,label=label,base_key=base,library_size=len(rr),nonfinite=len(rr)-len(finite),maximum_error=None if best is None else best['absolute_error'],maximizer=None if best is None else best['attack']))
    summaries=[]
    for profile,label in sorted(clean):
        rr=[r for r in maximums if (r['profile'],r['label'])==(profile,label)];vv=[r['maximum_error'] for r in rr if r['maximum_error'] is not None];cc=clean[(profile,label)]
        summaries.append(dict(profile=profile,label=label,base_inputs=len(rr),library_nonfinite=sum(r['nonfinite'] for r in rr),clean_rmse=math.sqrt(sum(r['squared_error'] for r in cc)/len(cc)),
            finite_library_worst_rmse=math.sqrt(sum(v*v for v in vv)/len(vv)),finite_library_maximum=max(vv),finite_library_mean_max_error=sum(vv)/len(vv)))
    folds=[]
    for item in RUN['bundles'].values():
        for grid in item['fit']['grids'].values():
            for cand in grid['candidates']:folds.extend(cand['folds'])
    failures=[r for r in safety if not all(r.get(k,False) for k in ['exact_feasible','float_state_match','contains_truth','width_bound','midpoint_bound','certified_float_radius_covers_truth','certified_float_radius_covers_hull'])]
    result=dict(scope='finite attack library on reused development inputs; not independent confirmation or global optimum attack',protocol_sha=sha(PF),rows=len(cells),conditions=len(safety),unique_inputs=len(RUN['bundles']),
        source_endpoint_worlds=worlds,depth_checks=depths,safety_failures=failures,projection_bound_failures=sum(not (r.get('exact_bound',False) and r.get('exported_radius_covers_truth',False)) for r in projection),projection_moves=sum(r.get('changed_exact',False) for r in projection),
        mlni_folds=len(folds),mlni_nonconverged=sum(not f['converged'] for f in folds),mlni_max_iterations=max(f['iterations'] for f in folds),
        final_statuses=dict(Counter(r['status'] for r in cells)),summary=summaries,maximums=maximums,safety=safety,projection=projection,boundaries=boundary_cases(),
        hashes={str(p.relative_to(B)):sha(p) for p in [PF,path('_runs.json'),path('_inputs.npz'),path('_evaluator_truth.json')]})
    dump(path('_summary.json'),result)
    with path('_cells.csv').open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(cells[0]));wr.writeheader();wr.writerows(cells)
    print(json.dumps({k:result[k] for k in ['rows','conditions','unique_inputs','source_endpoint_worlds','safety_failures','projection_moves','mlni_folds','mlni_nonconverged','mlni_max_iterations']},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','prepare','predict','evaluate']);a=p.parse_args();globals()[a.action]()
