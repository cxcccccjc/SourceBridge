"""Fixed-seed attack evaluation using shared inference functions."""
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
import sys,json,hashlib,argparse,math
from fractions import Fraction as F
from collections import defaultdict
import packet_attacks as engine
import additive_selection as fast
import acquisition_evaluation as old
import numpy as np
B=Path(__file__).resolve().parent;R=B/'results';N='attack_evaluation';PF=B/(N+'_protocol.json')
def path(s):return R/(N+s)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def dump(p,x):engine.dump(p,x)
def freeze():
    engine.verify()
    names=[Path(__file__).name,N+'_protocol.md','additive_selection.py','catalog_selection.py','packet_attacks.py','packet_attacks_protocol.json']
    p=dict(status='FROZEN_BEFORE_NEW_CONFIRMATION_SEEDS_EXECUTED',seeds=list(range(739001,739041)),span=450.,anchor_truths=[25.,115.,205.,295.,385.,475.],
        profiles=dict(exact=[0]*6,heterogeneous=[2,5,10,20,40,80]),costs=[1,2,1,2,1,2],budget=4,target_distribution='U[0,500]',
        primary='MLNI_JSAC2022__native',strong_secondary='MLNI_JSAC2022__weighted_source_affine',bootstrap_seed=819003,bootstrap_repetitions=10000,
        original_attack_protocol=read(engine.PF),source_hashes={n:sha(B/n) for n in names})
    if PF.exists():assert read(PF)==p
    else:dump(PF,p)
    print(json.dumps(dict(frozen=True,sha256=sha(PF))))
def verify():
    p=read(PF)
    for n,h in p['source_hashes'].items():assert verify_identity(B/n,h),n
    engine.span.verify()
    for n,h in p['original_attack_protocol']['source_hashes'].items():assert verify_identity(B/n,h),n
    return p
def prepare():
    p=verify();ap=p['original_attack_protocol'];aq=np.array(p['anchor_truths']);cases=[];truth={};inputs={};plans=[];contracts=[]
    for seed in p['seeds']:
        rng=np.random.default_rng(seed);a=rng.uniform(.7,1.3,9);b=rng.uniform(-20,20,9);e=rng.uniform(-4.5,4.5,(9,7));u=rng.uniform(-.9,.9,6);qt=float(rng.uniform(0,500))
        Q=np.r_[aq,qt];Yfull=a[:,None]*Q[None,:]+b[:,None]+e
        reportvalid=max(abs(F(float(Yfull[i,j]))-F(float(a[i]))*F(float(Q[j]))-F(float(b[i]))) for i in range(9) for j in range(7))<=5
        for profile,hh in p['profiles'].items():
            h=np.array(hh,dtype=float);c=aq+u*h;A=np.column_stack([c-h,c+h,np.ones(6)*5]);costs=np.array(p['costs'])
            r=fast.design_anchors_additive(A.tolist(),[0.,500.],[.7,1.3],5.,p['costs'],p['budget']);ids=list(r['anchor_ids'])
            oldids,_=old.select('MinimaxPairTable',A,c,h,costs,p['budget'],dict(day='confirmation',target_station='anonymous_fixed_target',original_ids=list(range(6))))
            assert ids==oldids,(seed,profile,ids,oldids)
            base=f's{seed}_{profile}';truth[base]=qt;Y=Yfull[:,ids+[6]];As=A[ids];q=c[ids];coarse=sum(64 if x==0 else max(1,math.ceil(20/x)) for x in h)
            totalcost=coarse+9*(1+int(costs[ids].sum()));plans.append(dict(base_key=base,seed=seed,profile=profile,selected_ids=ids,prequery_width=r['width'],total_cost=totalcost))
            coarsevalid=all(F(float(lo))<=F(float(val))<=F(float(hi)) for val,(lo,hi,_) in zip(aq,A));contracts.append(dict(base_key=base,report_valid=reportvalid,prior_valid=coarsevalid,target_valid=0<=qt<=500))
            library=[('clean','clean',Y.copy())]
            for shift in ap['target_shifts']:
                y=Y.copy();y[-2:,-1]+=shift;library.append((f'target_{shift:+}','target_shift',y))
            for gain in ap['spoof_gains']:
                for target in ap['spoof_targets']:
                    y=Y.copy();bias=y[-2:,-1]-gain*target;y[-2:,:-1]=gain*q[None,:]+bias[:,None];library.append((f'anchor_g{gain}_q{target}','anchor_spoof',y))
            y=Y.copy();y[-2:,:-1]=y[-2:,:-1][:,::-1];library.append(('reverse_anchors','inconsistency_control',y))
            for shift in ap['translation_controls']:
                y=Y.copy();y[-2:,:]+=shift;library.append((f'translate_{shift:+}','translation_control',y))
            for attack,family,y in library:
                assert np.array_equal(y[:7],Y[:7]);k=base+'__'+attack
                y=np.ascontiguousarray(y)
                inputs[k+'__reports']=y;inputs[k+'__anchors']=As;inputs[k+'__centers']=q
                cases.append(dict(base_key=base,key=k,profile=profile,target_slot=0,seed=seed,attack=attack,family=family,selected_count=len(ids),total_cost=totalcost,main_library=family!='translation_control'))
    assert len(cases)==1280 and all(all(r[k] for k in ['report_valid','prior_valid','target_valid']) for r in contracts)
    np.savez_compressed(path('_inputs.npz'),**inputs);dump(path('_evaluator_truth.json'),truth);dump(path('_plans.json'),plans);dump(path('_contracts.json'),contracts)
    dump(path('_manifest.json'),dict(cases=cases,input_sha=sha(path('_inputs.npz')),truth_sha=sha(path('_evaluator_truth.json')),protocol_sha=sha(PF)))
    print(json.dumps(dict(base_inputs=len(truth),conditions=len(cases),source_contracts_valid=True)))
def configure():
    engine.N=N;engine.PF=PF;engine.verify=verify;engine.boundary_cases=lambda:[]
def predict():configure();engine.predict()
def evaluate():
    configure();engine.evaluate();s=read(path('_summary.json'))
    s['scope']='Locked new-seed confirmation in the previously selected wide-anchor simulation scenario; no external real-world validation'
    dump(path('_summary.json'),s)
def analyze():
    p=verify();s=read(path('_summary.json'));lookup={(r['profile'],r['label'],r['base_key']):r for r in s['maximums']};rng=np.random.default_rng(p['bootstrap_seed']);idx=rng.integers(0,40,size=(p['bootstrap_repetitions'],40));out=[]
    assert s['protocol_sha']==sha(PF)
    assert len(lookup)==len(s['maximums'])
    assert all(r['library_size']==14 and r['nonfinite']==0 and r['maximum_error'] is not None and math.isfinite(r['maximum_error']) for r in s['maximums'])
    for profile in p['profiles']:
        keys=[f's{seed}_{profile}' for seed in p['seeds']];ours=np.array([lookup[(profile,'SourceBridge_packet',k)]['maximum_error'] for k in keys])
        for label in sorted({r['label'] for r in s['summary']} - {'SourceBridge_packet'}):
            baseline=np.array([lookup[(profile,label,k)]['maximum_error'] for k in keys]);o=np.sqrt(np.mean(ours**2));v=np.sqrt(np.mean(baseline**2))
            ob=np.sqrt(np.mean(ours[idx]**2,axis=1));vb=np.sqrt(np.mean(baseline[idx]**2,axis=1));difference=ob-vb;relative=1-ob/vb
            rr=dict(profile=profile,baseline=label,independent_seeds=40,ours_worst_rmse=float(o),baseline_worst_rmse=float(v),rmse_difference=float(o-v),
                paired_bootstrap_difference_95=np.quantile(difference,[.025,.975]).tolist(),relative_reduction=float(1-o/v),paired_bootstrap_relative_95=np.quantile(relative,[.025,.975]).tolist())
            sm={r['label']:r for r in s['summary'] if r['profile']==profile};rr['clean_relative_change']=sm['SourceBridge_packet']['clean_rmse']/sm[label]['clean_rmse']-1
            rr['primary_gate']=None if not (profile=='heterogeneous' and label==p['primary']) else bool(rr['rmse_difference']<0 and rr['paired_bootstrap_difference_95'][1]<0 and rr['clean_relative_change']<=.1)
            out.append(rr)
    result=dict(scope='Locked conditional simulation confirmation; secondary intervals descriptive, no familywise inference',protocol_sha=sha(PF),summary_sha=sha(path('_summary.json')),
        experiment_valid=not s['safety_failures'] and s['projection_bound_failures']==0 and s['mlni_nonconverged']==0,
        primary=next(r for r in out if r['primary_gate'] is not None),comparisons=out)
    result['confirmation_passed']=bool(result['experiment_valid'] and result['primary']['primary_gate'])
    dump(path('_paired_analysis.json'),result);print(json.dumps(result['primary'],ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','prepare','predict','evaluate','analyze']);a=p.parse_args();globals()[a.action]()
