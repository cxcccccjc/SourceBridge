"""Portable reanalysis from the complete prediction and procurement tables."""
import os,sys
sys.dont_write_bytecode=True
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
from fractions import Fraction as F
import json,csv,math,hashlib,collections
import numpy as np
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'
(ROOT/'generated').mkdir(exist_ok=True)
def read(n):return json.loads((resolve_resource(DATA/n)).read_text(encoding='utf-8-sig'))
manifest=json.loads((ROOT/'package_manifest.json').read_text(encoding='utf-8'))
hash_checks=0
for name,rec in manifest['included_data_files'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==rec['sha256'],name
    hash_checks+=1
with (DATA/'public_inference_cells.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
assert len(rows)==102240
events=collections.defaultdict(list);condition=collections.defaultdict(dict)
for r in rows:
    err=(float(r['prediction'])-float(r['truth']))**2
    assert math.isclose(err,float(r['squared_error']),rel_tol=1e-13,abs_tol=1e-11)
    events[(r['profile'],r['label'],r['event'])].append(r)
    condition[r['key']][r['label']]=float(r['prediction'])
for rr in condition.values():assert rr['SourceBridge_proper_MLNI_WLS']==rr['MLNI_JSAC2022__proper_public_WLS']
e=[]
for (p,l,k),rr in events.items():
    clean=[r for r in rr if r['attack']=='clean'];lib=[r for r in rr if r['main_library']=='True']
    assert len(rr)==16 and len(clean)==1 and len(lib)==14
    e.append(dict(profile=p,label=l,event=k,day=rr[0]['day'],clean_se=float(clean[0]['squared_error']),worst_se=max(float(r['squared_error']) for r in lib)))
P=read('public_inference_protocol.json');days=P['days'];labels=sorted({r['label'] for r in rows});profiles=['exact','heterogeneous'];ours='SourceBridge_proper_MLNI_WLS'
B=20000;rng=np.random.default_rng(2026093001);blocks={}
for L in [1,3,7]:
    starts=rng.integers(0,28,size=(B,math.ceil(28/L)));blocks[L]=((starts[:,:,None]+np.arange(L))%28).reshape(B,-1)[:,:28]
def ci(a):return np.quantile(a,[.025,.975]).tolist()
lookup={(r['profile'],r['label'],r['event']):r for r in e}
out=[]
for profile in profiles:
    for metric in ['clean','worst']:
        for label in labels:
            ee=[r for r in e if r['profile']==profile and r['label']==label]
            ss=np.array([sum(r[metric+'_se'] for r in ee if r['day']==d) for d in days]);nn=np.array([sum(r['day']==d for r in ee) for d in days]);oo=np.array([sum(lookup[(profile,ours,r['event'])][metric+'_se'] for r in ee if r['day']==d) for d in days])
            out.append(dict(profile=profile,metric=metric,baseline=label,ours_rmse=float(np.sqrt(oo.sum()/nn.sum())),baseline_rmse=float(np.sqrt(ss.sum()/nn.sum())),difference=float(np.sqrt(oo.sum()/nn.sum())-np.sqrt(ss.sum()/nn.sum())),block_bootstrap_95={str(L):ci(np.sqrt(oo[idx].sum(axis=1)/nn[idx].sum(axis=1))-np.sqrt(ss[idx].sum(axis=1)/nn[idx].sum(axis=1))) for L,idx in blocks.items()}))

pp=read('procurement_protocol.json');pr=read('procurement_runs.json');inputs={p['pool_id']:p for p in read('procurement_inputs.json')['pools']};W={};D={}
for c in pr['catalogs']:
    pool=c['pool_id'];cost=inputs[pool]['costs'];ww=read(c['w_file'])
    W[pool]=[dict(ids=r['ids'],w=F(r['width']),cost=sum(cost[i] for i in r['ids'])) for r in ww['subset_rows']]
    D[pool]=[dict(ids=r['ids'],d=F(r['determinant']),dn=F(r['nominal_determinant']),cost=F(r['cost'])) for r in c['jb_catalog']['entries']]
for r in pr['budget_rows']:
    p=r['pool_id'];w=min((v for v in W[p] if v['cost']<=r['budget']),key=lambda v:(v['w'],v['cost'],len(v['ids']),v['ids']));d=min((v for v in D[p] if v['cost']<=r['budget']),key=lambda v:(-v['d'],-v['dn'],v['cost'],len(v['ids']),v['ids']))
    assert w['ids']==r['ours_ids'] and d['ids']==r['jb_ids']
for r in pr['quotes']:
    pool=r['pool_id'];feas=[w for w in W[pool] if w['w']<=2*r['r']];assert bool(feas)==r['ours_feasible']
    if feas:
        w=min(feas,key=lambda v:(v['cost'],len(v['ids']),v['ids']));assert w['ids']==r['ours_ids'] and r['ours_total']==inputs[pool]['coarse_cost']+9*(1+w['cost'])
    opts=[]
    for budget in range(sum(inputs[pool]['costs'])+1):
        d=min((v for v in D[pool] if v['cost']<=budget),key=lambda v:(-v['d'],-v['dn'],v['cost'],len(v['ids']),v['ids']));w=next(v for v in W[pool] if v['ids']==d['ids'])
        if w['w']<=2*r['r']:opts.append(w)
    assert bool(opts)==r['jb_feasible']
    if opts:assert r['jb_total']==inputs[pool]['coarse_cost']+9*(1+min(v['cost'] for v in opts))
seeds=pp['seeds'];idx=rng.integers(0,20,size=(B,20));proc=[]
for rad in [20,50,100,150,'all']:
    qq=[q for q in pr['quotes'] if rad=='all' or q['r']==rad];a=[]
    for s in seeds:
        ss=[q for q in qq if q['seed']==s and q['ours_feasible'] and q['jb_feasible']]
        a.append([sum(float(F(q['saving'])) for q in ss),len(ss),sum(q['ours_total'] for q in ss),sum(q['jb_total'] for q in ss)])
    a=np.array(a,dtype=float);t=a.sum(axis=0);bs=a[idx].sum(axis=1);ok=bs[:,1]>0
    proc.append(dict(r=rad,requests=len(qq),both_feasible=int(t[1]),conditional_mean_fraction_saving=float(t[0]/t[1]),paired_seed_cluster_95=ci(bs[ok,0]/bs[ok,1]),ratio_of_cost_totals_saving=float(1-t[2]/t[3])))
expected=json.loads((ROOT/'metrics.json').read_text(encoding='utf-8'))
for r in out:
    exp=next(t for t in expected['public']['paired_statistics'] if t['profile']==r['profile'] and t['metric']==r['metric'] and t['baseline']==r['baseline'])
    assert math.isclose(r['difference'],exp['rmse_difference_ours_minus_baseline'],abs_tol=1e-12)
    for L in ['1','3','7']:assert np.allclose(r['block_bootstrap_95'][L],exp['block_bootstrap_95'][L],rtol=1e-12,atol=1e-12)
for r in proc:
    exp=next(t for t in expected['procurement']['by_radius_with_ci'] if t['r']==r['r'])
    assert r['both_feasible']==exp['both_feasible'] and np.allclose(r['paired_seed_cluster_95'],exp['paired_seed_cluster_95'],atol=1e-12)
result=dict(status='PASS',scope='Portable complete-table reanalysis; post-hoc sensitivity, not new blind trial.',hash_checks=hash_checks,error_rows=len(rows),equal_composition_pairs=len(condition),public_comparisons=out,procurement=proc,budget_rank_checks=len(pr['budget_rows']),precision_rank_checks=len(pr['quotes']))
(ROOT/'generated'/'reanalysis_results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['public_comparisons','procurement']}))
