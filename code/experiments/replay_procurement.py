"""Actually recompute JB-RobustD on every frozen pool using standard library."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
from fractions import Fraction as F
from itertools import combinations
import json,hashlib,time
ROOT=Path(__file__).resolve().parent;D=ROOT/'data';S=ROOT/'source'
sys.path.insert(0,str(S))
import robust_doptimal as jb
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def normalize(v):
    if isinstance(v,F):return str(v)
    if isinstance(v,(tuple,list)):return [normalize(x) for x in v]
    if isinstance(v,dict):return {k:normalize(x) for k,x in v.items()}
    return v
p=read(D/'procurement_protocol.json')
assert verify_identity(S/'robust_doptimal.py',p['source_sha256']['source_bridge_robust_doptimal.py'])
run=read(D/'procurement_runs.json');inputs={g['pool_id']:g for g in read(D/'procurement_inputs.json')['pools']}
checks=dict(pools=0,subset_determinants=0,subset_nominal_determinants=0,attaining_witnesses=0,stationarity_certificates=0,budget_choices=0,precision_queries=0,integer_budget_actions=0)
records=[];tic=time.perf_counter()
for saved in run['catalogs']:
    key=saved['pool_id'];g=inputs[key];fresh=jb.build_catalog(g['anchors'],g['costs']);checks['pools']+=1
    assert fresh['subset_count']==64 and normalize(fresh['entries'])==saved['jb_catalog']['entries'],key
    for e in fresh['entries']:
        ids=e['ids'];a=[g['anchors'][i] for i in ids];weights=[1/F(x[2])**2 for x in a];q=e['worst_values'];s0=sum(weights,F(0));s1=sum((w*x for w,x in zip(weights,q)),F(0));s2=sum((w*x*x for w,x in zip(weights,q)),F(0))
        assert s0*s2-s1*s1==e['determinant']
        centers=[(F(x[0])+F(x[1]))/2 for x in a]
        nominal=sum((weights[i]*weights[j]*(centers[i]-centers[j])**2 for i,j in combinations(range(len(ids)),2)),F(0))
        assert nominal==e['nominal_determinant'];checks['subset_determinants']+=1;checks['subset_nominal_determinants']+=1
        if ids:
            t=e['center'];assert all(F(x[0])<=v<=F(x[1]) and v==max(F(x[0]),min(F(x[1]),t)) for x,v in zip(a,q))
            assert sum((w*(t-v) for w,v in zip(weights,q)),F(0))==0
        checks['attaining_witnesses']+=1;checks['stationarity_certificates']+=1
    wc=read(resolve_resource(D/saved['w_file']));assert sha(resolve_resource(D/saved['w_file']))==saved['w_sha'];w={tuple(e['ids']):F(e['width']) for e in wc['subset_rows']}
    actions=[]
    for budget in range(sum(g['costs'])+1):
        a=jb.select_from_catalog(fresh,budget,source_count=9);actions.append(dict(budget=budget,ids=a['anchor_ids'],cost=a['cost'],width=w[a['anchor_ids']],drob=a['worst_determinant_per_source'],dnom=a['nominal_determinant_per_source'],degenerate=a['robust_degenerate']))
        checks['integer_budget_actions']+=1
    for b in [x for x in run['budget_rows'] if x['pool_id']==key]:
        a=actions[b['budget']];assert list(a['ids'])==b['jb_ids'] and a['cost']==F(b['jb_cost']) and a['width']==F(b['jb_width']) and a['degenerate']==b['jb_degenerate'];checks['budget_choices']+=1
    qout=[]
    for q in [x for x in run['quotes'] if x['pool_id']==key]:
        passing=[a for a in actions if a['width']<=2*q['r']];a=min(passing,key=lambda a:(a['cost'],len(a['ids']),a['ids'],a['budget'])) if passing else None
        assert bool(passing)==q['jb_feasible']
        if a:
            assert list(a['ids'])==q['jb_ids'] and a['width']==F(q['jb_width']) and g['coarse_cost']+9*(1+a['cost'])==q['jb_total']
            assert a['budget']==q['jb_budget'] and passing[0]['budget']==q['jb_first_budget']
        qout.append(dict(r=q['r'],feasible=bool(passing),ids=None if a is None else a['ids'],cost=None if a is None else g['coarse_cost']+9*(1+a['cost'])));checks['precision_queries']+=1
    records.append(dict(pool_id=key,subset_checks=64,all_determinants_exactly_equal=True,actions=actions,quotes=qout))
assert checks['pools']==120 and checks['subset_determinants']==7680 and checks['budget_choices']==240 and checks['precision_queries']==480
result=dict(status='PASS',scope='Actual standard-library replay of published JB-RobustD criterion adapter from original public intervals/costs. Recomputes robust and nominal determinants, attaining q and first-order certificates; exact budget/quote matching. Shared W values are read from the certified original W catalogs.',checks=checks,elapsed_seconds=time.perf_counter()-tic,source_sha256=sha(S/'robust_doptimal.py'),original_protocol_sha256=sha(D/'procurement_protocol.json'),original_runs_sha256=sha(D/'procurement_runs.json'),records=records)
(ROOT/'generated').mkdir(exist_ok=True);(ROOT/'generated'/'procurement_replay_results.json').write_text(json.dumps(normalize(result),indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='records'}))
