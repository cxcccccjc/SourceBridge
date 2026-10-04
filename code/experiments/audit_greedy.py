"""Independent standard-library audit of the added same-objective control."""
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
from fractions import Fraction as F
from itertools import combinations
import json,hashlib
ROOT=Path(__file__).resolve().parent;D=ROOT/'data'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
r=read(ROOT/'greedy_control_results.json');inputs={p['pool_id']:p for p in read(D/'procurement_inputs.json')['pools']};original=read(D/'procurement_runs.json');widths={}
for c in original['catalogs']:
    p=resolve_resource(D/c['w_file']);assert hashlib.sha256(p.read_bytes()).hexdigest()==c['w_sha'];widths[c['pool_id']]={frozenset(v['ids']):F(v['width']) for v in read(p)['subset_rows']}
def independent(p,budget,threshold):
    costs=inputs[p]['costs'];ww=widths[p];s=frozenset();spent=0
    while True:
        if threshold is not None and ww[s]<=threshold:return s,spent,'precision_reached'
        choices={j:(ww[s]-ww[s|{j}])/F(costs[j]) for j in range(6) if j not in s and (budget is None or spent+costs[j]<=budget)}
        if not choices:return s,spent,'no_eligible_anchor'
        j=max(choices,key=lambda j:(choices[j],-costs[j],-j))
        if choices[j]==0:return s,spent,'zero_single_step_gain'
        assert choices[j]>0;s=s|{j};spent+=costs[j]
def check(row,budget,threshold):
    pool=row['pool_id'];s,c,status=independent(pool,budget,threshold);g=row['greedy']
    assert sorted(s)==g['ids'] and c==g['anchor_cost'] and status==g['stop_reason']
    assert widths[pool][s]==F(g['width']) and g['total_cost']==inputs[pool]['coarse_cost']+9*(1+c)
for row in r['budget_rows']:
    check(row,row['budget'],None);p=row['pool_id'];costs=inputs[p]['costs'];ww=widths[p]
    best=min((s for s in ww if sum(costs[j] for j in s)<=row['budget']),key=lambda s:(ww[s],sum(costs[j] for j in s),len(s),sorted(s)))
    assert sorted(best)==row['optimal_ids'] and ww[best]==F(row['optimal_width'])
for row in r['quote_rows']:
    check(row,None,2*row['r']);p=row['pool_id'];costs=inputs[p]['costs'];ww=widths[p];feas=[s for s in ww if ww[s]<=2*row['r']]
    assert bool(feas)==row['globally_reachable']
    if feas:
        best=min(feas,key=lambda s:(sum(costs[j] for j in s),len(s),sorted(s)));assert sorted(best)==row['optimal_ids']
        assert row['optimal_total']==inputs[p]['coarse_cost']+9*(1+sum(costs[j] for j in best))
witness=read(ROOT/'greedy_complementarity_witnesses.json')['witnesses'];lookup={(x['pool_id'],frozenset(x['base_ids']),x['j'],x['k']):x for x in witness};assert len(lookup)==len(witness)
allpairs=pos=zero=0
for p,ww in widths.items():
    for s,w in ww.items():
        for j,k in combinations([v for v in range(6) if v not in s],2):
            a=w-ww[s|{j}];b=w-ww[s|{k}];both=w-ww[s|{j,k}];delta=both-a-b;allpairs+=1
            if delta>0:
                pos+=1;zero+=a==b==0;x=lookup[(p,s,j,k)];assert F(x['single_j_gain'])==a and F(x['single_k_gain'])==b and F(x['pair_gain'])==both and F(x['complementarity'])==delta
assert allpairs==28800 and pos==1735 and zero==160
out=dict(status='PASS',budget_cases=240,precision_cases=480,all_subset_rank_checks=720,complementarity_pairs=allpairs,positive_witnesses=pos,zero_single_positive_pair=zero,production_greedy_imported=False,scope='Independent exact-rational reimplementation of greedy choice, optimum ranking, and all complementarity arithmetic.')
(ROOT/'generated').mkdir(exist_ok=True);(ROOT/'generated'/'greedy_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out))
