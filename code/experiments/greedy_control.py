"""Same-ambiguity-objective greedy diagnostic, using exact frozen catalogs.

This is not a literature baseline and does not generate a new holdout dataset.
"""
import os,sys
sys.dont_write_bytecode=True
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
from fractions import Fraction as F
from itertools import combinations
from collections import Counter,defaultdict
import argparse,json,hashlib,math
import numpy as np

HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('--data-root',type=Path,default=HERE/'data')
ap.add_argument('--protocol',type=Path,default=HERE/'greedy_control_protocol.json')
ap.add_argument('--out',type=Path,default=HERE/'generated'/'greedy_control_results.json')
args=ap.parse_args();D=args.data_root.resolve()
def rd(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
protocol=rd(args.protocol)
orig_protocol=D/'procurement_protocol.json'
if not orig_protocol.exists():orig_protocol=D.parent/orig_protocol.name
assert sha(orig_protocol)==protocol['source_protocol_sha256']
run_file=D/'procurement_runs.json';input_file=D/'procurement_inputs.json'
assert sha(run_file)==protocol['source_runs_sha256'] and sha(input_file)==protocol['source_inputs_sha256']
run=rd(run_file);pools={r['pool_id']:r for r in rd(input_file)['pools']}
assert len(pools)==120 and len(run['budget_rows'])==240 and len(run['quotes'])==480
W={};catalog_hashes={}
for rec in run['catalogs']:
    f=resolve_resource(D/rec['w_file']);assert sha(f)==rec['w_sha'];catalog_hashes[f.name]=sha(f)
    cat=rd(f);W[rec['pool_id']]={tuple(r['ids']):F(r['width']) for r in cat['subset_rows']}
    assert len(W[rec['pool_id']])==64

def greedy(pool,budget=None,radius=None):
    costs=pools[pool]['costs'];widths=W[pool];ids=();cost=0;trace=[]
    while True:
        current=widths[ids]
        if radius is not None and current<=2*radius:reason='precision_reached';break
        choices=[]
        for j,cj in enumerate(costs):
            if j in ids or (budget is not None and cost+cj>budget):continue
            nxt=tuple(sorted(ids+(j,)));gain=current-widths[nxt];assert gain>=0
            choices.append(dict(id=j,cost=cj,gain=gain,score=gain/cj,next_ids=nxt,next_width=widths[nxt]))
        if not choices:reason='no_eligible_anchor';break
        choice=min(choices,key=lambda r:(-r['score'],r['cost'],r['id']))
        trace.append(dict(ids=list(ids),width=str(current),cost=cost,choices=[{k:str(v) if isinstance(v,F) else list(v) if isinstance(v,tuple) else v for k,v in r.items()} for r in choices],chosen=choice['id'] if choice['gain']>0 else None))
        if choice['gain']==0:reason='zero_single_step_gain';break
        ids=choice['next_ids'];cost+=choice['cost']
    return dict(ids=list(ids),width=str(widths[ids]),radius=float(widths[ids]/2),anchor_cost=cost,total_cost=pools[pool]['coarse_cost']+9*(1+cost),stop_reason=reason,trace=trace)

budgets=[]
for old in run['budget_rows']:
    pool=old['pool_id'];r=greedy(pool,budget=old['budget']);w=F(r['width']);opt=F(old['ours_width']);assert w>=opt
    budgets.append(dict(pool_id=pool,seed=old['seed'],span=old['span'],profile=old['profile'],budget_slot=old['budget_slot'],budget=old['budget'],greedy=r,optimal_ids=old['ours_ids'],optimal_width=old['ours_width'],optimal_anchor_cost=old['ours_cost'],equal_width=w==opt,radius_gap=float((w-opt)/2),relative_width_gap=float((w-opt)/opt),width_gap=str(w-opt)))
quotes=[]
for old in run['quotes']:
    pool=old['pool_id'];r=greedy(pool,radius=old['r']);feasible=F(r['width'])<=2*old['r'];reachable=min(W[pool].values())<=2*old['r'];assert reachable==old['ours_feasible']
    state='reached' if feasible else 'greedy_stall' if reachable else 'globally_unreachable'
    if feasible:assert r['total_cost']>=old['ours_total']
    opt_anchor=(old['ours_total']-pools[pool]['coarse_cost'])//9-1 if reachable else None
    quotes.append(dict(pool_id=pool,seed=old['seed'],span=old['span'],profile=old['profile'],r=old['r'],greedy=r,state=state,globally_reachable=reachable,greedy_feasible=feasible,optimal_ids=old['ours_ids'],optimal_total=old['ours_total'],optimal_anchor_cost=opt_anchor,total_cost_gap=r['total_cost']-old['ours_total'] if feasible else None,relative_total_cost_excess=(r['total_cost']/old['ours_total']-1) if feasible else None,optimal_saving_vs_greedy=(1-old['ours_total']/r['total_cost']) if feasible else None,anchor_cost_gap=r['anchor_cost']-opt_anchor if feasible else None))

synergy=[];synergy_counts=[];pairs_checked=0
for pool,widths in W.items():
    pos=0;zero=0;maximum=F(0);best=None
    for ids,w in widths.items():
        for j,k in combinations([j for j in range(6) if j not in ids],2):
            jids=tuple(sorted(ids+(j,)));kids=tuple(sorted(ids+(k,)));both=tuple(sorted(ids+(j,k)))
            gj=w-widths[jids];gk=w-widths[kids];gjk=w-widths[both];gain=gjk-gj-gk;pairs_checked+=1
            if gain>0:
                pos+=1;zero+=int(gj==gk==0)
                row=dict(pool_id=pool,seed=pools[pool]['seed'],span=pools[pool]['span'],profile=pools[pool]['profile'],base_ids=list(ids),j=j,k=k,base_width=str(w),single_j_gain=str(gj),single_k_gain=str(gk),pair_gain=str(gjk),complementarity=str(gain),zero_single_positive_pair=gj==gk==0)
                synergy.append(row)
                if gain>maximum:maximum=gain;best=row
    synergy_counts.append(dict(pool_id=pool,positive_pairs=pos,zero_single_positive_pair=zero,maximum_complementarity=float(maximum),maximum_witness=best))

def budget_summary(rr):
    return dict(cells=len(rr),equal=sum(r['equal_width'] for r in rr),strict_gap=sum(not r['equal_width'] for r in rr),mean_radius_gap=float(np.mean([r['radius_gap'] for r in rr])),median_radius_gap=float(np.median([r['radius_gap'] for r in rr])),max_radius_gap=max(r['radius_gap'] for r in rr),mean_relative_width_gap=float(np.mean([r['relative_width_gap'] for r in rr])),median_relative_width_gap=float(np.median([r['relative_width_gap'] for r in rr])),greedy_support_sizes=dict(Counter(len(r['greedy']['ids']) for r in rr)),optimal_support_sizes=dict(Counter(len(r['optimal_ids']) for r in rr)))
def quote_summary(rr):
    ff=[r for r in rr if r['greedy_feasible']]
    return dict(requests=len(rr),reached=sum(r['state']=='reached' for r in rr),greedy_stall=sum(r['state']=='greedy_stall' for r in rr),globally_unreachable=sum(r['state']=='globally_unreachable' for r in rr),cost_equal=sum(r['total_cost_gap']==0 for r in ff),cost_strict_gap=sum(r['total_cost_gap']>0 for r in ff),mean_total_cost_gap=float(np.mean([r['total_cost_gap'] for r in ff])) if ff else None,mean_relative_total_cost_excess=float(np.mean([r['relative_total_cost_excess'] for r in ff])) if ff else None,mean_optimal_saving_vs_greedy=float(np.mean([r['optimal_saving_vs_greedy'] for r in ff])) if ff else None,mean_anchor_cost_gap=float(np.mean([r['anchor_cost_gap'] for r in ff])) if ff else None,greedy_support_sizes=dict(Counter(len(r['greedy']['ids']) for r in ff)),optimal_support_sizes=dict(Counter(len(r['optimal_ids']) for r in rr if r['globally_reachable'])))
budget_groups=[dict(span=s,profile=p,budget_slot=b,**budget_summary([r for r in budgets if (r['span'],r['profile'],r['budget_slot'])==(s,p,b)])) for s in [20,150,450] for p in ['exact','heterogeneous'] for b in [0,1]]
quote_groups=[dict(span=s,profile=p,r=rad,**quote_summary([r for r in quotes if (r['span'],r['profile'],r['r'])==(s,p,rad)])) for s in [20,150,450] for p in ['exact','heterogeneous'] for rad in [20,50,100,150]]
seeds=sorted({r['seed'] for r in budgets});rng=np.random.default_rng(2026093002);idx=rng.integers(0,len(seeds),(20000,len(seeds)));CIs=[]
for kind in ['budget_radius_gap','budget_relative_width_gap','quote_relative_total_cost_excess','quote_optimal_saving_vs_greedy']:
    a=[]
    for seed in seeds:
        if kind.startswith('budget'):
            rr=[r for r in budgets if r['seed']==seed];field=kind[len('budget_'):]
        else:rr=[r for r in quotes if r['seed']==seed and r['greedy_feasible']];field=kind[len('quote_'):]
        a.append([sum(r[field] for r in rr),len(rr)])
    a=np.array(a);draw=a[idx].sum(axis=1);obs=a.sum(axis=0)
    CIs.append(dict(metric=kind,estimate=float(obs[0]/obs[1]),paired_seed_bootstrap_95=np.quantile(draw[:,0]/draw[:,1],[.025,.975]).tolist()))

mechanism=[]
for pool,p in pools.items():
    if p['seed']!=840101:continue
    ww=W[pool];costs=p['costs'];card=[]
    for kk in [0,1,2,3,4,5,6]:
        ids=min((ids for ids in ww if len(ids)==kk),key=lambda ids:(ww[ids],sum(costs[j] for j in ids),ids))
        card.append(dict(cardinality=kk,ids=list(ids),width=str(ww[ids]),radius=float(ww[ids]/2),anchor_cost=sum(costs[j] for j in ids),selected_centers=[p['centers'][j] for j in ids],selected_radii=[p['radii'][j] for j in ids]))
    mechanism.append(dict(pool_id=pool,span=p['span'],profile=p['profile'],anchors=p['anchors'],costs=p['costs'],best_by_exact_cardinality=card,greedy_budget_cases=[r for r in budgets if r['pool_id']==pool],greedy_precision_cases=[r for r in quotes if r['pool_id']==pool]))
result=dict(status='PASS',label='Same-objective cost-efficiency greedy (diagnostic control)',scope=protocol['scope'],protocol_sha256=sha(args.protocol),source_run_sha256=sha(run_file),counts=dict(pools=len(pools),budget_cells=len(budgets),precision_requests=len(quotes),complementarity_pairs_checked=pairs_checked,positive_complementarity_pairs=len(synergy),zero_single_positive_pair=sum(r['zero_single_positive_pair'] for r in synergy)),budget_summary=budget_summary(budgets),quote_summary=quote_summary(quotes),budget_groups=budget_groups,quote_groups=quote_groups,bootstrap=CIs,budget_rows=budgets,quote_rows=quotes,complementarity_by_pool=synergy_counts,mechanism_fixed_seed=mechanism,catalog_hashes=catalog_hashes)
public_plans_file=D/'public_inference_plans.json'
if public_plans_file.exists():
    public_plans=rd(public_plans_file)
    result['existing_public_support_sizes']=dict(scope='Descriptive audit of existing optimal actions, not a new experiment.',plans_sha256=sha(public_plans_file),budget=dict(Counter(len(r['selected_ids']) for r in public_plans['plans'])),precision=dict(Counter(len(r['selected_ids']) for r in public_plans['quotes'] if r['feasible'])),precision_by_r={str(rad):dict(Counter(len(r['selected_ids']) for r in public_plans['quotes'] if r['feasible'] and r['r']==rad)) for rad in [20,50,100,150]})
save(args.out,result);witnessfile=args.out.with_name('greedy_complementarity_witnesses.json');save(witnessfile,dict(protocol_sha256=sha(args.protocol),pairs_checked=pairs_checked,witnesses=synergy))
print(json.dumps({k:result[k] for k in ['status','counts','budget_summary','quote_summary','bootstrap']}))
