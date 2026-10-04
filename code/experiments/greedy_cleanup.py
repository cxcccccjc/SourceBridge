"""Strengthen the frozen greedy precision control by feasible deletion."""
import os,sys
sys.dont_write_bytecode=True
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
from fractions import Fraction as F
from collections import Counter
import argparse,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,default=HERE/'data');ap.add_argument('--greedy',type=Path,default=HERE/'greedy_control_results.json');ap.add_argument('--protocol',type=Path,default=HERE/'greedy_cleanup_protocol.json');ap.add_argument('--out',type=Path,default=HERE/'generated'/'greedy_cleanup_results.json');args=ap.parse_args();D=args.data_root
def rd(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protocol=rd(args.protocol);assert sha(args.greedy)==protocol['source_greedy_sha256']
run_file=D/'procurement_runs.json';inputs_file=D/'procurement_inputs.json'
assert sha(run_file)==protocol['source_runs_sha256'] and sha(inputs_file)==protocol['source_inputs_sha256']
run=rd(run_file);old=rd(args.greedy);pools={p['pool_id']:p for p in rd(inputs_file)['pools']};W={}
for c in run['catalogs']:
    path=resolve_resource(D/c['w_file']);assert sha(path)==c['w_sha'];W[c['pool_id']]={tuple(r['ids']):F(r['width']) for r in rd(path)['subset_rows']}
rows=[]
for r in old['quote_rows']:
    p=r['pool_id'];costs=pools[p]['costs'];ids=tuple(r['greedy']['ids']);trace=[];original=ids
    if r['greedy_feasible']:
        assert W[p][ids]<=2*r['r']
        while True:
            candidates=[]
            for j in ids:
                trial=tuple(k for k in ids if k!=j);width=W[p][trial]
                candidates.append(dict(remove=j,removed_cost=costs[j],remaining_ids=list(trial),remaining_width=str(width),eligible=width<=2*r['r']))
            valid=[x for x in candidates if x['eligible']]
            chosen=min(valid,key=lambda x:(-x['removed_cost'],x['remove'])) if valid else None
            trace.append(dict(current_ids=list(ids),current_width=str(W[p][ids]),candidates=candidates,deleted_id=None if chosen is None else chosen['remove']))
            if chosen is None:break
            ids=tuple(chosen['remaining_ids'])
        assert W[p][ids]<=2*r['r']
        assert all(W[p][tuple(k for k in ids if k!=j)]>2*r['r'] for j in ids)
        cost=sum(costs[j] for j in ids);total=pools[p]['coarse_cost']+9*(1+cost)
        assert r['optimal_total']<=total<=r['greedy']['total_cost']
    else:
        assert not r['globally_reachable'];cost=None;total=None
    rows.append(dict(pool_id=p,seed=r['seed'],span=r['span'],profile=r['profile'],r=r['r'],state=r['state'],feasible=r['greedy_feasible'],original_ids=list(original),original_anchor_cost=r['greedy']['anchor_cost'],original_total=r['greedy']['total_cost'] if r['greedy_feasible'] else None,cleaned_ids=list(ids),cleaned_width=str(W[p][ids]),cleaned_anchor_cost=cost,cleaned_total=total,deleted_count=len(original)-len(ids),optimal_ids=r['optimal_ids'],optimal_anchor_cost=r['optimal_anchor_cost'],optimal_total=r['optimal_total'],original_gap=r['total_cost_gap'],total_cost_gap=None if total is None else total-r['optimal_total'],optimal_saving_vs_cleaned=None if total is None else 1-r['optimal_total']/total,cleaned_excess_vs_optimum=None if total is None else total/r['optimal_total']-1,trace=trace))
assert len(rows)==480
def summary(rr):
    ff=[r for r in rr if r['feasible']]
    return dict(requests=len(rr),feasible=len(ff),globally_unreachable=sum(not r['feasible'] for r in rr),strict_gap=sum(r['total_cost_gap']>0 for r in ff),tied=sum(r['total_cost_gap']==0 for r in ff),improved_by_deletion=sum(r['deleted_count']>0 for r in ff),deleted_anchors=sum(r['deleted_count'] for r in ff),previous_gaps_resolved=sum(r['original_gap']>0 and r['total_cost_gap']==0 for r in ff),mean_total_cost_gap=float(np.mean([r['total_cost_gap'] for r in ff])) if ff else None,mean_optimal_saving_vs_cleaned=float(np.mean([r['optimal_saving_vs_cleaned'] for r in ff])) if ff else None,mean_cleaned_excess_vs_optimum=float(np.mean([r['cleaned_excess_vs_optimum'] for r in ff])) if ff else None,cleaned_support_sizes=dict(Counter(len(r['cleaned_ids']) for r in ff)))
seeds=sorted({r['seed'] for r in rows});rng=np.random.default_rng(protocol['statistics']['seed']);idx=rng.integers(0,20,(protocol['statistics']['replicates'],20));cis=[]
for field in ['total_cost_gap','optimal_saving_vs_cleaned','cleaned_excess_vs_optimum']:
    a=[]
    for s in seeds:
        rr=[r for r in rows if r['seed']==s and r['feasible']];a.append([sum(r[field] for r in rr),len(rr)])
    a=np.array(a);bs=a[idx].sum(axis=1);t=a.sum(axis=0)
    cis.append(dict(metric=field,estimate=float(t[0]/t[1]),paired_seed_cluster_percentile_95=np.quantile(bs[:,0]/bs[:,1],[.025,.975]).tolist()))
groups=[dict(span=s,profile=p,r=rad,**summary([r for r in rows if (r['span'],r['profile'],r['r'])==(s,p,rad)])) for s in [20,150,450] for p in ['exact','heterogeneous'] for rad in [20,50,100,150]]
result=dict(status='PASS',label=protocol['label'],scope=protocol['scope'],protocol_sha256=sha(args.protocol),source_greedy_sha256=sha(args.greedy),summary=summary(rows),by_radius=[dict(r=rad,**summary([r for r in rows if r['r']==rad])) for rad in [20,50,100,150]],by_stratum=groups,by_seed=[dict(seed=s,**summary([r for r in rows if r['seed']==s])) for s in seeds],bootstrap=cis,rows=rows)
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
print(json.dumps({k:result[k] for k in ['status','summary','bootstrap','by_radius']}))
