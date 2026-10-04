"""Derive figure aggregates from complete frozen procurement and prediction records."""
from pathlib import Path
import sys,os,json,hashlib,csv
from fractions import Fraction as F
import numpy as np
EXP=Path(__file__).resolve().parent
HERE=Path(os.environ.get('SOURCEBRIDGE_METRIC_OUTPUT_DIR',str(EXP/'generated')))
HERE.mkdir(parents=True,exist_ok=True)

def read(p):return json.loads(p.read_text('utf-8'))
d=read(EXP/'metrics.json')
g=read(EXP/'greedy_control_results.json')
c=read(EXP/'greedy_cleanup_results.json')
j=read(EXP/'data/procurement_runs.json')
plans=read(EXP/'data/public_inference_plans.json')
gi={(x['pool_id'],x['r']):x for x in g['quote_rows']}
ci={(x['pool_id'],x['r']):x for x in c['rows']}
requests=[]
for row in j['quotes']:
    key=(row['pool_id'],row['r']);gr=gi[key];cr=ci[key]
    assert row['ours_feasible']==row['jb_feasible']==gr['greedy_feasible']==cr['feasible']
    out={k:row[k] for k in ['pool_id','seed','span','profile','r']};out['feasible']=row['ours_feasible']
    if out['feasible']:
        costs={'SourceBridge':int(row['ours_total']),'JB-RobustD':int(row['jb_total']),'Greedy':int(gr['greedy']['total_cost']),'Greedy + cleanup':int(cr['cleaned_total'])}
        assert costs['SourceBridge']==gr['optimal_total']==cr['optimal_total']
        out.update(costs=costs,saving={k:float(F(v-costs['SourceBridge'],v)) for k,v in costs.items()})
    requests.append(out)
seeds=sorted({r['seed'] for r in requests})
rng=np.random.default_rng(2026093004);draws=rng.integers(0,len(seeds),size=(20000,len(seeds)))
stats=[]
for radius in [20,50,100,150,'all']:
    rr=[r for r in requests if radius=='all' or r['r']==radius]
    feasible=[r for r in rr if r['feasible']]
    for method in ['SourceBridge','JB-RobustD','Greedy','Greedy + cleanup']:
        vals=[r['saving'][method] for r in feasible]
        sums=np.array([sum(r['saving'][method] for r in feasible if r['seed']==s) for s in seeds])
        counts=np.array([sum(r['seed']==s for r in feasible) for s in seeds]);bs=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
        stats.append(dict(r=radius,method=method,requests=len(rr),feasible=len(feasible),matches=sum(r['costs'][method]==r['costs']['SourceBridge'] for r in feasible),mean_fraction_saving=float(np.mean(vals)),whole_seed_bootstrap_95=np.quantile(bs,[.025,.975]).tolist(),ratio_of_cost_totals_saving=1-sum(r['costs']['SourceBridge'] for r in feasible)/sum(r['costs'][method] for r in feasible)))
budget={'JB-RobustD':[float((F(r['jb_width'])-F(r['ours_width']))/2) for r in j['budget_rows']], 'Greedy':[r['radius_gap'] for r in g['budget_rows']]}
radius={(p['day'],p['profile']):float(F(p['width'])/2) for p in plans['plans']}
events=[dict(profile=r['profile'],day=r['day'],station=r['station'],empirical_worst_absolute_error=float(np.sqrt(r['worst_se'])),prequery_radius=radius[(r['day'],r['profile'])]) for r in d['public']['event_stats'] if r['label']=='SourceBridge_proper_MLNI_WLS']
csvpath=EXP/'data/public_inference_cells.csv'
with csvpath.open(encoding='utf-8-sig',newline='') as f:rows=[r for r in csv.DictReader(f) if r['label'] in ['SourceBridge_proper_MLNI_WLS','MLNI_JSAC2022__proper_public_WLS']]
paired={}
for r in rows:paired.setdefault(r['key'],{})[r['label']]=float(r['prediction'])
preds=[]
for key,row in paired.items():
    assert row['SourceBridge_proper_MLNI_WLS']==row['MLNI_JSAC2022__proper_public_WLS']
    preds.append(row['SourceBridge_proper_MLNI_WLS'])
assert len(preds)==6816
out=dict(protocol=dict(role='Post-hoc plotting analysis',seed=2026093004,whole_seed_resamples=20000,feasible_filter='All four methods agree, 329/480'),procurement_requests=requests,procurement_stats=stats,budget_radius_gaps=budget,public_radius_events=events,shared_MLNI=dict(conditions=len(preds),exactly_equal=len(preds),projection_changes=d['public']['projection_changed'],predictions=preds),source_hashes={str(p.relative_to(EXP)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [EXP/'metrics.json',EXP/'greedy_control_results.json',EXP/'greedy_cleanup_results.json',EXP/'data/procurement_runs.json',EXP/'data/public_inference_plans.json',csvpath]})
(HERE/'plot_metrics.json').write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps([x for x in stats if x['r']=='all'],indent=2))
