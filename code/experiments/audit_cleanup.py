"""Independent standard-library replay of every feasible deletion decision."""
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
from fractions import Fraction as F
import json,hashlib,math
ROOT=Path(__file__).resolve().parent;D=ROOT/'data'
def rd(p):return json.loads(p.read_text(encoding='utf-8-sig'))
protocol=rd(ROOT/'greedy_cleanup_protocol.json');greedy=rd(ROOT/'greedy_control_results.json');clean=rd(ROOT/'greedy_cleanup_results.json')
assert hashlib.sha256((ROOT/'greedy_control_results.json').read_bytes()).hexdigest()==protocol['source_greedy_sha256']
old={(r['pool_id'],r['r']):r for r in greedy['quote_rows']};inputs={p['pool_id']:p for p in rd(D/'procurement_inputs.json')['pools']};W={}
for c in rd(D/'procurement_runs.json')['catalogs']:
    path=resolve_resource(D/c['w_file']);assert hashlib.sha256(path.read_bytes()).hexdigest()==c['w_sha'];W[c['pool_id']]={frozenset(r['ids']):F(r['width']) for r in rd(path)['subset_rows']}
deletions=checks=0
for row in clean['rows']:
    pool=row['pool_id'];rad=row['r'];source=old[(pool,rad)];s=set(source['greedy']['ids']);c=inputs[pool]['costs'];steps=[]
    if source['greedy_feasible']:
        while True:
            eligible=[j for j in s if W[pool][frozenset(s-{j})]<=2*rad]
            if not eligible:break
            j=max(eligible,key=lambda j:(c[j],-j));s.remove(j);steps.append(j);deletions+=1
        assert steps==[t['deleted_id'] for t in row['trace'] if t['deleted_id'] is not None]
        assert all(W[pool][frozenset(s-{j})]>2*rad for j in s)
        total=inputs[pool]['coarse_cost']+9*(1+sum(c[j] for j in s))
        assert total==row['cleaned_total'] and row['optimal_total']<=total<=row['original_total']
        assert row['total_cost_gap']==total-row['optimal_total']
        assert math.isclose(row['optimal_saving_vs_cleaned'],1-row['optimal_total']/total,abs_tol=1e-15)
    else:assert not source['globally_reachable'] and row['cleaned_total'] is None and not row['trace']
    assert sorted(s)==row['cleaned_ids'] and W[pool][frozenset(s)]==F(row['cleaned_width']);checks+=1
rr=[r for r in clean['rows'] if r['feasible']];assert len(rr)==329 and len(clean['rows'])==480
assert sum(r['total_cost_gap']>0 for r in rr)==10 and sum(r['total_cost_gap']==0 for r in rr)==319 and deletions==26
out=dict(status='PASS',precision_queries=checks,feasible=329,globally_unreachable=151,deletions=deletions,remaining_cost_gaps=10,tied_costs=319,production_cleanup_imported=False,scope='Independent standard-library deletion replay and threshold/cost checks.')
(ROOT/'generated').mkdir(exist_ok=True);(ROOT/'generated'/'cleanup_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out))
