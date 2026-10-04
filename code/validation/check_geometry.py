"""Bounded structural audit; no data or historical result files are modified.

The reference below independently eliminates beta and optimizes a one-dimensional
piecewise affine envelope, then compares against the saved three-variable solver.
This is mathematical implementation QA, not predictive-performance evidence.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations
import argparse, hashlib, json, random

OUT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-dir', type=Path,
                    default=OUT.parent/'experiments'/'source',
                    help='Directory containing packaged SourceBridge source modules; relative paths use the current working directory.')
parser.add_argument('--output', type=Path, default=OUT/'geometry_check.json',
                    help='JSON result file; defaults to the script directory.')
args = parser.parse_args()
SRC = args.source_dir.resolve()
TRACK = ['reference_geometry.py', 'additive_selection.py',
         'catalog_selection.py', 'packet_geometry.py']
missing = [name for name in TRACK if not (SRC/name).is_file()]
if missing:
    parser.error('Missing source modules in %s: %s' % (SRC, ', '.join(missing)))
hashes_before = {s: hashlib.sha256((SRC/s).read_bytes()).hexdigest() for s in TRACK}
sys.path.insert(0, str(SRC))
from reference_geometry import worst_width
from additive_selection import select_from_catalog, threshold_minimum, _validate

def envelope(anchors, target, gain, et, positive):
    amin, amax = map(F, gain)
    L,U = map(F,target); c=L if positive else U; kt=2*F(et)/amin
    lower = [(F(lo), -(F(hi)+2*F(e)/amin)) for lo,hi,e in anchors]
    upper = [(F(hi), -(F(lo)-2*F(e)/amin)) for lo,hi,e in anchors]
    lower.append((L, -(c+kt)))
    upper.append((U, -(c-kt)))
    left,right = amin/amax,F(1)
    for b,m in lower:
        for d,n in upper:
            k=m-n; rhs=d-b
            if k>0: right=min(right,rhs/k)
            elif k<0: left=max(left,rhs/k)
            else: assert rhs>=0
    assert left<=right
    lines=[(U,F(0))] + [(b,m+c+kt) for b,m in upper] if positive else [(L,F(0))]+[(b,m+c-kt) for b,m in lower]
    points={left,right}
    for (b,m),(d,n) in combinations(lines,2):
        if m!=n:
            a=(d-b)/(m-n)
            if left<=a<=right: points.add(a)
    values=[min(b+m*a for b,m in lines) if positive else max(b+m*a for b,m in lines) for a in points]
    return max(values)-L if positive else U-min(values)

def subsets(n,k=None):
    return [p for z in range((n if k is None else min(n,k))+1) for p in combinations(range(n),z)]

def check_witness(r, anchors,target,gain,et):
    w=r['witness']; q=w['first_truth']; qp=w['second_truth']; y=w['reports']
    a,ap=w['first_gain'],w['second_gain']; b,bp=w['first_bias'],w['second_bias']
    assert gain[0]<=a<=gain[1] and gain[0]<=ap<=gain[1]
    for x,xp,yy,(lo,hi,e) in zip(q,qp,y,anchors+[(target[0],target[1],et)]):
        assert F(lo)<=x<=F(hi) and F(lo)<=xp<=F(hi)
        assert abs(yy-a*x-b)<=e and abs(yy-ap*xp-bp)<=e
    assert abs(q[-1]-qp[-1])==r['width']

rng=random.Random(20260930)
cases=[
    ('four_anchor_tight', [(-14,-10,1),(-2,0,1),(0,2,1),(10,14,1)],(-10,10),(1,4),1),
    ('equal_gain_points', [(-2,-2,0),(3,3,0)],(-3,8),(2,2),0),
    ('target_point', [(-2,4,1),(4,5,2)],(2,2),(1,3),1),
    ('nonsubmodular', [(0,0,0),(1,1,0)],(2,3),(1,2),0),
    ('narrow_span_two_directions',[(4,4,1),(6,6,1)],(0,10),(1,2),1),
]
for i in range(10):
    anchors=[]
    for _ in range(4):
        lo=rng.randrange(-12,12); anchors.append((lo,lo+rng.randrange(0,7),rng.randrange(0,4)))
    cases.append((f'fixed_seed_{i}', anchors,(-5,9),(1,3),rng.randrange(0,4)))
audit=[]; directional_checks=0; witness_checks=0
for name,anchors,target,gain,et in cases:
    cache={}
    for ids in subsets(len(anchors)):
        A=[anchors[j] for j in ids]
        plus=envelope(A,target,gain,et,True); minus=envelope(A,target,gain,et,False)
        got=worst_width(A,target,gain,et,exact=True)
        assert [r['width'] for r in got['branches']]==[plus,minus]
        for r in got['branches']:check_witness(r,A,target,gain,et);witness_checks+=1
        cache[ids]=(plus,minus);directional_checks+=2
    full=tuple(range(len(anchors))); vals=cache[full]
    ps=[p for p in subsets(len(anchors),2) if cache[p][0]==vals[0]]
    qs=[p for p in subsets(len(anchors),2) if cache[p][1]==vals[1]]
    assert ps and qs
    support=min((tuple(sorted(set(p)|set(q))) for p in ps for q in qs),key=lambda a:(len(a),a))
    assert cache[support]==vals
    row={'name':name,'full_directions':list(map(str,vals)),'full_width':str(max(vals)),
         'positive_support':ps[0],'negative_support':qs[0],'minimum_union_support':support,
         'all_subset_widths':{','.join(map(str,p)):str(max(v)) for p,v in cache.items()}}
    audit.append(row)
assert audit[0]['full_width']=='20/3'
assert [audit[0]['all_subset_widths'][','.join(str(j) for j in range(4) if j!=i)] for i in range(4)]==['8','48/7','48/7','8']

combination_queries=0; threshold_queries=0
for n in range(1,7):
    for rep in range(8):
        catalog=[{'ids':p,'positive':rng.randrange(4),'negative':rng.randrange(4)} for p in subsets(n,2)]
        costs=[rng.randrange(4) for _ in range(n)]
        for budget in range(sum(costs)+2):
            chosen=select_from_catalog(catalog,costs,budget)
            candidates=[]
            for p in catalog:
                for q in catalog:
                    union=tuple(sorted(set(p['ids'])|set(q['ids'])));cost=sum(costs[j] for j in union)
                    if cost<=budget:candidates.append((max(p['positive'],q['negative']),cost,len(union),union))
            best=min(candidates)
            assert (chosen['selection_bound'],chosen['cost'],len(chosen['anchor_ids']),chosen['anchor_ids'])==best
            combination_queries+=1
        entries,costs,_=_validate(catalog,costs,0)
        for tau in range(4):
            chosen=threshold_minimum(entries,costs,tau)
            candidates=[]
            for p in catalog:
                if p['positive']>tau:continue
                for q in catalog:
                    if q['negative']>tau:continue
                    union=tuple(sorted(set(p['ids'])|set(q['ids']))); candidates.append((sum(costs[j] for j in union),len(union),union))
            if candidates:
                assert (chosen['cost'],len(chosen['anchor_ids']),chosen['anchor_ids'])==min(candidates)
                assert chosen['representative_count']<=7 and chosen['negative_scans']<=7
            else:assert chosen is None
            threshold_queries+=1

projection_checks=0
for L,U,lo,hi,W in [(0,2,0,2,2),(0,2,0,1,2),(0,F(2,3),0,F(2,3),F(2,3)),(-2,7,-1,6,9)]:
    R=F(W)/2; sl=max(F(L),F(hi)-R);su=min(F(U),F(lo)+R)
    assert sl<=su
    for z in [-10,F(1,3),0,1,10]:
        p=min(su,max(sl,z));assert max(abs(p-lo),abs(p-hi))<=R;projection_checks+=1
hashes_after={s:hashlib.sha256((SRC/s).read_bytes()).hexdigest() for s in TRACK}
assert hashes_after==hashes_before
result={'status':'PASS','scope':'Independent exact structural audit; no empirical predictive evidence',
        'directional_exact_checks':directional_checks,'attainability_witness_checks':witness_checks,
        'model_cases':len(cases),'budget_pairing_queries':combination_queries,'threshold_pairing_queries':threshold_queries,
        'projection_checks':projection_checks,'source_hashes_unchanged':hashes_after,'model_results':audit}
args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['model_results','source_hashes_unchanged']},indent=2))
