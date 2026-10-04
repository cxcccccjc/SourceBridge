from release_integrity import verify_identity, resolve_resource
"""Budgeted anchor design via directional two-anchor certificates.

The sparse-support theorem and proof are provided in the manuscript and supplement.
This is a candidate conditional design mechanism, not a published baseline.
"""
from itertools import combinations
from reference_geometry import worst_width
import time

def pair_catalog(anchors,target,gain_bounds,epsilon_target):
    out=[]
    for k in range(min(2,len(anchors))+1):
        for ids in combinations(range(len(anchors)),k):
            result=worst_width([anchors[i] for i in ids],target,gain_bounds,epsilon_target)
            out.append(dict(ids=ids,positive=result['branches'][0]['width'],negative=result['branches'][1]['width']))
    return out

def design_anchors(anchors,target,gain_bounds,epsilon_target,costs,budget,catalog=None):
    if len(costs)!=len(anchors) or any(c<0 for c in costs) or budget<0:raise ValueError('nonnegative costs and budget required')
    tick=time.perf_counter()
    if catalog is None:catalog=pair_catalog(anchors,target,gain_bounds,epsilon_target)
    solver_time=time.perf_counter()-tick;best=None;feasible=0
    for P in catalog:
        for Q in catalog:
            ids=tuple(sorted(set(P['ids'])|set(Q['ids'])));cost=sum(costs[i] for i in ids)
            if cost>budget:continue
            feasible+=1;score=max(P['positive'],Q['negative'])
            key=(score,cost,len(ids),ids)
            if best is None or key<best[0]:best=(key,P['ids'],Q['ids'])
    assert best is not None
    key,pos,neg=best;score,cost,_,ids=key
    actual=worst_width([anchors[i] for i in ids],target,gain_bounds,epsilon_target)
    # Floating solver output is diagnostic; exact selected-world verification
    # and ties are reported independently, without calling this certified code.
    if abs(actual['width']-score)>1e-7:raise RuntimeError('directional support mismatch')
    return dict(anchor_ids=ids,cost=cost,selection_bound=score,width=actual['width'],
                positive_support=pos,negative_support=neg,catalog_size=len(catalog),
                catalog_LPs=2*len(catalog),feasible_pair_pairs=feasible,
                catalog_seconds=solver_time,total_seconds=time.perf_counter()-tick)
