from release_integrity import verify_identity, resolve_resource
"""Fast pairing of two-anchor certificates for additive nonnegative costs.

This changes the pairing algorithm, not the pre-query statistical model or
the directional LPs. It does not support a general monotone cost oracle.
"""
import math
import time
from itertools import combinations
from fractions import Fraction
from catalog_selection import pair_catalog
from reference_geometry import worst_width


def _validate(catalog, costs, budget):
    for x in list(costs)+[budget]:
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0:
            raise ValueError('Costs/budget must be finite nonnegative int/float values')
    # Interpret finite floating prices exactly, so overlap discounts really
    # obey additive-set identities. Usual integer simulation prices stay ints.
    costs=[x if isinstance(x,int) else Fraction(x) for x in costs]
    budget=budget if isinstance(budget,int) else Fraction(budget)
    entries=[];seen=set()
    for r in catalog:
        ids=tuple(r['ids'])
        if len(ids)>2 or tuple(sorted(set(ids)))!=ids or any(isinstance(i,bool) or not isinstance(i,int) or not 0<=i<len(costs) for i in ids):
            raise ValueError('Certificate ids must be sorted distinct indices, size <= 2')
        if ids in seen:raise ValueError('Duplicate certificate subset')
        seen.add(ids)
        for name in ['positive','negative']:
            x=r[name]
            if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0:
                raise ValueError('Directional values must be finite nonnegative numbers')
        entries.append(dict(ids=ids,positive=r['positive'],negative=r['negative'],cost=sum(costs[i] for i in ids)))
    return entries,costs,budget


def threshold_minimum(entries,costs,tau):
    """Find a minimum-cost union with both directional values <= tau.

    Ties use cardinality and lexicographic union ids. The representative family
    preserves a cheapest negative certificate avoiding any <= 2 vertices.
    """
    positive=[e for e in entries if e['positive']<=tau]
    negative=[e for e in entries if e['negative']<=tau]
    if not positive or not negative:return None
    key=lambda e:(e['cost'],len(e['ids']),e['ids'])
    scans=0
    def cheapest_avoiding(forbidden):
        nonlocal scans
        scans+=1
        return min((e for e in negative if not any(i in forbidden for i in e['ids'])),key=key,default=None)
    root=cheapest_avoiding(())
    reps={root['ids']:root}
    for v in root['ids']:
        child=cheapest_avoiding((v,))
        if child is None:continue
        reps[child['ids']]=child
        for w in child['ids']:
            leaf=cheapest_avoiding((v,w))
            if leaf is not None:reps[leaf['ids']]=leaf
    containing={};by_ids={e['ids']:e for e in negative}
    for e in negative:
        for i in e['ids']:
            if i not in containing or key(e)<key(containing[i]):containing[i]=e
    best=None;candidate_checks=0
    for p in positive:
        candidates=dict(reps)
        for i in p['ids']:
            if i in containing:candidates[containing[i]['ids']]=containing[i]
        # In particular Q=P needs an explicit check: its double-overlap cost
        # discount need not be preserved by the cheapest incident certificate.
        for k in range(len(p['ids'])+1):
            for subset in combinations(p['ids'],k):
                if subset in by_ids:candidates[subset]=by_ids[subset]
        for q in candidates.values():
            ids=tuple(sorted(set(p['ids'])|set(q['ids'])))
            candidate_checks+=1
            rank=(sum(costs[i] for i in ids),len(ids),ids)
            if best is None or rank<best[0]:best=(rank,p['ids'],q['ids'])
    return dict(cost=best[0][0],anchor_ids=best[0][2],positive_support=best[1],negative_support=best[2],
                representative_count=len(reps),negative_scans=scans,candidate_checks=candidate_checks,
                eligible_positive=len(positive),eligible_negative=len(negative))


def select_from_catalog(catalog,costs,budget):
    entries,costs,budget=_validate(catalog,costs,budget)
    thresholds=sorted({e[k] for e in entries for k in ['positive','negative']})
    if not thresholds:return dict(feasible=False,reason='empty_certificate_family')
    lo,hi=0,len(thresholds);cache={}
    def query(i):
        if i not in cache:cache[i]=threshold_minimum(entries,costs,thresholds[i])
        return cache[i]
    while lo<hi:
        mid=(lo+hi)//2;r=query(mid)
        if r is not None and r['cost']<=budget:hi=mid
        else:lo=mid+1
    if lo==len(thresholds):return dict(feasible=False,reason='no_budget_feasible_certificate_union',threshold_queries=len(cache))
    r=query(lo)
    return dict(feasible=True,selection_bound=thresholds[lo],**r,threshold_queries=len(cache),threshold_count=len(thresholds),
                total_candidate_checks=sum(q['candidate_checks'] for q in cache.values() if q is not None))


def design_anchors_additive(anchors,target,gain_bounds,epsilon_target,costs,budget,catalog=None):
    if len(anchors)!=len(costs):raise ValueError('One additive cost is required per anchor')
    tick=time.perf_counter()
    if catalog is None:catalog=pair_catalog(anchors,target,gain_bounds,epsilon_target)
    catalog_seconds=time.perf_counter()-tick
    start=time.perf_counter();r=select_from_catalog(catalog,costs,budget);pairing_seconds=time.perf_counter()-start
    if not r['feasible']:return dict(**r,catalog_seconds=catalog_seconds,pairing_seconds=pairing_seconds)
    actual=worst_width([anchors[i] for i in r['anchor_ids']],target,gain_bounds,epsilon_target)
    if abs(actual['width']-r['selection_bound'])>1e-7:
        raise RuntimeError('Directional LP versus selected union discrepancy')
    return dict(**r,width=actual['width'],catalog_size=len(catalog),catalog_LPs=2*len(catalog),
                catalog_seconds=catalog_seconds,pairing_seconds=pairing_seconds,total_seconds=time.perf_counter()-tick)
