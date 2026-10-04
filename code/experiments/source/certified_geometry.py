from release_integrity import verify_identity, resolve_resource
"""Exact rational reference for finite anchor choice and packet fusion.

Interprets stored floats as exact binary rationals. No tolerance in decisions.
Not proof-assistant verified; bit complexity and time are not constant.
"""
from fractions import Fraction as F
from itertools import combinations
import math,time,sys,operator,numbers
from reference_geometry import worst_width,determinant,jsonable

_MAX_FLOAT=sys.float_info.max
_MAX_FLOAT_RATIONAL=F(_MAX_FLOAT)

def _rational(value,name):
    try:return F(value)
    except (TypeError,ValueError,OverflowError,ZeroDivisionError) as exc:
        raise ValueError(name+' must be a finite rational number') from exc

def _model(anchors,target,gain_bounds,epsilon_target):
    try:
        L,U=target;amin,amax=gain_bounds
    except (TypeError,ValueError) as exc:
        raise ValueError('target and gain_bounds must each have two endpoints') from exc
    L,U=(_rational(v,'target endpoint') for v in (L,U))
    amin,amax=(_rational(v,'gain bound') for v in (amin,amax))
    et=_rational(epsilon_target,'target error bound')
    if not (0<amin<=amax and L<=U and et>=0):
        raise ValueError('Require 0 < gain_min <= gain_max, target_min <= target_max, and epsilon_target >= 0')
    parsed=[]
    for row in anchors:
        try:lo,hi,eps=row
        except (TypeError,ValueError) as exc:
            raise ValueError('Each anchor must have (lower, upper, error_bound)') from exc
        lo,hi,eps=(_rational(v,'anchor value') for v in (lo,hi,eps))
        if lo>hi or eps<0:raise ValueError('Each anchor needs lower <= upper and nonnegative error bound')
        parsed.append((lo,hi,eps))
    return parsed,(L,U),(amin,amax),et

def outward(value,lower):
    value=_rational(value,'exported value')
    # The directed bound outside binary64 range is finite on one side and
    # infinity on the other; Fraction(infinity) is never a valid operation.
    if value>_MAX_FLOAT_RATIONAL:return _MAX_FLOAT if lower else math.inf
    if value<-_MAX_FLOAT_RATIONAL:return -math.inf if lower else -_MAX_FLOAT
    x=float(value)
    if (lower and F(x)>value) or (not lower and F(x)<value):
        x=math.nextafter(x,-math.inf if lower else math.inf)
    assert F(x)<=value if lower else F(x)>=value
    return x

def _json_bound(value):
    return value if math.isfinite(value) else ('-Infinity' if value<0 else 'Infinity')

def exact_design(anchors,target,gain_bounds,epsilon_target,costs,budget):
    tick=time.perf_counter()
    anchors,target,gain_bounds,epsilon_target=_model(anchors,target,gain_bounds,epsilon_target)
    costs=[_rational(v,'anchor cost') for v in costs];budget=_rational(budget,'budget')
    if len(costs)!=len(anchors) or min(costs,default=F(0))<0 or budget<0:
        raise ValueError('Need one nonnegative cost per anchor and a nonnegative budget')
    catalog=[]
    for k in range(min(2,len(anchors))+1):
        for ids in combinations(range(len(anchors)),k):
            r=worst_width([anchors[i] for i in ids],target,gain_bounds,epsilon_target,exact=True)
            catalog.append((ids,r['branches'][0]['width'],r['branches'][1]['width']))
    best=None;combinations_checked=0
    for P,wp,_ in catalog:
        for Q,_,wm in catalog:
            ids=tuple(sorted(set(P)|set(Q)));cost=sum((costs[i] for i in ids),F(0))
            if cost>budget:continue
            combinations_checked+=1;key=(max(wp,wm),cost,len(ids),ids)
            if best is None or key<best[0]:best=(key,P,Q)
    key,P,Q=best;value,cost,_,ids=key
    result=worst_width([anchors[i] for i in ids],target,gain_bounds,epsilon_target,exact=True)
    assert result['width']==value
    exported_value=outward(value,False)
    return dict(anchor_ids=ids,cost=cost,width=value,positive_support=P,negative_support=Q,
                value_upper_float=_json_bound(exported_value),value_upper_float_finite=math.isfinite(exported_value),catalog_entries=len(catalog),
                pair_pairs_checked=combinations_checked,selected_exact_result=result,
                seconds=time.perf_counter()-tick)

def exact_polytope_p(rows):
    """All vertices of bounded rational 3-polytope; min/max third coordinate."""
    integer=[]
    for row in rows:
        row=tuple(_rational(v,'polytope coefficient') for v in row)
        if len(row)!=4:raise ValueError('Each 3D halfspace must have four coefficients')
        scale=math.lcm(*(v.denominator for v in row))
        vals=tuple(int(v*scale) for v in row);g=math.gcd(*vals)
        integer.append(tuple(v//g for v in vals) if g else vals)
    low=high=None;low_x=high_x=None
    for ids in combinations(range(len(integer)),3):
        rr=[integer[i] for i in ids];den=determinant(*(r[:3] for r in rr))
        if not den:continue
        nums=[]
        for col in range(3):
            mat=[list(r[:3]) for r in rr]
            for k in range(3):mat[k][col]=rr[k][3]
            nums.append(determinant(*mat))
        if den<0:den=-den;nums=[-v for v in nums]
        if any(sum(r[j]*nums[j] for j in range(3))>r[3]*den for r in integer):continue
        p=F(nums[2],den)
        if low is None or p<low:low=p;low_x=tuple(F(v,den) for v in nums)
        if high is None or p>high:high=p;high_x=tuple(F(v,den) for v in nums)
    return None if low is None else dict(interval=(low,high),lower_world=low_x,upper_world=high_x)

def exact_packet_fusion(reports,anchors,target,gain_bounds,epsilon_target,f):
    tick=time.perf_counter()
    anchors,target,gain_bounds,et=_model(anchors,target,gain_bounds,epsilon_target)
    amin,amax=gain_bounds;L,U=target;reports=list(reports);m=len(reports)
    if isinstance(f,bool) or not isinstance(f,numbers.Integral):
        raise ValueError('f must be an integer identity count, not a boolean or fractional value')
    try:f=operator.index(f)
    except TypeError as exc:raise ValueError('f must be an integer identity count') from exc
    if not 0<=f<m:raise ValueError('Need at least one report and 0 <= f < number of sources')
    single=[]
    for report in reports:
        y=[_rational(v,'report value') for v in report]
        if len(y)!=len(anchors)+1:raise ValueError('Every source must report all anchors and the target')
        rows=[(-1,0,0,-1/amax),(1,0,0,1/amin),(0,0,-1,-L),(0,0,1,U)]
        for j,(lo,hi,eps) in enumerate(anchors):
            rows.extend([(y[j]-eps,1,0,hi),(-y[j]-eps,-1,0,-lo)])
        rows.extend([(y[-1]-et,1,-1,0),(-y[-1]-et,-1,1,0)])
        single.append(exact_polytope_p(rows))
    ranges=[r['interval'] for r in single if r is not None]
    events={}
    for lo,hi in ranges:
        events.setdefault(lo,[0,0])[0]+=1
        events.setdefault(hi,[0,0])[1]+=1
    accepted=[];depth=0
    for p,(starts,ends) in sorted(events.items()):
        # Closed intervals: both new starts and current ends cover p, including
        # zero-width intervals. Only remove ends after testing this endpoint.
        depth+=starts
        if depth>=m-f:accepted.append(p)
        depth-=ends
    if not accepted:return dict(feasible=False,interval=None,point=None,source_results=single,seconds=time.perf_counter()-tick)
    lo,hi=accepted[0],accepted[-1];mid=(lo+hi)/2
    point=None if abs(mid)>_MAX_FLOAT_RATIONAL else float(mid)
    exported_radius=None
    if point is not None:
        radius=max(abs(F(point)-lo),abs(hi-F(point)))
        exported_radius=outward(radius,False)
    outer=[outward(lo,True),outward(hi,False)]
    return dict(feasible=True,interval=(lo,hi),width=hi-lo,exact_midpoint=mid,
                point_float=point,float_point_status='unrepresentable' if point is None else 'rounded_midpoint',
                float_point_error_radius_upper=None if exported_radius is None else _json_bound(exported_radius),
                float_outer_interval=list(map(_json_bound,outer)),
                float_export_finite=exported_radius is not None and math.isfinite(exported_radius) and all(map(math.isfinite,outer)),
                source_results=single,seconds=time.perf_counter()-tick)
