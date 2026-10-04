"""Sourcewise affine calibration followed by closed-interval depth.

This composes standard bounded-error calibration and robust interval fusion.
The acquisition theorem, not this composition in isolation, is the candidate
increment. Floating LP tolerances are explicit; no formal machine certificate.
"""
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parent/'deps'))
import numpy as np
from scipy.optimize import linprog

def packet_interval(report,anchors,target,gain_bounds,epsilon_target):
    """One source; anchors=(lo,hi,eps); report has anchor values then target."""
    y=np.asarray(report,dtype=float)
    if len(y)!=len(anchors)+1 or not np.isfinite(y).all():raise ValueError('complete finite packet required')
    amin,amax=gain_bounds;L,U=target
    if not (0<amin<=amax and L<=U and epsilon_target>=0):raise ValueError('invalid specification')
    A=[];rhs=[]
    for j,(lo,hi,eps) in enumerate(anchors):
        if lo>hi or eps<0:raise ValueError('invalid anchor box/specification')
        A.extend([[y[j]-eps,1,0],[-y[j]-eps,-1,0]]);rhs.extend([hi,-lo])
    A.extend([[y[-1]-epsilon_target,1,-1],[-y[-1]-epsilon_target,-1,1]]);rhs.extend([0,0])
    ends=[];statuses=[]
    for sign in [1,-1]:
        fit=linprog([0,0,sign],A_ub=A,b_ub=rhs,bounds=[(1/amax,1/amin),(None,None),(L,U)],method='highs')
        statuses.append(dict(status=fit.status,message=fit.message,iterations=fit.nit))
        if fit.status==2:return dict(feasible=False,interval=None,statuses=statuses)
        if not fit.success:raise RuntimeError(fit.message)
        ends.append(float(fit.x[2]))
    return dict(feasible=True,interval=ends,statuses=statuses)

def packet_fusion(reports,anchors,target,gain_bounds,epsilon_target,f):
    tick=time.perf_counter();reports=np.asarray(reports,dtype=float);m=len(reports)
    if reports.ndim!=2 or not isinstance(f,int) or not 0<=f<m:raise ValueError('require integer 0<=f<m')
    single=[packet_interval(row,anchors,target,gain_bounds,epsilon_target) for row in reports]
    ranges=[r['interval'] for r in single if r['feasible']];need=m-f
    # Feasibility of inconsistent sources counts against the source budget.
    if len(ranges)<need:return dict(feasible=False,interval=None,point=None,source_results=single,seconds=time.perf_counter()-tick)
    events={}
    for lo,hi in ranges:
        events.setdefault(lo,[0,0])[0]+=1
        events.setdefault(hi,[0,0])[1]+=1
    # At a closed endpoint add starts before checking and remove ends after.
    # Grouping equal coordinates preserves zero-width/touching intervals.
    accepted=[];active=0
    for p,(starts,ends) in sorted(events.items()):
        active+=starts
        if active>=need:accepted.append(p)
        active-=ends
    if not accepted:return dict(feasible=False,interval=None,point=None,source_results=single,seconds=time.perf_counter()-tick)
    lo,hi=accepted[0],accepted[-1]
    return dict(feasible=True,interval=[lo,hi],width=hi-lo,point=lo+(hi-lo)/2,source_results=single,
                inference='Floating source LPs and exact comparison of their returned endpoints; not outward-certified',
                seconds=time.perf_counter()-tick)
