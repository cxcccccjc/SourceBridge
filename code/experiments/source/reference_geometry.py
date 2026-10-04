"""Pre-query two-world ambiguity for homogeneous fresh affine sources.

An independently derived three-variable LP; the exact reference enumerates
vertices using rational arithmetic. This is a structural model component,
not a published baseline or a claim of novelty. Both exact rational and numerical LP implementations are provided.
"""
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
import math, sys, time
sys.path.insert(0,str(Path(__file__).resolve().parent/'deps'))
import numpy as np
from scipy.optimize import linprog

def rows_for(anchors, target, gain_bounds, epsilon_target, c):
    """Rows A*x <= b, x=(alpha,beta,p); anchors are (lo,hi,epsilon)."""
    amin,amax=map(F,gain_bounds);L,U=map(F,target);c=F(c)
    assert 0<amin<=amax and L<=U and F(epsilon_target)>=0
    rows=[];tags=[]
    def add(a,b,tag): rows.append((tuple(map(F,a)),F(b)));tags.append(tag)
    add((-1,0,0),-amin/amax,'alpha_min');add((1,0,0),1,'alpha_max')
    add((0,0,-1),-L,'target_min');add((0,0,1),U,'target_max')
    kt=2*F(epsilon_target)/amin
    add((-(c+kt),-1,1),0,'target_relation_upper')
    add((c-kt,1,-1),0,'target_relation_lower')
    for i,(lo,hi,eps) in enumerate(anchors):
        lo,hi,eps=F(lo),F(hi),F(eps);assert lo<=hi and eps>=0
        k=2*eps/amin
        add((-(hi+k),-1,0),-lo,('anchor_lower',i))
        add((lo-k,1,0),hi,('anchor_upper',i))
    return rows,tags

def determinant(a,b,c):
    return a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])

def solve_branch_exact(anchors,target,gain_bounds,epsilon_target,c,positive):
    rows,tags=rows_for(anchors,target,gain_bounds,epsilon_target,c)
    # Equivalent integer inequalities avoid repeatedly expanding denominators.
    integer=[]
    for a,b in rows:
        l=math.lcm(*(v.denominator for v in (*a,b)))
        vals=[int(v*l) for v in (*a,b)];g=math.gcd(*vals)
        integer.append(tuple(v//g for v in vals) if g else tuple(vals))
    best=None;vertex_count=0
    for ids in combinations(range(len(integer)),3):
        r=[integer[i] for i in ids];a=[t[:3] for t in r];den=determinant(*a)
        if not den:continue
        nums=[]
        for col in range(3):
            mat=[list(t[:3]) for t in r]
            for j in range(3):mat[j][col]=r[j][3]
            nums.append(determinant(*mat))
        if den<0:den=-den;nums=[-v for v in nums]
        if any(sum(t[j]*nums[j] for j in range(3))>t[3]*den for t in integer):continue
        vertex_count+=1;p=F(nums[2],den)
        if best is None or (p>best[0] if positive else p<best[0]):
            best=(p,tuple(F(v,den) for v in nums),ids)
    assert best is not None,'identity transform must be feasible'
    p,x,ids=best
    return dict(width=p-F(c) if positive else F(c)-p,x=x,active_rows=ids,
                active_tags=[tags[i] for i in ids],vertices_examined=vertex_count)

def witness(anchors,target,gain_bounds,epsilon_target,c,x):
    """Build one shared report vector compatible with both extremal worlds."""
    alpha,beta,p=x;amin=F(gain_bounds[0]);a=amin;ap=amin/alpha;b=F(0);bp=-ap*beta
    first=[];second=[];epsilons=[]
    for lo,hi,eps in anchors:
        lo,hi,eps=F(lo),F(hi),F(eps);radius=2*eps/a*alpha
        low=max(lo,(lo-beta-radius)/alpha);high=min(hi,(hi-beta+radius)/alpha)
        assert low<=high
        q=(low+high)/2
        plow=max(lo,alpha*q+beta-radius);phigh=min(hi,alpha*q+beta+radius)
        assert plow<=phigh
        first.append(q);second.append((plow+phigh)/2);epsilons.append(eps)
    first.append(F(c));second.append(p);epsilons.append(F(epsilon_target))
    Y=[(a*q+b+ap*qp+bp)/2 for q,qp in zip(first,second)]
    intervals=[t[:2] for t in anchors]+[target]
    for q,qp,y,eps,(lo,hi) in zip(first,second,Y,epsilons,intervals):
        assert F(lo)<=q<=F(hi) and F(lo)<=qp<=F(hi)
        assert abs(y-a*q-b)<=eps and abs(y-ap*qp-bp)<=eps
    assert F(gain_bounds[0])<=a<=F(gain_bounds[1]) and F(gain_bounds[0])<=ap<=F(gain_bounds[1])
    return dict(first_truth=first,second_truth=second,reports=Y,first_gain=a,second_gain=ap,
                first_bias=b,second_bias=bp,epsilon=epsilons)

def worst_width(anchors,target,gain_bounds,epsilon_target,exact=False):
    start=time.perf_counter();branches=[]
    for positive,c in [(True,target[0]),(False,target[1])]:
        if exact:
            r=solve_branch_exact(anchors,target,gain_bounds,epsilon_target,c,positive)
            r['witness']=witness(anchors,target,gain_bounds,epsilon_target,c,r['x'])
        else:
            rows,tags=rows_for(anchors,target,gain_bounds,epsilon_target,c)
            A=np.array([[float(v) for v in a] for a,b in rows]);b=np.array([float(b) for a,b in rows])
            out=linprog([0,0,-1 if positive else 1],A_ub=A,b_ub=b,bounds=[(None,None)]*3,method='highs')
            if not out.success:raise RuntimeError(out.message)
            r=dict(width=float(out.x[2]-c if positive else c-out.x[2]),x=out.x.tolist(),
                   active_rows=np.flatnonzero(abs(A@out.x-b)<=1e-8).tolist(),dual_marginals=out.ineqlin.marginals.tolist())
        r['direction']='positive' if positive else 'negative';branches.append(r)
    return dict(width=max(r['width'] for r in branches),branches=branches,exact=exact,seconds=time.perf_counter()-start)

def jsonable(value):
    if isinstance(value,F):return str(value)
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    return value
