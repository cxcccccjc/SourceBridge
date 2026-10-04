"""Numerical inference implementations and explicitly named statistical controls.
See group_protocol.json and group_method_contract.md for baseline scope.
"""
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
import sys,itertools,time
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B/'deps'))
import numpy as np
from scipy.optimize import minimize

def group_reports(y,groups):
    return np.column_stack([y[:,groups==g].mean(axis=1) for g in np.unique(groups)])

def second_moment(e,shrink=0.05):
    C=e.T@e/len(e)
    return (1-shrink)*C+shrink*np.diag(np.diag(C))+1e-9*np.eye(C.shape[0])

def nominal_weights(C):
    G=len(C);scale=max(np.trace(C),1e-12);C=C/scale
    sol=minimize(lambda w:float(w@C@w),np.ones(G)/G,jac=lambda w:2*C@w,
        bounds=[(0,1)]*G,constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(G)},
        method='SLSQP',options={'ftol':1e-12,'maxiter':300})
    if not sol.success:raise RuntimeError(sol.message)
    return sol.x/sol.x.sum()

def risk_weights(C,f=1,bound=1.,kind='adaptive',ncal=None,alpha=.05):
    import cvxpy as cp
    G=len(C)
    factor=1.
    if ncal is not None:
        gap=1-np.sqrt(G/ncal)-np.sqrt(2*np.log(1/alpha)/ncal)
        if gap<=0:raise ValueError('Too few calibration points for Gaussian covariance envelope')
        factor=1/gap**2
    M=C*factor
    vals,vecs=np.linalg.eigh(M);root=np.diag(np.sqrt(np.maximum(vals,0)))@vecs.T
    w=cp.Variable(G,nonneg=True);a=cp.Variable(nonneg=True);s=cp.Variable(nonneg=True)
    constraints=[cp.sum(w)==1,cp.norm(root@w,2)<=a]
    # Standard top-f epigraph, O(G) auxiliaries; no subset enumeration.
    constraints.append(cp.sum_largest(w,f)<=s)
    c=0. if kind=='oblivious' else np.sqrt(2/np.pi)
    objective=cp.square(np.sqrt(1-c*c)*a)+cp.square(bound*s+c*a)
    prob=cp.Problem(cp.Minimize(objective),constraints)
    prob.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_feas=1e-9)
    retried=False
    if prob.status!='optimal':
        retried=True
        prob.solve(solver='CLARABEL',tol_gap_abs=1e-7,tol_feas=1e-7,tol_gap_rel=1e-7,max_iter=500)
    if prob.status!='optimal':raise RuntimeError(prob.status)
    weight=np.maximum(w.value,0);weight/=weight.sum()
    actual_a=float(np.linalg.norm(root@weight));actual_s=float(np.sort(weight)[-f:].sum())
    actual=actual_a**2+2*c*bound*actual_a*actual_s+(bound*actual_s)**2
    assert abs(actual-prob.value)<1e-5*max(1.,actual)
    return weight,dict(objective=float(actual),solver_objective=float(prob.value),factor=float(factor),status=prob.status,retried=retried)

def mlni(Y,cal_y,cal_truth,window=16,prior_strength=2.,truth_prior=100.):
    # JSAC 2022 Eq.(14)-(16), numerical tasks only. One fixed task type.
    K=Y.shape[1];e=cal_y-cal_truth[:,None]
    h0=e.mean(axis=0);v0=np.maximum(np.var(e,axis=0),1e-5)
    lam2=v0/max(prior_strength,1e-5);aa=np.ones(K)*prior_strength
    bb=v0*(aa+1);mu=float(cal_truth.mean());tv=max(np.var(cal_truth)*truth_prior,1e-5)
    out=[];max_iter=0;unconverged=0
    for t in range(len(Y)):
        X=Y[max(0,t-window+1):t+1];h=h0.copy();v=v0.copy();z=X.mean(axis=1)
        for iteration in range(150):
            old=np.r_[z,h,v]
            z=(mu/tv+((X-h)/v).sum(axis=1))/(1/tv+(1/v).sum())
            h=(h0/lam2+((X-z[:,None])/v).sum(axis=0))/(1/lam2+len(X)/v)
            v=(bb+.5*((X-z[:,None]-h)**2).sum(axis=0))/(aa+1+len(X)/2)
            v=np.maximum(v,1e-10)
            if np.max(np.abs(np.r_[z,h,v]-old))<1e-6:break
        max_iter=max(max_iter,iteration+1);unconverged+=int(iteration==149)
        out.append(z[-1])
    return np.array(out),dict(max_iterations=max_iter,unconverged_rounds=unconverged)

def crh_blocks(X):
    # TKDE 2016 Sec.3.1.2 default: normalized absolute loss (15),
    # max-normalized log weights, weighted median (16). X: batch, tasks, sources.
    # Temporal window, 100-iteration cap and numerical floors are our adapter.
    z=X.mean(axis=2);active=np.ones(len(X),dtype=bool)
    scale=np.maximum(X.std(axis=2),1e-10)
    order=np.argsort(X,axis=2,kind='stable');ordered=np.take_along_axis(X,order,axis=2)
    for iteration in range(100):
        old=z.copy()
        loss=np.maximum((np.abs(X-z[:,:,None])/scale[:,:,None]).sum(axis=1),1e-10)
        w=np.log(loss.max(axis=1)[:,None]/loss)
        # Perfect equality makes all weights zero. Equal weights is the declared tie rule.
        equal=w.sum(axis=1)<=1e-12;w[equal]=1.
        sw=np.take_along_axis(np.broadcast_to(w[:,None,:],X.shape),order,axis=2)
        crossing=np.cumsum(sw,axis=2)>=.5*w.sum(axis=1)[:,None,None]
        ix=np.argmax(crossing,axis=2)
        nz=np.take_along_axis(ordered,ix[:,:,None],axis=2)[:,:,0]
        z[active]=nz[active]
        active &= np.max(np.abs(nz-old),axis=1)>=1e-6
        if not active.any():break
    return z

def crh(Y,window=16):
    return np.array([crh_blocks(Y[max(0,t-window+1):t+1][None,:,:])[0,-1]
                     for t in range(len(Y))])

def ptet(Y,eta=1000.,decay=.5,p=3):
    # Exact real-arithmetic weighted inference and ICRH update, p=3,d=1 ARIMA.
    # No numerical clipping of divergent ARIMA trajectories; failure stays visible.
    w=np.ones(Y.shape[1]);acc=np.zeros_like(w);coef=np.random.default_rng(31415).uniform(-.05,.05,p)
    out=[];prediction=None;previous_lags=None
    for t,x in enumerate(Y):
        z=x@w/w.sum()
        if prediction is not None:z=.5*(z+prediction)  # alpha=sum(w), Section V-E
        out.append(z)
        if len(out)>=p+2:
            if previous_lags is not None:
                coef-=2*(prediction-z)*previous_lags/(eta*np.sqrt(max(t+1-p,1)))
            lags=np.diff(np.array(out[-(p+1):]))[::-1]
            prediction=z+coef@lags;previous_lags=lags
        acc=decay*acc+(x-z)**2;loss=np.maximum(acc,1e-10)
        w=np.maximum(np.log(loss.sum()/loss),1e-10)
        if not np.isfinite(z) or abs(z)>1e12:
            return np.full(len(Y),np.nan)
    return np.array(out)

def vrpmtd(Y,cal_y,window=4,gamma=.8,sigma=.2,tau=.03,xi=1e-6):
    # IoTJ 2026 scalar numerical core, zero masking noise, all parties responsive.
    # First report initializes once; subsequent delta update exactly as V-D Step 5.
    scale=max(float(np.mean(abs(cal_y))),1e-8);truth=Y[0].mean();out=[truth];acc=np.zeros(Y.shape[1]);losses=[]
    for t in range(1,len(Y)):
        x=Y[t];loss=xi+1-np.exp(-.5*((x-truth)/(sigma*scale))**2)
        losses.append(loss);acc=gamma*acc+loss
        if len(losses)>window:acc-=gamma**window*losses[-window-1]
        total=np.clip(acc.sum(),xi,Y.shape[1]*(xi+1)*sum(gamma**i for i in range(window)))
        w=np.log(total/np.maximum(acc,xi));delta=x-Y[t-1]
        active=np.abs(delta)/scale>tau
        truth+=np.sum(w*delta*active)/w.sum();out.append(truth)
    return np.array(out)

def weighted_huber(U,C,threshold=1.5):
    sd=np.sqrt(np.maximum(np.diag(C),1e-8));w0=1/sd**2
    z=np.median(U,axis=1)
    for _ in range(30):
        r=np.abs((U-z[:,None])/sd)
        w=w0*np.minimum(1,threshold/np.maximum(r,1e-12))
        new=(w*U).sum(axis=1)/w.sum(axis=1)
        if np.max(abs(new-z))<1e-7:break
        z=new
    return z

def fit_rules(cal_y,cal_truth,groups,bound=1.,f=1):
    U=group_reports(cal_y,groups);e=U-cal_truth[:,None];C=second_moment(e)
    rules={'NominalCov':nominal_weights(C)};meta={}
    for name,kind,cal in [('GroupRiskOblivious','oblivious',None),('GroupRiskAdaptive','adaptive',None),('GroupRiskCalibrated','adaptive',len(U))]:
        # Calibrated theorem requires raw sample second moment, not shrinkage.
        cc=e.T@e/len(e) if cal else C
        try:rules[name],meta[name]=risk_weights(cc,f,bound,kind,cal)
        except ValueError as exc:
            if name!='GroupRiskCalibrated':raise
            rules[name]=None;meta[name]=dict(unavailable=str(exc))
    return C,rules,meta

def evaluate_methods(Y,cal_y,cal_truth,groups,C,rules,settings=None):
    settings=settings or {};U=group_reports(Y,groups);out={};times={};meta={}
    for name,w in rules.items():
        start=time.perf_counter();out[name]=U@w if w is not None else np.full(len(U),np.nan);times[name]=time.perf_counter()-start
    start=time.perf_counter();f=max(1,U.shape[1]//5);ordered=np.sort(U,axis=1)
    core=out['GroupRiskCalibrated'];core=np.where(np.isfinite(core),core,np.median(U,axis=1))
    out['GroupRiskGuarded']=np.clip(core,ordered[:,f],ordered[:,-f-1]);times['GroupRiskGuarded']=time.perf_counter()-start+times['GroupRiskCalibrated']
    for name,fn in [
        ('MLNI_JSAC22',lambda:mlni(Y,cal_y,cal_truth,**settings.get('mlni',{}))),
        ('PTET_TDSC25',lambda:ptet(Y,**settings.get('ptet',{}))),
        ('VRPMTD_IoTJ26',lambda:vrpmtd(Y,cal_y,**settings.get('vrpmtd',{}))),
        ('CRH_TKDE16',lambda:crh(Y)),('GroupMean',lambda:U.mean(axis=1)),
        ('GroupMedian',lambda:np.median(U,axis=1)),('GroupHuber',lambda:weighted_huber(U,C))]:
        start=time.perf_counter();v=fn();times[name]=time.perf_counter()-start
        if isinstance(v,tuple):out[name],meta[name]=v
        else:out[name]=v
    return out,times,meta
