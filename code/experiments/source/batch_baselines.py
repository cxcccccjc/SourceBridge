from release_integrity import verify_identity, resolve_resource
"""Static batch wrappers of already source-verified numerical paper equations.

CRH TKDE2016, FETD AAAI2023, EPTD TIFS2022 squared CRH, MLNI JSAC2022.
No time order or privacy protocol is fabricated. Shared calibration is added
explicitly outside the paper cores, not attributed to their original authors.
"""
import numpy as np
from group_methods import crh_blocks
from eptd_inference import eptd_crh

def affine_fit(x,q,ridge=.01):
    x=np.asarray(x);q=np.asarray(q);xm=x.mean(axis=0);qm=q.mean();xc=x-xm
    v=np.sum(xc*xc,axis=0);cov=np.sum(xc*(q-qm).reshape((-1,)+(1,)*(x.ndim-1)),axis=0)
    a=np.divide(cov+ridge*v,(1+ridge)*v,out=np.ones_like(v),where=v>1e-14)
    return a,qm-a*xm

def source_transform(all_reports,cal_reports,cal_truth,kind):
    if kind=='raw':return all_reports.copy()
    if kind=='offset':return all_reports-(cal_reports-cal_truth[:,None]).mean(axis=0)
    if kind=='individual_affine':
        a,b=affine_fit(cal_reports,cal_truth);return all_reports*a+b
    raise ValueError(kind)

def fetd_batch(X,variant):
    G=X.shape[1];sq=(X*X).mean(axis=0);D=np.maximum(sq[:,None]+sq[None,:]-2*X.T@X/len(X),0.)
    np.fill_diagonal(D,0.);p=D.sum(axis=1)/(G-1)
    if variant=='D':w=np.maximum(p.max()-p,0.)
    elif variant=='AK':w=1/np.maximum(p-G/(G-1)*p.mean()/2,1e-8)
    else:raise ValueError(variant)
    if w.sum()<=1e-12:w=np.ones(G)
    return X@(w/w.sum())

def mlni_batch(X,C,q,prior_strength=2.,truth_prior=100.):
    """JSAC Eq14-16 over one simultaneous unlabeled batch, fixed task type.

    Priors use only paid reference pairs. Task order is not interpreted as
    observation time. Prediction includes all reference/test reports;
    reference labels set priors, not hard-clamped iterates.
    """
    e=C-q[:,None];h0=e.mean(axis=0);v0=np.maximum(np.var(e,axis=0),1e-5)
    lam2=v0/prior_strength;aa=np.full(X.shape[1],prior_strength);bb=v0*(aa+1)
    mu=float(q.mean());tv=max(float(np.var(q))*truth_prior,1e-5)
    h=h0.copy();v=v0.copy();z=X.mean(axis=1)
    for it in range(150):
        old=np.r_[z,h,v];z=(mu/tv+((X-h)/v).sum(axis=1))/(1/tv+(1/v).sum())
        h=(h0/lam2+((X-z[:,None])/v).sum(axis=0))/(1/lam2+len(X)/v)
        v=(bb+.5*((X-z[:,None]-h)**2).sum(axis=0))/(aa+1+len(X)/2);v=np.maximum(v,1e-10)
        if np.max(abs(np.r_[z,h,v]-old))<1e-6:break
    return z,dict(iterations=it+1,converged=it<149)

def paper_predict(name,X,C,q,parameters=None):
    if name=='CRH_TKDE2016':return crh_blocks(X[None,:,:])[0],{}
    if name=='EPTD_TIFS2022_squared_CRH':return eptd_crh(X)
    if name.startswith('FETD_'):return fetd_batch(X,name.split('_')[1]),{}
    if name=='MLNI_JSAC2022':return mlni_batch(X,C,q,**(parameters or {}))
    raise ValueError(name)
