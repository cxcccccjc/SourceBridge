"""Four-paper plaintext comparison and separate extended-threat diagnostic."""
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import sys,json,hashlib,time,argparse,platform
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'deps'))
import numpy as np
import spptd_plaintext_core as sp
from group_methods import crh_blocks
from eptd_inference import eptd_crh
from batch_baselines import fetd_batch
R=B/'results';R.mkdir(exist_ok=True)
PFILE=B/'air_quality_baselines_protocol.json'
PREFIX='air_quality_baselines'
ENGINES=['CRH_TKDE2016','EPTD_TIFS2022','FETD_AK_AAAI2023','FETD_D_AAAI2023','arithmetic_mean','median']
def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def objhash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def arrhash(x):return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
def paths(suffix):return R/(PREFIX+suffix)
def grid():
    out=[dict(alpha=0.,spatial_variance=.01,schedule='synchronous')]
    for a in [.1,1.,10.]:
        for v in [.0001,.01]:
            for s in ['synchronous','grouped']:out.append(dict(alpha=a,spatial_variance=v,schedule=s))
    return out

def make_relations(station_indices,hours,coords):
    rad=np.radians(coords);dlat=rad[:,None,0]-rad[None,:,0];dlon=rad[:,None,1]-rad[None,:,1]
    dist=2*6371*np.arcsin(np.sqrt(np.clip(np.sin(dlat/2)**2+np.cos(rad[:,None,0])*np.cos(rad[None,:,0])*np.sin(dlon/2)**2,0,1)))
    near=np.zeros((len(coords),len(coords)),bool)
    for i in range(len(coords)):
        ids=sorted((j for j in range(len(coords)) if j!=i),key=lambda j:(dist[i,j],j))[:2];near[i,ids]=True
    near=near|near.T;si=np.array(station_indices);hh=np.array(hours)
    rel=np.zeros((len(si),len(si)),np.int8)
    temporal=(si[:,None]==si[None,:])&(abs(hh[:,None]-hh[None,:])==1)
    spatial=(hh[:,None]==hh[None,:])&near[si[:,None],si[None,:]]
    assert not (temporal&spatial).any();rel[temporal]=1;rel[spatial]=2
    return rel

def prepare():
    P=json.loads(PFILE.read_text(encoding='utf-8'));assert P['status']=='FROZEN_BEFORE_EXECUTION'
    dependencies=[PFILE,Path(__file__),B/'spptd_plaintext_core.py',B/'group_methods.py',B/'eptd_inference.py',B/'batch_baselines.py',B/P['dataset']['path']]
    source_hashes={str(p.relative_to(B)):digest(p) for p in dependencies}
    source=np.load(B/P['dataset']['path'],allow_pickle=False)
    stations=sorted(source['station_ids'][source['station_split']=='station_train'].tolist())[:8]
    si=[int(np.where(source['station_ids']==s)[0][0]) for s in stations]
    assert all(source['station_split'][i]=='station_train' for i in si)
    coords=source['coordinates'][si];pi=int(np.where(source['pollutants']=='PM25_Concentration')[0][0])
    inputs={};truths={};cases=[];days=[]
    for di,day in enumerate(P['dataset']['day_blocks']):
        indices=[];hours=[];station_indices=[];q=[];keys=[];missing=[]
        for hour in range(24):
            ts=np.datetime64(f'{day}T{hour:02}:00:00','s');ti=int(np.where(source['timestamps']==ts)[0][0])
            assert source['temporal_split'][ti]=='train'
            for local_i,source_i in enumerate(si):
                key=stations[local_i]+'|'+str(ts)
                if source['reference_observed'][source_i,ti,pi]:
                    hours.append(hour);station_indices.append(local_i);q.append(float(source['reference_values'][source_i,ti,pi]));keys.append(key)
                else:missing.append(key)
        q=np.array(q);hours=np.array(hours);station_indices=np.array(station_indices)
        ref=np.flatnonzero(hours<6);ev=np.flatnonzero(hours>=6)
        rank=sorted(ref,key=lambda j:hashlib.sha256(('spptd-msra-paid-reference-v1:'+keys[j]).encode()).hexdigest())
        rel=make_relations(station_indices,hours,coords)
        local=(station_indices<2)&(hours>=12)&(hours<=15)
        neighbor=np.any(rel[:,local]>0,axis=1)&~local&(hours>=6)
        far=(hours>=6)&~local&~neighbor
        days.append(dict(day=day,stations=stations,grid_count=192,available=len(q),missing_object_ids=missing,
            reference_phase_count=len(ref),evaluation_count=len(ev),reference_ids={str(k):[keys[j] for j in rank[:k]] for k in [3,12]},
            temporal_directed_edges=int((rel==1).sum()),spatial_directed_edges=int((rel==2).sum()),
            localized_regions=dict(attacked=int(local.sum()),neighbor=int(neighbor.sum()),far=int(far.sum()))))
        for seed in P['design']['seeds']:
            rng=np.random.default_rng(seed+1000*di)
            sigma=rng.uniform(.03,.25,12);gain=rng.uniform(.7,1.3,12);bias=rng.uniform(0,20,12);noise=rng.uniform(-1,1,(12,len(q)))
            homogeneous=q[None,:]+sigma[:,None]*q[None,:]*noise
            heterogeneous=gain[:,None]*q[None,:]+bias[:,None]+sigma[:,None]*q[None,:]*noise
            variants={'homogeneous_honest':homogeneous,'heterogeneous_honest':heterogeneous}
            coordinated=heterogeneous.copy();coordinated[-3:,ev]+=80;variants['heterogeneous_coordinated']=coordinated
            localized=heterogeneous.copy();localized[-3:,local]+=80;variants['heterogeneous_localized']=localized
            for scenario,X in variants.items():
                key=f'd{di}_s{seed}_{scenario}'
                valid=len(ref)>=12 and np.isfinite(X).all() and np.all(X>=0) and np.all(X<2048)
                inputs[key+'__reports']=X;inputs[key+'__relations']=rel;inputs[key+'__reference_phase']=ref;inputs[key+'__evaluation']=ev
                inputs[key+'__hours']=hours;inputs[key+'__station_indices']=station_indices;inputs[key+'__object_ids']=np.array(keys)
                inputs[key+'__local_region']=local;inputs[key+'__neighbor_region']=neighbor;inputs[key+'__far_region']=far
                for k in [3,12]:
                    chosen=np.array(rank[:k],dtype=int);inputs[key+f'__reference_indices_{k}']=chosen;inputs[key+f'__reference_labels_{k}']=q[chosen]
                truths[key+'__truth']=q
                cases.append(dict(key=key,day=day,day_index=di,seed=seed,scenario=scenario,status='ready' if valid else 'input_contract_failed',
                    objects=len(q),reference_phase_objects=len(ref),evaluation_objects=len(ev),report_sha256=arrhash(X),
                    reference_reports_sha256=arrhash(X[:,ref]),public_scale=2048.,report_min=float(X.min()),report_max=float(X.max()),
                    source_gain=([1.]*12 if scenario=='homogeneous_honest' else gain.tolist()),source_bias=([0.]*12 if scenario=='homogeneous_honest' else bias.tolist()),
                    source_sigma=sigma.tolist(),attack_source_indices=[] if scenario.endswith('honest') else [9,10,11],
                    changed_report_count=int(np.count_nonzero(X-heterogeneous)) if not scenario.endswith('honest') else 0))
    assert len(cases)==24
    np.savez_compressed(paths('_inputs.npz'),**inputs);np.savez_compressed(paths('_evaluator_truth.npz'),**truths)
    manifest=dict(scope=P['purpose'],cases=cases,day_ledgers=days,source_hashes=source_hashes,input_hash=digest(paths('_inputs.npz')),
        evaluator_hash=digest(paths('_evaluator_truth.npz')),created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        environment=dict(platform=platform.platform(),processor=platform.processor(),python=sys.version,numpy=np.__version__,threads=1),
        external_validation_test_data_used=False,development_only=True)
    dump(paths('_manifest.json'),manifest);print(json.dumps(dict(cases=24,ready=sum(c['status']=='ready' for c in cases),days=days),indent=2))

def crh_trace(X):
    z=X.mean(axis=1);scale=np.maximum(X.std(axis=1),1e-10);order=np.argsort(X,axis=1,kind='stable');ordered=np.take_along_axis(X,order,axis=1)
    for it in range(500):
        old=z.copy();loss=np.maximum((np.abs(X-z[:,None])/scale[:,None]).sum(axis=0),1e-10)
        w=np.log(loss.max()/loss)
        if w.sum()<=1e-12:w=np.ones(X.shape[1])
        sw=np.take_along_axis(np.broadcast_to(w,X.shape),order,axis=1)
        ix=np.argmax(np.cumsum(sw,axis=1)>=.5*w.sum(),axis=1);z=ordered[np.arange(len(X)),ix]
        delta=float(np.max(abs(z-old)))
        if delta<1e-6:return z,dict(status='converged',iterations=it+1,last_delta=delta)
    return z,dict(status='max_iterations',iterations=500,last_delta=delta)

def base_predict(name,reports):
    begin=time.perf_counter();X=reports.T/2048
    if name=='CRH_TKDE2016':z,meta=crh_trace(X)
    elif name=='EPTD_TIFS2022':
        z,m=eptd_crh(X,max_iterations=500,tolerance=1e-6);meta=dict(status='converged' if m['converged'] else 'max_iterations',iterations=m['iterations'])
    elif name.startswith('FETD_'):z=fetd_batch(X,name.split('_')[1]);meta=dict(status='closed_form',iterations=0)
    elif name=='arithmetic_mean':z=X.mean(axis=1);meta=dict(status='closed_form',iterations=0)
    elif name=='median':z=np.median(X,axis=1);meta=dict(status='closed_form',iterations=0)
    else:raise KeyError(name)
    return z*2048,{**meta,'inference_seconds':time.perf_counter()-begin}

def sp_predict(reports,relations,cfg,mode='mixed',reference_reports=None,reference_relations=None,scale=2048.,floor=False):
    begin=time.perf_counter();rel=relations.copy()
    if mode=='temporal_only':rel[rel==2]=0
    if mode=='spatial_only':rel[rel==1]=0
    config=sp.PlaintextConfig(alpha=cfg['alpha'],temporal_degree=1,spatial_variance=cfg['spatial_variance'],schedule=cfg['schedule'],
        max_iterations=500,absolute_tolerance=1e-6,zero_residual_policy='floor' if floor else 'strict',residual_floor=1e-10 if floor else None)
    groups=sp.independent_groups(rel) if cfg['schedule']=='grouped' else None
    constants=None;original_kernel=sp._kernel
    if mode=='static':
        z=reference_reports.mean(axis=0)/scale;constants={}
        for kind in [1,2]:
            ii,jj=np.where(reference_relations==kind)
            values=z[ii]*z[jj] if kind==1 else np.exp(-(z[ii]-z[jj])**2/(2*cfg['spatial_variance']))
            constants[kind]=float(values.mean()) if len(values) else 0.
        def static_kernel(j,neighbors,state,relations,config):return np.array([constants[int(k)] for k in relations[j,neighbors]])
        sp._kernel=static_kernel
    try:
        observed=np.ones(reports.shape,dtype=bool);x=sp.normalize_by_public_max(reports,observed,scale)
        result=sp.run_spptd_plaintext(x,observed,rel,config=config,groups=groups)
    finally:sp._kernel=original_kernel
    prediction=None if result['truth'] is None else result['truth']*scale
    history=result['history'];last=history[-1] if history else None
    meta=dict(status=result['status'],iterations=result['iterations'],reason=result['reason'],config=cfg,mode=mode,scale=scale,
        inference_seconds=time.perf_counter()-begin,static_constants=constants,
        last_delta=None if last is None else last['l1_change'],
        last_weight_delta=None if last is None else float(np.max(abs(last['weights_next']-last['weights_used']))),
        regularized_source_updates=sum(h['regularized_source_count'] for h in history))
    trace={}
    if history:
        trace={name:np.stack([h[name] for h in history]) for name in ['corrected_truth','weights_used','weights_next','source_squared_residuals','kernel_masses','correction_denominators']}
        trace['corrected_truth']*=scale
    return prediction,meta,trace

def affine(z,q):
    zm=float(np.mean(z));qm=float(np.mean(q));v=float(np.sum((z-zm)**2))
    a=1. if v<=1e-14 else (float(np.sum((z-zm)*(q-qm)))+.01*v)/(1.01*v)
    return float(a),float(qm-a*zm)

def select_candidates(predictions,reference_local,labels):
    scores=[]
    for index,pred in enumerate(predictions):
        if pred is None:scores.append(dict(index=index,raw_mse=None,loo_mse=None,folds=[]));continue
        z=pred[reference_local];folds=[]
        for k in range(len(labels)):
            keep=np.arange(len(labels))!=k;a,b=affine(z[keep],labels[keep]);value=a*z[k]+b
            folds.append(dict(held=k,prediction=float(value),loss=float((value-labels[k])**2),a=a,b=b))
        scores.append(dict(index=index,raw_mse=float(np.mean((z-labels)**2)),loo_mse=float(np.mean([f['loss'] for f in folds])),folds=folds))
    def choose(field):
        valid=[s for s in scores if s[field] is not None]
        return min(valid,key=lambda s:(s[field],s['index']))['index'] if valid else None
    return scores,choose('raw_mse'),choose('loo_mse')

def smoke():
    X=np.array([[.11,.13,.12,.16],[.2,.25,.24,.19],[.4,.42,.38,.44],[.6,.57,.62,.65]])
    ours,meta=crh_trace(X);existing=crh_blocks(X[None])[0]
    assert np.max(abs(ours-existing))<1e-12
    return dict(CRH_existing_max_difference=float(np.max(abs(ours-existing))),note='Only changed tracing wrapper checked; unmodified EPTD/FETD called directly')

def run():
    manifest=json.loads(paths('_manifest.json').read_text(encoding='utf-8'))
    for filename,h in manifest['source_hashes'].items():assert verify_identity(B/filename,h),(filename,'changed after source freeze')
    assert digest(paths('_inputs.npz'))==manifest['input_hash'];data=np.load(paths('_inputs.npz'),allow_pickle=False)
    adapter_check=smoke();preds={};traces={};records=[];selections=[];phase_cache={};full_cache={};grid_configs=grid();ref_predictions={};phase_metadata={};run_start=time.perf_counter()
    # Evaluator truth is deliberately not opened until every method finishes.
    for case_index,case in enumerate(manifest['cases']):
        key=case['key']
        if case['status']!='ready':records.append(dict(key=key,status=case['status']));continue
        X=data[key+'__reports'];rel=data[key+'__relations'];phase=data[key+'__reference_phase'];PX=X[:,phase];PR=rel[np.ix_(phase,phase)]
        phase_id=arrhash(PX)+arrhash(PR)
        if phase_id not in phase_cache:
            bp={};bm={}
            for method in ENGINES:bp[method],bm[method]=base_predict(method,PX)
            cp=[];cm=[]
            for cfg in grid_configs:
                p,meta,tr=sp_predict(PX,PR,cfg);cp.append(p);cm.append(meta)
            phase_cache[phase_id]=(bp,bm,cp,cm)
            for method,p in bp.items():ref_predictions[phase_id+'__'+method]=p
            for index,p in enumerate(cp):
                if p is not None:ref_predictions[phase_id+f'__SPPTD_candidate{index}']=p
            phase_metadata[phase_id]=dict(first_case_key=key,base_metadata=bm,SPPTD_grid_metadata=cm)
        bp,bm,cp,cm=phase_cache[phase_id]
        for budget in [3,12]:
            condition=key+f'__K{budget}';refs=data[key+f'__reference_indices_{budget}'];labels=data[key+f'__reference_labels_{budget}']
            local=np.array([int(np.where(phase==j)[0][0]) for j in refs]);tic=time.perf_counter()
            scores,native_index,post_index=select_candidates(cp,local,labels)
            selections.append(dict(condition=condition,reference_indices=refs.tolist(),reference_labels=labels.tolist(),reference_phase_id=phase_id,
                native_selected=native_index,post_selected=post_index,scores=scores,grid_metadata=cm,selection_seconds=time.perf_counter()-tic))
            def get_sp(index,mode='mixed',scale=2048.,floor=False):
                cachekey=(key,index,mode,scale,floor)
                if cachekey not in full_cache:
                    full_cache[cachekey]=sp_predict(X,rel,grid_configs[index],mode=mode,reference_reports=PX,reference_relations=PR,scale=scale,floor=floor)
                return full_cache[cachekey]
            def save(name,pred,meta,category,trace=None):
                outkey=condition+'__'+name
                record=dict(condition=condition,key=key,budget=budget,method=name,category=category,**meta)
                if pred is not None:
                    preds[outkey]=np.asarray(pred);record['prediction_key']=outkey;record['prediction_sha256']=arrhash(pred)
                if trace:
                    for field,value in trace.items():traces[outkey+'__'+field]=value
                    record['trace_prefix']=outkey
                records.append(record)
            for method in ENGINES:
                cachekey=(key,method)
                if cachekey not in full_cache:full_cache[cachekey]=base_predict(method,X)
                p,meta=full_cache[cachekey];save(method+'__native',p,meta,'paper_core' if method not in ['arithmetic_mean','median'] else 'statistical_control')
                a,b=affine(p[refs],labels);save(method+'__post_affine',a*p+b,{**meta,'affine_a':a,'affine_b':b},'shared_calibration_adapter')
            for tag,index in [('native',native_index),('post_affine',post_index)]:
                if index is None:save('SPPTD__'+tag,None,dict(status='all_candidates_undefined'),'paper_core');continue
                p,meta,tr=get_sp(index);meta={**meta,'selected_index':index}
                if tag=='post_affine' and p is not None:
                    a,b=affine(p[refs],labels);p=a*p+b;meta.update(affine_a=a,affine_b=b)
                save('SPPTD__'+tag,p,meta,'paper_core' if tag=='native' else 'shared_calibration_adapter',trace=tr if tag=='native' else None)
            # Fixed paper interpretations and component controls never become extra papers.
            for label,index in [('SPPTD_fixed_variance_0.0001',1),('SPPTD_fixed_variance_0.01',3),('SPPTD_alpha0',0)]:
                p,meta,tr=get_sp(index);save(label,p,meta,'paper_parameter_interpretation' if index else 'component_ablation',trace=tr if index==0 else None)
            if native_index is not None:
                for mode in ['temporal_only','spatial_only','static']:
                    p,meta,tr=get_sp(native_index,mode);save('SPPTD_'+mode,p,meta,'component_ablation',trace=tr if mode=='static' else None)
                p,meta,tr=get_sp(native_index,scale=4096.);save('SPPTD_scale4096',p,meta,'unit_sensitivity')
                p,meta,tr=get_sp(native_index)
                if meta['status']=='undefined' and meta.get('reason') and 'zero-residual' in meta['reason']:
                    fp,fm,ft=get_sp(native_index,floor=True);save('SPPTD_regularized_sensitivity',fp,fm,'declared_floor_sensitivity',trace=ft)
        print(json.dumps(dict(cases_completed=case_index+1,total=24,key=key,unique_reference_phases=len(phase_cache),native_fits_cached=len(full_cache))),flush=True)
        dump(paths('_run_progress.json'),dict(cases_completed=case_index+1,records=len(records)))
    np.savez_compressed(paths('_predictions.npz'),**preds);np.savez_compressed(paths('_traces.npz'),**traces)
    np.savez_compressed(paths('_reference_predictions.npz'),**ref_predictions);dump(paths('_reference_phases.json'),phase_metadata)
    # Commit selections and predictions before opening evaluator-only values.
    dump(paths('_selections.json'),selections)
    dump(paths('_runs.json'),dict(completed=True,manifest_hash=digest(paths('_manifest.json')),adapter_check=adapter_check,records=records,
        prediction_hash=digest(paths('_predictions.npz')),trace_hash=digest(paths('_traces.npz')),selections_hash=digest(paths('_selections.json')),
        reference_prediction_hash=digest(paths('_reference_predictions.npz')),reference_phase_metadata_hash=digest(paths('_reference_phases.json')),
        run_wall_seconds=time.perf_counter()-run_start,fit_cache_policy='Reuse bitwise-identical reference phases and identical final configurations; recorded per-fit timings are not an uncached full job total'))
    print(json.dumps(dict(completed=True,records=len(records),predictions=len(preds),trace_arrays=len(traces))))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);arg=parser.parse_args()
    prepare() if arg.mode=='prepare' else run()
