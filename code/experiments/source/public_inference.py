"""Proper-prior MLNI inference on a later public time block."""
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,hashlib,time,math,csv,zipfile,io,argparse,datetime
from fractions import Fraction as F
from collections import Counter,defaultdict
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'deps'))
import numpy as np
import public_source_generation as old
from weighted_inference import weighted_adapt
from certified_geometry import exact_packet_fusion
from safe_projection import project
from reference_geometry import jsonable
N='public_inference';R=B/'results';PF=B/(N+'_protocol.json')
PROPER='MLNI_JSAC2022__proper_public_WLS';COMPOSED='SourceBridge_proper_MLNI_WLS';LEGACY='MLNI_JSAC2022__weighted_source_affine'
def path(s):return R/(N+s)
def rd(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(jsonable(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def freeze():
    assert not PF.exists(),'Preserve frozen protocol';old.verify();base=rd(old.PF);files=[Path(__file__).name,N+'_protocol.md','mlni_prior_specification.md',
        'public_source_generation.py','public_source_generation_protocol.json']+list(base['source_hashes'])
    p={k:v for k,v in base.items() if k not in ['source_hashes','status','scope','dataset','dataset_sha','days','nominal']}
    p.update(status='FROZEN_BEFORE_NEW_TIME_BLOCK_VALUE_EXTRACTION',scope='Later public time holdout, previously seen stations; semi-synthetic sources and intervals',
        days=[(datetime.date(2014,5,29)+datetime.timedelta(days=i)).isoformat() for i in range(28)],nominal=PROPER,
        dataset='results/'+N+'_extracted_data.npz',dataset_sha='fixed_after_frozen_extraction_in_extracted_manifest',
        archive='data/msra_urban_air/Data-1.zip',archive_sha='f72e5c162bd86da145147dad62d8d039b237a9a48b1a5f72375ed8734ffbe873',
        extraction='Only fixed target station/hour12 and anchor station/hours11,13 on 28 declared dates. Preserve missing, conflict, negatives and domain violations. No value selection.',
        comparison=dict(legacy_interfaces='All original 13 interfaces retained',proper_nominal=PROPER,proper_composed=COMPOSED),
        proper_prior=dict(center='midpoint public T',scale='width public T',truth_mean=0.,truth_variance=1/12,shape=2.,bias_strength=2.,
            variance_floor='max(eps_target^2/(3*a_hat^2*s^2), mean_j((eps_j^2/a_hat^2+h_j^2)/(3*s^2)))',
            residual_variance='population variance after unchanged WLS and public-domain normalization',initial_variance='max(residual_variance,variance_floor)',
            variance_prior_scale='3*initial_variance',bias_prior_mean='mean calibration residual',bias_prior_variance='initial_variance/2',
            all_K='same rule, no CV/grid',cap=20000,tolerance=1e-6,machine_guard=1e-10),
        legacy_source_seed_preserved=True,legacy_source_seed_namespace='SourceBridge-public-confirmation-v1:',
        source_hashes={f:sha(B/f) for f in sorted(set(files))})
    save(PF,p);print(json.dumps(dict(frozen=True,protocol_sha=sha(PF))),flush=True)
def verify():
    p=rd(PF)
    for name,h in p['source_hashes'].items():assert verify_identity(B/name,h),name
    if path('_extracted_manifest.json').exists():
        dm=rd(path('_extracted_manifest.json'));assert dm['protocol_sha']==sha(PF);assert sha(resolve_resource(B/p['dataset']))==dm['dataset_sha'];p['dataset_sha']=dm['dataset_sha']
    return p
def configure():old.N=N;old.PF=PF;old.verify=verify
def extract():
    p=verify();assert not path('_extracted_manifest.json').exists();assert sha(B/p['archive'])==p['archive_sha'];dates=set(p['days']);target=set(p['target_stations']);anchors=set(p['anchor_stations']);ids=sorted(target|anchors)
    times=[day+f'T{hour:02}:00:00' for day in p['days'] for hour in [11,12,13]];imap={v:i for i,v in enumerate(ids)};tmap={v:i for i,v in enumerate(times)};values=defaultdict(set);raw=Counter();invalid=Counter();selected_records=0;total=0
    with zipfile.ZipFile(B/p['archive']) as z,z.open('Data/airquality.csv') as binary,io.TextIOWrapper(binary,encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            total+=1;station=r['station_id'];stamp=r['time']
            if station not in imap or stamp[:10] not in dates:continue
            hour=int(stamp[11:13]);wanted=(station in target and hour==12) or (station in anchors and hour in [11,13])
            if not wanted:continue
            assert stamp[13:]==':00:00';key=(station,stamp.replace(' ','T'));raw[key]+=1;selected_records+=1;token=r[p['pollutant']].strip()
            if token in ['', 'NULL']:invalid['missing_marker']+=1;continue
            try:v=float(token)
            except ValueError:invalid['nonnumeric']+=1;continue
            if not math.isfinite(v):invalid['nonfinite']+=1;continue
            values[key].add(v)
    arr=np.full((len(ids),len(times),1),np.nan);mask=np.zeros(arr.shape,dtype=bool);conflict=np.zeros(arr.shape,dtype=bool);rows=np.zeros(arr.shape[:2],dtype=int)
    for key,count in raw.items():
        i,j=imap[key[0]],tmap[key[1]];rows[i,j]=count;vv=values[key]
        if len(vv)==1:arr[i,j,0]=next(iter(vv));mask[i,j,0]=True
        elif len(vv)>1:conflict[i,j,0]=True
    np.savez_compressed(resolve_resource(B/p['dataset']),reference_values=arr,reference_observed=mask,reference_conflict=conflict,raw_row_count=rows,
        station_ids=np.array(ids),timestamps=np.array(times,dtype='datetime64[s]'),pollutants=np.array([p['pollutant']]),station_split=np.array(['station_test' if i in target else 'station_train' for i in ids]),temporal_split=np.array(['test']*len(times)))
    dm=dict(protocol_sha=sha(PF),archive_sha=p['archive_sha'],dataset_sha=sha(resolve_resource(B/p['dataset'])),time_block=[p['days'][0],p['days'][-1]],public_targets_planned=224,anchor_observations_planned=168,
        archive_records_scanned=total,selected_records=selected_records,observed_numeric=int(mask.sum()),conflicted=int(conflict.sum()),negative=int(np.sum(arr[mask]<0)),out_of_domain=int(np.sum((arr[mask]<0)|(arr[mask]>500))),invalid=dict(invalid),
        unit='station-time target, not worker',scope=p['scope'],target_stations=p['target_stations'],anchor_stations=p['anchor_stations'])
    save(path('_extracted_manifest.json'),dm);print(json.dumps(dm),flush=True)
def prepare():configure();old.prepare()
def plans():configure();old.plans()
def fulfill():configure();old.fulfill()
def proper_mlni(Y,A,q,p):
    start=time.perf_counter();X=Y.T;h=(A[:,1]-A[:,0])/2;eps=A[:,2];ref=np.arange(len(q));tx,am=weighted_adapt(X,q,ref,h,eps)
    center=(p['target'][0]+p['target'][1])/2;scale=p['target'][1]-p['target'][0];assert scale>0;x=(tx-center)/scale;qq=(q-center)/scale;n,g=x.shape
    a_hat=np.asarray(am['gain']);residual=x[ref]-qq[:,None];h0=residual.mean(axis=0);empirical=residual.var(axis=0)
    target_nominal=(p['epsilon']**2)/(3*a_hat*a_hat*scale*scale)
    reference_nominal=np.mean((eps[:,None]**2/(a_hat[None,:]**2)+h[:,None]**2)/(3*scale*scale),axis=0)
    floor=np.maximum(target_nominal,reference_nominal);v0=np.maximum(empirical,floor);hvar=v0/2;prior_a=np.full(g,2.);prior_b=3*v0;truthvar=1/12;mu=0.
    hfit=h0.copy();v=v0.copy();z=x.mean(axis=1);sumx=x.sum(axis=0);guard_count=0;maxrise=0.
    def objective():
        r=x-z[:,None]-hfit[None,:]
        return float(.5*np.sum((z-mu)**2)/truthvar+.5*np.sum((hfit-h0)**2/hvar)+np.sum((prior_a+1+n/2)*np.log(v)+(prior_b+.5*np.sum(r*r,axis=0))/v))
    initial=previous=objective()
    for step in range(1,20001):
        oz=z.copy();oh=hfit.copy();ov=v.copy();precision=1/v
        z=(x@precision-np.dot(hfit,precision)+mu/truthvar)/(precision.sum()+1/truthvar)
        hfit=(h0/hvar+(sumx-z.sum())*precision)/(1/hvar+n*precision)
        residual_fit=x-z[:,None]-hfit[None,:];candidate=(prior_b+.5*np.sum(residual_fit*residual_fit,axis=0))/(prior_a+1+n/2)
        guard_count+=int(np.sum(candidate<1e-10));v=np.maximum(candidate,1e-10)
        delta=max(float(np.max(abs(z-oz))),float(np.max(abs(hfit-oh))),float(np.max(abs(v-ov))))
        now=objective();maxrise=max(maxrise,now-previous);previous=now
        if delta<1e-6:break
    out=z*scale+center;assert np.isfinite(out).all()
    meta=dict(status='converged' if delta<1e-6 else 'max_iterations',converged=delta<1e-6,iterations=step,max_delta=delta,objective=previous,initial_objective=initial,max_objective_increase=maxrise,
        normalizer_center=center,normalizer_scale=scale,truth_prior_mean=mu,truth_prior_variance=truthvar,source_prior_bias_mean=h0.tolist(),empirical_residual_variance=empirical.tolist(),
        target_nominal_variance=target_nominal.tolist(),reference_nominal_variance=reference_nominal.tolist(),variance_floor=floor.tolist(),initial_source_variance=v0.tolist(),source_prior_bias_variance=hvar.tolist(),
        inverse_gamma_shape=prior_a.tolist(),inverse_gamma_scale=prior_b.tolist(),final_source_bias=hfit.tolist(),final_source_variance=v.tolist(),machine_guard_count=guard_count,adapter=am)
    sec=time.perf_counter()-start
    return dict(label=PROPER,method='MLNI_JSAC2022',adapter='proper_public_WLS',prediction=float(out[-1]),projected_prediction=float(np.clip(out[-1],0,500)),whole_prediction=out.tolist(),
        status=meta['status'],metadata=meta,selected_config='fixed_public_information_prior_no_CV',inference_times=[sec],inference_seconds=sec,selection_seconds=0.)
def predict():
    p=verify();assert not path('_runs.json').exists();m=rd(path('_purchased_manifest.json'));assert sha(path('_purchased_inputs.npz'))==m['input_sha'];original=np.load;loaded=[]
    def guard(file,*a,**kw):
        assert Path(file).resolve()==path('_purchased_inputs.npz').resolve();loaded.append(str(Path(file).resolve()));return original(file,*a,**kw)
    np.load=guard
    try:
        d=np.load(path('_purchased_inputs.npz'),allow_pickle=False);bundles={};rows=[];cache=B/'reading'/N;cache.mkdir(exist_ok=True);tic=time.perf_counter()
        for ci,c in enumerate(m['cases']):
            key=c['key'];Y=d[key+'__reports'];A=d[key+'__anchors'];q=d[key+'__centers'];tag=hashlib.sha256(Y.tobytes()+A.tobytes()+q.tobytes()+sha(PF).encode()).hexdigest();cp=cache/(tag+'.json')
            if tag not in bundles:
                if cp.exists():item=rd(cp)
                else:
                    fit=old.engine.fit(Y,A,q,timing=False);proper=proper_mlni(Y,A,q,p);fit['records'].append(proper)
                    t=time.perf_counter();exact=exact_packet_fusion(Y,A.tolist(),p['target'],p['gain_bounds'],5.,2);cert_sec=time.perf_counter()-t;projection={};prsec={}
                    for name in [LEGACY,PROPER]:
                        rec=next(r for r in fit['records'] if r['label']==name);t=time.perf_counter();z=rec['projected_prediction'];projection[name]=project(z,*exact['interval'],F(c['width']),p['target']) if exact['feasible'] else dict(status='no_certificate',prediction=None);prsec[name]=time.perf_counter()-t
                    item=dict(fit=fit,exact=exact,projections=projection,projection_seconds=prsec,extra_certificate_seconds=cert_sec);save(cp,item);item=rd(cp)
                bundles[tag]=item
            item=bundles[tag];rows.extend([dict(**c,fit_key=tag,**r) for r in item['fit']['records']])
            for label,nominal in [('SourceBridge_MLNI_WLS',LEGACY),(COMPOSED,PROPER)]:
                pr=item['projections'][nominal];rows.append(dict(**c,fit_key=tag,label=label,method='complete_composition',adapter='fixed_published_nominal_plus_certificate',status=pr['status'],prediction=pr.get('prediction'),projected_prediction=pr.get('prediction'),metadata=pr))
            if (ci+1)%128==0:print(json.dumps(dict(predicted=ci+1,total=len(m['cases']),unique=len(bundles),seconds=time.perf_counter()-tic)),flush=True)
        save(path('_runs.json'),dict(protocol_sha=sha(PF),input_sha=m['input_sha'],numeric_loads=loaded,rows=rows,bundles=bundles,total_seconds=time.perf_counter()-tic))
    finally:np.load=original
def evaluate():
    p=verify();m=rd(path('_manifest.json'));pm=rd(path('_purchased_manifest.json'));run=rd(path('_runs.json'));plans=rd(path('_plans.json'));assert run['input_sha']==sha(path('_purchased_inputs.npz'));assert sha(path('_evaluator_truth.json'))==m['truth_sha']
    truth=rd(path('_evaluator_truth.json'));contracts={(v['event'],v['profile']):v for v in rd(path('_contracts.json'))};rows_by=defaultdict(list)
    for r in run['rows']:rows_by[r['key']].append(r)
    cells=[];safety=[]
    for c in pm['cases']:
        rr=rows_by[c['key']];item=run['bundles'][rr[0]['fit_key']];ex=item['exact'];q=F(truth[c['event']]);w=F(c['width']);contract=contracts[(c['event'],c['profile'])];valid=all(contract[k] for k in ['coarse_valid','honest_report_valid','target_valid'])
        for nominal in [LEGACY,PROPER]:
            pr=item['projections'][nominal];check=dict(**c,nominal=nominal,condition_valid=valid,certificate_feasible=ex['feasible'],projection_defined=pr['status']=='defined',changed=pr.get('changed_exact',False),movement=pr.get('movement'))
            if ex['feasible']:
                lo,hi=map(F,ex['interval']);check.update(contains_truth=lo<=q<=hi,width_bound=hi-lo<=w,posterior_width=float(hi-lo))
            if pr['status']=='defined':check.update(exact_point_bound=abs(F(pr['exact_point'])-q)<=w/2,float_radius_covers_truth=abs(F(pr['prediction'])-q)<=F(pr['float_radius_upper']),exported_radius=pr['float_radius_upper'])
            check['conditional_failure']=valid and not all(check.get(k,False) for k in ['certificate_feasible','projection_defined','contains_truth','width_bound','exact_point_bound','float_radius_covers_truth']);safety.append(check)
        for r in rr:
            z=r.get('projected_prediction');raw=r.get('prediction');cells.append(dict(**{k:r[k] for k in ['event','base_key','pool','day','station','profile','key','attack','family','main_library','label','status','total_cost','selected_count']},truth=float(q),prediction=z,raw_prediction=raw,condition_valid=valid,
                squared_error=None if z is None else (z-float(q))**2,absolute_error=None if z is None else abs(z-float(q)),raw_squared_error=None if raw is None else (raw-float(q))**2))
    summary=old.aggregate(cells,['profile','label']);byday=old.aggregate(cells,['profile','day','label']);bystation=old.aggregate(cells,['profile','station','label'])
    folds=[f for item in run['bundles'].values() for grid in item['fit']['grids'].values() for cand in grid['candidates'] for f in cand['folds']];proper=[next(r for r in item['fit']['records'] if r['label']==PROPER) for item in run['bundles'].values()]
    result=dict(protocol_sha=sha(PF),scope=p['scope'],public_events=len(truth),observed_events=sum(v is not None for v in truth.values()),pools=len(m['pools']),base_conditions=len(m['cases']),inference_conditions=len(pm['cases']),rows=len(cells),skips=pm['skips'],unique_numeric_fits=len(run['bundles']),
        conditional_safety_failures=[v for v in safety if v['conditional_failure']],valid_conditions=sum(v['condition_valid'] for v in safety if v['nominal']==PROPER),projection_changed={n:sum(v['changed'] for v in safety if v['nominal']==n) for n in [LEGACY,PROPER]},
        procurement_failures=plans['failures'],legacy_cv_folds=len(folds),legacy_cv_nonconverged=sum(not f['converged'] for f in folds),proper_nonconverged=sum(not r['metadata']['converged'] for r in proper),proper_max_iterations=max(r['metadata']['iterations'] for r in proper),
        proper_machine_guard_count=sum(r['metadata']['machine_guard_count'] for r in proper),proper_max_objective_increase=max(r['metadata']['max_objective_increase'] for r in proper),
        summary=summary,by_day=byday,by_station=bystation,safety=safety,method_statuses=dict(Counter(v['status'] for v in cells)),
        hashes={str(v.relative_to(B)):sha(v) for v in [PF,path('_runs.json'),path('_plans.json'),path('_purchased_inputs.npz'),path('_evaluator_truth.json'),path('_contracts.json'),path('_observations.json'),path('_extracted_manifest.json')]})
    save(path('_summary.json'),result)
    for suffix,rows in [('_cells.csv',cells),('_overall.csv',summary),('_by_day.csv',byday),('_by_station.csv',bystation)]:
        with path(suffix).open('w',encoding='utf-8-sig',newline='') as f:writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:result[k] for k in ['public_events','observed_events','inference_conditions','rows','projection_changed','proper_nonconverged','proper_max_iterations','proper_machine_guard_count','proper_max_objective_increase']},ensure_ascii=False),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','extract','prepare','plans','fulfill','predict','evaluate']);args=parser.parse_args();globals()[args.action]()
