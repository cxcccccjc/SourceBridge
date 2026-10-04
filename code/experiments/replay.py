"""Read-only actual replay. No new test data and no claims of new blind trials."""
import os,sys
sys.dont_write_bytecode=True
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
import json,hashlib,time,math,collections
from fractions import Fraction as F
SRC=(Path(__file__).resolve().parent/'source').resolve()
FIX=Path(__file__).resolve().parent/'fixtures'
OUT=Path(__file__).resolve().parent
(OUT/'generated').mkdir(exist_ok=True)
def under_source(p):
    try:return Path(p).resolve().is_relative_to(SRC)
    except (TypeError,ValueError):return False
original_mkdir=Path.mkdir
def readonly_mkdir(self,*a,**kw):
    if under_source(self):
        return None # Imported modules declare unused caches; replay never writes them.
    return original_mkdir(self,*a,**kw)
Path.mkdir=readonly_mkdir
def guard(event,args):
    if event=='open' and under_source(args[0]):
        mode=args[1];flags=args[2]
        assert not (isinstance(mode,str) and any(c in mode for c in 'wax+')),('source write',args)
        assert not (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)),('source write',args)
    if event in ['os.remove','os.rename','os.rmdir'] and under_source(args[0]):raise RuntimeError('Source mutation blocked')
sys.addaudithook(guard)
sys.path.insert(0,str(SRC))
import public_inference as engine
import numpy as np
from certified_geometry import exact_packet_fusion
from safe_projection import project
from reference_geometry import jsonable

def load(name):return json.loads((SRC/name).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
P=load('public_inference_protocol.json')
for n,h in P['source_hashes'].items():assert verify_identity(SRC/n,h)
manifest=json.loads((FIX/'purchased_manifest.json').read_text(encoding='utf-8'))
run=json.loads((FIX/'saved_runs.json').read_text(encoding='utf-8'))
data_path=FIX/'purchased_inputs.npz'
assert sha(data_path)==manifest['input_sha']==run['input_sha']
events=sorted({c['event'] for c in manifest['cases']})
chosen=[c for c in manifest['cases'] if c['event'] in [events[0],events[-1]]]
assert len(chosen)==64
assert manifest['derived_subset'] and run['derived_subset']
saved=collections.defaultdict(dict)
for r in run['rows']:saved[r['key']][r['label']]=r
D=np.load(data_path,allow_pickle=False)
reports=[];tic=time.perf_counter();maximum=0;exact_equal=0;prediction_checks=0;status_checks=0
for i,c in enumerate(chosen):
    k=c['key'];Y,A,q=(D[k+'__'+s] for s in ['reports','anchors','centers'])
    fit=engine.old.engine.fit(Y,A,q,timing=False)
    fit['records'].append(engine.proper_mlni(Y,A,q,P))
    ex=exact_packet_fusion(Y,A.tolist(),P['target'],P['gain_bounds'],P['epsilon'],P['f'])
    tag=saved[k][engine.PROPER]['fit_key'];saved_ex=run['bundles'][tag]['exact']
    assert engine.old.engine.strip(jsonable(ex))==engine.old.engine.strip(saved_ex),k
    exact_equal+=1
    records=list(fit['records'])
    for label,nominal in [('SourceBridge_MLNI_WLS',engine.LEGACY),(engine.COMPOSED,engine.PROPER)]:
        r=next(x for x in records if x['label']==nominal)
        pr=project(r['projected_prediction'],*ex['interval'],F(c['width']),P['target'])
        records.append(dict(label=label,prediction=pr['prediction'],projected_prediction=pr['prediction'],status=pr['status']))
    assert len(records)==15
    for r in records:
        expected=saved[k][r['label']]
        diff=abs(r['projected_prediction']-expected['projected_prediction']);maximum=max(maximum,diff)
        assert math.isclose(r['projected_prediction'],expected['projected_prediction'],rel_tol=1e-11,abs_tol=1e-10),(k,r['label'],diff)
        assert r['status']==expected['status']
        prediction_checks+=1;status_checks+=1
        reports.append(dict(key=k,label=r['label'],prediction=r['projected_prediction'],saved_prediction=expected['projected_prediction'],difference=diff,status=r['status']))
    if (i+1)%16==0:print(json.dumps(dict(completed=i+1,total=len(chosen),seconds=time.perf_counter()-tic)),flush=True)
result=dict(status='PASS',scope='Actual recomputation of frozen purchased input. Recomputation from frozen observations.',cases=64,event_ids=[events[0],events[-1]],all_15_interfaces=True,prediction_checks=prediction_checks,status_checks=status_checks,exact_packet_certificates_equal=exact_equal,maximum_absolute_prediction_difference=maximum,elapsed_seconds=time.perf_counter()-tic,evaluator_truth_loaded=False,source_write_guard=True,protocol_sha=sha(SRC/'public_inference_protocol.json'),purchased_input_sha=sha(data_path),rows=reports)
(OUT/'generated'/'replay_results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


