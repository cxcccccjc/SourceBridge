"""A declared component experiment; saved published predictions are not refitted."""
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
from fractions import Fraction as F
from collections import defaultdict
import json, math, hashlib, argparse, time

B = Path(__file__).resolve().parent
R = B / 'results'
BASE = 'acquisition_evaluation'
NAME = 'safe_projection'

def read(p):
    return Path(p).read_text(encoding='utf-8')

def load(p):
    return json.loads(read(p))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, data):
    Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def upward(q):
    x = float(q)
    return math.nextafter(x, math.inf) if F(x) < q else x

def project(z, lo, hi, width, target):
    lo, hi, width, z = map(F, (lo, hi, width, z))
    lower, upper = map(F, target)
    if lo > hi or width < 0 or lo < lower or hi > upper:
        return {'status': 'invalid_certificate'}
    sl, su = max(lower, hi-width/2), min(upper, lo+width/2)
    if sl > su:
        return {'status': 'empty_safe_set'}
    p = min(max(z, sl), su)
    fp = float(p)
    radius = max(abs(F(fp)-lo), abs(F(fp)-hi))
    exact_radius = max(abs(p-lo), abs(p-hi))
    if exact_radius > width/2:
        raise AssertionError('Projection theorem failed')
    return dict(status='defined', exact_point=str(p), prediction=fp,
                exact_safe_set=[str(sl),str(su)], nominal_point=float(z),
                changed_exact=p!=z, changed_float=fp!=float(z), movement=abs(fp-float(z)),
                exact_radius=str(exact_radius), float_radius_upper=upward(radius),
                floating_bound_excess=float(max(F(0),radius-width/2)),
                width=float(width), exact_width=str(width), exact_interval=[str(lo),str(hi)])

def freeze():
    files=[Path(__file__),B/(NAME+'_protocol.md')]
    protocol=dict(status='FIXED_BEFORE_BASE_RESULT_READING',default_core='EPTD_TIFS2022__bounded_source_affine',
                  main_policy='MinimaxPairTable',all_other_interfaces='retained component sensitivities',
                  source_hashes={p.name:sha(p) for p in files})
    p=B/(NAME+'_protocol.json')
    if p.exists():
        if load(p)!=protocol:raise AssertionError('Frozen component protocol changed')
    else:save(p,protocol)
    print(json.dumps(protocol,ensure_ascii=False))

def verify():
    p=load(B/(NAME+'_protocol.json'))
    for name,h in p['source_hashes'].items():
        if not verify_identity(B/name,h):raise AssertionError(name)
    return p

def predict():
    verify()
    paths=[R/(BASE+'_runs.json'),R/(BASE+'_exact_checks.json'),B/(BASE+'_protocol.json')]
    runs,ex,protocol=map(load,paths)
    if runs['protocol_sha256']!=sha(paths[-1]) or ex['protocol_sha256']!=sha(paths[-1]):
        raise AssertionError('Mismatched acquisition protocol')
    target=protocol['spec']['target']
    plans={r['plan_key']:r for r in ex['plan_checks']}
    cert={r['inference_key']:ex['fit_references'][r['exact_fit_key']] for r in ex['fit_cases']}
    out=[]
    for row in runs['rows']:
        if row['label']=='SourceBridge_packet':continue
        keep=['inference_key','plan_key','key','profile','budget','policy','scenario','label','selected_count','status']
        r={k:row[k] for k in keep}
        r['original_status']=r.pop('status')
        z=row.get('projected_prediction')
        e=cert.get(row['inference_key'])
        if z is None or not math.isfinite(z):r.update(status='no_finite_nominal',prediction=None)
        elif e is None or not e['feasible']:r.update(status='no_feasible_certificate',prediction=None)
        else:
            tick=time.perf_counter()
            result=project(z,*e['interval'],plans[row['plan_key']]['selected_exact_width'],target)
            r.update(result);r['projection_seconds']=time.perf_counter()-tick
        out.append(r)
    save(R/(NAME+'_predictions.json'),dict(protocol_hash=sha(B/(NAME+'_protocol.json')),
          source_hashes={str(p.relative_to(B)):sha(p) for p in paths},rows=out))
    print(json.dumps(dict(rows=len(out),status_counts={s:sum(r['status']==s for r in out) for s in sorted(set(r['status'] for r in out))})))

def analyze():
    verify()
    predicted=load(R/(NAME+'_predictions.json'))
    for name,h in predicted['source_hashes'].items():
        if not verify_identity(B/name,h):raise AssertionError(name)
    import csv
    with (R/(BASE+'_cells.csv')).open(encoding='utf-8-sig',newline='') as f:
        base={(r['inference_key'],r['label']):r for r in csv.DictReader(f)}
    groups=defaultdict(list);rows=[];failures=[]
    for pred in predicted['rows']:
        row=dict(pred);b=base[(pred['inference_key'],pred['label'])]
        mid=base[(pred['inference_key'],'SourceBridge_packet')]
        if pred['status']=='defined' and b['truth']:
            q=F(float(b['truth']));fp=F(pred['prediction']);ep=F(pred['exact_point'])
            original=F(float(b['projected_prediction']));center=F(float(mid['prediction']))
            lo,hi=map(F,pred['exact_interval']);valid=b['condition_valid']=='True'
            row.update(truth=float(q),condition_valid=valid,
                squared_error=float((fp-q)**2),absolute_error=float(abs(fp-q)),
                nominal_squared_error=float((original-q)**2),midpoint_squared_error=float((center-q)**2),
                exact_error_bound_holds=abs(ep-q)<=F(pred['exact_width'])/2,
                exported_error_bound_holds=abs(fp-q)<=F(pred['float_radius_upper']),
                interval_contains_truth=lo<=q<=hi)
            if valid and not all(row[k] for k in ['exact_error_bound_holds','exported_error_bound_holds','interval_contains_truth']):
                failures.append((row['inference_key'],row['label']))
        rows.append(row)
        groups[tuple(row[k] for k in ['profile','budget','scenario','policy','label'])].append(row)
    summary=[]
    for keys,rr in sorted(groups.items()):
        use=[r for r in rr if 'squared_error' in r]
        result=dict(zip(['profile','budget','scenario','policy','label'],keys));result.update(total=len(rr),n=len(use))
        if use:
            result.update(rmse=math.sqrt(sum(r['squared_error'] for r in use)/len(use)),
                nominal_rmse=math.sqrt(sum(r['nominal_squared_error'] for r in use)/len(use)),
                midpoint_rmse=math.sqrt(sum(r['midpoint_squared_error'] for r in use)/len(use)),
                mae=sum(r['absolute_error'] for r in use)/len(use),max_error=max(r['absolute_error'] for r in use),
                changed=sum(r['changed_exact'] for r in use),movement_mean=sum(r['movement'] for r in use)/len(use),
                nominal_wins=sum(r['squared_error']<r['nominal_squared_error']-1e-12 for r in use),
                nominal_losses=sum(r['squared_error']>r['nominal_squared_error']+1e-12 for r in use),
                midpoint_wins=sum(r['squared_error']<r['midpoint_squared_error']-1e-12 for r in use),
                midpoint_losses=sum(r['squared_error']>r['midpoint_squared_error']+1e-12 for r in use),
                mean_half_width=sum(r['width']/2 for r in use)/len(use),
                max_floating_excess=max(r['floating_bound_excess'] for r in use))
        summary.append(result)
    save(R/(NAME+'_summary.json'),dict(prediction_hash=sha(R/(NAME+'_predictions.json')),
        evaluation_source_hash=sha(R/(BASE+'_cells.csv')),summary=summary,rows=rows,bound_failures=failures))
    print(json.dumps(dict(rows=len(rows),groups=len(summary),valid_bound_failures=len(failures),
                         projected_cases=sum(r.get('changed_exact',False) for r in rows))))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','predict','analyze']);args=ap.parse_args()
    {'freeze':freeze,'predict':predict,'analyze':analyze}[args.stage]()
