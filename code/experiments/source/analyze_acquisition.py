"""Evaluation and arithmetic audit after all fixed-input predictions were saved."""
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
import sys,json,hashlib,csv,math
from collections import defaultdict,Counter
from fractions import Fraction as F
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'deps'))
import numpy as np
R=B/'results';PREFIX='acquisition_evaluation'
def path(s):return R/(PREFIX+s)
def read(p):return json.loads(Path(p).read_text('utf-8'))
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    P=read(B/(PREFIX+'_protocol.json'));manifest=read(path('_manifest.json'));runs=read(path('_runs.json'))
    for name,dig in P['source_hashes'].items():assert verify_identity(B/name,dig),name
    for suffix,key in [('_evaluator_truth.npz','evaluator_hash'),('_evaluator_contracts.json','contracts_hash'),('_public_inputs.npz','public_hash')]:assert sha(path(suffix))==manifest[key],suffix
    T=np.load(path('_evaluator_truth.npz'),allow_pickle=False);D=np.load(path('_purchased_inputs.npz'),allow_pickle=False)
    exact=read(path('_exact_checks.json'));er={r['inference_key']:exact['fit_references'][r['exact_fit_key']] for r in exact['fit_cases']}
    contracts={(r['key'],r['scenario']):r for r in read(path('_evaluator_contracts.json'))};planchecks={r['plan_key']:r for r in exact['plan_checks']}
    cells=[];groups=defaultdict(list);safety=[];predictor_states=Counter();common_defined=0
    for row in runs['rows']:
        key=row['key'];contract=contracts[(key,row['scenario'])];truth=float(T[key+'__target'][0]);available=math.isfinite(truth)
        pred=row['prediction'];projected=row.get('projected_prediction');defined=pred is not None and math.isfinite(pred)
        label=row['label'];valid=contract['coarse_interval_valid'] and contract['honest_report_spec_valid'] and contract['target_domain_valid']
        if defined:
            assert projected is not None and math.isfinite(projected)
            assert projected==min(max(pred,0.),500.)
        cell={k:row[k] for k in ['inference_key','plan_key','key','day','target_station','seed','profile','budget','policy','scenario','label','method','adapter','selected_count','total_cost','status','fit_key']}
        cell.update(truth=truth if available else None,prediction=pred,projected_prediction=projected,
            absolute_error=abs(projected-truth) if defined and available else None,squared_error=(projected-truth)**2 if defined and available else None,
            raw_absolute_error=abs(pred-truth) if defined and available else None,raw_squared_error=(pred-truth)**2 if defined and available else None,
            condition_valid=valid,all_paper_interfaces_defined=row['selected_count']>0 and row['target_observed'],
            selection_seconds=row.get('selection_seconds',0.),inference_seconds=row.get('inference_seconds'),
            acquisition_seconds=row['selection_seconds'] if 'selection_times' in row and row['method']!='MLNI_JSAC2022' else None,
            prequery_width=row['prequery_width'],selected_exact_prequery_width=planchecks[row['plan_key']]['selected_exact_width_float'])
        # Row's method selection_seconds may shadow plan selection timing. Recover procurement separately from plans below.
        cells.append(cell);groups[(row['profile'],row['budget'],row['scenario'],row['policy'],label)].append(cell);predictor_states[(label,row['status'])]+=1
        if row['method']=='SourceBridge_packet':
            e=er.get(row['inference_key']);fl=row.get('metadata',{});s=dict(inference_key=row['inference_key'],condition_valid=valid,
                float_feasible=fl.get('feasible'),exact_feasible=None if e is None else e['feasible'],status_match=e is not None and fl.get('feasible')==e['feasible'])
            if e is not None and e['feasible']:
                lo,hi=map(F,e['interval']);w=hi-lo;wpre=F(planchecks[row['plan_key']]['selected_exact_width']);q=F(truth)
                fp=None if pred is None else F(pred)
                s.update(exact_contains_truth=lo<=q<=hi,exact_width=float(w),float_width=fl.get('width'),
                    endpoint_max_abs_difference=max(abs(float(lo)-fl['interval'][0]),abs(float(hi)-fl['interval'][1])) if fl.get('feasible') else None,
                    actual_width_le_prequery=w<=wpre,exact_midpoint_error_le_half_prequery=abs(F(e['exact_midpoint'])-q)<=wpre/2,
                    rounded_float_point_error_le_half_prequery=None if fp is None else abs(fp-q)<=wpre/2,
                    certified_float_radius_covers_rounding=None if e['point_float'] is None else F(e['float_point_error_radius_upper'])>=max(abs(F(e['point_float'])-lo),abs(hi-F(e['point_float']))))
            safety.append(s)
    plans=read(path('_plans.json'))['plans'];pc={p['plan_key']:p for p in plans}
    for c in cells:c['acquisition_seconds']=pc[c['plan_key']]['selection_seconds']
    summaries=[]
    for keys,rr in sorted(groups.items()):
        use=[r for r in rr if r['squared_error'] is not None];v=[r for r in use if r['condition_valid']];co=[r for r in use if r['all_paper_interfaces_defined']]
        def metrics(rows):
            if not rows:return dict(n=0,rmse=None,mae=None,max_absolute_error=None,raw_rmse=None)
            return dict(n=len(rows),rmse=math.sqrt(sum(r['squared_error'] for r in rows)/len(rows)),mae=sum(r['absolute_error'] for r in rows)/len(rows),
                max_absolute_error=max(r['absolute_error'] for r in rows),raw_rmse=math.sqrt(sum(r['raw_squared_error'] for r in rows)/len(rows)))
        summaries.append(dict(profile=keys[0],budget=keys[1],scenario=keys[2],policy=keys[3],label=keys[4],conditions=len(rr),
            status_counts=dict(Counter(r['status'] for r in rr)),all_conditions=metrics(use),valid_conditions=metrics(v),all_paper_interfaces_defined=metrics(co),
            mean_total_cost=sum(r['total_cost'] for r in rr)/len(rr),median_inference_seconds=float(np.median([r['inference_seconds'] for r in use])) if use else None,
            median_selection_seconds=float(np.median([r['selection_seconds'] for r in use])) if use else None))
    # Reference-only CV scores independently recomputed from saved fold predictions and purchased centers.
    grids=read(path('_mlni_grids.json'));case_byfit={r['fit_key']:r for r in runs['rows']};scores=0;folds=0;nonconv=0;maxiters=0;maxdiff=0.
    for fitkey,adapters in grids.items():
        case=case_byfit[fitkey];centers=D[case['inference_key']+'__centers']
        for adapter,grid in adapters.items():
            cand=grid['candidates']
            for candidate in cand:
                losses=[]
                for f in candidate['folds']:
                    train=f['training_indices'];assert f['held'] not in train
                    expected_center=float(centers[train].mean());sd=float(centers[train].std());expected_scale=sd if sd>1e-8 else 1.
                    assert f['normalizer_center']==expected_center and f['normalizer_scale']==expected_scale
                    l=(f['prediction']-float(centers[f['held']]))**2;losses.append(l);maxdiff=max(maxdiff,abs(l-f['loss']));folds+=1;nonconv+=not f['converged'];maxiters=max(maxiters,f['iterations'])
                score=sum(losses)/len(losses);maxdiff=max(maxdiff,abs(score-candidate['score']));scores+=1
            if cand:assert grid['selected_index']==int(np.argmin([c['score'] for c in cand]))
    # Procurement benefit is a separate axis, always relative to the SAME physical case/budget.
    pg=defaultdict(dict)
    for p in plans:pg[(p['key'],p['budget'])][p['policy']]=p
    purchase=[]
    for (key,budget),pp in pg.items():
        new=pp['MinimaxPairTable'];nw=planchecks[new['plan_key']]['selected_exact_width_float']
        for other in ['MaxSpan','NominalCOptimal','FixedIdBudget']:
            o=pp[other];ow=planchecks[o['plan_key']]['selected_exact_width_float'];purchase.append(dict(key=key,profile=new['profile'],budget=budget,control=other,
                selected_new=new['selected_count'],selected_control=o['selected_count'],new_width=nw,control_width=ow,
                reduction_fraction=(ow-nw)/ow if ow else 0.,exact_equal=F(planchecks[new['plan_key']]['selected_exact_width'])==F(planchecks[o['plan_key']]['selected_exact_width'])))
    # Predetermined same-purchase point-error pairing against each paper and adapter.
    bycase=defaultdict(dict)
    for c in cells:bycase[c['inference_key']][c['label']]=c
    pairs=[]
    for ck,rr in bycase.items():
        n=rr['SourceBridge_packet']
        for lab,o in rr.items():
            if lab=='SourceBridge_packet':continue
            comparable=n['squared_error'] is not None and o['squared_error'] is not None
            pairs.append(dict(inference_key=ck,profile=n['profile'],budget=n['budget'],scenario=n['scenario'],policy=n['policy'],baseline=lab,comparable=comparable,
                new_squared_error=n['squared_error'],baseline_squared_error=o['squared_error'],
                result=None if not comparable else ('win' if n['squared_error']<o['squared_error']-1e-12 else 'loss' if n['squared_error']>o['squared_error']+1e-12 else 'tie')))
    result=dict(protocol_sha256=sha(B/(PREFIX+'_protocol.json')),runs_hash=sha(path('_runs.json')),conditions=len(bycase),rows=len(cells),unique_bundles=runs['unique_bundles'],
        summary=summaries,purchase_comparisons=purchase,paper_pairings=pairs,safety=safety,
        integrity=dict(reference_cv_scores=scores,reference_cv_folds=folds,reference_cv_max_score_difference=maxdiff,reference_grid_nonconverged=nonconv,reference_grid_max_iterations=maxiters,
            selected_final_nonconverged=sum(r['status']=='max_iterations' for r in cells),
            exact_design_count=len(exact['designs']),exact_fit_count=len(exact['fit_references']),
            float_exact_state_disagreements=sum(not s['status_match'] for s in safety),valid_exact_coverage_failures=sum(s.get('exact_contains_truth') is False for s in safety if s['condition_valid']),
            valid_exact_prequery_width_failures=sum(s.get('actual_width_le_prequery') is False for s in safety if s['condition_valid']),
            valid_exact_midpoint_bound_failures=sum(s.get('exact_midpoint_error_le_half_prequery') is False for s in safety if s['condition_valid']),
            valid_float_point_bound_failures=sum(s.get('rounded_float_point_error_le_half_prequery') is False for s in safety if s['condition_valid']),
            minimax_float_selected_not_exact_optimal=sum(not r['same_exact_optimum'] for r in exact['plan_checks'] if pc[r['plan_key']]['policy']=='MinimaxPairTable'),
            max_endpoint_difference=max((s.get('endpoint_max_abs_difference') or 0 for s in safety),default=0),
            hidden_contract_status_counts=dict(Counter((str(c['coarse_interval_valid'])+'/'+str(c['honest_report_spec_valid'])+'/'+str(c['target_domain_valid'])) for c in contracts.values()))))
    dump(path('_summary.json'),result)
    with path('_cells.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(cells[0]));w.writeheader();w.writerows(cells)
    print(json.dumps(dict(rows=len(cells),conditions=len(bycase),integrity=result['integrity']),indent=2),flush=True)

if __name__=='__main__':main()
