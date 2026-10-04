"""Separate evaluation of the frozen two-axis pure-simulation diagnostic."""
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
from fractions import Fraction as F
from collections import defaultdict,Counter
import json,sys,csv,math,hashlib
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'deps'))
import numpy as np
import simulated_inference as b

def main():
    P=b.verify();M=b.read(b.path('_manifest.json'));RUN=b.read(b.path('_runs.json'));EX=b.read(b.path('_exact_checks.json'))
    assert b.sha(b.path('_evaluator_truth.npz'))==M['evaluator_hash'] and b.sha(b.path('_contracts.json'))==M['contracts_hash']
    T=np.load(b.path('_evaluator_truth.npz'),allow_pickle=False);D=np.load(b.path('_purchased_inputs.npz'),allow_pickle=False)
    contracts={(r['key'],r['scenario']):r for r in b.read(b.path('_contracts.json'))};pc={p['plan_key']:p for p in EX['plan_checks']};ec={r['inference_key']:EX['fit_references'][r['exact_fit_key']] for r in EX['fit_cases']}
    cells=[];groups=defaultdict(list);safety=[];depth_checked=set();source_worlds=0;newpoint={};status=Counter()
    for row in RUN['rows']:
        q=float(T[row['key']+'__target'][0]);pred=row['prediction'];projected=row.get('projected_prediction');valid=all(contracts[(row['key'],row['scenario'])][k] for k in ['coarse_valid','report_valid','target_valid'])
        c={k:row[k] for k in ['key','span','target_slot','profile','seed','plan_key','policy','budget','selected_count','total_cost','scenario','inference_key','fit_key','label','method','adapter','status']}
        if pred is not None:assert projected==min(max(pred,0.),500.)
        c.update(truth=q,prediction=pred,projected_prediction=projected,squared_error=None if pred is None else (projected-q)**2,absolute_error=None if pred is None else abs(projected-q),
            raw_squared_error=None if pred is None else (pred-q)**2,condition_valid=valid,prequery_exact_width=float(F(pc[row['plan_key']]['selected_exact_width'])),
            prequery_r20_reachable=pc[row['plan_key']]['prequery_r20_reachable'],inference_seconds=row.get('inference_seconds'),selection_seconds=row.get('selection_seconds',0.),acquisition_seconds=row['acquisition_seconds'])
        cells.append(c);groups[tuple(c[k] for k in ['span','target_slot','profile','budget','scenario','policy','label'])].append(c);status[row['status']]+=1
        if row['label']=='SourceBridge_packet':
            newpoint[row['inference_key']]=row;er=ec[row['inference_key']];fl=row['metadata'];wpre=F(pc[row['plan_key']]['selected_exact_width']);check=dict(inference_key=row['inference_key'],span=row['span'],target_slot=row['target_slot'],profile=row['profile'],budget=row['budget'],policy=row['policy'],scenario=row['scenario'],
                condition_valid=valid,float_feasible=fl['feasible'],exact_feasible=er['feasible'],state_match=fl['feasible']==er['feasible'],prequery_r20=wpre<=40)
            if er['feasible']:
                lo,hi=map(F,er['interval']);qq=F(q);width=hi-lo
                check.update(exact_width=float(width),exact_contains_truth=lo<=qq<=hi,width_le_prequery=width<=wpre,exact_midpoint_error_bound=abs(F(er['exact_midpoint'])-qq)<=wpre/2,
                    floating_point_error_bound=pred is not None and abs(F(pred)-qq)<=wpre/2,postquery_r20=width<=40,
                    endpoint_difference=max(abs(fl['interval'][0]-float(lo)),abs(fl['interval'][1]-float(hi))) if fl['feasible'] else None)
            safety.append(check)
    # Independent feasible-world and direct closed-depth checks, no imports of solver internals.
    for item in EX['fit_cases']:
        tag=item['exact_fit_key']
        if tag in depth_checked:continue
        depth_checked.add(tag);er=EX['fit_references'][tag];Y=D[item['inference_key']+'__reports'];A=D[item['inference_key']+'__anchors'];intervals=[]
        for source,s in enumerate(er['source_results']):
            if s is None:continue
            lo,hi=map(F,s['interval']);intervals.append((lo,hi))
            for which,endpoint in [('lower_world',lo),('upper_world',hi)]:
                u,v,qt=map(F,s[which]);assert F(1)/F(1.3)<=u<=F(1)/F(.7) and 0<=qt<=500 and qt==endpoint
                for j,(al,ah,e) in enumerate(A):
                    yy=F(float(Y[source,j]));ee=F(float(e));assert (yy-ee)*u+v<=F(float(ah)) and (yy+ee)*u+v>=F(float(al))
                assert abs(F(float(Y[source,-1]))*u+v-qt)<=5*u;source_worlds+=1
        points=sorted({x for iv in intervals for x in iv});accepted=[x for x in points if sum(lo<=x<=hi for lo,hi in intervals)>=7]
        assert bool(accepted)==er['feasible']
        if accepted:assert list(map(F,er['interval']))==[accepted[0],accepted[-1]]
    summary=[]
    for key,rr in sorted(groups.items()):
        good=[r for r in rr if r['squared_error'] is not None];s=dict(zip(['span','target_slot','profile','budget','scenario','policy','label'],key));s.update(total=len(rr),n=len(good),statuses=dict(Counter(r['status'] for r in rr)))
        if good:s.update(rmse=math.sqrt(sum(r['squared_error'] for r in good)/len(good)),mae=sum(r['absolute_error'] for r in good)/len(good),maximum_error=max(r['absolute_error'] for r in good),
            raw_rmse=math.sqrt(sum(r['raw_squared_error'] for r in good)/len(good)),median_inference_ms=float(np.median([r['inference_seconds'] for r in good]))*1000,
            median_selection_ms=float(np.median([r['selection_seconds'] for r in good]))*1000,mean_total_cost=float(np.mean([r['total_cost'] for r in good])))
        summary.append(s)
    # All MLNI candidates are scored again only from their purchased reference centers.
    grids=b.read(b.path('_grids.json'));byfit={r['fit_key']:r for r in RUN['rows']};foldcount=0;scores=0;maxdiff=0.;nonconv=0;maxiter=0
    for tag,adapters in grids.items():
        key=byfit[tag]['inference_key'];q=D[key+'__centers'];A=D[key+'__anchors'];h=(A[:,1]-A[:,0])/2
        for adapter,g in adapters.items():
            for candidate in g['candidates']:
                losses=[]
                for fold in candidate['folds']:
                    tr=fold['training_indices'];held=fold['held'];assert held not in tr
                    center=float(q[tr].mean());sd=float(q[tr].std());sd=sd if sd>1e-8 else 1.;assert fold['normalizer_center']==center and fold['normalizer_scale']==sd
                    if adapter=='weighted_source_affine':assert fold['training_h']==h[tr].tolist()
                    loss=(fold['prediction']-float(q[held]))**2;maxdiff=max(maxdiff,abs(loss-fold['loss']));losses.append(loss);foldcount+=1;nonconv+=not fold['converged'];maxiter=max(maxiter,fold['iterations'])
                maxdiff=max(maxdiff,abs(sum(losses)/len(losses)-candidate['score']));scores+=1
            if g['candidates']:assert g['selected_index']==int(np.argmin([c['score'] for c in g['candidates']]))
    plans=b.read(b.path('_plans.json'))['plans'];pr=[]
    for p in plans:
        ex=pc[p['plan_key']];pr.append({**{k:p[k] for k in ['key','span','target_slot','profile','seed','policy','budget','plan_key','selected_count','total_cost','report_anchor_cost_per_source','selection_seconds']},
            'selected_exact_width':float(F(ex['selected_exact_width'])),'exact_optimum':ex['exact_optimum'],'prequery_r20':ex['prequery_r20_reachable'],'selected_ids':p['local_anchor_ids']})
    projected=[];proj_groups=defaultdict(list)
    for r in b.read(b.path('_projection_predictions.json'))['rows']:
        q=F(float(T[r['key']+'__target'][0]));er=ec[r['inference_key']];v={**r,'truth':float(q)}
        if r['status']=='defined':
            lo,hi=map(F,er['interval']);v.update(squared_error=float((F(r['prediction'])-q)**2),nominal_squared_error=float((F(r['nominal_point'])-q)**2),
                midpoint_squared_error=float((F(newpoint[r['inference_key']]['prediction'])-q)**2),exact_bound_holds=abs(F(r['exact_point'])-q)<=F(r['exact_width'])/2,
                exported_bound_holds=abs(F(r['prediction'])-q)<=F(r['float_radius_upper']),contains_truth=lo<=q<=hi)
        projected.append(v);proj_groups[tuple(r[k] for k in ['span','target_slot','profile','budget','scenario','policy'])].append(v)
    ps=[]
    for key,rr in sorted(proj_groups.items()):
        good=[r for r in rr if r['status']=='defined'];v=dict(zip(['span','target_slot','profile','budget','scenario','policy'],key));v.update(n=len(good),total=len(rr))
        if good:v.update(rmse=math.sqrt(sum(r['squared_error'] for r in good)/len(good)),nominal_rmse=math.sqrt(sum(r['nominal_squared_error'] for r in good)/len(good)),
            midpoint_rmse=math.sqrt(sum(r['midpoint_squared_error'] for r in good)/len(good)),changed=sum(r['changed_exact'] for r in good),mean_radius=float(np.mean([r['width']/2 for r in good])))
        ps.append(v)
    integrity=dict(rows=len(cells),conditions=len(safety),unique_inputs=RUN['unique_inputs'],statuses=dict(status),mlni_reference_scores=scores,mlni_reference_folds=foldcount,
        mlni_reference_nonconverged=nonconv,mlni_reference_max_iterations=maxiter,mlni_score_recompute_maxdiff=maxdiff,final_nonconverged=sum(r['status']=='max_iterations' for r in cells),
        exact_designs=len(EX['designs']),exact_fits=len(EX['fit_references']),source_endpoint_worlds_checked=source_worlds,closed_depth_hulls_checked=len(depth_checked),
        float_exact_state_disagreement=sum(not s['state_match'] for s in safety),valid_coverage_failures=sum(s.get('exact_contains_truth') is False for s in safety if s['condition_valid']),
        valid_width_bound_failures=sum(s.get('width_le_prequery') is False for s in safety if s['condition_valid']),valid_midpoint_bound_failures=sum(s.get('exact_midpoint_error_bound') is False for s in safety if s['condition_valid']),
        max_endpoint_difference=max((s.get('endpoint_difference') or 0 for s in safety),default=0),minimax_nonoptimal=sum(not p['exact_optimum'] for p in pr if p['policy']=='MinimaxPairTable'),
        projection_bound_failures=sum(not (r['exact_bound_holds'] and r['exported_bound_holds'] and r['contains_truth']) for r in projected if r['status']=='defined'),projection_moves=sum(r.get('changed_exact',False) for r in projected))
    b.dump(b.path('_summary.json'),dict(protocol_sha256=b.sha(b.PF),runs_hash=b.sha(b.path('_runs.json')),integrity=integrity,summary=summary,purchases=pr,safety=safety,projection_summary=ps,projection_rows=projected))
    for suffix,rr in [('_cells.csv',cells),('_main_summary.csv',summary),('_purchases.csv',pr),('_projection_summary.csv',ps)]:
        with b.path(suffix).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    print(json.dumps(integrity,indent=2))

if __name__=='__main__':main()
