"""Independent exact certificate and saved-action audit; no production imports."""
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
import sys,json,hashlib,itertools,math,statistics
from fractions import Fraction as Q
from collections import Counter
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'deps'))
import numpy as np
R=B/'results';N='procurement';PF=B/(N+'_protocol.json')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def dets(anchors):
    if not anchors:return Q(0),Q(0)
    a=[(Q(lo),Q(hi),1/Q(e)**2) for lo,hi,e in anchors]
    ends=sorted({v for lo,hi,w in a for v in (lo,hi)});candidates=set(ends)
    for l,r in zip(ends,ends[1:]):
        probe=(l+r)/2;active=[(w,lo if probe<lo else hi) for lo,hi,w in a if not lo<=probe<=hi]
        if active:
            root=sum(w*v for w,v in active)/sum(w for w,v in active)
            if l<=root<=r:candidates.add(root)
    def obj(t):return sum((w*(lo-t if t<lo else t-hi if t>hi else Q(0))**2 for lo,hi,w in a),Q(0))
    dr=sum(w for lo,hi,w in a)*min(obj(t) for t in candidates)
    dn=sum((v[2]*w[2]*((v[0]+v[1]-w[0]-w[1])/2)**2 for v,w in itertools.combinations(a,2)),Q(0))
    return dr,dn


def linear_solution(columns,target):
    k=len(columns);m=[[Q(columns[j][i]) for j in range(k)]+[Q(target[i])] for i in range(3)]
    at=0
    for col in range(k):
        pivot=next((i for i in range(at,3) if m[i][col]),None)
        if pivot is None:return None
        m[at],m[pivot]=m[pivot],m[at];v=m[at][col];m[at]=[x/v for x in m[at]]
        for i in range(3):
            if i!=at:
                v=m[i][col];m[i]=[x-v*y for x,y in zip(m[i],m[at])]
        at+=1
    if any(all(v==0 for v in row[:k]) and row[k] for row in m):return None
    return [m[i][-1] for i in range(k)]


def independent_lp_rows(anchors,spec,endpoint):
    low,high=map(Q,spec['target']);amin,amax=map(Q,spec['gain_bounds']);noise=2*Q(spec['epsilon'])/amin;c=Q(endpoint)
    rows=[((-1,0,0),-amin/amax),((1,0,0),Q(1)),((0,0,-1),-low),((0,0,1),high),
          ((-(c+noise),-1,1),Q(0)),((c-noise,1,-1),Q(0))]
    for lo,hi,eps in anchors:
        lo,hi,eps=map(Q,(lo,hi,eps));r=2*eps/amin
        rows.extend([((-(hi+r),-1,0),-lo),((lo-r,1,0),hi)])
    return [(tuple(map(Q,a)),Q(b)) for a,b in rows]


def main():
    errors=[];counts=Counter();p=read(PF);inp=read(R/(N+'_inputs.json'));runs=read(R/(N+'_runs.json'));summary=read(R/(N+'_summary.json'))
    def ck(ok,label):
        counts[label.split(':')[0]]+=1
        if not ok:errors.append(label)
    for name,h in p['source_sha256'].items():ck(sha(B/name)==h,'frozen:'+name)
    ck(inp['protocol_sha']==runs['protocol_sha']==summary['protocol_sha']==sha(PF),'protocol_hash')
    ck(runs['inputs_sha']==sha(R/(N+'_inputs.json')),'inputs_hash');ck(summary['runs_sha']==sha(R/(N+'_runs.json')),'runs_hash')
    pools={x['pool_id']:x for x in inp['pools']};cats={x['pool_id']:x for x in runs['catalogs']}
    expect={(seed,span,profile) for seed in range(840101,840121) for span in [20,150,450] for profile in ['exact','heterogeneous']}
    ck({(g['seed'],g['span'],g['profile']) for g in pools.values()}==expect,'all120_input_cells')
    ck(len(inp['pools'])==len(pools)==len(cats)==len(runs['catalogs'])==120,'pool_uniqueness')
    ck(set(pools)==set(cats),'catalog_coverage')
    # Independent regeneration uses only the frozen randomization specification.
    for seed in range(840101,840121):
        u=np.random.default_rng(np.random.SeedSequence([seed,0])).uniform(-.9,.9,6)
        h=np.random.default_rng(np.random.SeedSequence([seed,1])).permutation([2.,5.,10.,20.,40.,80.])
        cost=np.random.default_rng(np.random.SeedSequence([seed,2])).integers(1,5,6).tolist()
        for span in [20,150,450]:
            q=250.+span*np.array([-.5,-.3,-.1,.1,.3,.5])
            for profile in ['exact','heterogeneous']:
                g=pools[f's{seed}_span{span}_{profile}'];rad=np.zeros(6) if profile=='exact' else h;c=q+u*rad
                ck(g['anchors']==np.column_stack([c-rad,c+rad,np.full(6,5.)]).tolist(),'regenerate_anchors:'+g['pool_id'])
                ck(g['costs']==cost,'regenerate_cost:'+g['pool_id'])
                ck(g['radii']==rad.tolist() and g['centers']==c.tolist(),'regenerate_metadata:'+g['pool_id'])
                ck(g['coarse_cost']==sum(64 if z==0 else max(1,math.ceil(20/z)) for z in rad),'regenerate_coarse:'+g['pool_id'])
                ck(g['budgets']==[sum(cost)//3,2*sum(cost)//3],'regenerate_budget:'+g['pool_id'])
    allids={ids for k in range(7) for ids in itertools.combinations(range(6),k)}
    widths={};audited_w={};tracks={};entries={};dual_support_counts=Counter()
    for pid,g in pools.items():
        cat=cats[pid];wf=R/cat['w_file'];ck(sha(wf)==cat['w_sha'],'w_hash:'+pid)
        ck(read(wf)['anchors']==g['anchors'],'cached_w_input:'+pid)
        if cat['w_file'] not in audited_w:
            wd=read(wf);ck(wd['protocol_sha']==sha(PF),'w_protocol:'+pid);ck(wd['anchors']==g['anchors'],'w_input:'+pid)
            wr={tuple(r['ids']):r for r in wd['subset_rows']};ck(set(wr)==allids and len(wd['subset_rows'])==64,'w_all_subsets:'+pid)
            for ids,row in wr.items():
                anchors=[g['anchors'][i] for i in ids];res=row['exact_result'];branchvals=[]
                for bi,branch in enumerate(res['branches']):
                    sign=1 if bi==0 else -1;endpoint=p['spec']['target'][bi];rows=independent_lp_rows(anchors,p['spec'],endpoint);x=tuple(map(Q,branch['x']))
                    lhs=[sum(v*y for v,y in zip(a,x)) for a,b in rows]
                    ck(all(v<=b for v,(a,b) in zip(lhs,rows)),'primal_feasible:'+pid+str(ids)+str(bi))
                    tight=[i for i,(v,(a,b)) in enumerate(zip(lhs,rows)) if v==b];dual=None
                    for k in range(1,4):
                        for active in itertools.combinations(tight,k):
                            lam=linear_solution([rows[i][0] for i in active],(0,0,sign))
                            if lam is not None and all(v>=0 for v in lam) and sum(v*rows[i][1] for v,i in zip(lam,active))==sign*x[2]:
                                dual=(active,lam);break
                        if dual is not None:break
                    ck(dual is not None,'dual_optimal:'+pid+str(ids)+str(bi))
                    if dual:dual_support_counts[len(dual[0])]+=1
                    value=sign*(x[2]-Q(endpoint));ck(Q(branch['width'])==value,'branch_value:'+pid+str(ids)+str(bi));branchvals.append(value)
                    w=branch['witness'];a,ap,b,bp=map(Q,[w['first_gain'],w['second_gain'],w['first_bias'],w['second_bias']])
                    a0,a1=map(Q,p['spec']['gain_bounds']);ck(a0<=a<=a1 and a0<=ap<=a1,'witness_gain:'+pid+str(ids)+str(bi))
                    tq=list(map(Q,w['first_truth']));tp=list(map(Q,w['second_truth']));y=list(map(Q,w['reports']));eps=list(map(Q,w['epsilon']))
                    ck(len(tq)==len(tp)==len(y)==len(eps)==len(ids)+1,'witness_length:'+pid+str(ids)+str(bi))
                    intervals=[z[:2] for z in anchors]+[p['spec']['target']]
                    ck(all(Q(lo)<=v<=Q(hi) and Q(lo)<=vp<=Q(hi) and abs(yy-a*v-b)<=ee and abs(yy-ap*vp-bp)<=ee for (lo,hi),v,vp,yy,ee in zip(intervals,tq,tp,y,eps)),'two_world_feasible:'+pid+str(ids)+str(bi))
                    ck(tq[-1]==Q(endpoint) and tp[-1]==x[2] and eps==[Q(z[2]) for z in anchors]+[Q(p['spec']['epsilon'])],'witness_target_and_eps:'+pid+str(ids)+str(bi))
                ck(Q(row['positive'])==branchvals[0] and Q(row['negative'])==branchvals[1] and Q(row['width'])==Q(res['width'])==max(branchvals),'width_max:'+pid+str(ids))
            audited_w[cat['w_file']]=wr
        wr=audited_w[cat['w_file']];widths[pid]={ids:Q(r['width']) for ids,r in wr.items()}
        jc=cat['jb_catalog'];jr={tuple(e['ids']):e for e in jc['entries']};ck(set(jr)==allids and len(jc['entries'])==64,'d_all_subsets:'+pid)
        ee=[]
        for ids in allids:
            dr,dn=dets([g['anchors'][i] for i in ids]);cost=sum(g['costs'][i] for i in ids);z=jr[ids]
            ck(Q(z['determinant'])==dr and Q(z['nominal_determinant'])==dn,'determinants:'+pid+str(ids));ck(Q(z['cost'])==cost,'subset_cost:'+pid+str(ids))
            witness=list(map(Q,z['worst_values']));aa=[g['anchors'][i] for i in ids]
            ck(len(witness)==len(ids) and all(Q(t[0])<=v<=Q(t[1]) for v,t in zip(witness,aa)),'d_witness_box:'+pid+str(ids))
            wd=sum(((witness[i]-witness[j])**2/(Q(aa[i][2])**2*Q(aa[j][2])**2) for i,j in itertools.combinations(range(len(ids)),2)),Q(0))
            ck(wd==dr and z['robust_full_rank']==(dr>0),'d_witness_attains:'+pid+str(ids))
            ee.append(dict(ids=ids,cost=cost,width=widths[pid][ids],drob=dr,dnom=dn,degenerate=dr==0))
        track=[]
        for budget in range(sum(g['costs'])+1):
            z=min((e for e in ee if e['cost']<=budget),key=lambda e:(-e['drob'],-e['dnom'],e['cost'],len(e['ids']),e['ids']))
            track.append(dict(budget=budget,**z))
        ck(len(track)==len(cat['trajectory']),'trajectory_length:'+pid)
        for expected,actual in zip(track,cat['trajectory']):
            ck(all(tuple(actual[k])==v if k=='ids' else Q(actual[k])==v if isinstance(v,Q) else actual[k]==v for k,v in expected.items()),'trajectory_choice:'+pid+str(expected['budget']))
        entries[pid]=ee;tracks[pid]=track
    budgets={(z['pool_id'],z['budget_slot']):z for z in runs['budget_rows']};quotes={(z['pool_id'],z['r']):z for z in runs['quotes']}
    ck(len(budgets)==len(runs['budget_rows'])==240,'budget_coverage');ck(len(quotes)==len(runs['quotes'])==480,'quote_coverage')
    for pid,g in pools.items():
        for slot,budget in enumerate(g['budgets']):
            z=budgets[pid,slot];gold=min((e for e in entries[pid] if e['cost']<=budget),key=lambda e:(e['width'],e['cost'],len(e['ids']),e['ids']));other=tracks[pid][budget]
            ck(tuple(z['ours_ids'])==gold['ids'] and tuple(z['jb_ids'])==other['ids'],'budget_ids:'+pid+str(slot))
            ck(Q(z['ours_width'])==gold['width'] and Q(z['jb_width'])==other['width'] and z['ours_cost']==gold['cost'] and Q(z['jb_cost'])==other['cost'],'budget_values:'+pid+str(slot))
            ck(z['strict']==(gold['width']<other['width']) and Q(z['saving'])==(0 if other['width']==0 else 1-gold['width']/other['width']),'budget_gain:'+pid+str(slot))
            ck(z['jb_degenerate']==other['degenerate'] and z['budget']==budget,'budget_metadata:'+pid+str(slot))
        for radius in [20,50,100,150]:
            z=quotes[pid,radius];a=[e for e in entries[pid] if e['width']<=2*radius];b=[e for e in tracks[pid] if e['width']<=2*radius]
            aa=min(a,key=lambda e:(e['cost'],len(e['ids']),e['ids'])) if a else None;bb=min(b,key=lambda e:(e['cost'],len(e['ids']),e['ids'],e['budget'])) if b else None
            ck(z['ours_feasible']==bool(a) and z['jb_feasible']==bool(b),'quote_feasible:'+pid+str(radius))
            for method,sel in [('ours',aa),('jb',bb)]:
                ck((None if z[method+'_ids'] is None else tuple(z[method+'_ids']))==(None if sel is None else sel['ids']),'quote_ids:'+pid+str(radius)+method)
                ck((None if z[method+'_total'] is None else Q(z[method+'_total']))==(None if sel is None else g['coarse_cost']+9*(1+sel['cost'])),'quote_total:'+pid+str(radius)+method)
                ck((None if z[method+'_width'] is None else Q(z[method+'_width']))==(None if sel is None else sel['width']),'quote_width:'+pid+str(radius)+method)
            ck(z['coarse_cost']==g['coarse_cost'] and z['jb_budget']==(None if bb is None else bb['budget']) and z['jb_first_budget']==(None if not b else b[0]['budget']),'quote_metadata:'+pid+str(radius))
            ck(z['saving'] is None if bb is None else Q(z['saving'])==1-Q(z['ours_total'])/Q(z['jb_total']),'quote_saving:'+pid+str(radius))
    # Validate the saved fast-selector outputs against independent full oracles.
    ck(len(runs['fast_oracle_checks'])==720,'fast_check_count')
    for z in runs['fast_oracle_checks']:
        pid=z['pool_id'];fast=z['fast'];g=pools[pid]
        if z['kind']=='budget':gold=budgets[pid,z['slot']];ids=tuple(gold['ours_ids']);bound=Q(gold['ours_width'])
        else:
            gold=quotes[pid,z['r']];ids=None if gold['ours_ids'] is None else tuple(gold['ours_ids']);bound=Q(2*z['r'])
        ck(z['rank_equal'] and (None if fast is None else tuple(fast['anchor_ids']))==ids,'fast_ids:'+pid+z['kind'])
        if fast is not None:
            plus,minus=tuple(fast['positive_support']),tuple(fast['negative_support']);wr=audited_w[cats[pid]['w_file']]
            ck(tuple(sorted(set(plus)|set(minus)))==ids and Q(wr[plus]['positive'])<=bound and Q(wr[minus]['negative'])<=bound,'fast_support:'+pid+z['kind'])
            ck(fast['cost']==sum(g['costs'][i] for i in ids),'fast_cost:'+pid+z['kind'])
    # Separately recompute all grouped arithmetic without the report module.
    def close(a,b):return a is None and b is None or a is not None and b is not None and math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12)
    for z in summary['budget_groups']:
        es=[r for r in budgets.values() if (r['span'],r['profile'],r['budget_slot'])==(z['span'],z['profile'],z['budget_slot'])];sav=[float(Q(r['saving'])) for r in es]
        expected=dict(seeds=len(es),strict=sum(r['strict'] for r in es),tied=sum(Q(r['ours_width'])==Q(r['jb_width']) for r in es),mean_ours_radius=statistics.mean(float(Q(r['ours_width']))/2 for r in es),mean_jb_radius=statistics.mean(float(Q(r['jb_width']))/2 for r in es),mean_saving=statistics.mean(sav),median_saving=statistics.median(sav),min_saving=min(sav),max_saving=max(sav),jb_degenerate=sum(r['jb_degenerate'] for r in es))
        ck(all(close(z[k],v) for k,v in expected.items()),'budget_summary:'+str((z['span'],z['profile'],z['budget_slot'])))
    for z in summary['quote_groups']:
        es=[r for r in quotes.values() if (r['span'],r['profile'],r['r'])==(z['span'],z['profile'],z['r'])];both=[r for r in es if r['ours_feasible'] and r['jb_feasible']];sav=[float(Q(r['saving'])) for r in both]
        expected=dict(seeds=len(es),ours_feasible=sum(r['ours_feasible'] for r in es),jb_feasible=sum(r['jb_feasible'] for r in es),both_feasible=len(both),only_ours=sum(r['ours_feasible'] and not r['jb_feasible'] for r in es),only_jb=sum(r['jb_feasible'] and not r['ours_feasible'] for r in es),neither=sum(not r['ours_feasible'] and not r['jb_feasible'] for r in es),cheaper=sum(r['ours_total']<r['jb_total'] for r in both),tied=sum(r['ours_total']==r['jb_total'] for r in both),mean_ours_total=None if not both else statistics.mean(r['ours_total'] for r in both),mean_jb_total=None if not both else statistics.mean(r['jb_total'] for r in both),mean_saving=None if not sav else statistics.mean(sav),median_saving=None if not sav else statistics.median(sav),min_saving=None if not sav else min(sav),max_saving=None if not sav else max(sav))
        ck(all(close(z[k],v) for k,v in expected.items()),'quote_summary:'+str((z['span'],z['profile'],z['r'])))
    ck(len(summary['budget_groups'])==12 and len(summary['quote_groups'])==24 and len(summary['seed_groups'])==20,'summary_group_counts')
    for z in summary['seed_groups']:
        bb=[r for r in budgets.values() if r['seed']==z['seed']];qq=[r for r in quotes.values() if r['seed']==z['seed']];both=[r for r in qq if r['ours_feasible'] and r['jb_feasible']]
        expected=dict(budget_cells=len(bb),strict_budget=sum(r['strict'] for r in bb),quote_requests=len(qq),both_feasible=len(both),cheaper=sum(r['ours_total']<r['jb_total'] for r in both),tied=sum(r['ours_total']==r['jb_total'] for r in both),only_ours=sum(r['ours_feasible'] and not r['jb_feasible'] for r in qq),neither=sum(not r['ours_feasible'] and not r['jb_feasible'] for r in qq),conditional_mean_saving=None if not both else statistics.mean(float(Q(r['saving'])) for r in both))
        ck(all(close(z[k],v) for k,v in expected.items()),'seed_summary:'+str(z['seed']))
    bb=list(budgets.values());qq=list(quotes.values());ss=summary['seed_groups']
    expected_totals=dict(strict_budget=sum(r['strict'] for r in bb),tied_budget=sum(Q(r['ours_width'])==Q(r['jb_width']) for r in bb),both_feasible=sum(r['ours_feasible'] and r['jb_feasible'] for r in qq),cheaper=sum(r['ours_feasible'] and r['jb_feasible'] and r['ours_total']<r['jb_total'] for r in qq),tied=sum(r['ours_feasible'] and r['jb_feasible'] and r['ours_total']==r['jb_total'] for r in qq),only_ours=sum(r['ours_feasible'] and not r['jb_feasible'] for r in qq),only_jb=sum(r['jb_feasible'] and not r['ours_feasible'] for r in qq),neither=sum(not r['ours_feasible'] and not r['jb_feasible'] for r in qq),seeds_with_budget_gain=sum(r['strict_budget']>0 for r in ss),seeds_with_cost_gain=sum(r['cheaper']>0 for r in ss),seeds_with_unique_feasibility=sum(r['only_ours']>0 for r in ss))
    ck(summary['totals']==expected_totals,'summary_totals')
    expected_counts=dict(pools=len(pools),subset_rows=64*len(pools),unique_w_catalogs=len(audited_w),unique_w_subset_solves=64*len(audited_w),exact_directional_lps=128*len(audited_w),fixed_budget_rows=len(budgets),precision_requests=len(quotes),fast_oracle_checks=len(runs['fast_oracle_checks']))
    ck(runs['counts']==summary['counts']==expected_counts,'summary_counts')
    out=dict(status='PASS' if not errors else 'FAIL',protocol_sha=sha(PF),runs_sha=sha(R/(N+'_runs.json')),summary_sha=sha(R/(N+'_summary.json')),checks=dict(counts),errors=errors,
             independent_scope='Regenerated new-seed inputs; exact primal/dual certificates and two-world witnesses for all unique W LPs; all robust/nominal determinant values and actions; all grouped arithmetic. No production imports.',
             counts=dict(pools=len(pools),unique_w_catalogs=len(audited_w),certified_directional_lps=counts['dual_optimal'],subset_determinants=counts['determinants'],budget_rows=len(budgets),quote_rows=len(quotes)),dual_support_counts=dict(dual_support_counts),totals=summary['totals'])
    (R/(N+'_independent_audit.json')).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False),flush=True)
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
