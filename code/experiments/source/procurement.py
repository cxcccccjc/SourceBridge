"""Simulation procurement under a fixed seed and candidate-pool protocol.

Only public candidate geometry and prices are generated. No predictions,
reports, attack outcomes, or evaluation truths are loaded by selection.
"""
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
import os
for env_name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[env_name] = '1'
import sys, json, hashlib, argparse, itertools, math, time, statistics, csv
from fractions import Fraction as F
B = Path(__file__).resolve().parent
sys.path.insert(0, str(B/'deps'))
import numpy as np
from reference_geometry import worst_width, jsonable
from additive_selection import threshold_minimum
import robust_doptimal as jb
R = B/'results'
N = 'procurement'
PF = B/(N+'_protocol.json')


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x): return hashlib.sha256(json.dumps(jsonable(x), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def path(s): return R/(N+s)
def dump(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(jsonable(x), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def freeze():
    assert not PF.exists() and not path('_inputs.json').exists(), 'Preserve existing freeze and inputs'
    names = [Path(__file__).name, N+'_audit.py', 'reference_geometry.py',
             'additive_selection.py', 'catalog_selection.py',
             'robust_doptimal.py', 'procurement_comparator_specification.md']
    p = dict(status='FROZEN_BEFORE_NEW_SEED_GENERATION', scope='New-seed pure-simulation procurement confirmation; not new public data or point-inference evidence',
             seeds=list(range(840101,840121)), spans=[20,150,450], profiles=['exact','heterogeneous'],
             center=250.0, positions=[-.5,-.3,-.1,.1,.3,.5], heterogeneous_radii=[2.,5.,10.,20.,40.,80.],
             randomness='NumPy default_rng SeedSequence([seed,stream]); stream0 uniform(-.9,.9,6), stream1 permutation(h), stream2 integers(1,5,6). Streams are independent; u, permutation and costs shared across all six paired cells of a seed.',
             public_geometry='nominal q=center+span*positions; public center=q+u*h; anchors=(center-h,center+h,5). Exact h=0. No truncation at target domain: anchor values need not lie in target interval.',
             costs='Six iid integers1..4 independent of h and u; same seed costs in all spans/profiles',
             spec=dict(target=[0.,500.], gain_bounds=[.7,1.3], epsilon=5., m=9, f=2),
             fixed_budgets=['floor(sum(costs)/3)','floor(2*sum(costs)/3)'], radii=[20,50,100,150],
             coarse_cost='sum(64 if h_i==0 else max(1,ceil(20/h_i)))', total_cost='coarse_cost+9*(1+actual per-source selected anchor cost)',
             sourcebridge='All64 exact W subsets, exact tie min(W,cost,count,IDs); threshold_minimum on <=2 supports independently compared for each budget and radius.',
             comparator='Joshi-Boyd IEEE TSP2009 DOI10.1109/TSP.2008.2007095 Eq21 robust D criterion; existing frozen adapter unchanged; full subsets, rank(-Drob,-Dnom,cost,count,IDs).',
             comparator_quote='All integer-budget D-opt actions evaluated by shared exact W; minimum actual-cost passing action. W never enters per-budget D objective.',
             numerics='Saved float inputs interpreted exactly as binary rational numbers. No tolerance for W<=2r or method ranking. Exact W cached only by complete anchor/spec identity.',
             retention='All120 pools,240 budget cells,480 radius requests; no selective deletion, outcome-based scene selection, seed replacement, or threshold change.',
             statistics='20 paired seeds are statistical units; describe each of6 span/profile strata and all4 radii/two budgets. Report seed consistency, mean/median/min/max conditional savings and feasibility. No p-values for exact-objective noninferiority.',
             timing='One complete pass; timings diagnostic only, no runtime superiority claim or repeat requirement.',
             audit='Independent saved-record check with separately coded input regeneration, exact LP primal/dual certificates, robust determinant recomputation, rankings/costs, and summary arithmetic; no production selector/evaluator imports.',
             expected=dict(pools=120, subset_rows=7680, fixed_budget_rows=240, precision_requests=480),
             source_sha256={name:sha(B/name) for name in names})
    dump(PF,p)
    print(json.dumps(dict(frozen=True, protocol_sha=sha(PF))), flush=True)


def verify():
    p=read(PF)
    for name,h in p['source_sha256'].items(): assert verify_identity(B/name,h), name
    return p


def prepare():
    p=verify(); assert not path('_inputs.json').exists(), 'Preserve generated inputs'
    pools=[]
    for seed in p['seeds']:
        u=np.random.default_rng(np.random.SeedSequence([seed,0])).uniform(-.9,.9,6)
        perm=np.random.default_rng(np.random.SeedSequence([seed,1])).permutation(p['heterogeneous_radii'])
        costs=np.random.default_rng(np.random.SeedSequence([seed,2])).integers(1,5,6).tolist()
        for span in p['spans']:
            q=p['center']+span*np.asarray(p['positions'])
            for profile in p['profiles']:
                h=np.zeros(6) if profile=='exact' else perm
                c=q+u*h
                anchors=np.column_stack([c-h,c+h,np.full(6,p['spec']['epsilon'])]).tolist()
                assert all(F(lo)<=F(float(v))<=F(hi) for v,(lo,hi,eps) in zip(q,anchors))
                row=dict(pool_id=f's{seed}_span{span}_{profile}', seed=seed, span=span, profile=profile, n=6,
                         anchors=anchors, centers=c.tolist(), radii=h.tolist(), costs=costs,
                         coarse_cost=sum(64 if v==0 else max(1,math.ceil(20/v)) for v in h),
                         budgets=[sum(costs)//3,2*sum(costs)//3])
                row['geometry_id']=digest(dict(anchors=anchors,costs=costs))
                pools.append(row)
    dump(path('_inputs.json'),dict(protocol_sha=sha(PF),pools=pools,counts=dict(pools=len(pools),distinct_geometry_and_cost=len({g['geometry_id'] for g in pools})),numeric_input_role='Public procurement input only'))
    print(json.dumps(dict(prepared=len(pools),inputs_sha=sha(path('_inputs.json')))),flush=True)


def fast_budget(entries,costs,budget):
    thresholds=sorted({e[k] for e in entries for k in ['positive','negative']})
    lo,hi=0,len(thresholds); cache={}
    def query(i):
        if i not in cache: cache[i]=threshold_minimum(entries,costs,thresholds[i])
        return cache[i]
    while lo<hi:
        mid=(lo+hi)//2; z=query(mid)
        if z is not None and z['cost']<=budget: hi=mid
        else: lo=mid+1
    assert lo<len(thresholds)
    return dict(bound=thresholds[lo],**query(lo),threshold_queries=len(cache))


def run():
    p=verify(); assert not path('_runs.json').exists(), 'Preserve completed runs'
    inp=read(path('_inputs.json')); assert inp['protocol_sha']==sha(PF)
    S=p['spec']; catalogs=[]; cache={}; budget_rows=[]; quotes=[]; checks=[]; start=time.perf_counter()
    for gi,g in enumerate(inp['pools']):
        wkey=digest(dict(anchors=g['anchors'],spec=S)); wp=path('_w_'+wkey+'.json')
        if wkey not in cache:
            if wp.exists():
                wdata=read(wp); assert wdata['protocol_sha']==sha(PF) and wdata['anchors']==g['anchors']
            else:
                tt=time.perf_counter(); rows=[]
                for k in range(7):
                    for ids in itertools.combinations(range(6),k):
                        w=worst_width([g['anchors'][i] for i in ids],S['target'],S['gain_bounds'],S['epsilon'],exact=True)
                        rows.append(dict(ids=ids,width=w['width'],positive=w['branches'][0]['width'],negative=w['branches'][1]['width'],exact_result=w))
                wdata=dict(protocol_sha=sha(PF),key=wkey,anchors=g['anchors'],subset_rows=rows,seconds=time.perf_counter()-tt)
                dump(wp,wdata)
            cache[wkey]=wdata
        wm={tuple(x['ids']):dict(ids=tuple(x['ids']),width=F(x['width']),positive=F(x['positive']),negative=F(x['negative']),cost=sum(g['costs'][i] for i in x['ids'])) for x in cache[wkey]['subset_rows']}
        entries=[x for x in wm.values() if len(x['ids'])<=2]
        jcat=jb.build_catalog(g['anchors'],g['costs']); track=[]
        for budget in range(sum(g['costs'])+1):
            a=jb.select_from_catalog(jcat,budget,source_count=S['m']); ids=a['anchor_ids']
            track.append(dict(budget=budget,ids=ids,cost=a['cost'],width=wm[ids]['width'],drob=a['worst_determinant_per_source'],dnom=a['nominal_determinant_per_source'],degenerate=a['robust_degenerate']))
        common={k:g[k] for k in ['pool_id','seed','span','profile','geometry_id']}
        catalogs.append(dict(**common,w_file=wp.name,w_sha=sha(wp),jb_catalog=jcat,trajectory=track))
        for bi,budget in enumerate(g['budgets']):
            gold=min((x for x in wm.values() if x['cost']<=budget),key=lambda x:(x['width'],x['cost'],len(x['ids']),x['ids']))
            fast=fast_budget(entries,g['costs'],budget)
            same=(fast['bound'],fast['cost'],len(fast['anchor_ids']),fast['anchor_ids'])==(gold['width'],gold['cost'],len(gold['ids']),gold['ids'])
            checks.append(dict(**common,kind='budget',budget=budget,slot=bi,fast=fast,rank_equal=same)); assert same,checks[-1]
            other=track[budget]
            budget_rows.append(dict(**common,budget_slot=bi,budget=budget,ours_ids=gold['ids'],jb_ids=other['ids'],ours_cost=gold['cost'],jb_cost=other['cost'],ours_width=gold['width'],jb_width=other['width'],strict=gold['width']<other['width'],saving=F(0) if other['width']==0 else 1-gold['width']/other['width'],jb_degenerate=other['degenerate']))
        for radius in p['radii']:
            candidates=[x for x in wm.values() if x['width']<=2*radius]
            gold=min(candidates,key=lambda x:(x['cost'],len(x['ids']),x['ids'])) if candidates else None
            fast=threshold_minimum(entries,g['costs'],F(2*radius))
            fkey=None if fast is None else (fast['cost'],len(fast['anchor_ids']),fast['anchor_ids'])
            gkey=None if gold is None else (gold['cost'],len(gold['ids']),gold['ids'])
            checks.append(dict(**common,kind='radius',r=radius,fast=fast,rank_equal=fkey==gkey)); assert fkey==gkey,checks[-1]
            passing=[x for x in track if x['width']<=2*radius]
            other=min(passing,key=lambda x:(x['cost'],len(x['ids']),x['ids'],x['budget'])) if passing else None
            ca=None if gold is None else g['coarse_cost']+S['m']*(1+gold['cost'])
            cb=None if other is None else g['coarse_cost']+S['m']*(1+int(other['cost']))
            assert other is None or (gold is not None and ca<=cb)
            quotes.append(dict(**common,r=radius,ours_feasible=gold is not None,jb_feasible=other is not None,
                               ours_ids=None if gold is None else gold['ids'],jb_ids=None if other is None else other['ids'],
                               ours_total=ca,jb_total=cb,coarse_cost=g['coarse_cost'],ours_width=None if gold is None else gold['width'],jb_width=None if other is None else other['width'],
                               saving=None if cb is None else F(cb-ca,cb),jb_degenerate=None if other is None else other['degenerate'],
                               jb_budget=None if other is None else other['budget'],jb_first_budget=None if not passing else passing[0]['budget']))
        print(json.dumps(dict(pools_done=gi+1,total=len(inp['pools']),unique_w_catalogs=len(cache),elapsed_seconds=time.perf_counter()-start)),flush=True)
    counts=dict(pools=len(catalogs),subset_rows=len(catalogs)*64,unique_w_catalogs=len(cache),unique_w_subset_solves=len(cache)*64,exact_directional_lps=len(cache)*128,fixed_budget_rows=len(budget_rows),precision_requests=len(quotes),fast_oracle_checks=len(checks))
    for k,v in p['expected'].items(): assert counts[k]==v,(k,counts[k],v)
    out=dict(status='PASS',protocol_sha=sha(PF),inputs_sha=sha(path('_inputs.json')),counts=counts,catalogs=catalogs,budget_rows=budget_rows,quotes=quotes,fast_oracle_checks=checks,seconds=time.perf_counter()-start)
    dump(path('_runs.json'),out); print(json.dumps(dict(status=out['status'],counts=counts)),flush=True)


def summarize(x,p):
    br=x['budget_rows']; qr=x['quotes']; bg=[]; qg=[]; sg=[]
    for span in p['spans']:
        for profile in p['profiles']:
            for slot in [0,1]:
                rr=[r for r in br if (r['span'],r['profile'],r['budget_slot'])==(span,profile,slot)]
                ss=[float(F(r['saving'])) for r in rr]
                bg.append(dict(span=span,profile=profile,budget_slot=slot,seeds=len(rr),strict=sum(r['strict'] for r in rr),tied=sum(F(r['ours_width'])==F(r['jb_width']) for r in rr),
                               mean_ours_radius=statistics.mean(float(F(r['ours_width']))/2 for r in rr),mean_jb_radius=statistics.mean(float(F(r['jb_width']))/2 for r in rr),
                               mean_saving=statistics.mean(ss),median_saving=statistics.median(ss),min_saving=min(ss),max_saving=max(ss),jb_degenerate=sum(r['jb_degenerate'] for r in rr)))
            for radius in p['radii']:
                rr=[r for r in qr if (r['span'],r['profile'],r['r'])==(span,profile,radius)]
                both=[r for r in rr if r['ours_feasible'] and r['jb_feasible']]; ss=[float(F(r['saving'])) for r in both]
                qg.append(dict(span=span,profile=profile,r=radius,seeds=len(rr),ours_feasible=sum(r['ours_feasible'] for r in rr),jb_feasible=sum(r['jb_feasible'] for r in rr),
                               both_feasible=len(both),only_ours=sum(r['ours_feasible'] and not r['jb_feasible'] for r in rr),only_jb=sum(r['jb_feasible'] and not r['ours_feasible'] for r in rr),
                               neither=sum(not r['ours_feasible'] and not r['jb_feasible'] for r in rr),cheaper=sum(r['ours_total']<r['jb_total'] for r in both),tied=sum(r['ours_total']==r['jb_total'] for r in both),
                               mean_ours_total=None if not both else statistics.mean(r['ours_total'] for r in both),mean_jb_total=None if not both else statistics.mean(r['jb_total'] for r in both),
                               mean_saving=None if not ss else statistics.mean(ss),median_saving=None if not ss else statistics.median(ss),min_saving=None if not ss else min(ss),max_saving=None if not ss else max(ss)))
    for seed in p['seeds']:
        bb=[r for r in br if r['seed']==seed]; qq=[r for r in qr if r['seed']==seed]; both=[r for r in qq if r['ours_feasible'] and r['jb_feasible']]
        sg.append(dict(seed=seed,budget_cells=len(bb),strict_budget=sum(r['strict'] for r in bb),quote_requests=len(qq),both_feasible=len(both),
                       cheaper=sum(r['ours_total']<r['jb_total'] for r in both),tied=sum(r['ours_total']==r['jb_total'] for r in both),
                       only_ours=sum(r['ours_feasible'] and not r['jb_feasible'] for r in qq),neither=sum(not r['ours_feasible'] and not r['jb_feasible'] for r in qq),
                       conditional_mean_saving=None if not both else statistics.mean(float(F(r['saving'])) for r in both)))
    return dict(status='COMPLETED',scope=p['scope'],protocol_sha=sha(PF),runs_sha=sha(path('_runs.json')),counts=x['counts'],
                budget_groups=bg,quote_groups=qg,seed_groups=sg,
                totals=dict(strict_budget=sum(r['strict'] for r in br),tied_budget=sum(F(r['ours_width'])==F(r['jb_width']) for r in br),
                            both_feasible=sum(r['ours_feasible'] and r['jb_feasible'] for r in qr),cheaper=sum(r['ours_feasible'] and r['jb_feasible'] and r['ours_total']<r['jb_total'] for r in qr),
                            tied=sum(r['ours_feasible'] and r['jb_feasible'] and r['ours_total']==r['jb_total'] for r in qr),
                            only_ours=sum(r['ours_feasible'] and not r['jb_feasible'] for r in qr),only_jb=sum(r['jb_feasible'] and not r['ours_feasible'] for r in qr),
                            neither=sum(not r['ours_feasible'] and not r['jb_feasible'] for r in qr),seeds_with_budget_gain=sum(r['strict_budget']>0 for r in sg),seeds_with_cost_gain=sum(r['cheaper']>0 for r in sg),
                            seeds_with_unique_feasibility=sum(r['only_ours']>0 for r in sg)))


def report():
    p=verify();x=read(path('_runs.json'));assert x['protocol_sha']==sha(PF) and x['inputs_sha']==sha(path('_inputs.json'))
    s=summarize(x,p);dump(path('_summary.json'),s)
    for suffix,rows in [('_budgets.csv',x['budget_rows']),('_quotes.csv',x['quotes']),('_seeds.csv',s['seed_groups'])]:
        with path(suffix).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# 新种子仿真采购确认：SourceBridge 与 JB-RobustD','',
           '预先冻结20个新种子、3种锚跨度、2种精度分布，完成120个池。费用为独立随机整数并在同seed各单元配对；未更改困难池、预算或精度请求。本实验扩展采购证据，未新增公开数据集、点估计、攻击或密码学实验。','',
           'JB为Joshi–Boyd IEEE TSP2009已发表鲁棒D设计准则的公开校准适配。每预算按原D目标选取；同精度报价使用共同W证书扫描。它不是近期群智SOTA，也不是原文凸松弛求解器的速度复现。','',
           '统计单位为20个seed，每个表格单元均覆盖20个seed；同seed的6池及多个阈值相关，不作为独立样本。W不劣来自精确目标优化，不做针对必然不劣的p值检验。','',
           '## 同预算保证','', '| span | profile | 预算占比 | 改善/持平 | 平均保证半径 ours/JB | 宽度缩小均值/中位数 | JB退化 |','|---|---|---|---:|---:|---:|---:|']
    for z in s['budget_groups']:
        lines.append(f"| {z['span']} | {z['profile']} | {'1/3' if z['budget_slot']==0 else '2/3'} | {z['strict']}/{z['tied']} | {z['mean_ours_radius']:.3f}/{z['mean_jb_radius']:.3f} | {100*z['mean_saving']:.2f}%/{100*z['median_saving']:.2f}% | {z['jb_degenerate']} |")
    lines+=['','## 同精度总成本','', '| span | profile | r | 双方可达 | ours独有 | 双方不可达 | 省费/持平 | 平均总成本 ours/JB | 省费均值/中位数 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for z in s['quote_groups']:
        cost='—' if z['mean_ours_total'] is None else f"{z['mean_ours_total']:.2f}/{z['mean_jb_total']:.2f}"
        saving='—' if z['mean_saving'] is None else f"{100*z['mean_saving']:.2f}%/{100*z['median_saving']:.2f}%"
        lines.append(f"| {z['span']} | {z['profile']} | {z['r']} | {z['both_feasible']} | {z['only_ours']} | {z['neither']} | {z['cheaper']}/{z['tied']} | {cost} | {saving} |")
    totals=s['totals'];lines+=['',f"跨seed：{totals['seeds_with_budget_gain']}/20 在至少一个预算单元严格改善；{totals['seeds_with_cost_gain']}/20 在至少一个共同可达请求省费。不能将“至少一个”解释成所有单元都获益。各seed完整明细保留在结果文件。",'',
         '总成本包括粗测费用及9个来源的目标报告和所购锚；是规定成本单位，不是人民币。不可达费用为null，未记0；成本均值/节约只在双方可达请求计算。', '',
         'exact跨seed可能共享相同锚几何，但价格不同；完整64子集均保留，完全相同锚值的W精确求解仅缓存一次。此缓存不改变选择、困难池或统计单位。','',
         f"协议SHA256：`{sha(PF)}`。独立核验结果另列，不能将本执行完成状态预先写作审计通过。",'']
    (B/(N+'_note.md')).write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(counts=s['counts'],totals=totals,budget_groups=s['budget_groups'],quote_groups=s['quote_groups']),ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','run','report']);args=ap.parse_args();globals()[args.action]()
