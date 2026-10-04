"""Read-only arithmetic/hash audit. No solver calls, timing reruns or summary imports."""
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
from collections import Counter, defaultdict
import ast
import csv
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
PREFIX = 'selection_scaling'
EXPECTED_PROTOCOL = '9e84f0a55f805f096c8134aa28645a7b1cbd00530515f2c057292449f7fec7d3'
EXPECTED_TIMINGS = '19c3948db8ccf34152df6d8fc39efd19b1f56ae8da77a5896fe71dbc1cc8d2e1'
EXPECTED_SELECTION_SOURCE_SOURCE = '83988817aa1313f7469ebd5a9453c3949b2e3e870727bbaf6b83437ad5f1bdca'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def median(values):
    a = sorted(values)
    k = len(a)//2
    return a[k] if len(a)%2 else (a[k-1]+a[k])/2


def main():
    failures = []
    def check(ok, label):
        if not ok:
            failures.append(label)
    protocol_path = HERE / (PREFIX+'_protocol.json')
    timing_path = RESULTS / (PREFIX+'_timings.json')
    summary_path = RESULTS / (PREFIX+'_summary.json')
    verify_path = RESULTS / (PREFIX+'_verification.json')
    protocol = read(protocol_path); raw = read(timing_path); summary = read(summary_path); verify = read(verify_path)
    check(digest(protocol_path) == EXPECTED_PROTOCOL, 'original protocol hash')
    check(digest(timing_path) == EXPECTED_TIMINGS, 'original timing hash')
    check(raw['protocol_sha256'] == EXPECTED_PROTOCOL, 'raw-to-protocol hash')
    check(verify['timings_sha256'] == digest(timing_path), 'verification-to-timing hash')
    check(summary['timings_sha256'] == digest(timing_path), 'summary-to-timing hash')
    check(summary['verification_sha256'] == digest(verify_path), 'summary-to-verification hash')
    source_checks = []
    for name, expected in protocol['source_sha256'].items():
        actual = digest(HERE/name)
        source_checks.append(dict(file=name, expected=expected, actual=actual, matches=expected==actual))
        check(expected == actual, 'frozen source '+name)
    check(digest(HERE/'additive_selection.py') == EXPECTED_SELECTION_SOURCE_SOURCE, 'theory-reviewed implementation hash')
    inputs = {r['id']:r for r in read(HERE/(PREFIX+'_inputs.json'))['cases']}
    catalogs = {}; catalog_checks = []
    for meta in raw['catalog_records']:
        path = RESULTS/meta['file']; document = read(path); cat = document['catalog']
        canonical = hashlib.sha256(json.dumps(cat, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
        n = meta['n']; valid_count = 1+n+n*(n-1)//2
        checks = dict(case_id=meta['case_id'], file_hash_match=digest(path)==meta['file_sha256'],
                      canonical_hash_match=canonical==meta['canonical_sha256']==document['catalog_sha256'],
                      entry_count_match=len(cat)==valid_count==meta['entries'],
                      lp_count_match=meta['LPs']==2*valid_count,
                      measured_seconds=meta['seconds'])
        catalog_checks.append(checks)
        check(all(checks[k] for k in ['file_hash_match','canonical_hash_match','entry_count_match','lp_count_match']), 'catalog '+meta['case_id'])
        check(math.isfinite(meta['seconds']) and meta['seconds']>0, 'catalog positive timing '+meta['case_id'])
        catalogs[meta['case_id']] = (meta, {tuple(x['ids']):x for x in cat})
    grouped_raw = defaultdict(list)
    for r in raw['records']:
        grouped_raw[r['condition_id']].append(r)
    check(raw['status']=='TIMING_COMPLETE', 'timing completion')
    check(len(catalogs)==8 and len(inputs)==8 and len(grouped_raw)==16 and len(raw['records'])==128, 'overall cardinalities')
    check(sum(r['phase']=='timed' for r in raw['records'])==96, '96 timed calls')
    check(sum(r['phase']=='warmup' for r in raw['records'])==32, '32 warmup calls')
    condition_checks = []; recomputed = []
    for cid, calls in grouped_raw.items():
        case = inputs[calls[0]['case_id']]; budget = calls[0]['budget']; meta, by_id = catalogs[case['id']]
        expected_calls = {(m,r) for m in ['exhaustive_pair_pair','additive_representative'] for r in [-1,0,1,2]}
        check(len(calls)==8 and {(x['method'],x['repetition']) for x in calls}==expected_calls, 'call identities '+cid)
        keys = []; call_details = []
        for call in calls:
            check(call['status']=='OK', 'status '+cid)
            check(math.isfinite(call['seconds']) and call['seconds']>0, 'positive timing '+cid)
            check(call['catalog_sha256']==meta['canonical_sha256'], 'same catalog '+cid)
            a = call['result']; ids = tuple(a['anchor_ids']); p = tuple(a['positive_support']); q = tuple(a['negative_support'])
            score = max(by_id[p]['positive'],by_id[q]['negative'])
            cost = sum(case['costs'][i] for i in ids)
            key = (a['selection_bound'],a['cost'],len(ids),ids)
            valid = (ids==tuple(sorted(set(ids))) and len(ids)<=4 and tuple(sorted(set(p)|set(q)))==ids
                     and score==a['selection_bound'] and cost==a['cost'] and cost<=budget
                     and abs(a['width']-score)<=1e-7)
            check(valid, 'selection tuple/support/cost '+cid)
            keys.append(key)
            call_details.append(dict(method=call['method'], repetition=call['repetition'], key=key, support_score=score, valid=valid))
        same = len(set(keys))==1
        check(same, 'all eight selection tuples '+cid)
        condition_checks.append(dict(condition_id=cid, all_eight_selection_tuples_equal=same, calls=call_details))
        times = {m: [x['seconds'] for x in calls if x['method']==m and x['phase']=='timed'] for m in ['exhaustive_pair_pair','additive_representative']}
        om = median(times['exhaustive_pair_pair']); nm = median(times['additive_representative']); cs = meta['seconds']
        recomputed.append(dict(condition_id=cid, n=case['n'], seed=case['seed'], budget=budget,
            catalog_seconds=cs, old_phase_median_seconds=om, new_phase_median_seconds=nm,
            phase_speedup=om/nm, old_phase_min_seconds=min(times['exhaustive_pair_pair']),
            old_phase_max_seconds=max(times['exhaustive_pair_pair']),new_phase_min_seconds=min(times['additive_representative']),
            new_phase_max_seconds=max(times['additive_representative']),old_cold_estimate_seconds=cs+om,
            new_cold_estimate_seconds=cs+nm,cold_estimate_speedup=(cs+om)/(cs+nm),
            selection_bound=keys[0][0],cost=keys[0][1],selected_count=keys[0][2],selected_ids=list(keys[0][3])))
    originals = {r['condition_id']:r for r in summary['records']}
    with (RESULTS/(PREFIX+'_summary.csv')).open(encoding='utf-8-sig',newline='') as handle:
        csv_rows = {r['condition_id']:r for r in csv.DictReader(handle)}
    for row in recomputed:
        for key, value in row.items():
            check(originals[row['condition_id']][key]==value, f'summary cell {row["condition_id"]}/{key}')
            c = csv_rows[row['condition_id']][key]
            observed = ast.literal_eval(c) if key=='selected_ids' else (float(c) if isinstance(value,(int,float)) else c)
            check(observed==value, f'csv cell {row["condition_id"]}/{key}')
    aggregate = []
    for n in [8,16,32,64]:
        part = [r for r in recomputed if r['n']==n]
        row = dict(n=n,conditions=len(part),catalog_median_seconds=median([c[0]['seconds'] for c in catalogs.values() if c[0]['n']==n]),
            old_phase_median_of_medians_seconds=median([r['old_phase_median_seconds'] for r in part]),
            new_phase_median_of_medians_seconds=median([r['new_phase_median_seconds'] for r in part]),
            phase_speedup_median=median([r['phase_speedup'] for r in part]),phase_speedup_min=min(r['phase_speedup'] for r in part),
            phase_speedup_max=max(r['phase_speedup'] for r in part),cold_estimate_speedup_median=median([r['cold_estimate_speedup'] for r in part]),
            old_cold_estimate_median_seconds=median([r['old_cold_estimate_seconds'] for r in part]),
            new_cold_estimate_median_seconds=median([r['new_cold_estimate_seconds'] for r in part]))
        saved = next(x for x in summary['groups'] if x['n']==n)
        check(saved==row, 'aggregate all fields n='+str(n));aggregate.append(row)
    strong_path = RESULTS/'simulated_inference_strong_comparison.json'
    strong = read(strong_path)['rows']; counts = {}
    for prefix in ['same_purchase','all_purchase']:
        statuses = []
        for r in strong:
            delta = r['new_rmse']-r[prefix+'_best_rmse']
            status = 'win' if delta<0 else ('loss' if delta>0 else 'tie')
            check(status==r[prefix+'_result'], 'strong envelope label '+prefix)
            statuses.append(status)
        counts[prefix] = dict(Counter(statuses))
    check(len(strong)==72 and counts['same_purchase']=={'loss':48,'win':24} and counts['all_purchase']=={'loss':61,'win':11}, 'strong envelope counts')
    claims = [
        dict(claim='n64 repeated-budget phase speedup about133.76x', status='supported', evidence='raw paired API timings; same actual LP catalog'),
        dict(claim='n64 estimated cold-computation speedup about1.73x', status='supported_as_composite_estimate_only', evidence='measured catalog seconds + each method warm phase median'),
        dict(claim='full selection tuple agrees for16conditions and128calls', status='supported_on_saved_floating_catalogs', evidence='raw selected supports, catalog directional scores, costs and identities'),
        dict(claim='133.76x end-to-end or published-paper speedup', status='unsupported', evidence='catalog cost excluded from that ratio; old method is internal exhaustive control'),
        dict(claim='new method generally more accurate than strongest published configurations', status='unsupported', evidence='72 correlated development cells:24/48 same-purchase win/loss;11/61 all-policy envelope'),
        dict(claim='novelty, broad safety, and publication maturity settled by scaling', status='unsupported', evidence='not tested by arithmetic/hash/scaling checks')]
    audit = dict(status='PASS' if not failures else 'FAIL', mode='quick numeric-audit + claim-audit',
        independence='Separate read-only checker rederives raw arithmetic without importing original driver/report logic; same agent authored execution driver, so this is not independent-person replication.',
        no_timing_reruns=True, no_solver_calls=True, no_source_or_raw_result_modifications=True,
        original_protocol_sha256=EXPECTED_PROTOCOL,original_timings_sha256=EXPECTED_TIMINGS,
        checker_sha256=digest(Path(__file__)),source_checks=source_checks,catalog_checks=catalog_checks,
        checked_conditions=16,checked_call_tuples=128,recomputed_conditions=recomputed,recomputed_aggregates=aggregate,
        selection_checks=condition_checks,strong_comparison_sha256=digest(strong_path),strong_comparison_cells=72,
        strong_comparison_counts=counts,claim_evidence_matrix=claims,failures=failures,
        severity='No numerical inconsistency found' if not failures else 'See failures',
        skipped=['No new source-paper citation verification','No new solver/exact-LP certification','No new theory proof','No new experiment or timing'],
        no_invention_status='All numbers derived from saved artifacts; no new measurements',next_owner='Parent research owner')
    output = RESULTS/(PREFIX+'_independent_audit.json')
    output.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    text = ['# 加性费用规模实验：原始证据复核', '',
        f"结果：**{audit['status']}**。模式：轻量数值与主张核验。复核程序从原始计时重新计算，不调用原汇总程序；实验驱动和此次复核由同一执行者编写，因此不称为独立人员复现。没有重跑计时、调用求解器、修改模型或原始证据。", '',
        f"已核验冻结协议、{len(source_checks)} 个依赖/输入哈希及 8 个目录文件/规范内容哈希；16 条件共 128 个调用的分数、费用、并集大小与 ID 完全一致，正负证书可在目录中重建该分数。逐条件中位数、配对加速、目录组合估计，以及 JSON/CSV 的对应值全部复算一致。", '',
        '| n | 原阶段 ms | 新阶段 ms | 配对阶段加速中位数 | 目录中位 s | 配对冷启动组合估计加速 |',
        '|---:|---:|---:|---:|---:|---:|']
    for g in aggregate:
        text.append(f"| {g['n']} | {1000*g['old_phase_median_of_medians_seconds']:.3f} | {1000*g['new_phase_median_of_medians_seconds']:.3f} | {g['phase_speedup_median']:.2f}× | {g['catalog_median_seconds']:.4f} | {g['cold_estimate_speedup_median']:.2f}× |")
    text += ['', '阶段比较在共同目录已经给定时测量完整 API，双方均含选定并集的两个 LP 检查，因此计时口径一致。冷启动列使用实测目录成本加预热阶段中位数，仅为组合估计。n=64 的 133.76×只能写成相对自有穷举实现的目录复用阶段提速；含目录成本的组合估计约 1.73×。', '',
        '**贡献边界评价：** 快速选择把当前机制的计算可扩展性补强了：在规定的加性费用模型下，保留同一最优选择并降低枚举开销。它没有补上点估计的整体竞争力。跨度实验的 72 个相关开发条件中，新中点对同采购的最优论文配置为 24 胜、48 负，对同预算全部采购的事后最佳为 11 胜、61 负；这些参考界由评价结果事后挑选，不能当可部署算法，也不能用于独立显著性结论。当前更有证据的定位是“有条件误差界与预算选锚的结构性结果、加性费用下的快速求解”，不能改写成普遍更准确、端到端快 133.76 倍、超越论文方法或已完成安全/新颖性验证。', '',
        '严重度：未发现本次范围内的数值或哈希不一致。若省略目录成本或将内部枚举控制写成文献基线，则会构成重大主张失配。建议保留上面的限定措辞。此次未重新核验论文来源、理论证明或模型外安全性；这些属于原研究负责人的后续研究门槛。', '',
        f'核验 JSON：`results/{PREFIX}_independent_audit.json`；可复算检查器：`{PREFIX}_independent_audit.py`。所有原始计时与跨度记录原样保留。', '']
    note = HERE/(PREFIX+'_independent_note.md')
    note.write_text('\n'.join(text),encoding='utf-8')
    check('\ufffd' not in note.read_text(encoding='utf-8'), 'UTF8 replacement character')
    print(json.dumps(dict(status=audit['status'],sources=len(source_checks),catalogs=len(catalog_checks),conditions=16,calls=128,
                         strong_counts=counts,failures=failures),ensure_ascii=False))


if __name__=='__main__':
    main()
