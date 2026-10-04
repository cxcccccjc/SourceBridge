"""Summarize already completed scaling timings, after saved-result verification."""
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
import csv
import hashlib
import json
import statistics as st

HERE = Path(__file__).resolve().parent
PREFIX = 'selection_scaling'
RESULTS = HERE / 'results'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    timing_path = RESULTS / (PREFIX + '_timings.json')
    verify_path = RESULTS / (PREFIX + '_verification.json')
    timings = json.loads(timing_path.read_text(encoding='utf-8'))
    audit = json.loads(verify_path.read_text(encoding='utf-8'))
    assert audit['status'] == 'PASS'
    assert audit['timings_sha256'] == sha(timing_path)
    cats = {r['case_id']: r for r in timings['catalog_records']}
    rows = []
    for check in audit['checks']:
        cid = check['condition_id']
        raw = [r for r in timings['records'] if r['condition_id'] == cid and r['phase'] == 'timed']
        old = [r for r in raw if r['method'] == 'exhaustive_pair_pair']
        new = [r for r in raw if r['method'] == 'additive_representative']
        om = st.median(r['seconds'] for r in old)
        nm = st.median(r['seconds'] for r in new)
        cat = cats[raw[0]['case_id']]
        a = new[0]['result']
        rows.append(dict(condition_id=cid, n=raw[0]['n'], seed=raw[0]['seed'], budget=raw[0]['budget'],
                         catalog_entries=cat['entries'], catalog_LPs=cat['LPs'], catalog_seconds=cat['seconds'],
                         old_phase_median_seconds=om, new_phase_median_seconds=nm,
                         phase_speedup=om/nm, old_phase_min_seconds=min(r['seconds'] for r in old),
                         old_phase_max_seconds=max(r['seconds'] for r in old),
                         new_phase_min_seconds=min(r['seconds'] for r in new),
                         new_phase_max_seconds=max(r['seconds'] for r in new),
                         new_internal_pairing_median_seconds=st.median(r['result']['pairing_seconds'] for r in new),
                         old_cold_estimate_seconds=cat['seconds']+om,
                         new_cold_estimate_seconds=cat['seconds']+nm,
                         cold_estimate_speedup=(cat['seconds']+om)/(cat['seconds']+nm),
                         selection_bound=a['selection_bound'], cost=a['cost'], selected_count=len(a['anchor_ids']),
                         selected_ids=a['anchor_ids'], threshold_queries=a['threshold_queries'],
                         total_candidate_checks=a['total_candidate_checks']))
    grouped = []
    for n in [8, 16, 32, 64]:
        part = [r for r in rows if r['n'] == n]
        cat_seconds = [c['seconds'] for c in cats.values() if c['n'] == n]
        grouped.append(dict(n=n, conditions=len(part),
            catalog_median_seconds=st.median(cat_seconds),
            old_phase_median_of_medians_seconds=st.median(r['old_phase_median_seconds'] for r in part),
            new_phase_median_of_medians_seconds=st.median(r['new_phase_median_seconds'] for r in part),
            phase_speedup_median=st.median(r['phase_speedup'] for r in part),
            phase_speedup_min=min(r['phase_speedup'] for r in part),
            phase_speedup_max=max(r['phase_speedup'] for r in part),
            cold_estimate_speedup_median=st.median(r['cold_estimate_speedup'] for r in part),
            old_cold_estimate_median_seconds=st.median(r['old_cold_estimate_seconds'] for r in part),
            new_cold_estimate_median_seconds=st.median(r['new_cold_estimate_seconds'] for r in part)))
    summary = dict(status='PASS', scope='Mathematical enumeration control; zero new literature baseline papers',
                   timings_sha256=sha(timing_path), verification_sha256=sha(verify_path),
                   conditions=len(rows), real_catalogs=len(cats), real_catalog_LPs=sum(c['LPs'] for c in cats.values()),
                   timed_calls=audit['timed_calls'], warmup_calls=audit['warmup_calls'],
                   exact_tuple_agreement_conditions=audit['compared_conditions'],
                   selected_anchor_counts=sorted({r['selected_count'] for r in rows}),
                   every_condition_new_phase_faster=all(r['phase_speedup']>1 for r in rows),
                   groups=grouped, records=rows, environment=timings['environment'])
    (RESULTS / (PREFIX + '_summary.json')).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    with (RESULTS / (PREFIX + '_summary.csv')).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ['# 加性费用选锚算法：规模实验结果', '',
             '本实验比较同一公开设计模型、同一真实 LP 目录上的两种数学实现：原始证书对穷举与加性费用快速选择。新增文献基线为 **0 篇**；它不验证真相发现准确率、隐私安全或相对已发表论文的整系统速度。', '',
             '## 冻结设计与结果一致性', '',
             'n=8/16/32/64，种子 731991/731992，预算 4/8，共 16 条件。每条件每方法预热 1 次、正式计时 3 次并交替顺序。8 个真实 LP 目录仅计算一次并在同一 n/seed 内供全部调用复用。全部计时保存后才执行独立结果比对。', '',
             f"共 {summary['real_catalog_LPs']:,} 次目录 LP 求解，96 次正式计时与 32 次预热。16/16 条件的全部调用在最优分数、并集费用、并集锚数及有序 ID 上完全一致；无运行错误。所选锚数集合为 {summary['selected_anchor_counts']}。这证明保存的浮点目录上选择结果一致，不等于对原 LP 进行了有理数精确认证。", '',
             '## 实测目录复用阶段', '',
             '两方法均测量提供既有目录后的完整 API 墙钟时间，包含最终所选并集的两个 LP 核查。表中阶段毫秒数是每条件三次中位数，再在该 n 的四个条件间取中位数；加速列是四个配对时间比的中位数及范围。不能把该加速比解读为包含目录构建的全流程加速。', '',
             '| n | 目录规模 | 原穷举阶段 ms | 快速选择阶段 ms | 阶段加速中位数 [范围] |',
             '|---:|---:|---:|---:|---:|']
    for g in grouped:
        n = g['n']
        lines.append(f"| {n} | {1+n+n*(n-1)//2} | {1000*g['old_phase_median_of_medians_seconds']:.3f} | {1000*g['new_phase_median_of_medians_seconds']:.3f} | {g['phase_speedup_median']:.2f}× [{g['phase_speedup_min']:.2f}, {g['phase_speedup_max']:.2f}] |")
    lines += ['', '## 目录构建与冷启动组合估计', '',
              '目录构建时间为实际测量。后两列使用每个条件的“实测目录时间 + 该方法阶段中位时间”再聚合，是冷启动的组合估计，**不是重新计时的完整流程**。共享目录在多个预算或重复求解间可摊销，但不可在只执行一次时删除目录成本。', '',
              '| n | 实测目录中位数 s | 原方法冷启动估计 s | 新方法冷启动估计 s | 配对估计加速中位数 |',
              '|---:|---:|---:|---:|---:|---:|']
    for g in grouped:
        lines.append(f"| {g['n']} | {g['catalog_median_seconds']:.4f} | {g['old_cold_estimate_median_seconds']:.4f} | {g['new_cold_estimate_median_seconds']:.4f} | {g['cold_estimate_speedup_median']:.2f}× |")
    lines += ['', '## 可支撑与不可支撑的结论', '',
              '在非负加性费用、每个方向至多二锚证书的条件下，新选择算法已获独立组合检查，理论选择复杂度为 O(n² log n)，相对旧枚举的 O(n⁴) 有结构性改进。此次固定随机设计验证了浮点 LP 目录上的一致性并给出了规模实测。目录本身仍有 O(n²) 个固定小 LP，实际耗时不能隐去。', '',
              '这组实验不增加论文基线数量，不证明新颖性检索已经充分，不证明真值估计更准，不证明恶意源攻击下安全，也不比较其他经过优化的通用求解器。仅两个随机种子和一次机器运行，三次计时用于减少偶发波动，不是独立统计样本。机器负载和 Python 实现影响绝对时间。一般非加性费用不适用该加速。', '',
              '## 证据文件', '',
              f'- 协议 SHA256：`{timings["protocol_sha256"]}`。',
              f'- 计时 SHA256：`{sha(timing_path)}`。',
              f'- 结果核查 SHA256：`{sha(verify_path)}`。',
              f'- 完整原始计时：`results/{PREFIX}_timings.json`。',
              f'- 逐条件对比：`results/{PREFIX}_summary.csv`。',
              f'- 固定输入：`{PREFIX}_inputs.json`；8 个目录：`results/{PREFIX}_catalog_*.json`。',
              f'- 环境：Python {timings["environment"]["python"].split()[0]}，NumPy {timings["environment"]["numpy"]}，SciPy {timings["environment"]["scipy"]}；完整机器环境见计时文件。', '']
    (HERE / (PREFIX + '_note.md')).write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k not in ['records', 'environment']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
