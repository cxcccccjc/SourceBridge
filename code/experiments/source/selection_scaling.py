"""Frozen, public-design-only scaling control; no paper baseline or accuracy data."""
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'deps'))
import numpy as np
import scipy
from catalog_selection import pair_catalog, design_anchors
from additive_selection import design_anchors_additive

PREFIX = 'selection_scaling'
PROTOCOL = HERE / (PREFIX + '_protocol.json')
INPUTS = HERE / (PREFIX + '_inputs.json')
RESULTS = HERE / 'results'
TIMINGS = RESULTS / (PREFIX + '_timings.json')
VERIFY = RESULTS / (PREFIX + '_verification.json')
EXPECTED_SELECTION_SOURCE = '83988817aa1313f7469ebd5a9453c3949b2e3e870727bbaf6b83437ad5f1bdca'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def freeze():
    if PROTOCOL.exists() or INPUTS.exists():
        raise RuntimeError('Existing freeze preserved; refusing overwrite')
    theory_path = RESULTS / 'additive_combinatorics_check.json'
    theory = json.loads(theory_path.read_text(encoding='utf-8'))
    assert theory['status'] == 'PASS' and theory['implementation_sha256'] == EXPECTED_SELECTION_SOURCE
    assert verify_identity(HERE / 'additive_selection.py', EXPECTED_SELECTION_SOURCE)
    cases = []
    for n in [8, 16, 32, 64]:
        for seed in [731991, 731992]:
            rng = np.random.default_rng(np.random.SeedSequence([seed, n]))
            centers = rng.uniform(0.0, 500.0, n).tolist()
            halfwidths = rng.choice([0, 2, 5, 10, 20, 50], size=n).tolist()
            costs = rng.integers(1, 5, size=n).tolist()
            cases.append(dict(id=f'n{n}_s{seed}', n=n, seed=seed, centers=centers,
                              halfwidths=halfwidths, costs=costs,
                              anchors=[[c-h, c+h, 5.0] for c, h in zip(centers, halfwidths)]))
    write(INPUTS, dict(cases=cases))
    deps = [Path(__file__).name, 'catalog_selection.py',
            'additive_selection.py', 'reference_geometry.py',
            'check_additive_combinatorics.py',
            'results/additive_combinatorics_check.json', INPUTS.name,
            PREFIX + '_protocol.md']
    protocol = dict(
        status='FROZEN_BEFORE_EXECUTION', scope='Public-design scaling; mathematical enumeration control; zero literature baselines',
        model=dict(target=[0.0, 500.0], gain_bounds=[0.7, 1.3], epsilon_target=5.0),
        n=[8, 16, 32, 64], seeds=[731991, 731992], budgets=[4, 8],
        generation='NumPy default_rng(SeedSequence([seed,n])); centers uniform[0,500]; independent radii choice[0,2,5,10,20,50]; integer costs1..4; retain generated ID order; no interval clipping',
        warmups_per_method_condition=1, timed_repeats_per_method_condition=3,
        methods=['exhaustive_pair_pair', 'additive_representative'],
        comparable_time='External wall time of full public API with same precomputed actual LP catalog; includes selected-union two-LP verification in both APIs',
        order='Alternate method order using (condition_index+phase_repeat) parity, including warmup phase_repeat=-1',
        catalog_policy='One actual pair_catalog per n/seed, same in-memory object reused for both methods and both budgets; measure catalog separately',
        estimate_policy='cold_estimate_seconds = real_catalog_seconds + median_warm_phase_seconds; an estimate, not measured full rerun',
        verification_policy='Save all timing results first; separate verify invocation checks complete tuple(score,cost,union_size,IDs), all repeats including warmup, and catalog identity',
        failures='Preserve raw exceptions, do not tune or silently replace inputs; source changes abort',
        no_accuracy_or_truth_inputs=True, expected_conditions=16, expected_catalogs=8,
        source_sha256={name: sha(HERE / name) for name in deps})
    write(PROTOCOL, protocol)
    print(json.dumps(dict(frozen=True, protocol_sha256=sha(PROTOCOL), cases=len(cases), conditions=16)), flush=True)


def validate_freeze():
    p = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    for name, expected in p['source_sha256'].items():
        if not verify_identity(HERE/name,expected):
            raise RuntimeError('Frozen source mismatch: ' + name)
    return p


def run():
    p = validate_freeze()
    if TIMINGS.exists():
        raise RuntimeError('Existing timing evidence preserved; refusing overwrite')
    inputs = json.loads(INPUTS.read_text(encoding='utf-8'))
    result = dict(status='RUNNING', protocol_sha256=sha(PROTOCOL),
                  environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                                   platform=platform.platform(), processor=platform.processor(), logical_cpus=os.cpu_count(),
                                   threading_env={k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']}),
                  catalog_records=[], records=[])
    write(TIMINGS, result)
    methods = dict(exhaustive_pair_pair=design_anchors, additive_representative=design_anchors_additive)
    model = p['model']
    start_all = time.perf_counter()
    condition_index = 0
    for case in inputs['cases']:
        start = time.perf_counter()
        catalog = pair_catalog(case['anchors'], model['target'], model['gain_bounds'], model['epsilon_target'])
        catalog_seconds = time.perf_counter() - start
        catalog_hash = hashlib.sha256(json_bytes(catalog)).hexdigest()
        catalog_path = RESULTS / f"{PREFIX}_catalog_{case['id']}.json"
        write(catalog_path, dict(case_id=case['id'], catalog_sha256=catalog_hash, catalog=catalog))
        result['catalog_records'].append(dict(case_id=case['id'], n=case['n'], seed=case['seed'],
            seconds=catalog_seconds, entries=len(catalog), LPs=2*len(catalog),
            canonical_sha256=catalog_hash, file=catalog_path.name, file_sha256=sha(catalog_path)))
        write(TIMINGS, result)
        print(f"Catalog {case['id']}: {len(catalog)} entries; {catalog_seconds:.6f}s", flush=True)
        for budget in p['budgets']:
            condition_id = f"{case['id']}_B{budget}"
            # Calls finish before any cross-method comparison or endpoint audit.
            for repetition in [-1, 0, 1, 2]:
                order = list(methods)
                if (condition_index + repetition) % 2:
                    order.reverse()
                for order_index, method_name in enumerate(order):
                    record = dict(condition_id=condition_id, case_id=case['id'], n=case['n'], seed=case['seed'],
                                  budget=budget, phase='warmup' if repetition < 0 else 'timed', repetition=repetition,
                                  method=method_name, order_index=order_index, catalog_sha256=catalog_hash)
                    tick = time.perf_counter()
                    try:
                        answer = methods[method_name](case['anchors'], model['target'], model['gain_bounds'],
                                                      model['epsilon_target'], case['costs'], budget, catalog=catalog)
                        elapsed = time.perf_counter() - tick
                        record.update(status='OK', seconds=elapsed, result=answer)
                    except Exception as exc:
                        elapsed = time.perf_counter() - tick
                        record.update(status='ERROR', seconds=elapsed, error=str(exc), traceback=traceback.format_exc())
                    result['records'].append(record)
                    write(TIMINGS, result)
                print(f"{condition_id} repeat={repetition} complete", flush=True)
            condition_index += 1
    result['status'] = 'TIMING_COMPLETE'
    result['wall_seconds_including_catalog_io_and_warmups'] = time.perf_counter() - start_all
    result['verification_not_run'] = True
    write(TIMINGS, result)
    print(json.dumps(dict(status=result['status'], records=len(result['records']), timings_sha256=sha(TIMINGS))), flush=True)


def verify():
    p = validate_freeze()
    if VERIFY.exists():
        raise RuntimeError('Existing verification evidence preserved; refusing overwrite')
    result = json.loads(TIMINGS.read_text(encoding='utf-8'))
    checks = []; failures = []
    if result['status'] != 'TIMING_COMPLETE':
        failures.append('timing incomplete')
    if result['protocol_sha256'] != sha(PROTOCOL):
        failures.append('protocol hash mismatch')
    inputs = json.loads(INPUTS.read_text(encoding='utf-8'))['cases']
    for c in result['catalog_records']:
        path = RESULTS / c['file']
        actual = json.loads(path.read_text(encoding='utf-8'))['catalog']
        if sha(path) != c['file_sha256'] or hashlib.sha256(json_bytes(actual)).hexdigest() != c['canonical_sha256']:
            failures.append('catalog hash mismatch: ' + c['case_id'])
        if len(actual) != 1+c['n']+c['n']*(c['n']-1)//2:
            failures.append('catalog cardinality mismatch: ' + c['case_id'])
    for case in inputs:
        catalog_meta = next(c for c in result['catalog_records'] if c['case_id'] == case['id'])
        for budget in p['budgets']:
            cid = f"{case['id']}_B{budget}"
            rows = [r for r in result['records'] if r['condition_id'] == cid]
            expected = {(method, repetition) for method in p['methods'] for repetition in [-1, 0, 1, 2]}
            issues = []
            if len(rows) != 8 or {(r['method'], r['repetition']) for r in rows} != expected:
                issues.append('incomplete calls')
            keys = []
            for r in rows:
                if r['status'] != 'OK':
                    issues.append('method error: ' + r['method'])
                    continue
                a = r['result']; ids = tuple(a['anchor_ids'])
                keys.append((a['selection_bound'], a['cost'], len(ids), ids))
                if a['cost'] != sum(case['costs'][i] for i in ids) or a['cost'] > budget:
                    issues.append('cost mismatch')
                if abs(a['width']-a['selection_bound']) > 1e-7:
                    issues.append('selected union LP mismatch')
                if r['catalog_sha256'] != catalog_meta['canonical_sha256']:
                    issues.append('different catalog')
            if len(set(keys)) != 1:
                issues.append('full selection tuple disagreement')
            checks.append(dict(condition_id=cid, status='PASS' if not issues else 'FAIL',
                               calls=len(rows), selected_key=list(keys[0]) if keys else None, issues=issues))
            failures.extend(cid+': '+s for s in issues)
    if len(result['catalog_records']) != 8 or len(result['records']) != 128:
        failures.append('overall counts mismatch')
    audit = dict(status='PASS' if not failures else 'FAIL', protocol_sha256=sha(PROTOCOL),
                 timings_sha256=sha(TIMINGS), timing_finished_before_verification=True,
                 scope='Agreement on saved floating LP catalogs, not an exact proof of original LP optima',
                 compared_conditions=len(checks), warmup_calls=32, timed_calls=96,
                 fields=['selection_bound', 'union_cost', 'union_cardinality', 'sorted_union_ids'],
                 checks=checks, failures=failures)
    write(VERIFY, audit)
    print(json.dumps({k:v for k,v in audit.items() if k != 'checks'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['freeze', 'run', 'verify'])
    args = parser.parse_args()
    globals()[args.mode]()
