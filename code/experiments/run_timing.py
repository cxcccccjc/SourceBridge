"""Verify saved timing evidence, check a small catalog, or measure query batches."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'source'))
import measure_catalog_queries as measurement


def smoke():
    """Recompute the eight-reference catalog and two complete query actions."""
    from catalog_selection import pair_catalog, design_anchors
    from additive_selection import design_anchors_additive
    protocol = json.loads(measurement.PROTO.read_text('utf-8'))
    cases = json.loads((ROOT / 'source/selection_scaling_inputs.json').read_text('utf-8'))['cases']
    case = next(c for c in cases if c['n'] == 8 and c['seed'] == 731991)
    catalog = pair_catalog(case['anchors'], **protocol['model'])
    records = json.loads((ROOT / 'timing_reference/end_to_end_raw.json').read_text('utf-8'))['records']
    signature = hashlib.sha256(json.dumps(catalog, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert signature == next(r['catalog_sha256'] for r in records if r['n'] == 8)
    for budget in (4, 8):
        selections = [f(case['anchors'], costs=case['costs'], budget=budget,
                        catalog=catalog, **protocol['model'])
                      for f in (design_anchors, design_anchors_additive)]
        keys = [[r['selection_bound'], r['cost'], len(r['anchor_ids']), list(r['anchor_ids'])] for r in selections]
        record = next(r for r in records if r['n'] == 8 and budget in r['budgets'])
        assert keys[0] == keys[1] == record['result_tuples'][record['budgets'].index(budget)]
    print(json.dumps({'status': 'PASS', 'references': 8, 'budget_queries': [4, 8], 'algorithms': 2}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify', 'smoke', 'measure'))
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.action == 'smoke':
        smoke()
    elif args.action == 'verify':
        measurement.verify()
    else:
        output = args.output_dir.resolve() if args.output_dir else ROOT / 'generated/timing'
        output.mkdir(parents=True, exist_ok=True)
        measurement.HERE = output
        measurement.RAW = output / 'end_to_end_raw.json'
        measurement.run()
        measurement.verify()


if __name__ == '__main__':
    main()
