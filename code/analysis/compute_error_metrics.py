"""Recompute the manuscript's error summaries from bundled frozen inputs.

Python standard library only. Default paths are relative to this script.
Optional --predictions-csv independently verifies the complete source record.
This program writes numerical reports, never manuscript files.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent
METRICS = {
    'clean_rmse': 'Square root of mean clean squared error over 213 targets.',
    'pooled_rmse': 'sqrt(sum_i sum_a e_i,a^2 / (N*|A|)); N=213 and |A|=14. Each of the 2982 target-condition cells has equal weight.',
    'targetwise_worst_rmse': 'R_A=sqrt(mean_i d_i^2), where d_i=max_{a in A}|e_i,a|.',
    'clean_mae': 'Mean clean absolute error over 213 targets.',
    'mean_targetwise_worst_absolute_error': 'M_A=mean_i d_i over all 213 targets.',
    'q95_targetwise_worst_absolute_error': 'Type-7 linear sample 95th percentile of d_i; for N=213 this is 0.6*d_(202)+0.4*d_(203), sorted ascending.',
    'observed_max_absolute_error': 'max_i d_i=max_{i,a in A}|e_i,a|, the observed maximum on the declared finite library.',
    'upper_ten_percent_mean': 'Mean of the largest ceil(0.10*213)=22 values d_i. This covers 10.3286% of targets and is not a fractional exact-10% CVaR.',
    'clean_max_absolute_error': 'Largest absolute clean error over 213 targets.',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def quantile(values, p):
    values = sorted(values)
    index = (len(values)-1)*p
    low = math.floor(index)
    weight = index-low
    return values[low]*(1-weight)+values[min(low+1,len(values)-1)]*weight


def verify_predictions(path, provenance, targets, conditions):
    assert digest(path) == provenance['original_sources']['predictions_csv']['sha256'], 'Original prediction CSV hash mismatch.'
    library = set(provenance['library'])
    controls = set(provenance['excluded_controls'])
    bundles = defaultdict(dict)
    raw = read_csv(path)
    assert len(raw) == 102240
    for r in raw:
        key = (r['profile'], r['label'], r['event'])
        assert r['attack'] not in bundles[key]
        e = abs(float(r['prediction'])-float(r['truth']))
        assert close(e, float(r['absolute_error']))
        assert close(e*e, float(r['squared_error']))
        assert (r['main_library']=='True') == (r['attack'] in library)
        bundles[key][r['attack']] = (e, float(r['prediction']))
    assert len(bundles) == 6390
    for rows in bundles.values():
        assert set(rows) == library | controls
    for key, target in targets.items():
        rows = bundles[key]
        assert close(max(rows[a][0] for a in library)**2, target['worst_squared_error'])
        assert close(rows['clean'][0], target['clean_absolute_error'])
    for key, expected in conditions.items():
        profile, label, condition = key
        errors = [rows[condition][0] for k,rows in bundles.items() if k[:2]==(profile,label)]
        assert len(errors) == expected['targets'] == 213
        assert close(math.fsum(e*e for e in errors), expected['sum_squared_error'])
        assert close(math.fsum(errors), expected['sum_absolute_error'])
        assert close(max(errors), expected['max_absolute_error'])
    sb = provenance['proper_mlni_equivalence']['sourcebridge_label']
    mlni = provenance['proper_mlni_equivalence']['comparator_label']
    comparisons = []
    for (profile,label,event), rows in bundles.items():
        if label == sb:
            for attack in library | controls:
                comparisons.append(abs(rows[attack][1]-bundles[profile,mlni,event][attack][1]))
    assert len(comparisons)==6816 and max(comparisons)==0
    return dict(status='PASS', prediction_rows=102240, complete_bundles=6390,
                proper_mlni_equal_predictions=6816, max_prediction_difference=0.0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results')
    parser.add_argument('--predictions-csv',type=Path)
    args=parser.parse_args()
    provenance=json.loads((args.data_dir/'error_metric_provenance.json').read_text('utf-8'))
    for filename, expected in provenance['bundled_inputs'].items():
        assert digest(args.data_dir/filename)==expected['sha256'], (filename,'Input hash mismatch')
    library=set(provenance['library'])
    assert len(library)==14
    targets={}
    for row in read_csv(args.data_dir/'target_error_summaries.csv'):
        key=(row['profile'],row['source_label'],row['event'])
        assert key not in targets
        parsed={k:float(row[k]) for k in ['clean_absolute_error','worst_squared_error']}
        assert all(math.isfinite(x) and x>=0 for x in parsed.values())
        targets[key]=parsed
    conditions={}
    for row in read_csv(args.data_dir/'condition_error_summaries.csv'):
        key=(row['profile'],row['source_label'],row['condition'])
        assert key not in conditions
        parsed={k:float(row[k]) for k in ['sum_squared_error','sum_absolute_error','max_absolute_error']}
        parsed['targets']=int(row['targets'])
        assert all(math.isfinite(x) and x>=0 for x in parsed.values())
        conditions[key]=parsed
    assert len(targets)==4686 and len(conditions)==308
    values=[]
    for item in provenance['interfaces']:
        interface,label=item['interface'],item['source_label']
        for profile in ['exact','heterogeneous']:
            events=[r for key,r in targets.items() if key[:2]==(profile,label)]
            blocks={key[2]:r for key,r in conditions.items() if key[:2]==(profile,label)}
            assert len(events)==213 and set(blocks)==library
            assert all(r['targets']==213 for r in blocks.values())
            clean=blocks['clean']
            clean_errors=[r['clean_absolute_error'] for r in events]
            d=[math.sqrt(r['worst_squared_error']) for r in events]
            assert close(math.fsum(clean_errors),clean['sum_absolute_error'])
            assert close(math.fsum(x*x for x in clean_errors),clean['sum_squared_error'])
            assert close(max(clean_errors),clean['max_absolute_error'])
            assert close(max(d),max(r['max_absolute_error'] for r in blocks.values()))
            values.append(dict(interface=interface,source_label=label,profile=profile,
                clean_rmse=math.sqrt(clean['sum_squared_error']/213),
                pooled_rmse=math.sqrt(math.fsum(r['sum_squared_error'] for r in blocks.values())/(213*14)),
                targetwise_worst_rmse=math.sqrt(statistics.fmean(r['worst_squared_error'] for r in events)),
                clean_mae=clean['sum_absolute_error']/213,
                mean_targetwise_worst_absolute_error=statistics.fmean(d),
                q95_targetwise_worst_absolute_error=quantile(d,.95),
                observed_max_absolute_error=max(d),
                upper_ten_percent_mean=statistics.fmean(sorted(d)[-22:]),
                clean_max_absolute_error=max(clean_errors)))
    rankings=[]
    for profile in ['exact','heterogeneous']:
        group=[r for r in values if r['profile']==profile]
        for metric in METRICS:
            ordered=sorted(group,key=lambda r:r[metric])
            rows=[dict(interface=r['interface'],value=r[metric],rank=1+sum(other[metric]<r[metric] and not close(other[metric],r[metric]) for other in group)) for r in ordered]
            rankings.append(dict(profile=profile,metric=metric,ranking=rows))
    raw_check=dict(status='NOT_REQUESTED',description='Bundled-input recomputation completed. Supply the original CSV for independent extraction verification.')
    if args.predictions_csv:
        raw_check=verify_predictions(args.predictions_csv,provenance,targets,conditions)
    result=dict(status='PASS',units='micrograms per cubic meter',targets_per_profile=213,
        main_library_conditions=14,pooled_cells_per_interface_profile=2982,
        definitions=METRICS,values=values,rankings=rankings,
        ranking_policy='Lower is better; all 11 displayed interfaces; competition ranking on unrounded values, equality tolerance 1e-12.',
        source_provenance='data/error_metric_provenance.json',raw_prediction_verification=raw_check,
        proper_mlni_equivalence=provenance['proper_mlni_equivalence'],
        interpretation='Frozen-prediction descriptive summaries. The pooled metric weights the 14 declared conditions equally, not by a real-world attack probability. Observed maxima are not uniform certificates. No significance test is implied. All prediction-error metrics coincide with proper MLNI on this record.')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'error_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (args.output_dir/'error_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    print(json.dumps(dict(status='PASS',interfaces=11,profiles=2,metrics=len(METRICS),raw_prediction_verification=raw_check,output_dir=str(args.output_dir)),ensure_ascii=False))


if __name__=='__main__':
    main()
