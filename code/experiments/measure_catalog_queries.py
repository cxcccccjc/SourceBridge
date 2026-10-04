"""Frozen bounded measurement of a freshly built floating LP catalog plus query batch."""
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
import os, sys, json, hashlib, time, datetime, platform, statistics
sys.dont_write_bytecode = True
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
HERE = Path(__file__).resolve().parent
SOURCE = HERE/'source'
sys.path[:0] = [str(SOURCE/'deps'), str(SOURCE)]
PROTO = HERE/'timing_reference'/'end_to_end_protocol.json'
RAW = HERE/'timing_reference'/'end_to_end_raw.json'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, d): p.write_text(json.dumps(d, indent=2, allow_nan=False), encoding='utf-8')
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def freeze():
    assert not PROTO.exists(), 'Do not overwrite the frozen protocol.'
    files = ['catalog_selection.py', 'additive_selection.py', 'reference_geometry.py', 'selection_scaling_inputs.json']
    d = dict(status='FROZEN_BEFORE_EXECUTION', created_utc=now(),
        scope='Bounded catalog-construction and query-batch timing with fixed geometry.',
        n=[8,32,64], seed=731991, K=[1,10], query_sequences={'1':[4], '10':[4,8]*5},
        model={'target':[0.,500.], 'gain_bounds':[.7,1.3], 'epsilon_target':5.},
        warmups_per_method_condition=1, retained_repeats_per_method_condition=3,
        condition_order='Ascending n, then ascending K',
        method_order='Alternate enumeration/fast using (condition_index + repeat) parity, warm-up repeat=-1.',
        measured_interval='External perf_counter around actual pair_catalog construction followed by all K API calls; one newly built catalog is reused within each batch, never across timed batches; includes selected-union two-LP check for every query.',
        excluded_from_timer='Imports, input parsing, canonical catalog hashing, result serialization, verification, and disk I/O.',
        verification='After recording all runs, require catalog hashes identical for a given n, and identical complete query tuple (selection_bound,cost,number_of_ids,ordered_ids) across methods and repetitions at each budget. Also compare reported widths.',
        arithmetic='Floating-point LP catalogs and floating-point API checks. This timing experiment is not rational certification.',
        failures='Retain errors and stop; no dropped repetitions or input replacements.',
        threads={'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
        summary='Report retained-run medians, ranges, and median of paired whole-batch enumeration/fast ratios; whole-batch amortized time is measured wall time divided by K.',
        hashes={name:sha(SOURCE/name) for name in files}, runner_sha256=sha(Path(__file__)))
    write(PROTO,d)
    print('FROZEN', sha(PROTO), flush=True)
def run():
    p=json.loads(PROTO.read_text('utf-8'))
    assert verify_identity(Path(__file__),p['runner_sha256'])
    for n,h in p['hashes'].items(): assert verify_identity(SOURCE/n,h), n
    assert not RAW.exists(), 'Preserve existing measurements.'
    from catalog_selection import pair_catalog, design_anchors
    from additive_selection import design_anchors_additive
    import numpy, scipy
    methods={'enumeration':design_anchors,'fast':design_anchors_additive}
    cases=json.loads((SOURCE/'selection_scaling_inputs.json').read_text('utf-8'))['cases']
    data=dict(status='RUNNING', started_utc=now(), protocol_sha256=sha(PROTO), environment={'python':sys.version,'platform':platform.platform(),'numpy':numpy.__version__,'scipy':scipy.__version__},records=[])
    write(RAW,data)
    idx=0
    try:
        for n in p['n']:
            case=next(c for c in cases if c['n']==n and c['seed']==p['seed'])
            for K in p['K']:
                budgets=p['query_sequences'][str(K)]
                for rep in [-1,0,1,2]:
                    order=['enumeration','fast'] if (idx+rep)%2==0 else ['fast','enumeration']
                    for method in order:
                        record=dict(n=n,K=K,seed=p['seed'],repeat=rep,warmup=rep==-1,method=method,order=order,budgets=budgets)
                        tick=time.perf_counter()
                        cat=pair_catalog(case['anchors'],**p['model'])
                        catalog_end=time.perf_counter()
                        results=[methods[method](case['anchors'],costs=case['costs'],budget=b,catalog=cat,**p['model']) for b in budgets]
                        end=time.perf_counter()
                        record.update(catalog_seconds=catalog_end-tick,query_batch_seconds=end-catalog_end,whole_batch_seconds=end-tick,
                            catalog_size=len(cat),catalog_sha256=hashlib.sha256(json.dumps(cat,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
                            result_tuples=[[r['selection_bound'],r['cost'],len(r['anchor_ids']),list(r['anchor_ids'])] for r in results], widths=[r['width'] for r in results])
                        data['records'].append(record); write(RAW,data)
                        print(n,K,rep,method,round(record['whole_batch_seconds'],4),flush=True)
                idx+=1
        data['status']='MEASURED';data['finished_utc']=now();write(RAW,data)
    except Exception as e:
        data['status']='ERROR';data['error']=repr(e);write(RAW,data);raise
def verify():
    d=json.loads(RAW.read_text('utf-8'));p=json.loads(PROTO.read_text('utf-8'))
    assert d['status']=='MEASURED' and len(d['records'])==48
    assert d['protocol_sha256']==sha(PROTO)
    summary=[];matched=0
    for n in p['n']:
        rs=[r for r in d['records'] if r['n']==n]
        assert len({r['catalog_sha256'] for r in rs})==1
        reference={}
        for r in rs:
            assert r['catalog_size']==1+n+n*(n-1)//2
            for b,t,w in zip(r['budgets'],r['result_tuples'],r['widths']):
                if b not in reference: reference[b]=(t,w)
                assert reference[b]==(t,w), (n,b,r['method'])
                matched+=1
        for K in p['K']:
            cr=[r for r in rs if r['K']==K and not r['warmup']]
            row=dict(n=n,K=K,catalog_size=rs[0]['catalog_size'],methods={})
            for method in ['enumeration','fast']:
                mr=[r for r in cr if r['method']==method]
                row['methods'][method]={key:{'median':statistics.median(r[key] for r in mr),'min':min(r[key] for r in mr),'max':max(r[key] for r in mr)} for key in ['catalog_seconds','query_batch_seconds','whole_batch_seconds']}
                row['methods'][method]['amortized_seconds_per_query']=statistics.median(r['whole_batch_seconds']/K for r in mr)
            ratios=[next(r['whole_batch_seconds'] for r in cr if r['repeat']==rep and r['method']=='enumeration')/next(r['whole_batch_seconds'] for r in cr if r['repeat']==rep and r['method']=='fast') for rep in range(3)]
            row['paired_whole_batch_speedup']={'median':statistics.median(ratios),'min':min(ratios),'max':max(ratios),'values':ratios}
            summary.append(row)
    out=dict(status='PASS',scope=p['arithmetic'],protocol_sha256=sha(PROTO),raw_sha256=sha(RAW),measured_batches=48,retained_batches=36,matched_queries_including_warmups=matched,groups=summary)
    (HERE/'generated').mkdir(exist_ok=True);write(HERE/'generated'/'end_to_end_verified.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__': {'freeze':freeze,'run':run,'verify':verify}[sys.argv[1]]()
