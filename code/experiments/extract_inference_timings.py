"""Stream original saved fit bundles; extract real timing observations, never rerun them."""
from pathlib import Path
import argparse,json,hashlib,re,statistics,math
HERE=Path(__file__).resolve().parent
EXPECTED='0fe996f5a08e7e48427a0da919dfdd24b8a8b0422f21b3fe5b1f283942b19231'
PROPER='MLNI_JSAC2022__proper_public_WLS'
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True,help='Original 588,620,036-byte inference runs JSON; not bundled with the portable package.');p.add_argument('--output',type=Path,default=HERE/'inference_timing_distribution.json');p.add_argument('--metrics',type=Path,required=True,help='Preserved metrics.json used to verify the extracted summaries.');a=p.parse_args()
    digest=hashlib.sha256();in_bundles=False;buffer=None;records=[]
    with a.source.open('rb') as f:
        for raw in f:
            digest.update(raw)
            if not in_bundles:
                if raw.strip()==b'"bundles": {':in_bundles=True
                continue
            line=raw.decode('utf-8')
            if buffer is None:
                if re.match(r'^    "[0-9a-f]{64}": \{\s*$',line):buffer=[line]
            else:
                buffer.append(line)
                if line.rstrip() in ['    },','    }']:
                    obj=json.loads('{'+''.join(buffer).rstrip().rstrip(',')+'}')
                    assert len(obj)==1
                    key,item=next(iter(obj.items()));rec=next(r for r in item['fit']['records'] if r['label']==PROPER)
                    records.append({'bundle_key':key,'proper_core_ms':rec['inference_seconds']*1000,'certificate_ms':item['extra_certificate_seconds']*1000,'projection_ms':item['projection_seconds'][PROPER]*1000})
                    buffer=None
    assert digest.hexdigest()==EXPECTED
    assert buffer is None and len(records)==6469 and len({r['bundle_key'] for r in records})==6469
    old=json.loads(a.metrics.read_text('utf-8'))['public']['timings'];summary={}
    for field in ['proper_core_ms','certificate_ms','projection_ms']:
        vals=[r[field] for r in records];assert all(math.isfinite(v) and v>0 for v in vals)
        out={'n':len(vals),'median':statistics.median(vals),'mean':statistics.mean(vals),'min':min(vals),'max':max(vals)}
        for k,v in out.items():assert math.isclose(v,old[field][k],rel_tol=1e-13,abs_tol=1e-13),(field,k,v,old[field][k])
        q=statistics.quantiles(vals,n=4,method='inclusive');out.update(q1=q[0],q3=q[2]);summary[field]=out
    out={'status':'PASS','source_file':a.source.name,'source_sha256':digest.hexdigest(),'scope':'Saved one-shot timings of all 6,469 distinct numeric fit bundles; conditions may reuse a bundle. No new runs, no synthetic samples, no independent-repetition claim.','extraction':'Same fields and 1000x seconds-to-ms conversion as public_inference_report.py lines 113-114.','summary_matches_prior_audit':True,'summary':summary,'records':records}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'status':'PASS','source_sha256':digest.hexdigest(),'summary':summary},indent=2))
if __name__=='__main__':main()
