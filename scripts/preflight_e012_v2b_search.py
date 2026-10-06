"""Measure batched protected search on fixed historical sequences, never train."""
from pathlib import Path
import hashlib
import importlib.util
import json
import resource
import time
import pyarrow.parquet as pq

REPO=Path('/home/simostocco/proteinGen-causal-rope')
ROOT=Path('/mnt/d/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2')
spec=importlib.util.spec_from_file_location('v2b',REPO/'scripts/prepare_e012_sequence_corpus_v2b.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def main():
    b=m.Build()
    report=b.report/'protected_search_preflight.json'
    if report.exists():
        print(report.read_text())
        return
    folder=ROOT/'mmseqs/resource_preflight'
    folder.mkdir(parents=True,exist_ok=True)
    train=REPO/'outputs/e012_causal_rope_sequence/pilot_v1/train.parquet'
    allseq={}
    for batch in pq.ParquetFile(train).iter_batches(columns=['sample_id','sequence']):
        for row in batch.to_pylist():
            allseq.setdefault(row['sequence'],str(row['sample_id']))
    chosen=sorted(allseq,key=lambda seq:(m.priority(seq),seq))[:10_000]
    fasta=folder/'historical_10000.fasta'
    with fasta.open('w') as f:
        for seq in chosen:
            f.write(f'>h_{m.sequence_digest(seq).hex()}\n{seq}\n')
    inputs={'train_sha256':m.digest(train),'protected_sha256':m.digest(b.protected),'pilot_sha256':m.digest(fasta)}
    with b.stage('protected_search_resource_preflight'):
        started=time.monotonic()
        hits=folder/'hits.tsv'
        cmd=b.search_command(fasta,hits,folder/'tmp')
        b.run(cmd,folder/'command.log')
        query_hits=set()
        with hits.open() as f:
            for line in f:
                query_hits.add(line.split('\t')[0])
        protected=set()
        with b.protected.open() as f:
            for line in f:
                if not line.startswith('>'):
                    protected.add(line.strip())
        clean=[seq for seq in chosen if 'h_'+m.sequence_digest(seq).hex() not in query_hits and seq not in protected]
        clean_fasta=folder/'clean.fasta'
        with clean_fasta.open('w') as f:
            for seq in reversed(clean):
                f.write(f'>h_{m.sequence_digest(seq).hex()}\n{seq}\n')
        verify=folder/'clean_hits.tsv'
        b.run(b.search_command(clean_fasta,verify,folder/'verify_tmp'),folder/'verify.log')
        assert verify.stat().st_size==0
        disk=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
        info={'inputs':inputs,'pilot_n':10_000,'pilot_source':'historical E012 unique TRAIN; not partial raw source',
            'wall_seconds_two_searches':time.monotonic()-started,
            'retained_logical_disk_bytes':disk,
            'projected_100k_batch_disk_bytes_with_4x_slack':disk*10*4,
            'batch_disk_allowance_bytes':8_000_000_000,
            'batch_disk_pass':disk*10*4<=8_000_000_000,
            'children_maxrss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            'commands':[list(map(str,cmd)),list(map(str,b.search_command(clean_fasta,verify,folder/'verify_tmp')))],
            'result_sha256':m.digest(hits),'verification_sha256':m.digest(verify),'zero_clean_matches':True,
            'estimated_extrapolation_limit':'Historical query composition differs from external candidates; runtime and hits cannot be predicted scientifically from this pilot.'}
        m.save(report,info)
        m.save(ROOT/'stats/protected_search_preflight.json',info)
        cert=ROOT/'manifests/search_resource_preflight.complete.json'
        m.save(cert,info)
        b.cleanup([folder],cert)
        print(json.dumps(info,indent=2))
        if not info['batch_disk_pass']:
            raise RuntimeError('DATA2-E: preflight batch temporary disk exceeds redesigned stage budget')

if __name__=='__main__':
    main()
