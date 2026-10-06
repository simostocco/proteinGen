"""E012 DATA V2B preflight; no training. All writes scoped to the new version."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone
import pyarrow.parquet as pq

REPO = Path('/home/simostocco/proteinGen-causal-rope')
REPORT = REPO / 'reports/experiments/E012_causal_rope_sequence/data_v2b_uniref50_lowdisk'
ROOT = Path('/mnt/d/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2')
AA = set('ACDEFGHIKLMNPQRSTVWY')

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()

def save(p, obj):
    p = Path(p)
    temp = p.with_suffix(p.suffix+'.partial')
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n')
    temp.replace(p)

def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    for part in ['raw','checksums','filtered','protected','mmseqs','clusters','manifests','corpus','stats','tmp']:
        (ROOT/part).mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage('/mnt/d').free
    # Mixture of strict count/length bounds and conservative per-stage storage allowances.
    raw = 8_780_552_383
    stage = {
        'source_download': {'raw': raw, 'partial_and_metadata_allowance': 1_000_000_000},
        'stream_filter_reservoir': {'raw': raw, 'hash_offset_index': 6_000_000_000,
            'reservoir_sqlite_sequences_24m': 30_000_000_000, 'journals_and_pages': 12_000_000_000},
        'batched_protected_screen': {'raw':raw, 'reservoir_sqlite':30_000_000_000,
            'chunk_fasta_and_mmseqs_dbs_results_tmp':8_000_000_000,'metadata_and_journals':5_000_000_000},
        'finalization_and_verification': {'raw':raw, 'reservoir_sqlite':30_000_000_000,
            'canonical_gzip_max_length_allowance':11_000_000_000,
            'parquet_shards_max_length_allowance':12_000_000_000,
            'metadata_and_indices':8_000_000_000,'bounded_external_shard_sort':32_000_000_000,
            'shard_transaction_and_audit':1_000_000_000},
        'sampled_diversity_after_certified_reservoir_cleanup': {'raw':raw,
            'final_gzip_shards_metadata':28_000_000_000,
            'one_million_sample_and_mmseqs_tmp':20_000_000_000,'audit':1_000_000_000},
    }
    totals = {k:sum(v.values()) for k,v in stage.items()}
    plan = {'starting_free_bytes':free,'safety_fraction':0.70,
        'allowed_simultaneous_bytes':free*7//10,'stage_components_bytes':stage,
        'stage_totals_bytes':totals,'planned_peak_bytes':max(totals.values()),
        'pass':max(totals.values()) <= free*7//10,
        'assumptions':'No permanent decompression or full filtered FASTA. SHA256+offset index only, '
          'full-sequence collision checks by gzip source seek. At most 24M external sequences in SQLite. '
          'Screen 100000-sequence batches sequentially; at most one batch database/temp tree exists. '
          'No full corpus clustering. Compressed FASTA and zstd Parquet shards retained. '
          'Temporary index cleaned only after reservoir completion marker. Reservoir cleaned only '
          'after final independent homology verification, hashes, and certification marker.',
        'cpu_threads':8,'mmseqs_version':subprocess.check_output(['mmseqs','version'],text=True).strip(),
        'timestamp':datetime.now(timezone.utc).isoformat(), 'training':False}
    prior=ROOT/'stats/resource_plan.json'
    if prior.exists():
        plan=json.loads(prior.read_text())
        assert plan['pass'] and plan['planned_peak_bytes']<=plan['allowed_simultaneous_bytes']
        assert shutil.disk_usage('/mnt/d').free>=plan['starting_free_bytes']*3//10
    else:
        save(prior, plan)
    save(REPORT/'resource_plan.json', plan)
    if not plan['pass']:
        raise RuntimeError('DATA2-E: low disk plan exceeds 70% free space')
    train_path=REPO/'outputs/e012_causal_rope_sequence/pilot_v1/train.parquet'
    valid_path=train_path.with_name('validation.parquet')
    contract=json.loads((REPO/'reports/experiments/E012_causal_rope_sequence/pilot_v1/contract.json').read_text())
    for p in [train_path, valid_path]:
        assert sha(p)==contract['protected_hashes'][str(p.relative_to(REPO))], str(p)
    old=Path('/mnt/d/Simone/proteinGen/data')
    sources=[valid_path]+sorted(set(old.glob('**/test*.parquet'))|set(old.glob('**/validation*.parquet')))
    # E011 validation cache and both E011/E012 panels explicitly resolved and checked.
    e011=Path('/home/simostocco/proteinGen-sequence-context/outputs/e011_sequence_context_only/pilot_v1/validation_sequences.parquet')
    sources.append(e011)
    all_sequences={}
    inventory=[]
    e012_ids={}
    for p in sources:
        assert p.is_file(), str(p)
        columns=pq.read_schema(p).names
        assert 'sequence' in columns and 'sample_id' in columns, str(p)
        count=0
        for batch in pq.ParquetFile(p).iter_batches(batch_size=8192,columns=['sample_id','sequence']):
            for r in batch.to_pylist():
                seq=''.join(r['sequence'].split()).upper()
                if not seq or not set(seq)<=AA:
                    raise RuntimeError(f'DATA2-D: noncanonical protected sequence in {p}: {r["sample_id"]}')
                all_sequences.setdefault(seq,[]).append({'source':str(p),'sample_id':r['sample_id']})
                if p==valid_path:
                    e012_ids[str(r['sample_id'])]=seq
                count+=1
        inventory.append({'path':str(p),'rows':count,'sha256':sha(p)})
    panels=json.loads((REPO/'reports/experiments/E012_causal_rope_sequence/pilot_v1/panels.json').read_text())
    for key,ids in panels.items():
        assert all(str(s) in e012_ids for s in ids), key
    for name,p in [('primary',REPO/'reports/experiments/E011_sequence_context_only/pilot_v1/primary_panel.json'),
                   ('independent',REPO/'reports/experiments/E011_sequence_context_only/diagnostic_panel.json')]:
        ids=json.loads(p.read_text())['sample_ids']
        assert set(map(str,ids))==set(map(str,panels[name]))
        inventory.append({'path':str(p),'rows':len(ids),'sha256':sha(p),'subset_of_e012_validation':True})
    protected=ROOT/'protected/protected.fasta'
    membership=ROOT/'protected/membership.jsonl.gz'
    if (ROOT/'protected/complete.json').exists():
        marker=json.loads((ROOT/'protected/complete.json').read_text())
        assert marker['input_inventory']==inventory
        assert marker['fasta_sha256']==sha(protected)
        assert marker['membership_sha256']==sha(membership)
        print('Protected stage already sealed')
    else:
        import gzip
        with protected.open('x') as f, gzip.open(membership,'wt') as m:
            for seq in sorted(all_sequences, key=lambda s:(hashlib.sha256(s.encode()).digest(),s)):
                sid=hashlib.sha256(seq.encode()).hexdigest()
                f.write(f'>p_{sid}\n{seq}\n')
                m.write(json.dumps({'id':'p_'+sid,'sequence_sha256':sid,'memberships':all_sequences[seq]},sort_keys=True)+'\n')
        marker={'schema':'e012_v2b_protected_v1','input_inventory':inventory,
            'unique_sequences':len(all_sequences),'fasta_sha256':sha(protected),
            'membership_sha256':sha(membership),
            'definition':'DATA-E minimum E012/E011 populations plus all historical repository validation/test manifests discovered under canonical proteinGen/data. No TRAIN-only membership.',
            'scope_note':'DATA-E froze scope only and explicitly left membership unbuilt; V2B resolves and seals it before source acquisition.'}
        save(ROOT/'protected/complete.json',marker)
    save(REPORT/'protected_definition.json',marker)
    historical={}
    for batch in pq.ParquetFile(train_path).iter_batches(columns=['sample_id','sequence']):
        for r in batch.to_pylist():
            seq=r['sequence']
            assert 20<=len(seq)<=500 and set(seq)<=AA
            historical[seq]=min(str(r['sample_id']),historical.get(seq,str(r['sample_id'])))
    assert len(historical)==87930
    info={'nominal_train':231743,'exact_unique_train':len(historical),
        'unique_total_residues':sum(map(len,historical)),
        'exact_historical_train_protected_overlap':len(set(historical)&set(all_sequences)),
        'train_sha256':sha(train_path),'protected_unique':len(all_sequences)}
    save(REPORT/'historical_population.json',info)
    print(json.dumps({'resource_plan':plan,'historical':info},indent=2))

if __name__=='__main__':
    main()
