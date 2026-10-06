"""Low-disk, resumable E012 UniRef50 2026_03 corpus build. Never trains.

Each subcommand is an explicit stage; inputs are hash pinned. No completion is
inferred from file existence. Bulk outputs live on D: and small reports in Git.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import subprocess
import threading
import time

AA = 'ACDEFGHIKLMNPQRSTVWY'
ALPHABET = frozenset(AA)
SEED = 12014
RAW_BYTES = 8_780_552_383
RAW_MD5 = '0492e3cf4093276ae4319ba24d52514f'
RAW_COUNT = 38_840_027
HISTORICAL_UNIQUE = 87_930
CHECKPOINT_EVERY = 100_000
MIN_FINAL = 10_000_000
SHARD_COUNT = 512
DIAGNOSTIC_SAMPLE_N = 1_000_000
AUDIT_SAMPLE_N = 10_000
REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path('/mnt/d/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2')
REPORT_REL = 'reports/experiments/E012_causal_rope_sequence/data_v2b_uniref50_lowdisk'

def now():
    return datetime.now(timezone.utc).isoformat()

def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)

def priority(sequence):
    return hashlib.sha256(b'E012-V2B:12014\0' + sequence.encode('ascii')).digest()

def sequence_digest(sequence):
    return hashlib.sha256(sequence.encode('ascii')).digest()

def fasta_records(file, start=0):
    """Yield (header offset, full header, sequence, malformed) without loading FASTA.

    Binary gzip handles support tell/seek in uncompressed offsets. Only ASCII FASTA
    formatting whitespace is removed; other symbols remain invalid. An orphan
    sequence line is a malformed record rather than silently discarded.
    """
    file.seek(start)
    header = None
    offset = start
    chunks = []
    malformed = False
    while True:
        pos = file.tell()
        line = file.readline()
        if not line:
            break
        if line.startswith(b'>'):
            if header is not None:
                yield offset, header, ''.join(chunks).upper(), malformed
            try:
                header = line[1:].strip().decode('utf-8')
                malformed = not bool(header) or not header.split()[0].startswith('UniRef50_')
            except UnicodeDecodeError:
                header, malformed = '', True
            chunks = []
            offset = pos
        elif line.strip():
            if header is None:
                yield pos, '', '', True
                continue
            try:
                chunks.append(line.translate(None, b' \t\r\n\v\f').decode('ascii'))
            except UnicodeDecodeError:
                malformed = True
    if header is not None:
        yield offset, header, ''.join(chunks).upper(), malformed

def valid(sequence):
    return 20 <= len(sequence) <= 500 and set(sequence) <= ALPHABET

def connection(path):
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=DELETE')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('PRAGMA cache_size=-524288')  # At most 512 MiB, allocated on demand.
    db.execute('PRAGMA temp_store=MEMORY')
    return db

class Stats:
    def __init__(self, data=None):
        self.n = 0
        self.residues = 0
        self.lengths = Counter()
        self.aa = Counter()
        if data:
            self.n = data['n']
            self.residues = data['residues']
            self.lengths.update({int(k):v for k,v in data['lengths'].items()})
            self.aa.update(data['aa'])

    def add(self, sequence):
        self.n += 1
        self.residues += len(sequence)
        self.lengths[len(sequence)] += 1
        self.aa.update(sequence)

    def checkpoint(self):
        return {'n':self.n,'residues':self.residues,'lengths':dict(self.lengths),'aa':dict(self.aa)}

    def summary(self):
        def quantile(q):
            if not self.n:
                return None
            rank = max(1, math.ceil(q * self.n))
            cumulative = 0
            for length,count in sorted(self.lengths.items()):
                cumulative += count
                if cumulative >= rank:
                    return length
        return {'sequences':self.n,'total_residues':self.residues,
                'mean_length':self.residues/self.n if self.n else None,
                'median_length':quantile(.5),'p90_length':quantile(.9),
                'p95_length':quantile(.95),'p99_length':quantile(.99),
                'length_strata':{f'{lo}-{hi}':sum(v for k,v in self.lengths.items() if lo<=k<=hi)
                    for lo,hi in [(20,64),(65,128),(129,256),(257,384),(385,500)]},
                'aa_counts':{a:self.aa[a] for a in AA},
                'aa_frequencies':{a:self.aa[a]/self.residues if self.residues else 0 for a in AA}}

class Build:
    def __init__(self, root=DEFAULT_ROOT, repo=REPO, cap=24_000_000, target=20_000_000):
        self.root, self.repo = Path(root), Path(repo)
        self.report = self.repo / REPORT_REL
        self.cap, self.target = cap, target
        self.raw = self.root / 'raw/uniref50.fasta.gz'
        self.reservoir = self.root / 'filtered/reservoir.sqlite'
        self.protected = self.root / 'protected/protected.fasta'
        self.plan = json.loads((self.root / 'stats/resource_plan.json').read_text())
        self.floor = self.plan['starting_free_bytes'] * 3 // 10
        self.monitor_stop = threading.Event()
        self.low_space = False
        self.min_free = shutil.disk_usage(self.root).free

    def guard(self):
        free = shutil.disk_usage(self.root).free
        self.min_free = min(self.min_free, free)
        if free < self.floor or self.low_space:
            raise RuntimeError('DATA2-E: runtime disk reserve breached; stage not completed')

    def telemetry(self, stage, before, start):
        data = {'stage':stage,'start':start['timestamp'],'end':now(),
                'wall_seconds':time.monotonic()-start['clock'],
                'free_before_bytes':before,'free_after_bytes':shutil.disk_usage(self.root).free,
                'minimum_free_sampled_bytes':self.min_free,
                'measured_peak_consumption_bytes':max(0,self.plan['starting_free_bytes']-self.min_free),
                'measurement':'Filesystem free sampled every 10s; includes unrelated D: activity; not exact per-process peak.',
                'parent_maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'children_maxrss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                'cpu_threads':8,'training':False}
        with (self.root / 'stats/telemetry.jsonl').open('a') as f:
            f.write(json.dumps(data,sort_keys=True)+'\n')
        shutil.copyfile(self.root/'stats/telemetry.jsonl',self.report/'resource_telemetry.jsonl')

    @contextmanager
    def stage(self, name):
        self.guard()
        before = shutil.disk_usage(self.root).free
        start = {'timestamp':now(),'clock':time.monotonic()}
        self.monitor_stop.clear()
        def monitor():
            while not self.monitor_stop.wait(10):
                try:
                    self.guard()
                except RuntimeError:
                    self.low_space = True
        worker = threading.Thread(target=monitor,daemon=True)
        worker.start()
        try:
            yield
        finally:
            self.monitor_stop.set()
            worker.join()
            self.telemetry(name,before,start)

    def marker(self, stage):
        return self.root / 'manifests' / (stage+'.complete.json')

    def completed(self, stage, inputs):
        path = self.marker(stage)
        if not path.exists():
            return None
        obj = json.loads(path.read_text())
        if obj['inputs'] != inputs:
            raise RuntimeError(f'{stage}: pinned inputs changed')
        for path, sha in obj['outputs'].items():
            if not Path(path).is_file():
                authorized=False
                cleanup_log=self.root/'stats/cleanup.jsonl'
                if cleanup_log.exists():
                    for line in cleanup_log.read_text().splitlines():
                        event=json.loads(line)
                        cert=Path(event['certificate'])
                        if path in event['deleted'] and cert.is_file() and digest(cert)==event['certificate_sha256']:
                            authorized=True
                            break
                if authorized:
                    continue
                raise RuntimeError(f'{stage}: missing completed output {path}')
            if digest(path) != sha:
                raise RuntimeError(f'{stage}: missing/corrupt completed output {path}')
        return obj

    def complete(self, stage, inputs, outputs, **details):
        data = {'stage':stage,'timestamp':now(),'inputs':inputs,
                'script_sha256':digest(__file__),
                'outputs':{str(p):digest(p) for p in outputs},**details}
        save(self.marker(stage),data)
        return data

    def cleanup(self, paths, certificate):
        assert Path(certificate).is_file()
        event = {'timestamp':now(),'certificate':str(certificate),'certificate_sha256':digest(certificate),'deleted':[]}
        for p in map(Path,paths):
            resolved = p.resolve()
            allowed = [self.root/'tmp',self.root/'mmseqs',self.root/'filtered']
            diagnostic_tmp=resolved.is_relative_to((self.root/'clusters').resolve()) and resolved.name in {'pilot_tmp','diagnostic_tmp'}
            if not any(resolved.is_relative_to(a.resolve()) for a in allowed) and not diagnostic_tmp:
                raise RuntimeError('cleanup outside derivable temporary roots')
            if p.exists():
                event['deleted'].append(str(p))
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
        with (self.root/'stats/cleanup.jsonl').open('a') as f:
            f.write(json.dumps(event,sort_keys=True)+'\n')
        shutil.copyfile(self.root/'stats/cleanup.jsonl',self.report/'cleanup.jsonl')

    def run(self, command, log):
        self.guard()
        with Path(log).open('a') as handle:
            handle.write(json.dumps({'timestamp':now(),'command':list(map(str,command))})+'\n')
            handle.flush()
            p = subprocess.Popen(list(map(str,command)),stdout=handle,stderr=subprocess.STDOUT)
            while p.poll() is None:
                try:
                    self.guard()
                except RuntimeError:
                    p.terminate()
                    p.wait()
                    raise
                time.sleep(1)
            if p.returncode:
                raise RuntimeError(f'Command failed ({p.returncode}); see {log}')

    def verify_raw(self):
        inputs = {'release':'2026_03','expected_bytes':RAW_BYTES,'expected_md5':RAW_MD5}
        if self.completed('raw',inputs):
            return
        with self.stage('verify_raw'):
            part = self.raw.with_name(self.raw.name+'.partial')
            src = self.raw if self.raw.exists() else part
            if src.stat().st_size != RAW_BYTES or digest(src,'md5') != RAW_MD5:
                raise RuntimeError('DATA2-F: raw archive size/MD5 mismatch')
            # gzip CRC/end-of-stream is also checked during mandatory full filter pass.
            sha = digest(src)
            birth = datetime.fromtimestamp(src.stat().st_ctime,timezone.utc).isoformat()
            if src == part:
                part.replace(self.raw)
            provenance = {'release':'2026_03','source':'UniRef50 representatives',
                'source_url':'https://ftp.uniprot.org/pub/databases/uniprot/current_release/uniref/uniref50/uniref50.fasta.gz',
                'mirror':'ftp.uniprot.org','filename':self.raw.name,'server_reported_bytes':RAW_BYTES,
                'local_bytes':src.stat().st_size if src.exists() else self.raw.stat().st_size,
                'md5':RAW_MD5,'official_md5_verified':True,'sha256':sha,
                'verification_end':now(),'source_file_ctime_observed':birth,
                'pin':'Frozen release-specific metalink MD5 and size; current_release is transport only.',
                'compression':'gzip','gzip_crc':'Pending full streaming parse',
                'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.repo,text=True).strip(),
                'processing_script_sha256':digest(__file__)}
            save(self.root/'checksums/source_provenance.json',provenance)
            save(self.report/'source_provenance.json',provenance)
            self.complete('raw',inputs,[self.raw,self.root/'checksums/source_provenance.json'])
            transport=self.root/'tmp/download_ranges'
            if transport.is_dir():
                self.cleanup([transport],self.marker('raw'))

    def filter_reservoir(self):
        raw = json.loads(self.marker('raw').read_text())
        inputs = {'raw_sha256':raw['outputs'][str(self.raw)],'seed':SEED,'cap':self.cap,
            'train_sha256':digest(self.repo/'outputs/e012_causal_rope_sequence/pilot_v1/train.parquet')}
        if self.completed('reservoir',inputs):
            return
        import pyarrow.parquet as pq
        seen_path = self.root/'tmp/exact_hash_offsets.sqlite'
        state_path = self.root/'tmp/filter_checkpoint.json'
        seen = connection(seen_path)
        db = connection(self.reservoir)
        db.execute('ATTACH DATABASE ? AS seen',(str(seen_path),))
        db.execute('PRAGMA seen.cache_size=-2097152')  # Compact hash/offset index only: 2 GiB.
        seen.close()
        db.execute('CREATE TABLE IF NOT EXISTS seen.hashes(h BLOB, off INTEGER, PRIMARY KEY(h,off)) WITHOUT ROWID')
        db.execute('CREATE TABLE IF NOT EXISTS candidates(h BLOB, off INTEGER, p BLOB, seq TEXT, source_id TEXT, multiplicity INTEGER, historical INTEGER, external INTEGER, PRIMARY KEY(h,off)) WITHOUT ROWID')
        db.execute('CREATE INDEX IF NOT EXISTS priority_index ON candidates(p,h,off)')
        db.execute('CREATE TABLE IF NOT EXISTS progress(k TEXT PRIMARY KEY, v TEXT)')
        row = db.execute("SELECT v FROM progress WHERE k='checkpoint'").fetchone()
        state = json.loads(row[0]) if row else {'offset':0,'raw':0,'length_valid':0,'canonical':0,
            'malformed':0,'unique':0,'duplicates':0,'stage':'external', 'inputs':inputs,
            'raw_stats':Stats().checkpoint(),'valid_stats':Stats().checkpoint()}
        if state['inputs'] != inputs:
            raise RuntimeError('filter checkpoint input mismatch')
        historical={}
        train=self.repo/'outputs/e012_causal_rope_sequence/pilot_v1/train.parquet'
        for batch in pq.ParquetFile(train).iter_batches(columns=['sample_id','sequence']):
            for row in batch.to_pylist():
                seq=row['sequence']
                if not valid(seq):
                    raise RuntimeError('historical TRAIN outside corpus contract')
                historical[seq]=min(str(row['sample_id']),historical.get(seq,str(row['sample_id'])))
        if len(historical)!=HISTORICAL_UNIQUE:
            raise RuntimeError('historical unique count changed')
        state.setdefault('historical_matches',{})
        raw_stats,valid_stats = Stats(state['raw_stats']),Stats(state['valid_stats'])
        count = db.execute('SELECT count(*) FROM candidates').fetchone()[0]
        cutoff = db.execute('SELECT p FROM candidates ORDER BY p DESC,h DESC,off DESC LIMIT 1').fetchone()
        cutoff = cutoff[0] if cutoff and count>=self.cap else None
        with self.stage('stream_filter_reservoir'), gzip.open(self.raw,'rb') as reader, gzip.open(self.raw,'rb') as verifier:
            def source_at(off):
                return next(fasta_records(verifier,off))
            def same_hash_offsets(h):
                return [r[0] for r in db.execute('SELECT off FROM seen.hashes WHERE h=?',(h,))]
            def sequence_at(h, off):
                item = db.execute('SELECT seq FROM candidates WHERE h=? AND off=?',(h,off)).fetchone()
                return item[0] if item else source_at(off)[2]
            def checkpoint(next_offset):
                nonlocal count,cutoff
                # Trim one bounded batch above cap using an indexed priority order.
                if count>self.cap:
                    boundary=db.execute('SELECT p,h,off FROM candidates ORDER BY p DESC,h DESC,off DESC LIMIT 1 OFFSET ?', (count-self.cap-1,)).fetchone()
                    db.execute('DELETE FROM candidates WHERE (p,h,off)>=(?,?,?)',boundary)
                    count=self.cap
                cutoff = db.execute('SELECT p FROM candidates ORDER BY p DESC,h DESC,off DESC LIMIT 1').fetchone()
                cutoff = cutoff[0] if cutoff and count>=self.cap else None
                state.update(offset=next_offset,raw_stats=raw_stats.checkpoint(),valid_stats=valid_stats.checkpoint())
                db.execute("INSERT OR REPLACE INTO progress VALUES('checkpoint',?)",(json.dumps(state),))
                db.commit()  # Attached DELETE journals atomically checkpoint both databases.
                save(state_path,state)
                self.guard()
                print(json.dumps({'stage':'filter','raw':state['raw'],'unique':state['unique'],'reservoir':count,'free':shutil.disk_usage(self.root).free}),flush=True)
            if state['stage']=='external':
                for off,header,seq,malformed in fasta_records(reader,state['offset']):
                    if off == state.get('skip_completed_offset'):
                        continue
                    state['raw']+=1
                    raw_stats.add(seq)
                    if malformed:
                        state['malformed']+=1
                    elif 20<=len(seq)<=500:
                        state['length_valid']+=1
                        if set(seq)<=ALPHABET:
                            state['canonical']+=1
                            valid_stats.add(seq)
                            if seq in historical:
                                rec=state['historical_matches'].setdefault(seq,[off,header.split()[0],0])
                                rec[1]=min(rec[1],header.split()[0])
                                rec[2]+=1
                            h=sequence_digest(seq)
                            duplicate=None
                            for prior in same_hash_offsets(h):
                                if sequence_at(h,prior)==seq:
                                    duplicate=prior
                                    break
                            if duplicate is not None:
                                state['duplicates']+=1
                                db.execute('UPDATE candidates SET multiplicity=multiplicity+1, source_id=min(source_id,?) WHERE h=? AND off=?',(header.split()[0],h,duplicate))
                            else:
                                state['unique']+=1
                                db.execute('INSERT INTO seen.hashes VALUES(?,?)',(h,off))
                                p=priority(seq)
                                if cutoff is None or p<=cutoff:
                                    db.execute('INSERT INTO candidates VALUES(?,?,?,?,?,1,0,1)',(h,off,p,seq,header.split()[0]))
                                    count+=1
                    if state['raw']%CHECKPOINT_EVERY==0:
                        # Yield consumes the next header. Resume from this completed record's
                        # header, skip that one record, then continue with the next record.
                        state['skip_completed_offset']=off
                        checkpoint(off)
                if state['raw'] != RAW_COUNT:
                    raise RuntimeError(f'DATA2-F: parsed {state["raw"]} records, expected {RAW_COUNT}')
                state['stage']='historical'
                state.pop('skip_completed_offset',None)
                checkpoint(reader.tell())
            # Historical strings are known before streaming. Match them during that
            # pass, avoiding thousands of random gzip seeks in the historical union.
            if state['stage']!='complete':
                external_reservoir=db.execute('SELECT count(*) FROM candidates WHERE external=1').fetchone()[0]
                historical_added=0
                cross_source=0
                for ordinal,(seq,sid) in enumerate(sorted(historical.items())):
                    h=sequence_digest(seq)
                    match=state['historical_matches'].get(seq)
                    if match is not None:
                        off,source_id,multiplicity=match
                        item=db.execute('SELECT seq FROM candidates WHERE h=? AND off=?',(h,off)).fetchone()
                        if item:
                            assert item[0]==seq
                            db.execute('UPDATE candidates SET historical=1 WHERE h=? AND off=?',(h,off))
                            cross_source+=1
                            continue
                        external=1
                    else:
                        source_id=sid
                        external=0
                        off=-ordinal-1
                        multiplicity=1
                    db.execute('INSERT INTO candidates VALUES(?,?,?,?,?,?,1,?)',(h,off,priority(seq),seq,source_id,multiplicity,external))
                    historical_added+=1
                state['union_counts']={'external_reservoir':external_reservoir,'historical_added':historical_added,'cross_source':cross_source}
                state['stage']='complete'
                db.execute('UPDATE progress SET v=? WHERE k=?',(json.dumps(state),'checkpoint'))
                db.commit()
            union=state['union_counts']
            external_reservoir=union['external_reservoir']
            historical_added=union['historical_added']
            cross_source=union['cross_source']
            n=db.execute('SELECT count(*) FROM candidates').fetchone()[0]
            db.close()
            stats={'raw_count':state['raw'],'malformed':state['malformed'],
                'length_valid':state['length_valid'],'canonical_valid':state['canonical'],
                'external_exact_unique_encountered':state['unique'],'exact_duplicates_removed':state['duplicates'],
                'external_reservoir_count':external_reservoir,'historical_unique':HISTORICAL_UNIQUE,
                'historical_added_to_reservoir':historical_added,'cross_source_reservoir_overlap':cross_source,
                'union_count':n,'raw_statistics':raw_stats.summary(),'canonical_statistics':valid_stats.summary(),
                'gzip_crc_verified':True,'seed':SEED,
                'priority_rule':'SHA256(E012-V2B:12014 NUL || canonical sequence); ascending (priority, sequence SHA256, source offset)',
                'representative_rule':'First source offset for the exact sequence; lexicographically smallest UniRef ID among exact duplicates encountered',
                'collision_verification':'Same SHA256 always compared by full string; source gzip seek for discarded reservoir records'}
            save(self.root/'stats/filtering_stats.json',stats)
            save(self.report/'filtering_stats.json',stats)
            self.complete('reservoir',inputs,[self.reservoir,self.root/'stats/filtering_stats.json'],statistics=stats)
            self.cleanup([seen_path,state_path],self.marker('reservoir'))

    def search_command(self, query, out, tmp):
        indexed=self.root/'mmseqs/protected_index/db' if hasattr(self,'root') else None
        target=indexed if indexed and Path(str(indexed)+'.idx').is_file() and self.marker('target_index').is_file() else self.protected
        return ['mmseqs','easy-search',query,target,out,tmp,
            '--min-seq-id','0.30','-c','0.80','--cov-mode','0',
            '-s','7.5','--seq-id-mode','0','--alignment-mode','3',
            '--threads','8','--split-memory-limit','4G',
            '--max-seqs','1000','--remove-tmp-files','0',
            '--format-output','query,target,fident,qcov,tcov,evalue']

    def target_index(self):
        """Reuse the small protected target index across all candidate batches."""
        inputs={'protected_sha256':digest(self.protected),'sensitivity':7.5,
            'mmseqs_version':subprocess.check_output(['mmseqs','version'],text=True).strip()}
        if self.completed('target_index',inputs):
            return
        folder=self.root/'mmseqs/protected_index'
        folder.mkdir(exist_ok=True,parents=True)
        prefix=folder/'db'
        commands=[['mmseqs','createdb',self.protected,prefix,'--dbtype','1','--shuffle','0'],
            ['mmseqs','createindex',prefix,folder/'tmp','-s','7.5','--threads','8','--split-memory-limit','4G','--remove-tmp-files','0']]
        with self.stage('protected_target_index'):
            for cmd in commands:
                self.run(cmd,folder/'command.log')
            outputs=[p for p in folder.glob('db*') if p.is_file()]
            info={'commands':[list(map(str,c)) for c in commands],
                'bytes':sum(p.stat().st_size for p in outputs),'inputs':inputs,
                'reference':'https://github.com/soedinglab/MMseqs2/wiki/Home/726c6045e91bb3554ec2dbac02cf16272d17a83c#searching'}
            if info['bytes']>2_000_000_000:
                raise RuntimeError('DATA2-E: protected target index exceeds 2GB sub-budget')
            save(self.report/'protected_target_index.json',info)
            self.complete('target_index',inputs,outputs,statistics=info)
            self.cleanup([folder/'tmp'],self.marker('target_index'))

    def screen(self, independent=False):
        if self.root==DEFAULT_ROOT:
            pilot=json.loads((self.root/'stats/protected_search_preflight.json').read_text())
            if not pilot['batch_disk_pass']:
                raise RuntimeError('DATA2-E: mandatory batched search resource preflight failed')
            assert pilot['inputs']['protected_sha256']==digest(self.protected)
            self.target_index()
        reservoir_marker=json.loads(self.marker('reservoir').read_text())
        inputs={'reservoir_sha256':reservoir_marker['outputs'][str(self.reservoir)],
            'protected_sha256':digest(self.protected),'policy':{'identity':.30,'coverage':.80,'cov_mode':0,'sensitivity':7.5},
            'verification':independent}
        name='verify_protected' if independent else 'screen'
        if self.completed(name,inputs):
            return
        db=connection(self.reservoir)
        exclusions=self.root/'manifests/exclusions.sqlite'
        progress_path=self.root/'manifests/verification_progress.sqlite' if independent else exclusions
        ex=connection(progress_path)
        ex.execute('CREATE TABLE IF NOT EXISTS removed(h BLOB,off INTEGER,protected TEXT,identity REAL,PRIMARY KEY(h,off)) WITHOUT ROWID')
        ex.execute('CREATE TABLE IF NOT EXISTS batches(name TEXT, batch INTEGER, n INTEGER, marker TEXT, PRIMARY KEY(name,batch))')
        ex.commit()
        protected_sequences={}
        with self.protected.open() as f:
            sid=None
            for line in f:
                if line.startswith('>'):
                    sid=line[1:].strip()
                elif line.strip():
                    protected_sequences[line.strip()]=sid
        db.execute('ATTACH DATABASE ? AS excluded',(str(exclusions),))
        db.execute('PRAGMA excluded.cache_size=-65536')
        if independent:
            where='WHERE NOT EXISTS(SELECT 1 FROM excluded.removed e WHERE e.h=c.h AND e.off=c.off)'
            n=db.execute('SELECT count(*) FROM candidates c '+where).fetchone()[0]
            limit=min(n,self.target)
        else:
            where=''
            n=db.execute('SELECT count(*) FROM candidates').fetchone()[0]
            limit=n
        cursor=db.execute('SELECT c.h,c.off,c.seq,c.historical,c.external FROM candidates c '+where+' ORDER BY c.p,c.h,c.off LIMIT ?', (limit,))
        matches_total=0
        with self.stage(name):
            batch_number=0
            while rows:=cursor.fetchmany(100_000):
                batch_number+=1
                folder=self.root/'mmseqs'/name/f'batch-{batch_number:04d}'
                cert=self.root/'manifests'/f'{name}-batch-{batch_number:04d}.json'
                already=ex.execute('SELECT marker FROM batches WHERE name=? AND batch=?',(name,batch_number)).fetchone()
                if already:
                    data=json.loads(cert.read_text())
                    assert data['input_rows_sha256']==hashlib.sha256(b''.join(h+str(off).encode()+b'\0' for h,off,*_ in rows)).hexdigest()
                    matches_total+=data['hits']
                    continue
                folder.mkdir(parents=True,exist_ok=True)
                fasta=folder/'query.fasta'
                hits=folder/'matches.tsv'
                with fasta.open('w') as f:
                    for h,off,seq,*_ in rows:
                        f.write(f'>s_{h.hex()}_{off}\n{seq}\n')
                command=self.search_command(fasta,hits,folder/'tmp')
                # Independent pass changes batch order by reversing FASTA, avoiding
                # reliance on exclusion parsing or original search output.
                if independent:
                    with fasta.open('w') as f:
                        for h,off,seq,*_ in reversed(rows):
                            f.write(f'>s_{h.hex()}_{off}\n{seq}\n')
                self.run(command,folder/'command.log')
                mapping={f's_{h.hex()}_{off}':(h,off,historical,external) for h,off,seq,historical,external in rows}
                nearest={f's_{h.hex()}_{off}':(protected_sequences[seq],1.0)
                    for h,off,seq,*_ in rows if seq in protected_sequences}
                with hits.open() as f:
                    for line in f:
                        query,target,ident,qcov,tcov,evalue=line.rstrip().split('\t')
                        assert query in mapping
                        ident,qcov,tcov=map(float,(ident,qcov,tcov))
                        # MMseqs output rounds values; filtering is performed internally.
                        if ident < .299 or qcov < .799 or tcov < .799:
                            raise RuntimeError('MMseqs result outside historical boundary')
                        if query not in nearest or ident>nearest[query][1]:
                            nearest[query]=(target,ident)
                if independent and nearest:
                    raise RuntimeError('DATA2-D: independent final protected violations detected')
                for query,(target,ident) in nearest.items():
                    h,off,_,_=mapping[query]
                    ex.execute('INSERT OR REPLACE INTO removed VALUES(?,?,?,?)',(h,off,target,ident))
                data={'stage':name,'batch':batch_number,'inputs':inputs,
                    'input_rows_sha256':hashlib.sha256(b''.join(h+str(off).encode()+b'\0' for h,off,*_ in rows)).hexdigest(),
                    'n':len(rows),'hits':len(nearest),'matches_sha256':digest(hits),
                    'query_sha256':digest(fasta),'command':list(map(str,command)),
                    'mmseqs_version':subprocess.check_output(['mmseqs','version'],text=True).strip(),
                    'timestamp':now()}
                # Keep complete checksummed results (small compared with batch databases).
                retained=self.root/'manifests'/f'{name}-batch-{batch_number:04d}.tsv.gz'
                with hits.open('rb') as f, retained.open('wb') as out, gzip.GzipFile(filename='',mode='wb',fileobj=out,mtime=0) as g:
                    shutil.copyfileobj(f,g)
                data['retained_result_sha256']=digest(retained)
                save(cert,data)
                # Search tmp cleanup is postponed until independent final certification.
                # Bounded batches must not accumulate databases: the per-batch zero-match
                # re-search below certifies its clean survivors before deletion.
                if not independent:
                    clean=[r for r in rows if f's_{r[0].hex()}_{r[1]}' not in nearest]
                    clean_fasta=folder/'clean.fasta'
                    with clean_fasta.open('w') as f:
                        for h,off,seq,*_ in reversed(clean):
                            f.write(f'>s_{h.hex()}_{off}\n{seq}\n')
                    clean_hits=folder/'clean_matches.tsv'
                    if clean:
                        self.run(self.search_command(clean_fasta,clean_hits,folder/'verify_tmp'),folder/'verify_command.log')
                        if clean_hits.stat().st_size:
                            raise RuntimeError('DATA2-D: batch survivor verification failed')
                        data['batch_zero_overlap_verified']=True
                        data['batch_verify_result_sha256']=digest(clean_hits)
                    else:
                        data['batch_zero_overlap_verified']=True
                        data['batch_verify_result_sha256']=hashlib.sha256(b'').hexdigest()
                    save(cert,data)
                ex.execute('INSERT INTO batches VALUES(?,?,?,?)',(name,batch_number,len(rows),str(cert)))
                ex.commit()
                self.cleanup([folder],cert)
                matches_total+=len(nearest)
                print(json.dumps({'stage':name,'batch':batch_number,'screened':batch_number*100_000,'hits':matches_total}),flush=True)
            ex.commit()
            removed=ex.execute('SELECT count(*) FROM removed').fetchone()[0]
            source_counts=db.execute('SELECT sum(c.historical),sum(c.external) FROM candidates c JOIN excluded.removed e USING(h,off)').fetchone()
            info={'candidates_screened':limit,'removed':0 if independent else removed,
                'removal_fraction':removed/n if n and not independent else 0,
                'historical_removals':source_counts[0] or 0,'external_removals':source_counts[1] or 0,
                'source_counts_overlap_for_dual_provenance':True,
                'clean_pool':n if independent else n-removed,
                'zero_detected_violations':True if independent else None,
                'query_target_policy':inputs['policy'],'search_commands':'Per-batch manifests',
                'identity_histogram_of_detected_nearest_matches':{str(i):ex.execute('SELECT count(*) FROM removed WHERE identity>=? AND identity<?',(i/10,(i+1)/10 if i<9 else 1.001)).fetchone()[0] for i in range(3,10)},
                'limitation':'Zero detected violations under recorded MMseqs heuristic search; not an exhaustive proof of all possible alignments.'}
            ex.close()
            db.close()
            p=self.root/'stats'/(name+'.json')
            save(p,info)
            save(self.report/(name+'.json'),info)
            self.complete(name,inputs,[p,progress_path],statistics=info)

    def final(self):
        """Checkpoint assignments, then publish each shard and gzip member separately."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        inputs={'reservoir_marker':digest(self.marker('reservoir')),
            'verification_marker':digest(self.marker('verify_protected')),
            'target':self.target,'shards':SHARD_COUNT,'script_sha256':digest(__file__)}
        if self.completed('final',inputs):
            return
        pool=json.loads((self.root/'stats/screen.json').read_text())['clean_pool']
        n=min(pool,self.target)
        if n<MIN_FINAL:
            save(self.report/'status.json',{'classification':'DATA2-C','clean_pool':pool,'training':False})
            raise RuntimeError('DATA2-C: clean pool below 10M; no other database allowed')
        db=connection(self.reservoir)
        db.execute('ATTACH DATABASE ? AS excluded',(str(self.root/'manifests/exclusions.sqlite'),))
        assignment_path=self.root/'tmp/shard_assignments.sqlite'
        assignments=connection(assignment_path)
        assignments.execute('CREATE TABLE IF NOT EXISTS assignment(idx INTEGER PRIMARY KEY,h BLOB,off INTEGER,shard INTEGER,row INTEGER)')
        assignments.execute('CREATE INDEX IF NOT EXISTS shard_rows ON assignment(shard,row)')
        assignments.execute('CREATE UNIQUE INDEX IF NOT EXISTS assignment_unique ON assignment(h,off)')
        assignments.execute('CREATE TABLE IF NOT EXISTS progress(k TEXT PRIMARY KEY,v TEXT)')
        prev=assignments.execute("SELECT v FROM progress WHERE k='state'").fetchone()
        ast=json.loads(prev[0]) if prev else {'inputs':inputs,'count':0,'last':None,'counts':[0]*SHARD_COUNT,'residues':[0]*SHARD_COUNT,'complete':False}
        if ast['inputs']!=inputs:
            raise RuntimeError('assignment checkpoint inputs changed')
        with self.stage('residue_balanced_assignment'):
            if not ast['complete']:
                boundary=''
                params=[]
                if ast['last']:
                    boundary='AND (c.p,c.h,c.off)>(?,?,?)'
                    params=[bytes.fromhex(ast['last'][0]),bytes.fromhex(ast['last'][1]),ast['last'][2]]
                params.append(n-ast['count'])
                cursor=db.execute('SELECT c.h,c.off,c.p,length(c.seq) FROM candidates c WHERE NOT EXISTS(SELECT 1 FROM excluded.removed e WHERE e.h=c.h AND e.off=c.off) '+boundary+' ORDER BY c.p,c.h,c.off LIMIT ?',params)
                heap=[(load,i) for i,load in enumerate(ast['residues'])]
                heapq.heapify(heap)
                for h,off,p,length in cursor:
                    load,shard=heapq.heappop(heap)
                    assignments.execute('INSERT INTO assignment VALUES(?,?,?,?,?)',(ast['count'],h,off,shard,ast['counts'][shard]))
                    ast['count']+=1
                    ast['counts'][shard]+=1
                    ast['residues'][shard]+=length
                    heapq.heappush(heap,(ast['residues'][shard],shard))
                    ast['last']=[p.hex(),h.hex(),off]
                    if ast['count']%100_000==0:
                        assignments.execute("INSERT OR REPLACE INTO progress VALUES('state',?)",(json.dumps(ast),))
                        assignments.commit()
                        self.guard()
                        print(json.dumps({'stage':'assignment','n':ast['count']}),flush=True)
                assert ast['count']==n
                ast['complete']=True
                assignments.execute("INSERT OR REPLACE INTO progress VALUES('state',?)",(json.dumps(ast),))
                assignments.commit()
        assignments.close()
        db.execute('ATTACH DATABASE ? AS assignments',(str(assignment_path),))
        out=self.root/'corpus'
        shardroot=out/'shards'
        manifest=out/'manifest'
        shardroot.mkdir(exist_ok=True)
        manifest.mkdir(exist_ok=True)
        schema=pa.schema([('sample_id',pa.string()),('sequence',pa.string()),('length',pa.int16())])
        mschema=pa.schema([('index',pa.int64()),('selection_rank',pa.int64()),('sample_id',pa.string()),('sequence_sha256',pa.binary(32)),
            ('priority',pa.binary(32)),('uniref_id',pa.string()),('source_offset',pa.int64()),
            ('source_historical_e012',pa.bool_()),('source_uniref50_2026_03',pa.bool_()),
            ('multiplicity',pa.int32()),('length',pa.int16()),('shard',pa.int16()),('shard_row',pa.int32())])
        fasta=out/f'e012_uniref50_2026_03_{n}.fasta.gz'
        ids=out/'ids.txt.gz'
        sample=self.root/'clusters/diversity_sample_1000000.fasta'
        fstate=self.root/'tmp/final_checkpoint.json'
        state=json.loads(fstate.read_text()) if fstate.exists() else {'inputs':inputs,'next_shard':0,'count':0,
            'fasta_bytes':0,'ids_bytes':0,'sample_bytes':0,'stats':Stats().checkpoint(),
            'composition':{},'shards':[],'sample_n':0,'audit_n':0}
        if state['inputs']!=inputs:
            raise RuntimeError('final checkpoint input mismatch')
        for path,key in [(fasta,'fasta_bytes'),(ids,'ids_bytes'),(sample,'sample_bytes')]:
            path.parent.mkdir(exist_ok=True,parents=True)
            if not path.exists():
                if state[key]:
                    raise RuntimeError('missing completed final checkpoint output')
                path.touch()
            with path.open('r+b') as f:
                if path.stat().st_size<state[key]:
                    raise RuntimeError('truncated completed final checkpoint output')
                f.truncate(state[key])  # Discard only uncertified tail from interrupted append.
        def in_sample(index,size):
            j=(index*size+n-1)//n
            return j<size and (j*n)//size==index
        stats=Stats(state['stats'])
        composition=Counter(state['composition'])
        with self.stage('finalization'):
            for shard in range(state['next_shard'],SHARD_COUNT):
                folder=self.root/'tmp'/f'final-shard-{shard:03d}'
                folder.mkdir(exist_ok=True)
                pf=shardroot/f'part-{shard:03d}.parquet'
                pm=manifest/f'part-{shard:03d}.parquet'
                audit_part=self.root/'manifests'/f'audit-part-{shard:03d}.parquet'
                part=folder/'canonical.fasta.gz'
                ipart=folder/'ids.txt.gz'
                spart=folder/'sample.fasta'
                rows=[]
                meta=[]
                audit_rows=[]
                shard_stats=Stats()
                shard_composition=Counter()
                sample_n=0
                pw=pq.ParquetWriter(pf.with_suffix('.parquet.partial'),schema,compression='zstd')
                mw=pq.ParquetWriter(pm.with_suffix('.parquet.partial'),mschema,compression='zstd')
                cursor=db.execute('SELECT a.idx,a.row,c.h,c.off,c.p,c.seq,c.source_id,c.multiplicity,c.historical,c.external FROM assignments.assignment a JOIN candidates c USING(h,off) WHERE a.shard=? ORDER BY a.row',(shard,))
                with part.open('wb') as fo,gzip.GzipFile(filename='',mode='wb',fileobj=fo,mtime=0,compresslevel=6) as fg, ipart.open('wb') as io,gzip.GzipFile(filename='',mode='wb',fileobj=io,mtime=0) as ig,spart.open('w') as sf:
                    for rank,srow,h,off,p,seq,sid,multi,hist,ext in cursor:
                        index=state['count']+shard_stats.n
                        assert valid(seq) and h==sequence_digest(seq) and p==priority(seq)
                        sample_id=f's_{h.hex()}_{off}'
                        fg.write(f'>{sample_id}\n{seq}\n'.encode())
                        ig.write(f'{index}\t{sample_id}\n'.encode())
                        row={'sample_id':sample_id,'sequence':seq,'length':len(seq)}
                        rows.append(row)
                        meta.append({'index':index,'selection_rank':rank,'sample_id':sample_id,'sequence_sha256':h,
                            'priority':p,'uniref_id':sid if ext else None,'source_offset':off,
                            'source_historical_e012':bool(hist),'source_uniref50_2026_03':bool(ext),
                            'multiplicity':multi,'length':len(seq),'shard':shard,'shard_row':srow})
                        if in_sample(index,DIAGNOSTIC_SAMPLE_N):
                            sf.write(f'>{sample_id}\n{seq}\n')
                            sample_n+=1
                        if in_sample(index,AUDIT_SAMPLE_N):
                            audit_rows.append(row)
                        shard_stats.add(seq)
                        shard_composition[f'historical={hist},external={ext}']+=1
                        if len(rows)>=4096:
                            pw.write_table(pa.Table.from_pylist(rows,schema=schema))
                            mw.write_table(pa.Table.from_pylist(meta,schema=mschema))
                            rows.clear()
                            meta.clear()
                    if rows:
                        pw.write_table(pa.Table.from_pylist(rows,schema=schema))
                        mw.write_table(pa.Table.from_pylist(meta,schema=mschema))
                pw.close()
                mw.close()
                assert shard_stats.n==ast['counts'][shard]
                assert shard_stats.residues==ast['residues'][shard]
                pf.with_suffix('.parquet.partial').replace(pf)
                pm.with_suffix('.parquet.partial').replace(pm)
                pq.write_table(pa.Table.from_pylist(audit_rows,schema=schema),audit_part,compression='zstd')
                # Shard sequences and manifest hashes must agree before the append marker.
                shard_h=hashlib.sha256()
                meta_h=hashlib.sha256()
                for batch in pq.ParquetFile(pf).iter_batches(columns=['sequence']):
                    for seq in batch.column(0).to_pylist():
                        shard_h.update(sequence_digest(seq))
                for batch in pq.ParquetFile(pm).iter_batches(columns=['sequence_sha256']):
                    for h in batch.column(0).to_pylist():
                        meta_h.update(h)
                assert shard_h.digest()==meta_h.digest()
                shard_info={'path':str(pf),'manifest_path':str(pm),'sequences':shard_stats.n,
                    'residues':shard_stats.residues,'bytes':pf.stat().st_size,'sha256':digest(pf),
                    'manifest_sha256':digest(pm),'canonical_member_sha256':digest(part),
                    'ids_member_sha256':digest(ipart),'sequence_order_sha256':shard_h.hexdigest()}
                for source,destination,key in [(part,fasta,'fasta_bytes'),(ipart,ids,'ids_bytes'),(spart,sample,'sample_bytes')]:
                    with source.open('rb') as r,destination.open('ab') as w:
                        shutil.copyfileobj(r,w)
                        w.flush()
                        os.fsync(w.fileno())
                    state[key]=destination.stat().st_size
                stats.n+=shard_stats.n
                stats.residues+=shard_stats.residues
                stats.lengths.update(shard_stats.lengths)
                stats.aa.update(shard_stats.aa)
                composition.update(shard_composition)
                state['next_shard']=shard+1
                state['count']=stats.n
                state['stats']=stats.checkpoint()
                state['composition']=dict(composition)
                state['shards'].append(shard_info)
                state['sample_n']+=sample_n
                state['audit_n']+=len(audit_rows)
                save(fstate,state)
                cert=self.root/'manifests'/f'final-shard-{shard:03d}.complete.json'
                save(cert,{'inputs':inputs,'outputs':shard_info,'checkpoint_sha256':digest(fstate),
                    'protected_verification_marker_sha256':digest(self.marker('verify_protected')),'timestamp':now()})
                self.cleanup([folder],cert)
                self.guard()
                print(json.dumps({'stage':'final','shard':shard,'n':stats.n}),flush=True)
        assert stats.n==n and state['sample_n']==DIAGNOSTIC_SAMPLE_N and state['audit_n']==AUDIT_SAMPLE_N
        audit=out/'audit_10000.parquet'
        aw=pq.ParquetWriter(audit,schema,compression='zstd')
        for shard in range(SHARD_COUNT):
            aw.write_table(pq.read_table(self.root/'manifests'/f'audit-part-{shard:03d}.parquet'))
        aw.close()
        db.close()
        summary={'classification':'DATA2-B' if n==20_000_000 else 'DATA2-A',
            'statistics':stats.summary(),'source_composition':dict(composition),
            'fasta':str(fasta),'fasta_bytes':fasta.stat().st_size,'manifest':str(manifest),
            'shard_root':str(shardroot),'shards':state['shards'],
            'shard_bytes_total':sum(s['bytes'] for s in state['shards']),
            'scale_vs_nominal':n/231743,'scale_vs_unique':n/87930,
            'historical_unique_residue_scale':stats.residues/17211507,
            'selection':'Ascending frozen priority then sequence hash and source offset; all clean records if fewer than target. Output order is residue-balanced shard order.',
            'diversity_sample_n':DIAGNOSTIC_SAMPLE_N,'diversity_sample_rule':'Systematic final-output ranks floor(j*N/1000000)',
            'audit_sample_n':AUDIT_SAMPLE_N,'audit_sample_sha256':digest(audit),'training':False}
        save(out/'summary.json',summary)
        outputs=[fasta,ids,audit,sample,out/'summary.json']+sorted(shardroot.glob('*.parquet'))+sorted(manifest.glob('*.parquet'))
        with (out/'SHA256SUMS').open('w') as f:
            for p in outputs:
                f.write(f'{digest(p)}  {p.relative_to(self.root)}\n')
        outputs.append(out/'SHA256SUMS')
        save(self.report/'final_manifest_summary.json',summary)
        self.complete('final',inputs,outputs,statistics=summary)

    def certify(self):
        """Read every frozen representation; verify loader behavior without a model."""
        import pyarrow.parquet as pq
        import sys
        os.environ['CUDA_VISIBLE_DEVICES']=''
        os.environ.setdefault('OMP_NUM_THREADS','2')
        sys.path.insert(0,str(self.repo/'src'))
        from protein_sequence_generation.dataset import ProteinSequenceDataset
        from protein_sequence_generation.collate import collate_sequences
        from protein_sequence_generation.e012 import batch
        import torch
        torch.set_num_threads(2)
        inputs={'final_marker_sha256':digest(self.marker('final')),
            'protected_verification_sha256':digest(self.marker('verify_protected'))}
        if self.completed('certified',inputs):
            return
        final=json.loads(self.marker('final').read_text())
        for p,h in final['outputs'].items():
            if digest(p)!=h:
                raise RuntimeError(f'Final checksum mismatch: {p}')
        summary=final['statistics']
        assert digest(self.reservoir)==json.loads(self.marker('reservoir').read_text())['outputs'][str(self.reservoir)]
        db=connection(self.reservoir)
        ap=self.root/'tmp/shard_assignments.sqlite'
        db.execute('ATTACH DATABASE ? AS assignments',(str(ap),))
        db.execute('ATTACH DATABASE ? AS excluded',(str(self.root/'manifests/exclusions.sqlite'),))
        assert db.execute('SELECT count(*) FROM assignments.assignment').fetchone()[0]==summary['statistics']['sequences']
        assert db.execute('SELECT count(*) FROM assignments.assignment a JOIN excluded.removed e USING(h,off)').fetchone()[0]==0
        fasta=Path(summary['fasta'])
        ids=fasta.parent/'ids.txt.gz'
        stats=Stats()
        audit_seqs=set()
        sequence_order=hashlib.sha256()
        with self.stage('certify'),gzip.open(fasta,'rt') as ff,gzip.open(ids,'rt') as fi:
            for shard in summary['shards']:
                pf=Path(shard['path'])
                pm=Path(shard['manifest_path'])
                actual_residues=0
                cursor=db.execute('SELECT a.idx,a.row,c.h,c.off,c.p,c.seq,c.historical,c.external FROM assignments.assignment a JOIN candidates c USING(h,off) WHERE a.shard=? ORDER BY a.row',(int(pf.stem.split('-')[1]),))
                metadata=pq.ParquetFile(pm).iter_batches(batch_size=8192)
                sequences=pq.ParquetFile(pf).iter_batches(batch_size=8192)
                matched=0
                for sb,mb in zip(sequences,metadata,strict=True):
                    sr,mr=sb.to_pylist(),mb.to_pylist()
                    assert len(sr)==len(mr)
                    for row,meta in zip(sr,mr,strict=True):
                        expected=cursor.fetchone()
                        assert expected is not None
                        rank,srow,h,off,p,seq,hist,ext=expected
                        assert valid(seq) and row['sequence']==seq
                        assert row['sample_id']==meta['sample_id']==f's_{h.hex()}_{off}'
                        assert meta['sequence_sha256']==sequence_digest(seq)==h
                        assert meta['priority']==priority(seq)==p
                        assert row['length']==meta['length']==len(seq)
                        assert meta['index']==stats.n and meta['selection_rank']==rank and meta['shard_row']==srow
                        assert meta['source_historical_e012']==bool(hist) and meta['source_uniref50_2026_03']==bool(ext)
                        assert ff.readline().rstrip()=='>'+row['sample_id']
                        assert ff.readline().rstrip()==seq
                        assert fi.readline().rstrip()==f'{stats.n}\t{row["sample_id"]}'
                        stats.add(seq)
                        sequence_order.update(h)
                        actual_residues+=len(seq)
                        matched+=1
                assert cursor.fetchone() is None
                assert matched==shard['sequences'] and actual_residues==shard['residues']
                self.guard()
            assert not ff.readline() and not fi.readline()
        assert stats.summary()==summary['statistics']
        audit_path=fasta.parent/'audit_10000.parquet'
        dataset=ProteinSequenceDataset(audit_path)
        assert len(dataset)==AUDIT_SAMPLE_N and dataset.audit['exact_duplicate_count']==0
        started=time.monotonic()
        audit_residues=0
        for first in range(0,len(dataset),128):
            items=[dataset[i] for i in range(first,min(first+128,len(dataset)))]
            collated=collate_sequences(items)
            historical_batch=batch([{'sample_id':i['sample_id'],'sequence':i['sequence']} for i in items],'cpu')
            for key in ['input_ids','target_ids','attention_mask','lengths']:
                assert torch.equal(collated[key],historical_batch[key])
            for j,item in enumerate(items):
                length=int(item['length'])
                assert collated['input_ids'][j,0]==dataset.vocabulary.bos_id
                assert torch.equal(collated['input_ids'][j,1:length],collated['target_ids'][j,:length-1])
                assert collated['attention_mask'][j].sum()==length
                audit_residues+=length
                audit_seqs.add(item['sequence'])
        elapsed=time.monotonic()-started
        assert len(audit_seqs)==AUDIT_SAMPLE_N
        # Audit sequences are a verified subset of independently screened final inputs.
        audit_hashes=[sequence_digest(s) for s in audit_seqs]
        for h in audit_hashes:
            assert db.execute('SELECT count(*) FROM assignments.assignment WHERE h=?',(h,)).fetchone()[0]>=1
        db.close()
        tracked=subprocess.check_output(['git','ls-files','-z'],cwd=self.repo).split(b'\0')
        bulk=[os.fsdecode(p) for p in tracked if p and (self.repo/os.fsdecode(p)).is_file() and (self.repo/os.fsdecode(p)).stat().st_size>100_000_000]
        assert not bulk,bulk
        changes=subprocess.check_output(['git','diff','e3cd8ad337407a6de968fde9a24fb027f1336fda','--name-only'],cwd=self.repo,text=True).splitlines()
        forbidden=[p for p in changes if p.startswith('reports/') and not p.startswith(REPORT_REL+'/')]
        assert not forbidden,forbidden
        cert={'certified':True,'classification':summary['classification'],
            'sequences':stats.n,'residues':stats.residues,'representation_counts_agree':True,
            'sequence_order_sha256':sequence_order.hexdigest(),
            'final_exact_unique':True,'shard_exclusivity':True,'manifest_index_consistency':True,
            'all_sha256_verified':True,'protected_exact_overlap':0,'protected_detected_homology_overlap':0,
            'audit_sample_n':AUDIT_SAMPLE_N,'audit_sha256':digest(audit_path),'tokenizer_compatible':True,
            'causal_collate_historical_parity':True,'bos_target_shift':True,
            'loader_sequences_per_second':AUDIT_SAMPLE_N/elapsed,'loader_residues_per_second':audit_residues/elapsed,
            'loader_audit_wall_seconds':elapsed,'large_git_tracked':bulk,
            'historical_reports_unchanged':True,'training_launched':False}
        cp=self.root/'corpus/certification.json'
        save(cp,cert)
        save(self.report/'certification.json',cert)
        self.complete('certified',inputs,[cp],statistics=cert)
        # Reservoir is derivable, but retain it until diagnostic preparation and
        # durable frozen-corpus certification are complete. No raw/final cleanup.

    def diversity(self):
        """Diagnose sample only, with a predeclared pilot scaling rule."""
        sample=self.root/'clusters/diversity_sample_1000000.fasta'
        inputs={'certification_marker_sha256':digest(self.marker('certified')),
            'sample_sha256':digest(sample),'rule_version':'v2b_sample_resource_rule_v1'}
        if self.completed('diversity',inputs):
            return
        self.cleanup([self.reservoir,self.root/'tmp/shard_assignments.sqlite',self.root/'tmp/final_checkpoint.json'],self.marker('certified'))
        def cluster_command(fasta,prefix,tmp,ident):
            return ['mmseqs','easy-linclust',fasta,prefix,tmp,'--min-seq-id',str(ident),
                '-c','0.80','--cov-mode','0','--cluster-mode','0','--threads','8',
                '--remove-tmp-files','0']
        def write_subset(target, size):
            with sample.open() as src,Path(target).open('w') as dst:
                for i in range(1_000_000):
                    head,seq=src.readline(),src.readline()
                    if not head or not seq:
                        raise RuntimeError('diversity sample truncated')
                    # Same fixed rank rule for both pilots and reduced diagnostics.
                    j=(i*size+1_000_000-1)//1_000_000
                    if j<size and j*1_000_000//size==i:
                        dst.write(head+seq)
        stats={}
        with self.stage('sampled_diversity'):
            for ident in [.50,.30]:
                label=f'identity_{int(ident*100)}'
                work=self.root/'clusters'/label
                work.mkdir(exist_ok=True)
                pilot=work/'pilot_10000.fasta'
                write_subset(pilot,10_000)
                cert=work/'pilot.complete.json'
                if cert.exists():
                    pilot_stats=json.loads(cert.read_text())
                    assert pilot_stats['sample_sha256']==inputs['sample_sha256']
                else:
                    start=time.monotonic()
                    command=cluster_command(pilot,work/'pilot',work/'pilot_tmp',ident)
                    self.run(command,work/'pilot.log')
                    seconds=time.monotonic()-start
                    disk=sum(p.stat().st_size for p in work.rglob('*') if p.is_file())
                    pilot_stats={'sample_sha256':inputs['sample_sha256'],'pilot_n':10_000,
                        'wall_seconds':seconds,'measured_retained_disk_bytes':disk,
                        'children_peak_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                        'command':list(map(str,command)),'pilot_cluster_tsv_sha256':digest(work/'pilot_cluster.tsv')}
                    save(cert,pilot_stats)
                # Predeclared conservative linear extrapolation with 4x slack.
                # Try 1M, then 250k, then 100k, then 10k. Keep <=2 hours, <=8GiB
                # estimated RAM and <=20GB temp while preserving runtime disk reserve.
                chosen=10_000
                estimates=[]
                for size in [1_000_000,250_000,100_000,10_000]:
                    factor=size/10_000
                    estimate={'n':size,'wall_seconds':pilot_stats['wall_seconds']*factor*4,
                        'disk_bytes':pilot_stats['measured_retained_disk_bytes']*factor*4,
                        'ram_bytes':pilot_stats['children_peak_rss_kib']*1024*factor*4}
                    estimate['pass']=estimate['wall_seconds']<=7200 and estimate['disk_bytes']<=20_000_000_000 and estimate['ram_bytes']<=8*1024**3 and estimate['disk_bytes']<shutil.disk_usage(self.root).free-self.floor
                    estimates.append(estimate)
                    if estimate['pass']:
                        chosen=size
                        break
                if chosen==10_000:
                    tsv=work/'pilot_cluster.tsv'
                    command=pilot_stats['command']
                else:
                    diagnostic=work/f'diagnostic_{chosen}.fasta'
                    write_subset(diagnostic,chosen)
                    prefix=work/f'diagnostic_{chosen}'
                    tsv=Path(str(prefix)+'_cluster.tsv')
                    command=cluster_command(diagnostic,prefix,work/'diagnostic_tmp',ident)
                    diag_cert=work/'diagnostic.complete.json'
                    if diag_cert.exists():
                        old=json.loads(diag_cert.read_text())
                        assert old['tsv_sha256']==digest(tsv)
                        assert old['sample_sha256']==inputs['sample_sha256']
                    else:
                        self.run(command,work/'diagnostic.log')
                        save(diag_cert,{'sample_sha256':inputs['sample_sha256'],'tsv_sha256':digest(tsv),'n':chosen,'command':list(map(str,command))})
                sizes=Counter()
                seen=0
                with tsv.open() as f:
                    for line in f:
                        cluster,member=line.rstrip().split('\t')
                        sizes[cluster]+=1
                        seen+=1
                assert seen==chosen
                values=sorted(sizes.values())
                k=len(values)
                entropy=-sum((v/seen)*math.log(v/seen) for v in values)
                gini=sum((2*i-k-1)*v for i,v in enumerate(values,1))/(k*seen)
                result={'measurement_scope':'DETERMINISTIC DIAGNOSTIC SAMPLE ONLY; not full corpus cluster counts',
                    'sample_n':chosen,'available_diagnostic_sample_n':1_000_000,'identity':ident,
                    'coverage':.80,'cov_mode':0,'clusters':k,'singleton_clusters':sum(v==1 for v in values),
                    'singleton_cluster_fraction':sum(v==1 for v in values)/k,
                    'singleton_sequence_fraction':sum(v==1 for v in values)/seen,
                    'p95_cluster_size':values[max(0,math.ceil(.95*k)-1)],
                    'p99_cluster_size':values[max(0,math.ceil(.99*k)-1)],'max_cluster_size':max(values),
                    'gini':gini,'inverse_simpson_effective_count':seen**2/sum(v*v for v in values),
                    'entropy_effective_count':math.exp(entropy),'cluster_size_histogram':dict(Counter(values)),
                    'pilot':pilot_stats,'resource_estimates':estimates,'command':list(map(str,command)),
                    'mmseqs_version':subprocess.check_output(['mmseqs','version'],text=True).strip(),
                    'tsv_sha256':digest(tsv),'limitation':'No full-corpus local clustering; UniRef50 representative membership is the coarse diversity substrate. Linclust is a scalable heuristic.'}
                stats[label]=result
                save(work/'statistics.json',result)
                # Keep diagnostic TSV and sample; remove only reproducible MMseqs tmp.
                self.cleanup([work/'pilot_tmp',work/'diagnostic_tmp'],self.marker('certified'))
        p=self.root/'stats/diversity_stats.json'
        save(p,stats)
        save(self.report/'diversity_stats.json',stats)
        self.complete('diversity',inputs,[p],statistics=stats)

    def handoff(self):
        required=['raw','reservoir','screen','verify_protected','final','certified','diversity']
        if not all(self.marker(s).is_file() for s in required):
            raise RuntimeError('Cannot publish final handoff before all required stage markers')
        source=json.loads((self.report/'source_provenance.json').read_text())
        filtering=json.loads((self.report/'filtering_stats.json').read_text())
        screened=json.loads((self.report/'screen.json').read_text())
        protected=json.loads((self.report/'protected_definition.json').read_text())
        final=json.loads((self.report/'final_manifest_summary.json').read_text())
        certification=json.loads((self.report/'certification.json').read_text())
        diversity=json.loads((self.report/'diversity_stats.json').read_text())
        assert certification['certified'] and certification['all_sha256_verified']
        telemetry=[json.loads(line) for line in (self.root/'stats/telemetry.jsonl').read_text().splitlines()]
        n=final['statistics']['sequences']
        method='Ascending frozen SHA256 priority (seed 12014), after exact deduplication and protected exclusion; all clean candidates if below 20M.'
        report={'result_commit':'Git commit containing this handoff (self-reference omitted)',
            'branch':'e012-causal-rope-sequence','d_root':str(self.root),
            'starting_free_disk_bytes':self.plan['starting_free_bytes'],
            'redesigned_planned_peak_bytes':self.plan['planned_peak_bytes'],
            'measured_peak_consumption_bytes':max(t['measured_peak_consumption_bytes'] for t in telemetry),
            'measured_peak_method':'Sampled filesystem free; includes unrelated D: activity, not an exact per-job working-set peak.',
            'source_archive_sha256':source['sha256'],'official_md5_verified':source['official_md5_verified'],
            'raw_representative_count':filtering['raw_count'],'length_valid':filtering['length_valid'],
            'canonical_valid':filtering['canonical_valid'],'exact_unique_encountered':filtering['external_exact_unique_encountered'],
            'exact_duplicates_removed':filtering['exact_duplicates_removed'],
            'reservoir_size':filtering['external_reservoir_count'],'historical_unique':filtering['historical_unique'],
            'historical_unique_union_size':filtering['union_count'],'protected_count':protected['unique_sequences'],
            'protected_removals':screened['removed'],'historical_removals':screened['historical_removals'],
            'external_removals':screened['external_removals'],'clean_pool':screened['clean_pool'],
            'final_corpus_n':n,'total_residues':final['statistics']['total_residues'],
            'compressed_final_corpus_bytes':final['fasta_bytes'],'shard_size_total_bytes':final['shard_bytes_total'],
            'remaining_d_free_bytes':shutil.disk_usage(self.root).free,
            'scale_vs_nominal_e012':final['scale_vs_nominal'],'scale_vs_exact_unique_e012':final['scale_vs_unique'],
            'diversity_sample_50':diversity['identity_50'],'diversity_sample_30':diversity['identity_30'],
            'full_corpus_cluster_counts':'NOT MEASURED; UniRef50 representative membership is coarse substrate.',
            'protected_overlap_zero':True,'shard_verification':certification,
            'length_strata':final['statistics']['length_strata'],
            'aa_frequencies':final['statistics']['aa_frequencies'],
            'selection_method':method,'final_fasta':final['fasta'],'final_manifest':final['manifest'],
            'shard_root':final['shard_root'],'shard_count':512,
            'tests':'14 V2B integrity tests including real MMseqs policy boundary, collision, crash/resume and loader parity; all frozen bytes independently verified.',
            'training_launched':False,'classification':final['classification'],
            'recommended_next_experiment':'One fixed-budget E012 causal-RoPE data-scaling comparison against historical unique TRAIN, with unchanged architecture and protected evaluation panels.'}
        save(self.report/'chatgpt_handoff.json',report)
        checksums={'source_sha256':source['sha256'],'official_md5':source['md5'],
            'protected_fasta_sha256':protected['fasta_sha256'],
            'final_sha256sums_sha256':digest(self.root/'corpus/SHA256SUMS'),
            'stage_markers':{s:digest(self.marker(s)) for s in required},
            'metadata':{p.name:digest(p) for p in (self.root/'checksums').iterdir() if p.is_file()}}
        save(self.report/'checksums.json',checksums)
        lines=[f"# {report['classification']} — E012 V2B frozen corpus",'',
            f"Final corpus: {n:,} exact-unique canonical sequences, lengths 20–500, {report['total_residues']:,} residues.",
            f"Protected set: {report['protected_count']:,} unique sequences; {report['protected_removals']:,} union candidates excluded.",
            'Independent search detected zero protected cross-boundary violations. Exact overlaps are explicitly excluded.',
            f"Source: UniRef50 2026_03, SHA256 {source['sha256']}; official MD5 passes.",'',
            f"Disk: {report['starting_free_disk_bytes']/1e9:.2f} GB initially free; {report['redesigned_planned_peak_bytes']/1e9:.2f} GB planned peak.",
            f"Sampled peak filesystem consumption: {report['measured_peak_consumption_bytes']/1e9:.2f} GB; includes unrelated D: activity.",
            f"Remaining free: {report['remaining_d_free_bytes']/1e9:.2f} GB.",'',
            f"Scale vs nominal 231,743: {report['scale_vs_nominal_e012']:.2f}×; vs exact unique 87,930: {report['scale_vs_exact_unique_e012']:.2f}×.",
            f"Selection: {method}",'',
            'No full-corpus local clustering was performed. UniRef50 representative membership supplies coarse diversity.',
            'Local diversity metrics characterize the explicitly reported deterministic sample, never the full corpus.']
        for key,label in [('identity_50','50%'),('identity_30','30%')]:
            d=diversity[key]
            lines.append(f"{label}: sample N={d['sample_n']:,}, clusters={d['clusters']:,}, inverse-Simpson={d['inverse_simpson_effective_count']:.2f}, entropy-effective={d['entropy_effective_count']:.2f}, Gini={d['gini']:.4f}.")
        lines+=['',f"Canonical FASTA: {final['fasta']}",f"Metadata manifest: {final['manifest']}",
            f"512 residue-balanced zstd Parquet shards: {final['shard_root']}",
            'All representation counts, residues, indices, shard exclusivity and checksums pass.',
            '10k audit passes tokenizer, historical causal collate, BOS/target shift and protected-subset checks.',
            'See resource_telemetry.jsonl, cleanup.jsonl, PROTOCOL.md and machine-readable chatgpt_handoff.json.',
            'Historical E012 and DATA-E artifacts remain unchanged. Training launched: NO.', '',
            'Recommended next experiment: '+report['recommended_next_experiment']]
        (self.report/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
        print(json.dumps(report,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['verify-raw','reservoir','target-index','screen','verify-protected','final','certify','diversity','handoff','all'])
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    a=p.parse_args()
    b=Build(a.root)
    stages={'verify-raw':b.verify_raw,'reservoir':b.filter_reservoir,'target-index':b.target_index,'screen':b.screen,
        'verify-protected':lambda:b.screen(True),'final':b.final,'certify':b.certify,'diversity':b.diversity,'handoff':b.handoff}
    if a.stage=='all':
        for stage in stages.values():
            stage()
    else:
        stages[a.stage]()

if __name__=='__main__':
    main()
