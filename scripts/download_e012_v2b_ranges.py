"""Four resumable curl ranges from one official mirror; raw MD5 pins identity."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path('/mnt/d/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2')
REPO=Path('/home/simostocco/proteinGen-causal-rope')
URL='https://ftp.uniprot.org/pub/databases/uniprot/current_release/uniref/uniref50/uniref50.fasta.gz'
SIZE=8_780_552_383
MD5='0492e3cf4093276ae4319ba24d52514f'

def now():
    return datetime.now(timezone.utc).isoformat()

def save(p,obj):
    tmp=p.with_name(p.name+'.partial')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    tmp.replace(p)

def hash_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()

def main():
    folder=ROOT/'tmp/download_ranges'
    folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'layout.json'
    partial=ROOT/'raw/uniref50.fasta.gz.partial'
    raw=ROOT/'raw/uniref50.fasta.gz'
    plan=json.loads((ROOT/'stats/resource_plan.json').read_text())
    if marker.exists():
        layout=json.loads(marker.read_text())
    else:
        # Stop only this task's exact sequential downloader, leaving its valid prefix.
        for proc in Path('/proc').iterdir():
            if not proc.name.isdecimal():
                continue
            try:
                argv=(proc/'cmdline').read_bytes().split(b'\0')
                if argv and Path(os.fsdecode(argv[0])).name=='curl' and os.fsencode(str(partial)) in argv and URL.encode() in argv and b'--continue-at' in argv:
                    os.kill(int(proc.name),signal.SIGTERM)
            except (OSError,ProcessLookupError):
                continue
        time.sleep(2)
        prefix=folder/'prefix.bin'
        partial.replace(prefix)
        start=prefix.stat().st_size
        assert 0<start<SIZE
        step=(SIZE-start+3)//4
        layout={'url':URL,'release':'2026_03','expected_bytes':SIZE,'expected_md5':MD5,
            'start_observed':now(),'prefix_bytes':start,'prefix_sha256':hash_file(prefix),
            'ranges':[[start+i*step,min(SIZE-1,start+(i+1)*step-1)] for i in range(4)],
            'download_connections':4}
        save(marker,layout)
    plan['stage_components_bytes']['source_download']={'source_and_retained_resumable_ranges':SIZE*2,'metadata_allowance':1_000_000_000}
    plan['stage_totals_bytes']['source_download']=SIZE*2+1_000_000_000
    plan['planned_peak_bytes']=max(plan['stage_totals_bytes'].values())
    assert plan['planned_peak_bytes']<=plan['allowed_simultaneous_bytes']
    save(ROOT/'stats/resource_plan.json',plan)
    save(REPO/'reports/experiments/E012_causal_rope_sequence/data_v2b_uniref50_lowdisk/resource_plan.json',plan)
    def segment(i,bounds):
        begin,end=bounds
        path=folder/f'range-{i}.bin'
        for attempt in range(10):
            local=path.stat().st_size if path.exists() else 0
            expected=end-begin+1
            if local==expected:
                result={'range':bounds,'bytes':local,'sha256':hash_file(path)}
                save(folder/f'range-{i}.complete.json',result)
                return result
            if local>expected:
                raise RuntimeError('Range response overflow; no preprocessing authorized')
            headers=folder/f'range-{i}.headers'
            cmd=['curl','-fsSL','--connect-timeout','30','--range',f'{begin+local}-{end}',
                '--speed-time','60','--speed-limit','1000','--dump-header',str(headers),URL]
            with path.open('ab') as out:
                code=subprocess.call(cmd,stdout=out)
            header=headers.read_text().lower()
            expected_header=f'content-range: bytes {begin+local}-{end}/{SIZE}'
            if expected_header not in header:
                # Restore the byte prefix verified before this request, never use an
                # unverified response or interpret file size as completion.
                with path.open('r+b') as f:
                    f.truncate(local)
                raise RuntimeError('Official server did not honor the exact byte range')
            print(json.dumps({'segment':i,'attempt':attempt,'curl_exit':code,'bytes':path.stat().st_size}),flush=True)
        raise RuntimeError('Repeated range transfer failures; partial chunks retained for resume')
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(segment,i,b) for i,b in enumerate(layout['ranges'])]
        results=[f.result() for f in futures]
    # Reassemble from immutable verified chunks; an interrupted assembly restarts
    # only this derivable output, never the completed ranges.
    if raw.exists():
        raise RuntimeError('Verified raw source already exists; do not overwrite it')
    with partial.open('wb') as out:
        for source in [folder/'prefix.bin']+[folder/f'range-{i}.bin' for i in range(4)]:
            with source.open('rb') as f:
                for block in iter(lambda:f.read(8*1024**2),b''):
                    out.write(block)
        out.flush()
        os.fsync(out.fileno())
    assert partial.stat().st_size==SIZE
    provenance=dict(layout,end=now(),segments=results,assembled_bytes=partial.stat().st_size,
        source_http_headers='Last-Modified 03-Sep-2026 18:00:00 GMT; ETag 20b5c98bf-65a97eba0a800',
        official_md5_verified=False,sha256=None)
    save(ROOT/'checksums/download_transport.json',provenance)
    save(REPO/'reports/experiments/E012_causal_rope_sequence/data_v2b_uniref50_lowdisk/download_transport.json',provenance)
    print('Transfer assembled; mandatory verify-raw stage must pass before preprocessing',flush=True)

if __name__=='__main__':
    main()
