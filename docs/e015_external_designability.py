"""Resumable external E015 designability controller; production science is untouched.

Every generated file and subprocess cache is on D:. Official tools are invoked
as subprocesses. Scientific decisions and panel selection precede refolding.
"""
import os
from pathlib import Path

WORK = Path('/mnt/d/Simone/proteinGen/.e015_worktree')
ROOT = Path('/mnt/d/Simone/proteinGen/e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1')
OUT = ROOT / 'external_designability_loop_v1'
PY = '/home/simostocco/miniforge3/envs/proteingen/bin/python'
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
for k in ('TMPDIR','TMP','TEMP','XDG_CACHE_HOME','MPLCONFIGDIR','TORCH_HOME','HF_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','PIP_CACHE_DIR','JAX_COMPILATION_CACHE_DIR','COLABFOLD_CACHE_DIR'):
    os.environ[k] = str(OUT / 'cache' / k.lower())
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='0.80'
os.environ['TF_FORCE_UNIFIED_MEMORY']='0'
import argparse, datetime, hashlib, json, math, shutil, subprocess, sys, time, urllib.request
os.nice(10)

def read(p): return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p); assert p.is_relative_to(Path('/mnt/d'))
    p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.partial')
    with q.open('w') as f: json.dump(x,f,indent=2,allow_nan=False); f.flush(); os.fsync(f.fileno())
    q.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(1024**2): h.update(b)
    return h.hexdigest()
def sync_outputs(files):
    for name in files:
        with Path(name).open('rb') as f:os.fsync(f.fileno())

def progress(phase,**kw):
    p=OUT/'progress.json'; x=read(p) if p.exists() else {}
    x.update(timestamp_unix=time.time(),phase=phase,**kw);write(p,x)
def run(cmd,name,timeout=1800):
    d=OUT/'logs'; d.mkdir(parents=True,exist_ok=True)
    t=time.monotonic()
    with (d/(name+'.log')).open('w') as f:
        cpus=sorted(os.sched_getaffinity(0))[:2]
        p=subprocess.Popen(cmd,cwd=WORK,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT,preexec_fn=(lambda:os.sched_setaffinity(0,cpus)) if Path(cmd[0]).name=='colabfold_batch' else None)
        last=t
        while p.poll() is None:
            now=time.monotonic()
            if now-last>=30:
                x=resource(name);last=now
                owners=[int(row.split(',')[0]) for row in x['gpu_compute_apps'].splitlines() if row.strip()]
                if x['ram_available_bytes']<3*1024**3 or (shutil.which('powershell.exe') and x.get('windows_ram_available_bytes',0)<3*1024**3) or x['d_free_bytes']<30*1024**3 or any(pid!=p.pid for pid in owners):
                    import signal
                    write(OUT/'STOP_EXTERNAL.json',dict(reason='Resource pressure',stage=name,resources=x));p.send_signal(signal.SIGINT);p.wait()
                    raise RuntimeError('External stage cancelled cleanly for resource pressure')
            if now-t>timeout:
                import signal
                p.send_signal(signal.SIGINT);p.wait()
                raise TimeoutError(name+' timed out; standard interrupt cancellation, no production signal')
            time.sleep(1)
    write(d/(name+'.command.json'),dict(command=cmd,seconds=time.monotonic()-t,returncode=p.returncode))
    if p.returncode: raise RuntimeError(f'{name}: exit {p.returncode}; see {d/(name+".log")}')
    return (d/(name+'.log')).read_text()[-2000:]
def resource(phase,heavy=False):
    from e015_serial_recovery import resources
    # Recovery helper sets its own D: caches on import. Restore this assessment's
    # dedicated cache paths before invoking any external tool.
    for k in ('TMPDIR','TMP','TEMP','XDG_CACHE_HOME','MPLCONFIGDIR','TORCH_HOME','HF_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','PIP_CACHE_DIR','JAX_COMPILATION_CACHE_DIR','COLABFOLD_CACHE_DIR'):
        os.environ[k]=str(OUT/'cache'/k.lower())
    x=resources();x.update(phase=phase,controller_pid=os.getpid())
    if shutil.which('powershell.exe'):
        try:
            result=subprocess.check_output(['powershell.exe','-NoProfile','-Command','(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory'],text=True,timeout=20)
            x['windows_ram_available_bytes']=int(result.strip())*1024
        except Exception as exc:
            x['windows_ram_measurement_error']=str(exc)
    with (OUT/'resource_monitor.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
    if heavy:
        if shutil.which('powershell.exe'):
            assert x.get('windows_ram_available_bytes',0)>=3*1024**3,'Native Windows RAM below3GiB or unavailable'
        assert x['ram_available_bytes']>=3*1024**3,'RAM below 3GiB'
        assert x['d_free_bytes']>=30*1024**3,'D reserve below 30GiB'
        assert not x['gpu_compute_apps'],'Existing CUDA owner'
        assert not [r for r in x['python_processes'] if any(s in ' '.join(r['command']) for s in ('run_e015_integrated.py','supervise_e015_training.py','run_e007','supervise_e007','evaluate_e007','colabfold_batch','protein_mpnn_run.py'))],'Existing heavy owner'
        assert not [r for r in x['python_processes'] if r['pid']!=os.getpid() and 'e015_external_designability.py' in ' '.join(r['command'])],'Another assessment controller is active'
        assert not list((ROOT/'production/checkpoints').glob('*.partial')),'Checkpoint write active'
    return x
def download(url,p):
    p=Path(p)
    if p.exists(): return
    p.parent.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(url,timeout=60) as r, p.with_suffix(p.suffix+'.partial').open('wb') as f:
        size=r.headers.get('Content-Length')
        if size: assert shutil.disk_usage(OUT).free-int(size)>=30*1024**3
        while b:=r.read(1024**2):
            assert shutil.disk_usage(OUT).free>=30*1024**3
            f.write(b)
        f.flush();os.fsync(f.fileno())
    p.with_suffix(p.suffix+'.partial').replace(p)
def init():
    import numpy as np
    import pyarrow.parquet as pq
    for name in ('tools','sanitized_backbones','proteinmpnn','sequences','colabfold','alignments','foldseek','candidates','known_fold_like_gallery','potentially_novel_gallery','likely_artifact_gallery','positive_control_gallery','same_seed_50_vs_latest_gallery','logs','cache'):
        (OUT/name).mkdir(parents=True,exist_ok=True)
    for k in ('TMPDIR','XDG_CACHE_HOME','MPLCONFIGDIR','TORCH_HOME','HF_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','PIP_CACHE_DIR','JAX_COMPILATION_CACHE_DIR','COLABFOLD_CACHE_DIR'):Path(os.environ[k]).mkdir(parents=True,exist_ok=True)
    if (OUT/'backbone_panel.json').exists():
        print('Existing frozen panel retained');return
    candidates=[]
    for e in (ROOT/'evaluations').iterdir():
        if not e.is_dir() or not all((e/f).exists() for f in ('REPORT.md','MANIFEST.json','generation_manifest.parquet','generation_scores.parquet')): continue
        m=read(e/'MANIFEST.json')
        if m.get('status')!='COMPLETE':continue
        rows=pq.read_table(e/'generation_manifest.parquet',use_threads=False).to_pylist()
        if not all(all((Path(r['path'])/f).exists() for f in ('final_coordinates.npz','sampler_configuration.json')) for r in rows): continue
        candidates.append((m['primary']['source']['position']['update'],e,m,rows))
    _,e,m,rows=max(candidates,key=lambda z:z[0]); scores=pq.read_table(e/'generation_scores.parquet',use_threads=False).to_pylist()
    bykey={r['sample_key']:r for r in rows}; sm={r['sample_key']:r['score'] for r in scores}
    raw=sorted([r for r in rows if r['weights']=='raw'],key=lambda r:(sm[r['sample_key']],r['sample_key']))
    chosen=[min([r for r in raw if r['requested_length']==n],key=lambda r:sm[r['sample_key']]) for n in (48,96,192,320,448)]
    panel=[]
    for category,rs in [('latest_best',chosen),('historical_same_seed',chosen),('latest_median',[raw[len(raw)//2]]),('latest_failure',[raw[-1]])]:
        for r in rs:
            d=Path(r['historical_raw_path'] if category=='historical_same_seed' else r['path'])
            a=np.load(d/'final_coordinates.npz'); xyz=a['coordinates_angstrom']; assert len(xyz)==r['requested_length']
            metrics=read(d/'metrics.json') if (d/'metrics.json').exists() else {}
            panel.append(dict(backbone_id=category+'_'+r['sample_key'],source_category=category,length=len(xyz),seed=r['seed'],checkpoint_sha256=('2986d44ed22aa1c2fd0dea2fe57fa1b50c0808893fd3608142240841e08cbc03' if category=='historical_same_seed' else r['checkpoint_sha256']),source_coordinates=str(d/'final_coordinates.npz'),source_pdb=str(d/'trace_ca_only.pdb'),numerical_score=(sm[r['sample_key']] if category!='historical_same_seed' else None),original_metrics=metrics,sampler=read(Path(r['path'])/'sampler_configuration.json'),same_x499_verified=r['same_saved_x499']))
    pop=Path('/mnt/d/Simone/proteinGen/e014_artifacts')
    dev=pq.read_table(pop/'population/development.parquet',use_threads=False).to_pylist()
    identities=read(pop/'coordinate_cache/identities.json'); idx={v:i for i,v in enumerate(identities)}
    offsets=np.load(pop/'coordinate_cache/offsets.npy',mmap_mode='r'); lengths=np.load(pop/'coordinate_cache/lengths.npy',mmap_mode='r')
    coords=np.memmap(pop/'coordinate_cache/coordinates.f32',mode='r',dtype='float32').reshape(-1,3)
    for n,lo,hi in [(48,20,64),(96,65,128),(192,129,256),(320,257,384),(448,385,500)]:
        rr=min([r for r in dev if lo<=r['length']<=hi],key=lambda r:(abs(r['length']-n),r['sample_id']))
        i=idx[rr['sample_id']]; start=int(offsets[i]); size=int(lengths[i]); assert size==rr['length']
        p=OUT/'sanitized_backbones'/('real_'+rr['sample_id']+'.npz');np.savez_compressed(p,coordinates_angstrom=np.array(coords[start:start+size]))
        panel.append(dict(backbone_id='real_'+rr['sample_id'],source_category='real_heldout',length=size,matched_length=n,seed=None,checkpoint_sha256=None,source_coordinates=str(p),source_pdb=None,numerical_score=None,original_metrics={},reference_identity=rr['sample_id'],coordinate_source=str(pop/'coordinate_cache/coordinates.f32'),coordinate_row_range=[start,start+size],control_selection='Frozen held-out split; nearest length within stratum; lexicographic tie-break; no external-performance selection'))
    assert len(panel)==17 and len({r['backbone_id'] for r in panel})==17
    sanitized=[]
    for r in panel:
        xyz=np.load(r['source_coordinates'])['coordinates_angstrom'];assert xyz.shape==(r['length'],3) and np.isfinite(xyz).all()
        p=OUT/'sanitized_backbones'/(r['backbone_id']+'.pdb')
        with p.open('w') as f:
            f.write('REMARK C-ALPHA ONLY; ALA PLACEHOLDER IS NOT A DESIGNED SEQUENCE\nREMARK NO SIDE CHAINS; NOT AN ALL-ATOM STRUCTURE OR EXPERIMENTAL VALIDATION\n')
            for j,(x,y,z) in enumerate(xyz,1):f.write(f'ATOM  {j:5d}  CA  ALA A{j:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n')
            f.write('TER\nEND\n')
        parsed=np.array([[float(line[a:b]) for a,b in [(30,38),(38,46),(46,54)]] for line in p.read_text().splitlines() if line.startswith('ATOM')])
        assert np.max(np.abs(parsed-xyz))<=0.000501
        r['sanitized_pdb']=str(p);r['backbone_sha256']=sha(p)
        sanitized.append(dict(backbone_id=r['backbone_id'],input=r['source_coordinates'],input_sha256=sha(r['source_coordinates']),output=str(p),output_sha256=r['backbone_sha256'],length=len(xyz),max_rounding_error_angstrom=float(np.max(np.abs(parsed-xyz))),geometry_repaired=False))
    write(OUT/'backbone_panel.json',dict(status='FROZEN',evaluation=str(e),checkpoint=m['primary']['source'],quality_contract=read(e/'numerical/quality_score_contract.json'),backbones=panel,screening_cap=82,count_conflict='Requested per-category allocation sums to84; clarification requested. No refolds dispatched until allocation resolved.',paired_comparison='Matched 4-design budget (2 lowest-score+2 diverse) primary paired endpoint; additional4 latest designs reported separately.'))
    write(OUT/'sanitized_backbone_manifest.json',sanitized)
    cols=['backbone_id','source_category','length','seed','checkpoint_sha256','numerical_score','sanitized_pdb']
    (OUT/'backbone_panel.tsv').write_text('\t'.join(cols)+'\n'+'\n'.join('\t'.join(str(r.get(k,'')) for k in cols) for r in panel)+'\n')
    progress('PANEL_FROZEN',backbones=17,predictions_completed=0)
    resource('lightweight_panel_preparation')
    write(OUT/'MANIFEST.json',dict(status='IN_PROGRESS',panel_sha256=sha(OUT/'backbone_panel.json'),all_artifacts_on_d=True,scientific_claims='Computational designability only; no experimental stability, solubility or function claim'))
    print(json.dumps(dict(panel=str(OUT/'backbone_panel.json'),backbones=17,evaluation=str(e))))

def audit_tools():
    tools={}
    for exe in ('colabfold_batch','foldseek','TMalign','USalign','mmseqs','mkdssp'):
        p=shutil.which(exe); tools[exe]=dict(path=p)
        if p:
            q=subprocess.run([p,'--help'],capture_output=True,text=True,timeout=30)
            (OUT/'logs'/(exe+'_help.txt')).write_text(q.stdout+q.stderr)
            tools[exe]['executable_sha256']=sha(p)
    commit='8907e6671bfbfc92303b5f79c4b5e6ce47cdef57';d=OUT/'tools'/'ProteinMPNN'
    for f in ('protein_mpnn_run.py','protein_mpnn_utils.py','LICENSE','README.md','ca_model_weights/v_48_020.pt'):
        download('https://raw.githubusercontent.com/dauparas/ProteinMPNN/'+commit+'/'+f,d/f)
    tools['ProteinMPNN']=dict(repository='https://github.com/dauparas/ProteinMPNN',commit=commit,model='CA-only v_48_020',weight_sha256=sha(d/'ca_model_weights/v_48_020.pt'),license='MIT',python=PY,environment_policy='Existing proteingen interpreter used read-only, without installing or modifying any packages; all assessment tools and caches on D.',files={str(p.relative_to(d)):sha(p) for p in d.rglob('*') if p.is_file()})
    # Record upstream versions/dependencies before environment admission.
    for repo,filename in [('sokrypton/ColabFold','pyproject.toml'),('pylelab/USalign','README')]:
        url='https://api.github.com/repos/'+repo+'/commits?per_page=1'
        with urllib.request.urlopen(url,timeout=30) as r:j=json.load(r)
        commit2=j[0]['sha']; name=repo.split('/')[-1]
        try: download('https://raw.githubusercontent.com/'+repo+'/'+commit2+'/'+filename,OUT/'tools'/name/filename)
        except urllib.error.HTTPError as exc:
            if exc.code!=404: raise
        tools[name]=dict(repository='https://github.com/'+repo,commit=commit2,status='AUDITED_NOT_INSTALLED')
    write(OUT/'toolchain_manifest.json',tools)
    (OUT/'TOOLCHAIN.md').write_text('# External toolchain audit\n\n'+json.dumps(tools,indent=2)+'\n\nNo GPU inference performed during the audit. ColabFold installation and folding require serialized admission after E015 graceful pause. Foldseek local PDB download is not admitted without a documented size and 30 GiB reserve.\n')
    progress('TOOLCHAIN_AUDITED')
    print(json.dumps(tools))

def pause():
    import e015_serial_recovery as recovery
    before=resource('before_graceful_pause')
    assert before['ram_available_bytes']>=3*1024**3 and before['d_free_bytes']>=30*1024**3
    sup=read(ROOT/'SUPERVISOR.json'); state=read(ROOT/'STATE.json')
    if state['phase']!='training_production':
        resource('already_stopped',heavy=True);return
    assert sup['child_pid']==state['pid']
    # Genuine initial audit observation, before the next rolling boundary at11687.
    report=OUT/'logs'/'production_pause'
    if not (report/'launch.json').exists():
        write(report/'launch.json',dict(supervisor_pid=sup['pid'],timestamp_unix=sup['timestamp_unix'],source='Existing SUPERVISOR.json actual launch; not a synthetic launch'))
        write(report/'recovery_audit.json',dict(selected=dict(position=dict(update=11614,exposures=352070)),source='Initial read-only audit observation; used solely as next-boundary baseline, not as a verified checkpoint'))
    assert read(ROOT/'COMMITTED.json')==read(ROOT/'production/checkpoints/LATEST.json')
    recovery.pause(report)
    progress('WAITING_FOR_GRACEFUL_EXIT',supervisor_pid=sup['pid'],trainer_pid=state['pid'])
    deadline=time.monotonic()+600
    while time.monotonic()<deadline:
        alive=[]
        for pid in (sup['pid'],state['pid']):
            try:
                if Path(f'/proc/{pid}/cmdline').read_bytes():alive.append(pid)
            except FileNotFoundError:pass
        if not alive:break
        time.sleep(5)
    assert not alive,'Graceful exit pending; no external work admitted'
    resource('after_graceful_exit',heavy=True)
    audit=recovery.audit(only_latest=True)
    proof=read(audit/'recovery_audit.json');cp=proof['selected'];s=read(ROOT/'STATE.json')
    assert s['exposures']==cp['position']['exposures'] and s['update']==cp['position']['update']
    write(OUT/'production_pause_proof.json',dict(audit=str(audit),checkpoint=cp,zero_uncommitted_exposures=True,trainer_exited=True,supervisor_exited=True,next_updates=proof['next_updates'],resume_command=proof['command']))
    progress('PRODUCTION_PAUSED_VERIFIED')
    print(json.dumps(dict(pause_checkpoint=cp,audit=str(audit))))

def install():
    resource('environment_installation',heavy=True)
    tc=read(OUT/'toolchain_manifest.json');d=OUT/'tools'/'ColabFold'
    commit=tc['ColabFold']['commit']
    if not (d/'source').exists():
        run(['git','clone','https://github.com/sokrypton/ColabFold',str(d/'source')],'colabfold_clone',300)
        run(['git','-C',str(d/'source'),'checkout','--detach',commit],'colabfold_pin',30)
    env=OUT/'tools'/'colabfold_env'
    if not (env/'bin/python').exists():run([PY,'-m','venv','--copies',str(env)],'colabfold_venv',120)
    ep=str(env/'bin/python')
    spec=str(d/'source')+'[alphafold]'
    cmd=[ep,'-m','pip','install','--dry-run','--report',str(OUT/'logs'/'colabfold_pip_plan.json'),spec,'jax[cuda12]==0.6.2']
    run(cmd,'colabfold_dependencies_preflight',1800)
    plan=read(OUT/'logs'/'colabfold_pip_plan.json')
    # Conservative admission: extracted packages, wheels/cache/build overhead,
    # model parameters and transient download tar each counted separately.
    already=sum(p.stat().st_size for folder in [OUT/'tools',OUT/'cache/pip_cache_dir'] for p in folder.rglob('*') if p.is_file())
    budget=dict(environment_estimate_bytes=8*1024**3,already_installed_or_cached_bytes=already,model_weights_final_bytes=4*1024**3,model_download_peak_extra_bytes=128*1024**2,model_download_protocol='Official ColabFold download.py streams the tar directly into extraction; no local tar archive',outputs_and_msa_budget_bytes=1024**3,pdb_local_database_bytes=None,reserve_bytes=30*1024**3,current_free_bytes=shutil.disk_usage(OUT).free)
    write(OUT/'disk_admission.json',budget)
    remaining=max(0,8*1024**3-already)+budget['model_weights_final_bytes']+budget['model_download_peak_extra_bytes']+budget['outputs_and_msa_budget_bytes']
    assert budget['current_free_bytes']-remaining>=30*1024**3,'Environment and model peak budget exceeds reserve'
    run([ep,'-m','pip','install',spec,'jax[cuda12]==0.6.2'],'colabfold_install',2400)
    run([ep,'-m','pip','freeze'],'colabfold_freeze',60)
    helptext=run([str(env/'bin/colabfold_batch'),'--help'],'colabfold_help',120)
    tc['ColabFold'].update(status='INSTALLED',environment=str(env),version='1.6.3',jax='0.6.2',dependency_lock=str(OUT/'logs/colabfold_freeze.log'),license='MIT; AlphaFold parameters have separate upstream license')
    write(OUT/'toolchain_manifest.json',tc)
    # Install US-align from the immutable official source, one compiler process.
    u=OUT/'tools'/'USalign';c=tc['USalign']['commit']
    if not (u/'source').exists():
        run(['git','clone','https://github.com/pylelab/USalign',str(u/'source')],'usalign_clone',300)
        run(['git','-C',str(u/'source'),'checkout','--detach',c],'usalign_pin',30)
    run(['g++','-O3','-o',str(u/'USalign'),str(u/'source/USalign.cpp')],'usalign_build',300)
    run([str(u/'USalign'),'-h'],'usalign_help',30)
    tc['USalign'].update(status='INSTALLED',executable=str(u/'USalign'),executable_sha256=sha(u/'USalign'))
    write(OUT/'toolchain_manifest.json',tc);progress('TOOLS_INSTALLED')
    print(helptext)

def mpnn():
    import numpy as np
    d=OUT/'tools/ProteinMPNN'; panel=read(OUT/'backbone_panel.json')['backbones']
    resource('proteinmpnn_help',heavy=True)
    run([PY,'-B',str(d/'protein_mpnn_run.py'),'--help'],'proteinmpnn_help',60)
    jobs=read(OUT/'proteinmpnn/jobs.json') if (OUT/'proteinmpnn/jobs.json').exists() else {}
    for i,b in enumerate(panel):
        for j,temp in enumerate(['0.10','0.20']):
            key=b['backbone_id']+'_t'+temp; od=OUT/'proteinmpnn'/key
            seed=(151500+2*[48,96,192,320,448].index(b['length'])+j if b['source_category'] in ('latest_best','historical_same_seed') else 151600+i*2+j)
            command=[PY,'-B',str(d/'protein_mpnn_run.py'),'--ca_only','--model_name','v_48_020','--path_to_model_weights',str(d/'ca_model_weights'),'--pdb_path',b['sanitized_pdb'],'--pdb_path_chains','A','--out_folder',str(od),'--num_seq_per_target','16','--batch_size','1','--sampling_temp',temp,'--seed',str(seed),'--save_score','1','--save_probs','1']
            fingerprint=hashlib.sha256(json.dumps(dict(command=command,backbone=b['backbone_sha256'],weight=sha(d/'ca_model_weights/v_48_020.pt')),sort_keys=True).encode()).hexdigest()
            old=jobs.get(key,{})
            if old.get('status')=='COMPLETE' and old.get('fingerprint')==fingerprint and all(Path(p).exists() and sha(p)==h for p,h in old['files'].items()):continue
            r=resource('proteinmpnn_'+key,heavy=True);progress('PROTEINMPNN',current_backbone=b['backbone_id'],temperature=temp)
            start=time.monotonic()
            try:
                run(command,key,1200)
                fasta=od/'seqs'/(b['backbone_id']+'.fa')
                lines=fasta.read_text().splitlines();assert sum(l.startswith('>T=') for l in lines)==16
                files={str(p):sha(p) for p in od.rglob('*') if p.is_file()}
                sync_outputs(files)
                jobs[key]=dict(status='COMPLETE',fingerprint=fingerprint,command=command,files=files,runtime_seconds=time.monotonic()-start,resources_before=r)
            except Exception as exc:
                jobs[key]=dict(status='FAILED',fingerprint=fingerprint,error=str(exc),command=command)
            write(OUT/'proteinmpnn/jobs.json',jobs)
            print(json.dumps(dict(job=key,status=jobs[key]['status'])),flush=True)
    progress('PROTEINMPNN_COMPLETE',mpnn_jobs_complete=sum(v['status']=='COMPLETE' for v in jobs.values()))

def sequences():
    import re,collections,itertools
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    resource('sequence_bookkeeping',heavy=True)
    panel=read(OUT/'backbone_panel.json')['backbones'];rows=[];selected=[];diversity={}
    def ident(a,b):return sum(x==y for x,y in zip(a,b))/len(a)
    for b in panel:
        br=[]
        for temp in ('0.10','0.20'):
            d=OUT/'proteinmpnn'/(b['backbone_id']+'_t'+temp)
            fp=d/'seqs'/(b['backbone_id']+'.fa')
            if not fp.exists():continue
            lines=fp.read_text().splitlines();prpath=d/'probs'/(b['backbone_id']+'.npz')
            probs=np.load(prpath)['probs'] if prpath.exists() else None
            scorepath=d/'scores'/(b['backbone_id']+'.npz')
            precise=np.load(scorepath) if scorepath.exists() else None
            designs=[(lines[i],lines[i+1]) for i in range(0,len(lines),2) if lines[i].startswith('>T=')]
            for j,(h,s) in enumerate(designs):
                assert len(s)==b['length'] and set(s)<=set('ACDEFGHIKLMNPQRSTVWY')
                counts=collections.Counter(s); freqs={a:counts[a]/len(s) for a in 'ACDEFGHIKLMNPQRSTVWY'}
                ent=float(-(probs[j]*np.log(np.maximum(probs[j],1e-30))).sum(-1).mean()) if probs is not None else None
                digest=hashlib.sha256(s.encode()).hexdigest()
                r=dict(backbone_id=b['backbone_id'],source_category=b['source_category'],sequence_id=b['backbone_id']+'_t'+temp+'_s'+str(j+1),sequence=s,sequence_sha256=digest,length=len(s),temperature=float(temp),score=float(re.search(r'(?:^|, )score=([\d.e+-]+)',h).group(1)),global_score=float(re.search(r'global_score=([\d.e+-]+)',h).group(1)),mean_position_entropy=ent,hydrophobic_fraction=sum(freqs[a] for a in 'AVILMFWY'),charged_fraction=sum(freqs[a] for a in 'DEKR'),glycine_fraction=freqs['G'],proline_fraction=freqs['P'],cysteine_fraction=freqs['C'],longest_homopolymer=max(len(list(g)) for _,g in itertools.groupby(s)),composition_entropy=float(-sum(v*math.log(v) for v in freqs.values() if v)),aa_frequencies=json.dumps(freqs),backbone_sha256=b['backbone_sha256'],probability_file=str(prpath),probability_row=j)
                br.append(r)
                if precise is not None:
                    r['score']=float(precise['score'][j]);r['global_score']=float(precise['global_score'][j])
        unique=[];seen=set()
        for r in sorted(br,key=lambda r:(r['score'],r['sequence_sha256'])):
            if r['sequence_sha256'] not in seen: unique.append(r);seen.add(r['sequence_sha256'])
        leaders=[]
        for r in unique:
            cluster=next((i for i,v in enumerate(leaders) if ident(r['sequence'],v['sequence'])>=.8),None)
            if cluster is None:cluster=len(leaders);leaders.append(r)
            r['identity80_cluster']=cluster
        annotation={r['sequence_sha256']:r['identity80_cluster'] for r in unique}
        for r in br:r['identity80_cluster']=annotation[r['sequence_sha256']]
        top=4 if b['source_category']=='latest_best' else 2
        picks=[dict(r,selection_reason='lowest_score') for r in unique[:top]]
        used={r['sequence_sha256'] for r in picks}
        for _ in range(top):
            used_clusters={r['identity80_cluster'] for r in picks}
            pool=[r for r in unique if r['sequence_sha256'] not in used and r['identity80_cluster'] not in used_clusters]
            fallback=False
            if not pool:pool=[r for r in unique if r['sequence_sha256'] not in used];fallback=True
            if not pool:break
            rr=max(pool,key=lambda r:(min(1-ident(r['sequence'],v['sequence']) for v in picks),-r['score'],r['sequence_sha256']))
            picks.append(dict(rr,selection_reason='diverse_within_cluster_fallback' if fallback else 'diverse_cluster'));used.add(rr['sequence_sha256'])
        pairs=[ident(x['sequence'],y['sequence']) for i,x in enumerate(unique) for y in unique[i+1:]]
        diversity[b['backbone_id']]=dict(raw_sequences=len(br),unique_sequences=len(unique),identity80_clusters=len(leaders),selected=len(picks),pairwise_identity_mean=float(np.mean(pairs)) if pairs else None,pairwise_identity_min=min(pairs) if pairs else None,distinct_selected_clusters=len({r['identity80_cluster'] for r in picks}),selection_shortfall=max(0,2*top-len(picks)),clustering='Deterministic score-ordered greedy leader clustering at80% index-matched sequence identity; no external outcome selection')
        # Four-design matched endpoint: two lowest score + first two diversity
        # selections; latest additional four reported as expanded-budget yield.
        matched=set(r['sequence_id'] for r in picks[:2]+picks[top:top+2])
        for r in picks:r['paired_four_design_endpoint']=r['sequence_id'] in matched
        rows.extend(br);selected.extend(picks)
    for name,data in [('proteinmpnn_all_sequences.parquet',rows),('proteinmpnn_selected_sequences.parquet',selected)]:
        pq.write_table(pa.Table.from_pylist(data),OUT/name)
    write(OUT/'sequence_diversity.json',diversity)
    fasta=''.join('>'+r['sequence_id']+'\n'+r['sequence']+'\n' for r in selected)
    (OUT/'sequences/selected_sequences.fasta').write_text(fasta);(OUT/'selected_sequences.fasta').write_text(fasta)
    progress('SEQUENCES_SELECTED',raw_sequences=len(rows),unique_selected=len(selected))
    print(json.dumps(dict(raw_sequences=len(rows),selected=len(selected),diversity=diversity)))

def folding(screen=False):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    selected=pq.read_table(OUT/'proteinmpnn_selected_sequences.parquet',use_threads=False).to_pylist()
    panels={b['backbone_id']:b for b in read(OUT/'backbone_panel.json')['backbones']}
    jobs=read(OUT/'colabfold/jobs.json') if (OUT/'colabfold/jobs.json').exists() else {}
    env=OUT/'tools/colabfold_env'; exe=str(env/'bin/colabfold_batch')
    helptext=(OUT/'logs/colabfold_help.log').read_text()
    for flag in ('--msa-mode','--model-type','--num-models','--num-recycle','--model-order','--data','--max-msa'):
        assert flag in helptext,flag+' unsupported'
    if screen:
        # Honor the hard82 cap: primary40, historical20, real20, median1,failure1.
        queue=[]
        for cat in ('real_heldout','latest_best','historical_same_seed','latest_median','latest_failure'):
            for bid,b in panels.items():
                if b['source_category']!=cat:continue
                rs=[r for r in selected if r['backbone_id']==bid]
                limit=8 if cat=='latest_best' else (1 if cat in ('latest_median','latest_failure') else 4)
                queue.extend(rs[:limit])
        assert len(queue)<=82
        write(OUT/'colabfold/screening_protocol.json',dict(maximum_sequences=82,actual_selected=len(queue),allocation=dict(latest_best=40,historical_same_seed=20,real_heldout=20,latest_median=1,latest_failure=1),count_conflict_resolution='Hard82 cap takes precedence; median/failure screened once each; their second selected design remains available for later authorized expansion.',primary_pair_endpoint='2 lowest-score+2 diverse on both sides; latest additional4 are separately reported',model_type='alphafold2_ptm',models=1,model_order=[1],recycles=3,msa_mode='single_sequence',templates=False,amber=False,msas='No MSA server requests during screening'))
    else:
        queue=[min([r for r in selected if r['source_category']=='latest_best' and r['length']==n],key=lambda r:r['score']) for n in (48,192,448)]
    for r in queue:
        seqid=r['sequence_id'];bd=panels[r['backbone_id']];od=OUT/'colabfold/screening'/seqid
        fa=OUT/'sequences'/(seqid+'.fasta');fa.write_text('>'+seqid+'\n'+r['sequence']+'\n')
        assert '--disable-unified-memory' in helptext
        cmd=[exe,str(fa),str(od),'--msa-mode','single_sequence','--model-type','alphafold2_ptm','--num-models','1','--model-order','1','--num-recycle','3','--recycle-early-stop-tolerance','0.0','--num-seeds','1','--random-seed','151502','--max-msa','1:1','--recompile-padding','0','--disable-unified-memory','--data',str(OUT/'tools/alphafold_weights')]
        fingerprint=hashlib.sha256(json.dumps(dict(sequence=r['sequence_sha256'],target=bd['backbone_sha256'],command=cmd,tool_commit=read(OUT/'toolchain_manifest.json')['ColabFold']['commit']),sort_keys=True).encode()).hexdigest()
        job=jobs.get(seqid,{})
        if job.get('status')=='COMPLETE' and job.get('fingerprint')==fingerprint and all(Path(p).exists() and sha(p)==h for p,h in job['files'].items()):continue
        before=resource('colabfold_'+seqid,heavy=True);progress('COLABFOLD_SCREEN' if screen else 'COLABFOLD_PREFLIGHT',current_sequence=seqid)
        t=time.monotonic();od.mkdir(parents=True,exist_ok=True)
        jobs[seqid]=dict(status='RUNNING',fingerprint=fingerprint,sequence=r,target_backbone=bd['sanitized_pdb'],command=cmd,runtime_seconds=0)
        write(OUT/'colabfold/jobs.json',jobs)
        log=OUT/'logs'/(seqid+'_colabfold.log'); peak_ram=0;peak_gpu=0;failure=None
        with log.open('w') as f:
            # Bound JAX compilation/CPU pools to two cores for external folding.
            # Production resume never inherits this child-only affinity.
            cpus=sorted(os.sched_getaffinity(0))[:2]
            p=subprocess.Popen(cmd,cwd=WORK,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT,preexec_fn=lambda:os.sched_setaffinity(0,cpus))
            last=0
            while p.poll() is None:
                if time.monotonic()-last>=30:
                    x=resource('colabfold_'+seqid);x['evaluator_pid']=p.pid
                    peak_ram=max(peak_ram,before['ram_available_bytes']-x['ram_available_bytes']);peak_gpu=max(peak_gpu,int(x['gpu_csv'].split(',')[0]))
                    owners=[int(row.split(',')[0]) for row in x['gpu_compute_apps'].splitlines() if row.strip()]
                    if x['ram_available_bytes']<3*1024**3 or (shutil.which('powershell.exe') and x.get('windows_ram_available_bytes',0)<3*1024**3) or x['d_free_bytes']<30*1024**3 or any(pid!=p.pid for pid in owners):
                        # SIGINT is standard external Python evaluator cancellation;
                        # never sent to production, which has already exited.
                        import signal
                        write(OUT/'STOP_EXTERNAL.json',dict(reason='Resource threshold',resources=x,sequence=seqid))
                        p.send_signal(signal.SIGINT);failure='RESOURCE_PRESSURE';break
                    last=time.monotonic()
                time.sleep(2)
            rc=p.wait()
        try:
            assert rc==0,(rc,log.read_text()[-2000:])
            pdbs=list(od.glob('*unrelaxed_rank_001*.pdb'));scores=list(od.glob('*scores_rank_001*.json'))
            assert len(pdbs)==len(scores)==1,'Missing unique rank1 output'
            pred=np.array([[float(line[a:b]) for a,b in [(30,38),(38,46),(46,54)]] for line in pdbs[0].read_text().splitlines() if line.startswith('ATOM') and line[12:16].strip()=='CA'])
            assert pred.shape==(r['length'],3) and np.isfinite(pred).all(),'Incomplete/nonfinite prediction'
            sc=read(scores[0]);jobs[seqid]=dict(status='COMPLETE',fingerprint=fingerprint,sequence=r,target_backbone=bd['sanitized_pdb'],pdb=str(pdbs[0]),score_json=str(scores[0]),mean_plddt=float(np.mean(sc['plddt'])),ptm=float(sc.get('ptm',0)),runtime_seconds=time.monotonic()-t,peak_gpu_used_mib=peak_gpu,system_ram_delta_peak_bytes=peak_ram,command=cmd,files={str(q):sha(q) for q in od.rglob('*') if q.is_file()})
        except Exception as exc:jobs[seqid]=dict(status='FAILED',fingerprint=fingerprint,sequence=r,error=str(exc),failure=failure,command=cmd,runtime_seconds=time.monotonic()-t,peak_gpu_used_mib=peak_gpu,system_ram_delta_peak_bytes=peak_ram)
        if jobs[seqid]['status']=='COMPLETE':sync_outputs(jobs[seqid]['files'])
        write(OUT/'colabfold/jobs.json',jobs)
        compact=[dict(sequence_id=k,backbone_id=v['sequence']['backbone_id'],length=v['sequence']['length'],status=v['status'],mean_plddt=v.get('mean_plddt'),ptm=v.get('ptm'),pdb=v.get('pdb'),runtime_seconds=v['runtime_seconds'],error=v.get('error')) for k,v in jobs.items()]
        pq.write_table(pa.Table.from_pylist(compact),OUT/'colabfold_screening.parquet')
        print(json.dumps(dict(sequence=seqid,status=jobs[seqid]['status'],seconds=jobs[seqid]['runtime_seconds'],peak_gpu_used_mib=peak_gpu)),flush=True)
        if not screen and jobs[seqid]['status']!='COMPLETE':
            progress('FOLDING_PREFLIGHT_FAILED',failed_sequence=seqid);return
        if failure:progress('STOPPED_RESOURCE_PRESSURE');return
    progress('SCREENING_COMPLETE' if screen else 'FOLDING_PREFLIGHT_COMPLETE',completed_refolds=sum(j['status']=='COMPLETE' for j in jobs.values()))

def align():
    import re
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    resource('structural_alignment',heavy=True)
    jobs=read(OUT/'colabfold/jobs.json');panel=read(OUT/'backbone_panel.json')['backbones']
    exe=str(OUT/'tools/USalign/USalign');rows=[]
    def ca(p):return np.array([[float(l[a:b]) for a,b in [(30,38),(38,46),(46,54)]] for l in Path(p).read_text().splitlines() if l.startswith('ATOM') and l[12:16].strip()=='CA'])
    for sid,j in jobs.items():
        if j['status']!='COMPLETE':continue
        run([exe,j['target_backbone'],j['pdb'],'-mol','prot'],sid+'_align',60)
        text=(OUT/'logs'/(sid+'_align.log')).read_text()
        scores=re.findall(r'TM-score=\s*([0-9.]+).*normalized by length of (?:Structure|Chain)_([12])',text)
        assert len(scores)>=2, text
        ts={int(k):float(v) for v,k in scores};m=re.search(r'Aligned length=\s*(\d+),\s*RMSD=\s*([0-9.]+)',text);assert m,text
        n=int(m[1]);rmsd=float(m[2]);length=j['sequence']['length'];cov=n/length
        x,y=ca(j['target_backbone']),ca(j['pdb']);assert x.shape==y.shape
        x=x-x.mean(0);y=y-y.mean(0);u,s,vt=np.linalg.svd(x.T@y);rot=u@np.diag([1,1,np.linalg.det(u@vt)])@vt
        kr=float(np.sqrt(np.mean(np.sum((x@rot-y)**2,axis=-1))))
        strong=ts[1]>=.7 and cov>=.8 and rmsd<=3
        if length<80:strong=strong and cov>=.9 and rmsd<=2.5
        very=ts[1]>=.8 and cov>=.9 and rmsd<=2.5
        rows.append(dict(sequence_id=sid,backbone_id=j['sequence']['backbone_id'],source_category=j['sequence']['source_category'],length=length,sequence_sha256=j['sequence']['sequence_sha256'],score=j['sequence']['score'],identity80_cluster=j['sequence']['identity80_cluster'],selection_reason=j['sequence']['selection_reason'],paired_four_design_endpoint=j['sequence']['paired_four_design_endpoint'],target_tm_score=ts[1],prediction_tm_score=ts[2],aligned_ca_rmsd_angstrom=rmsd,index_matched_kabsch_rmsd_angstrom=kr,aligned_residues=n,target_coverage=cov,prediction_coverage=cov,mean_plddt=j['mean_plddt'],ptm=j['ptm'],strong_recovery=strong,very_strong_recovery=very,confidence_support=j['mean_plddt']>=70 and j['ptm']>=.6,tm_short_chain_sensitive=length<80,pdb=j['pdb'],score_json=j['score_json']))
    pq.write_table(pa.Table.from_pylist(rows),OUT/'structural_alignment_results.parquet')
    summary=[]
    for b in panel:
        br=[r for r in rows if r['backbone_id']==b['backbone_id']];scheduled=[v for v in jobs.values() if v['sequence']['backbone_id']==b['backbone_id']]
        ns=len(scheduled);valid=len(br);sc=sum(r['strong_recovery'] for r in br);vs=sum(r['very_strong_recovery'] for r in br)
        best=max((r['target_tm_score'] for r in br),default=None);clusters=len({r['identity80_cluster'] for r in br if r['strong_recovery']});d7=sc/ns if ns else None
        if not ns or valid!=ns:cls='INCONCLUSIVE'
        elif d7>=.5:cls='STRONGLY_DESIGNABLE'
        elif d7>=.25 or (best>=.8 and clusters>=2):cls='PROMISINGLY_DESIGNABLE'
        elif d7>0:cls='WEAK_EVIDENCE'
        elif best<.7:cls='NONDESIGNABLE_IN_THIS_TEST'
        else:cls='INCONCLUSIVE' # TM alone cannot bypass coverage/RMSD gates.
        summary.append(dict(backbone_id=b['backbone_id'],source_category=b['source_category'],length=b['length'],selected_denominator=ns,valid_predictions=valid,strong_count=sc,very_strong_count=vs,D_0_7=d7,D_0_8=vs/ns if ns else None,best_target_tm_score=best,median_target_tm_score=float(np.median([r['target_tm_score'] for r in br])) if br else None,mean_target_tm_score=float(np.mean([r['target_tm_score'] for r in br])) if br else None,best_rmsd_angstrom=min((r['aligned_ca_rmsd_angstrom'] for r in br),default=None),median_rmsd_angstrom=float(np.median([r['aligned_ca_rmsd_angstrom'] for r in br])) if br else None,successful_clusters=clusters,classification=cls))
    real=[r for r in summary if r['source_category']=='real_heldout'];passcount=sum(r['best_target_tm_score'] is not None and r['best_target_tm_score']>=.7 for r in real)
    controls=dict(status='VALIDATED' if passcount>=4 else 'TOOLCHAIN_NOT_VALIDATED',controls=real,best_tm_ge_07_count=passcount,D07_ge_025_count=sum(r['D_0_7'] is not None and r['D_0_7']>=.25 for r in real),required=4,novelty_claims_allowed=passcount>=4)
    write(OUT/'controls_summary.json',controls)
    if passcount<4:
        for r in summary:
            if r['source_category']!='real_heldout':r['classification']='INCONCLUSIVE'
    pq.write_table(pa.Table.from_pylist(summary),OUT/'designability_summary.parquet')
    progress('ALIGNMENT_COMPLETE',controls_status=controls['status']);print(json.dumps(dict(controls=controls,backbones=summary)))

def novelty_audit():
    import tarfile
    resource('foldseek_install',heavy=True)
    tc=read(OUT/'toolchain_manifest.json');d=OUT/'tools/Foldseek'
    with urllib.request.urlopen('https://api.github.com/repos/steineggerlab/foldseek/releases/latest',timeout=30) as r: release=json.load(r)
    assets=[x for x in release['assets'] if x['name']=='foldseek-linux-avx2.tar.gz'];assert len(assets)==1
    asset=assets[0];p=d/asset['name'];download(asset['browser_download_url'],p)
    with tarfile.open(p) as t:t.extractall(d,filter='data')
    exe=d/'foldseek/bin/foldseek';version=run([str(exe),'version'],'foldseek_version',30).strip()
    run([str(exe),'easy-search','-h'],'foldseek_search_help',30)
    run([str(exe),'databases','-h'],'foldseek_databases_help',30)
    tc['Foldseek']=dict(repository='https://github.com/steineggerlab/foldseek',release=release['tag_name'],release_url=release['html_url'],version=version,archive_sha256=sha(p),executable_sha256=sha(exe),executable=str(exe),license='GPLv3; see official distribution',database_policy='No full AlphaFoldDB; PDB requires size admission')
    write(OUT/'toolchain_manifest.json',tc)
    heads=[]
    for url in ('https://mmseqs.com/foldseek/pdb100.tar.gz','https://mmseqs.com/foldseek/pdb.tar.gz'):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=30) as response:
                heads.append(dict(url=url,redirected_url=response.url,headers=dict(response.headers)))
        except Exception as exc:heads.append(dict(url=url,error=str(exc)))
    write(OUT/'foldseek/local_pdb_download_preflight.json',dict(official_endpoint_probes=heads,free_bytes=shutil.disk_usage(OUT).free,reserve_bytes=30*1024**3,admitted=False,reason='No local database extraction budget admitted; remote official PDB route used if supported. HEAD-only; no PDB database downloaded.'))
    # Official server metadata requires no structure submission. Persist the
    # available database snapshot before submitting any novelty query.
    for endpoint in ('https://search.foldseek.com/api/databases/all','https://search.foldseek.com/api/databases'):
        try:
            with urllib.request.urlopen(endpoint,timeout=30) as r:meta=json.load(r)
            write(OUT/'foldseek/remote_database_inventory.json',dict(endpoint=endpoint,retrieved_unix=time.time(),databases=meta));break
        except Exception as exc:write(OUT/'foldseek/remote_audit_error.json',dict(endpoint=endpoint,error=str(exc)))
    print(json.dumps(tc['Foldseek']))

def runtime_audit():
    import importlib.metadata as md,ast
    tc=read(OUT/'toolchain_manifest.json')
    tc['runtime']=dict(python=sys.version,existing_proteingen_torch=md.version('torch'),existing_proteingen_numpy=md.version('numpy'),production_environment_unchanged=True,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip())
    vp=Path(md.distribution('torch').locate_file('torch/version.py'))
    module=ast.parse(vp.read_text())
    for node in module.body:
        if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id in ('cuda','__version__','git_version'):
            try:tc['runtime']['torch_'+node.target.id]=ast.literal_eval(node.value)
            except ValueError:pass
    tc['mmseqs'].update(version=subprocess.check_output(['/usr/bin/mmseqs','version'],text=True).strip(),repository='https://github.com/soedinglab/MMseqs2',license='GPLv3',policy='Existing executable inspected read-only')
    tc['USalign']['version']=(OUT/'logs/usalign_help.log').read_text()[:400]
    weights=OUT/'tools/alphafold_weights/params'
    # Weight hashing is disk work; only admitted once no evaluator owns CUDA.
    resource('model_weight_hash_audit',heavy=True)
    tc['ColabFold']['model_weights']={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in weights.glob('*.npz')}
    tc['ColabFold']['weight_source']='https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar'
    tc['ColabFold']['alphafold_package_version']='2.3.20'
    write(OUT/'toolchain_manifest.json',tc)
    print(json.dumps(tc['runtime']))

def cleanup_installer():
    # Only the first, timed-out installer used recovery's pre-existing D: temp
    # root. Its private pip directories are expendable copies of cached wheels,
    # not scientific evidence. Never touch checkpoint/evaluation/tool caches.
    from e015_serial_recovery import processes
    oldroot=ROOT/'runtime/tmpdir';records=[json.loads(l) for l in (OUT/'resource_monitor.jsonl').read_text().splitlines()]
    starts=[r['timestamp_unix'] for r in records if r['phase']=='environment_installation'];assert len(starts)>=2
    begin,end=starts[:2]
    for p in processes():
        if '-m pip' in ' '.join(p['command']):
            env=Path(f"/proc/{p['pid']}/environ").read_bytes().split(b'\0')
            assert ('TMPDIR='+str(oldroot)).encode() not in env,'Old installer still active'
    doomed=[p for p in oldroot.glob('pip-*') if p.is_dir() and begin<=p.stat().st_mtime<end]
    items=[dict(path=str(p),bytes=sum(q.stat().st_size for q in p.rglob('*') if q.is_file())) for p in doomed]
    write(OUT/'logs/orphaned_installer_cleanup.json',dict(first_install_window=[begin,end],directories=items,source='Owned temporary staging from timed-out first ColabFold dependency preflight; reusable download cache retained'))
    for p in doomed:shutil.rmtree(p)
    print(json.dumps(dict(removed_own_temporary_directories=len(items),reclaimed_bytes=sum(r['bytes'] for r in items))))

def protocol():
    import numpy as np
    import pyarrow.parquet as pq
    panel=read(OUT/'backbone_panel.json');proof=[]
    for b in panel['backbones']:
        if b['source_category']!='latest_best':continue
        latest=Path(b['source_coordinates']).parent/'sparse_trajectory.npz';old=Path(b['sampler']['initial_gaussian_source'])
        x,y=np.load(latest)['t499'],np.load(old)['t499'];assert np.array_equal(x,y)
        proof.append(dict(length=b['length'],seed=b['seed'],latest_path=str(latest),historical_path=str(old),latest_file_sha256=sha(latest),historical_file_sha256=sha(old),x499_tensor_sha256=hashlib.sha256(x.tobytes()).hexdigest(),bitwise_equal=True))
    write(OUT/'same_initial_gaussian_proof.json',proof)
    controls=[b['reference_identity'] for b in panel['backbones'] if b['source_category']=='real_heldout']
    train=pq.read_table('/mnt/d/Simone/proteinGen/e014_artifacts/population/train.parquet',columns=['sample_id'],use_threads=False).column('sample_id').to_pylist()
    assert not set(controls)&set(train)
    write(OUT/'assessment_protocol.json',dict(frozen_before_external_predictions=True,panel_sha256=sha(OUT/'backbone_panel.json'),real_controls_disjoint_from_train=True,proteinmpnn=dict(model='CA-only v_48_020',temperatures=[.1,.2],sequences_per_temperature=16,batch_size=1,no_sequence_bias=True,no_fixed_residues=True,paired_design_rng='Same temperature-specific seeds for latest/historical length pairs'),screening=dict(maximum_sequences=82,latest_best=40,historical=20,real=20,median=1,failure=1,allocation_deviation='Requested allocations sum84; hard82 cap honored by screening one design each for median and failure',single_sequence=True,templates=False,model='alphafold2_ptm',model_order=[1],recycles=3,msa_depth='1:1',relax=False),structural_gates=dict(strong=dict(target_tm_score_min=.7,target_coverage_min=.8,rmsd_max_angstrom=3),short_chain_extra=dict(length_lt=80,coverage_min=.9,rmsd_max_angstrom=2.5),very_strong=dict(target_tm_score_min=.8,coverage_min=.9,rmsd_max_angstrom=2.5),confidence_support=dict(plddt_min=70,ptm_min=.6),positive_controls=dict(required_real_backbones_best_tm_ge_07=4,total_real_controls=5)),geometry_prefilter=dict(finite_complete=True,no_duplicate_ca=True,adjacent_valid_fraction_min=.8,frozen_adjacent_range_angstrom=[3.4,4.2],clash_fraction_max=.005,role='Descriptive prefilter only; all17 panel backbones still designed and screened. Does not replace target refolding gates.',high_priority_novel_extra='Rg,covariance,local geometry compared with matched held-out reference ranges; no novelty from low similarity alone'),fair_comparison='Matched4 designs:2 score-selected+2 diversity-selected per latest/historical backbone; expanded8-design latest results separately',novelty=dict(database='Official PDB snapshot only',known_query_tm_min=.5,remote_fallback='Official search.foldseek.com; server snapshot must be recorded and nearest hits independently confirmed using US-align',positive_control_gate_required=True),claims='Computational target recovery only. No claim of experimental stability, solubility, functionality or proven novel fold.'))
    print(json.dumps(dict(pairs_bitwise_equal=len(proof),real_controls_disjoint=True)))

def confirm():
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    assert read(OUT/'controls_summary.json')['status']=='VALIDATED','Positive-control gate must pass before confirmation'
    resource('confirmatory_refolding',heavy=True)
    aligned=pq.read_table(OUT/'structural_alignment_results.parquet',use_threads=False).to_pylist()
    designs={r['sequence_id']:r for r in pq.read_table(OUT/'proteinmpnn_selected_sequences.parquet',use_threads=False).to_pylist()}
    panel={r['backbone_id']:r for r in read(OUT/'backbone_panel.json')['backbones']};queue=[]
    for bid,b in panel.items():
        if b['source_category']=='real_heldout':continue
        eligible=sorted([r for r in aligned if r['backbone_id']==bid],key=lambda r:(-r['target_tm_score'],r['sequence_id']))
        if not eligible or eligible[0]['target_tm_score']<.5:continue
        picks=[eligible[0]];other=next((r for r in eligible[1:] if r['identity80_cluster']!=picks[0]['identity80_cluster']),None)
        if other:picks.append(other)
        queue.extend(designs[r['sequence_id']] for r in picks)
    write(OUT/'colabfold/confirmation_plan.json',dict(sequence_ids=[r['sequence_id'] for r in queue],selection='Two best screen designs from distinct80%-identity clusters where available; only E015 bestTM>=.5; no replacement controls',model='alphafold2_ptm',models=5,recycles=6,max_msa='16:32',templates=False,relaxation=False,msa_source='Public ColabFold server; one local serial job; cache by sequenceSHA'))
    jobs=read(OUT/'colabfold/confirmation_jobs.json') if (OUT/'colabfold/confirmation_jobs.json').exists() else {}
    for r in queue:
        sid=r['sequence_id'];od=OUT/'colabfold/confirmation'/sid;cache=OUT/'cache/msa'/(r['sequence_sha256']+'.a3m')
        fa=OUT/'sequences'/(sid+'.fasta');a3m=OUT/'sequences'/(sid+'.a3m')
        if cache.exists():shutil.copyfile(cache,a3m);inp=a3m
        else:inp=fa
        cmd=[str(OUT/'tools/colabfold_env/bin/colabfold_batch'),str(inp),str(od),'--msa-mode','mmseqs2_uniref_env','--model-type','alphafold2_ptm','--num-models','5','--num-recycle','6','--max-msa','16:32','--num-seeds','1','--random-seed','151502','--disable-unified-memory','--data',str(OUT/'tools/alphafold_weights')]
        prior=jobs.get(sid,{})
        if prior.get('status')=='COMPLETE' and prior['sequence_sha256']==r['sequence_sha256'] and all(Path(f).exists() and sha(f)==h for f,h in prior['files'].items()):continue
        resource('confirmation_'+sid,heavy=True);progress('COLABFOLD_CONFIRMATION',current_sequence=sid)
        start=time.monotonic();error=None
        try:run(cmd,sid+'_confirmation',3600)
        except Exception as exc:error=str(exc)
        msas=list(od.glob('*.a3m'))
        if msas:
            cache.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(msas[0],cache)
            write(cache.with_suffix('.json'),dict(sequence_sha256=r['sequence_sha256'],msa_sha256=sha(cache),source='Public ColabFold MSA server; returned by official client',msa_header_count=sum(l.startswith('>') for l in cache.read_text().splitlines())))
        predictions=[]
        if error is None:
            for p in sorted(od.glob('*unrelaxed_rank_*.pdb')):
                suffix=p.name.split('_unrelaxed_')[1].removesuffix('.pdb');sp=p.with_name(p.name.split('_unrelaxed_')[0]+'_scores_'+suffix+'.json');sc=read(sp)
                predictions.append(dict(sequence_id=sid,backbone_id=r['backbone_id'],model=suffix,pdb=str(p),score_json=str(sp),mean_plddt=float(np.mean(sc['plddt'])),ptm=float(sc['ptm']),msa_sha256=sha(cache) if cache.exists() else None,msa_source='public_colabfold',recycles=6,templates=False))
            if len(predictions)!=5:error='Expected five complete model predictions'
        jobs[sid]=dict(status='FAILED' if error else 'COMPLETE',sequence_sha256=r['sequence_sha256'],target_sha256=r['backbone_sha256'],command=cmd,runtime_seconds=time.monotonic()-start,error=error,predictions=predictions,files={str(p):sha(p) for p in od.rglob('*') if p.is_file()})
        if jobs[sid]['status']=='COMPLETE':sync_outputs(jobs[sid]['files'])
        write(OUT/'colabfold/confirmation_jobs.json',jobs)
        allpred=[p for j in jobs.values() for p in j['predictions']]
        if allpred:pq.write_table(pa.Table.from_pylist(allpred),OUT/'colabfold_confirmation.parquet')
        print(json.dumps(dict(confirmation=sid,status=jobs[sid]['status'])),flush=True)
        if (OUT/'STOP_EXTERNAL.json').exists():break
    progress('CONFIRMATION_COMPLETE',confirmation_sequences=len(queue))

def confirm_align():
    import re
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    resource('confirmatory_structural_alignment',heavy=True)
    predictions=pq.read_table(OUT/'colabfold_confirmation.parquet',use_threads=False).to_pylist() if (OUT/'colabfold_confirmation.parquet').exists() else []
    panel={r['backbone_id']:r for r in read(OUT/'backbone_panel.json')['backbones']}
    designs={r['sequence_id']:r for r in pq.read_table(OUT/'proteinmpnn_selected_sequences.parquet',use_threads=False).to_pylist()}
    jobs=read(OUT/'alignments/confirmation_jobs.json') if (OUT/'alignments/confirmation_jobs.json').exists() else {};rows=[]
    exe=OUT/'tools/USalign/USalign';binary_hash=sha(exe)
    def ca(p):return np.array([[float(line[a:b]) for a,b in [(30,38),(38,46),(46,54)]] for line in Path(p).read_text().splitlines() if line.startswith('ATOM') and line[12:16].strip()=='CA'])
    for pred in predictions:
        key=pred['sequence_id']+'_'+pred['model'];target=panel[pred['backbone_id']]['sanitized_pdb'];fp=hashlib.sha256((sha(target)+sha(pred['pdb'])+binary_hash).encode()).hexdigest()
        old=jobs.get(key,{})
        if old.get('fingerprint')==fp and old.get('status')=='COMPLETE':rows.append(old['row']);continue
        run([str(exe),target,pred['pdb'],'-mol','prot'],key+'_confirmation_align',60)
        text=(OUT/'logs'/(key+'_confirmation_align.log')).read_text();tm={int(k):float(v) for v,k in re.findall(r'TM-score=\s*([0-9.]+).*normalized by length of (?:Structure|Chain)_([12])',text)};match=re.search(r'Aligned length=\s*(\d+),\s*RMSD=\s*([0-9.]+)',text);assert match and len(tm)>=2
        x,y=ca(target),ca(pred['pdb']);complete=x.shape==y.shape and np.isfinite(y).all();n=len(x);coverage=int(match[1])/n;rmsd=float(match[2]);strong=complete and tm[1]>=.7 and coverage>=.8 and rmsd<=3
        if n<80:strong=strong and coverage>=.9 and rmsd<=2.5
        kabsch=None
        if complete:
            x=x-x.mean(0);y=y-y.mean(0);u,_,vt=np.linalg.svd(x.T@y);rot=u@np.diag([1,1,np.linalg.det(u@vt)])@vt;kabsch=float(np.sqrt(np.mean(np.sum((x@rot-y)**2,axis=-1))))
        row=dict(pred,target_tm_score=tm[1],prediction_tm_score=tm[2],target_coverage=coverage,aligned_ca_rmsd_angstrom=rmsd,aligned_residues=int(match[1]),index_matched_kabsch_rmsd_angstrom=kabsch,strong_recovery=bool(strong),finite_complete_prediction=bool(complete),identity80_cluster=designs[pred['sequence_id']]['identity80_cluster'],assessment='MSA-assisted ensemble result; separately reported from single-sequence primary gate')
        rows.append(row);jobs[key]=dict(status='COMPLETE',fingerprint=fp,row=row);write(OUT/'alignments/confirmation_jobs.json',jobs)
        pq.write_table(pa.Table.from_pylist(rows),OUT/'colabfold_confirmation.parquet')
    if rows:pq.write_table(pa.Table.from_pylist(rows),OUT/'colabfold_confirmation.parquet')
    progress('CONFIRMATION_ALIGNMENT_COMPLETE',models=len(rows))

def novelty():
    import re,tarfile,io,urllib.parse
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    assert read(OUT/'controls_summary.json')['status']=='VALIDATED','No novelty interpretation without working real controls'
    resource('pdb_novelty_search',heavy=True)
    inventory=read(OUT/'foldseek/remote_database_inventory.json');dbs=inventory['databases']
    if isinstance(dbs,dict):dbs=dbs['databases']
    pdbs=[r for r in dbs if 'pdb' in (r.get('name','')+' '+r.get('path','')).lower()]
    assert pdbs,'No official remote PDB snapshot available'
    db=sorted(pdbs,key=lambda r:('100' not in r.get('name','').lower(),r['path']))[0]
    write(OUT/'foldseek/database_selection.json',dict(route='Official Foldseek search server',snapshot=db,local_database='Not downloaded; tool+weights leave insufficient conservative reserve for a local full PDB snapshot',api_documentation='https://search.mmseqs.com/docs/',mode='tmalign',remote_fields_not_exposed=['qtmscore','ttmscore','lddt'],confirmation='US-align top10 nonredundant PDB entries; absence among retrieved candidates is not exhaustive proof of a novel fold'))
    panel=read(OUT/'backbone_panel.json')['backbones'];summ={r['backbone_id']:r for r in pq.read_table(OUT/'designability_summary.parquet',use_threads=False).to_pylist()};aligned=pq.read_table(OUT/'structural_alignment_results.parquet',use_threads=False).to_pylist();queue=[]
    for b in panel:
        if b['source_category']=='latest_best' or (b['source_category'] in ('latest_median','latest_failure') and summ[b['backbone_id']]['classification'] in ('STRONGLY_DESIGNABLE','PROMISINGLY_DESIGNABLE')):queue.append(dict(query_id=b['backbone_id']+'_target',backbone_id=b['backbone_id'],path=b['sanitized_pdb'],kind='target'))
        successful=sorted([r for r in aligned if r['backbone_id']==b['backbone_id'] and r['strong_recovery']],key=lambda r:-r['target_tm_score'])
        if successful and b['source_category'].startswith('latest'):queue.append(dict(query_id=b['backbone_id']+'_best_refold',backbone_id=b['backbone_id'],path=successful[0]['pdb'],kind='single_sequence_refold'))
    jobs=read(OUT/'foldseek/jobs.json') if (OUT/'foldseek/jobs.json').exists() else {}
    allhits=pq.read_table(OUT/'foldseek_hits.parquet',use_threads=False).to_pylist() if (OUT/'foldseek_hits.parquet').exists() else []
    confirmed=pq.read_table(OUT/'confirmed_pdb_hits.parquet',use_threads=False).to_pylist() if (OUT/'confirmed_pdb_hits.parquet').exists() else []
    novel=pq.read_table(OUT/'novelty_summary.parquet',use_threads=False).to_pylist() if (OUT/'novelty_summary.parquet').exists() else []
    for q in queue:
        qid=q['query_id'];folder=OUT/'foldseek'/qid;folder.mkdir(parents=True,exist_ok=True);fingerprint=hashlib.sha256((sha(q['path'])+json.dumps(db,sort_keys=True)+'tmalign').encode()).hexdigest();job=jobs.get(qid,{})
        if job.get('fingerprint')!=fingerprint:job=dict(fingerprint=fingerprint,status='NEW',query=q,database=db)
        if job.get('status')=='COMPLETE' and job.get('fingerprint')==fingerprint and all(Path(p).exists() and sha(p)==h for p,h in job.get('files',{}).items()):
            saved_hits=[r for r in allhits if r.get('query')==qid];saved_confirmed=[r for r in confirmed if r.get('query_id')==qid]
            saved_novel=[r for r in novel if r.get('backbone_id')==q['backbone_id']]
            if len(saved_hits)==job['top_hits'] and len(saved_confirmed)==job['confirmed_nonredundant_hits'] and (q['kind']!='target' or saved_novel):continue
        # Incomplete scalar publication is reconstructed from cached query data;
        # retain unrelated completed queries and never duplicate their rows.
        allhits=[r for r in allhits if r.get('query')!=qid];confirmed=[r for r in confirmed if r.get('query_id')!=qid]
        if q['kind']=='target':novel=[r for r in novel if r.get('backbone_id')!=q['backbone_id']]

        if job.get('status')=='SUBMITTING' and not job.get('ticket'):raise RuntimeError('Ambiguous interrupted remote submission; will not submit a duplicate blindly')
        if not job.get('ticket'):
            job['status']='SUBMITTING';jobs[qid]=job;write(OUT/'foldseek/jobs.json',jobs)
            data=urllib.parse.urlencode({'q':Path(q['path']).read_text(),'database[]':db['path'],'mode':'tmalign'}).encode()
            req=urllib.request.Request('https://search.foldseek.com/api/ticket',data=data)
            with urllib.request.urlopen(req,timeout=60) as response:ticket=json.load(response)
            assert 'id' in ticket,ticket;job['ticket']=ticket['id'];job['status']='SUBMITTED';jobs[qid]=job;write(OUT/'foldseek/jobs.json',jobs)
        archive=folder/'results.tar.gz'
        raw=folder/'remote_results.json'
        tid=job['ticket']
        if not archive.exists():
            deadline=time.monotonic()+1800
            while time.monotonic()<deadline:
                with urllib.request.urlopen('https://search.foldseek.com/api/ticket/'+tid,timeout=30) as response:status=json.load(response)
                job['remote_status']=status;write(OUT/'foldseek/jobs.json',jobs)
                if status['status'] in ('COMPLETE','ERROR','UNKNOWN'):break
                resource('foldseek_wait_'+qid);time.sleep(15)
            assert status['status']=='COMPLETE',status
            if not raw.exists():
                with urllib.request.urlopen('https://search.foldseek.com/api/result/'+tid+'/0',timeout=60) as response:result=json.load(response)
                write(raw,result)
        result=read(raw) if raw.exists() else {};hits=[]
        for rr in result.get('results',[]):hits.extend(rr.get('alignments',[]))
        write(folder/'remote_top_hits.json',hits[:100])
        archive=folder/'results.tar.gz';download('https://search.foldseek.com/api/result/download/'+tid,archive)
        # The downloadable archive retains all returned alignments even when
        # the interactive endpoint paginates its JSON view.
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                if member.isfile() and (member.name.endswith('.m8') or 'alis_' in member.name) and member.size<128*1024**2:
                    stream=tar.extractfile(member)
                    if stream is None:continue
                    for line in io.TextIOWrapper(stream):
                        cols=line.rstrip('\n').split('\t')
                        if len(cols)>=18:
                            hit=dict(target=cols[1],alnlen=int(cols[3]),prob=float(cols[10]),evalue=float(cols[11]),bits=float(cols[12]),qlen=int(cols[13]),tlen=int(cols[14]),qAln=cols[15],dbAln=cols[16],tCa=cols[17],tSeq=cols[18] if len(cols)>18 else '')
                            hits.append(hit)
        unique=[];seen=set()
        for hit in sorted(hits,key=lambda h:-float(h.get('bits',h.get('score',0)))):
            target=hit.get('target',hit.get('targetId',hit.get('targetid','')))
            if not target or target in seen:continue
            seen.add(target);hit['target']=target;unique.append(hit)
        unique=unique[:100];write(folder/'top100.json',unique)
        for rank,h in enumerate(unique,1):allhits.append(dict(query=qid,target=h['target'],rank=rank,alnlen=h.get('alnlen',h.get('alnLength')),evalue=h.get('evalue'),prob=h.get('prob'),raw_bits=h.get('bits'),qtmscore=None,ttmscore=None,alntmscore=None,lddt=None,database_version=db.get('version'),ticket=tid))
        entries=set();seqhashes=set();local=[]
        for h in unique:
            name=h['target'];entry=name[:4].lower();seq=h.get('tSeq',h.get('tseq',h.get('targetSeq','')))
            coords=h.get('tCa',h.get('tca',h.get('targetCa','')))
            if entry in entries or not coords:continue
            seqhash=hashlib.sha256(seq.encode()).hexdigest()
            if seq and seqhash in seqhashes:continue
            xyz=np.fromstring(coords,sep=',').reshape(-1,3) if isinstance(coords,str) else np.asarray(coords).reshape(-1,3)
            if not len(xyz) or not np.isfinite(xyz).all():continue
            p=folder/('neighbor_'+str(len(local))+'.pdb')
            with p.open('w') as f:
                f.write('REMARK C-alpha neighbor coordinates from the official searched PDB snapshot\n')
                for i,(x,y,z) in enumerate(xyz,1):f.write(f'ATOM  {i:5d}  CA  ALA A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n')
                f.write('TER\nEND\n')
            name2=qid+'_neighbor'+str(len(local));run([str(OUT/'tools/USalign/USalign'),q['path'],str(p),'-mol','prot'],name2,120)
            text=(OUT/'logs'/(name2+'.log')).read_text();scores=re.findall(r'TM-score=\s*([0-9.]+).*normalized by length of (?:Structure|Chain)_([12])',text);tm={int(k):float(v) for v,k in scores};a=re.search(r'Aligned length=\s*(\d+),\s*RMSD=\s*([0-9.]+)',text);assert a and len(tm)>=2
            qlen=sum(l.startswith('ATOM') and l[12:16].strip()=='CA' for l in Path(q['path']).read_text().splitlines());row=dict(query_id=qid,backbone_id=q['backbone_id'],query_kind=q['kind'],pdb_neighbor=name,query_tm_score=tm[1],target_tm_score=tm[2],aligned_residues=int(a[1]),query_coverage=int(a[1])/qlen,target_coverage=int(a[1])/len(xyz),rmsd_angstrom=float(a[2]),neighbor_ca_pdb=str(p),neighbor_sha256=sha(p),database_version=db.get('version'),source='Official Foldseek server PDB coordinates; independently US-aligned')
            local.append(row);confirmed.append(row);entries.add(entry);seqhashes.add(seqhash)
            if len(local)>=10:break
        job.update(status='COMPLETE',top_hits=len(unique),confirmed_nonredundant_hits=len(local),files={str(p):sha(p) for p in folder.iterdir() if p.is_file()});write(OUT/'foldseek/jobs.json',jobs)
        if q['kind']=='target':
            nearest=max(local,key=lambda r:r['query_tm_score']) if local else None;design=summ[q['backbone_id']]
            if nearest is None:cls='INCONCLUSIVE'
            elif nearest['query_tm_score']>=.5 and nearest['query_coverage']>=.5:cls='KNOWN_FOLD_LIKE'
            elif design['classification'] in ('STRONGLY_DESIGNABLE','PROMISINGLY_DESIGNABLE'):cls='POTENTIALLY_NOVEL_RELATIVE_TO_PDB'
            else:cls='ARTIFACT_LIKELY'
            novel.append(dict(backbone_id=q['backbone_id'],classification=cls,nearest_pdb=nearest['pdb_neighbor'] if nearest else None,query_tm_score=nearest['query_tm_score'] if nearest else None,target_tm_score=nearest['target_tm_score'] if nearest else None,query_coverage=nearest['query_coverage'] if nearest else None,confirmed_hits=len(local),neighbor_ca_pdb=nearest['neighbor_ca_pdb'] if nearest else None,designability=design['classification'],database_version=db.get('version'),qualification='Relative to searched PDB snapshot and retrieved candidates; not a proven novel fold'))
        for filename,rs in [('foldseek_hits.parquet',allhits),('confirmed_pdb_hits.parquet',confirmed),('novelty_summary.parquet',novel)]:
            if rs:pq.write_table(pa.Table.from_pylist(rs),OUT/filename)
        progress('PDB_NOVELTY_SEARCH',current_query=qid,completed_queries=sum(j.get('status')=='COMPLETE' for j in jobs.values()))
    progress('NOVELTY_COMPLETE')

def resume():
    import e015_serial_recovery as recovery
    resource('before_production_resume',heavy=True)
    pause_proof=read(OUT/'production_pause_proof.json')
    state=read(ROOT/'STATE.json')
    if state.get('phase') in ('complete','training_complete','production_complete'):
        write(OUT/'production_resume_proof.json',dict(resumed=False,reason='Frozen production contract is complete; no extension'));return
    # Strict restore also reconstructs the CUDA Philox stream. Run that audit
    # in an isolated subprocess so its CUDA context fully exits before training.
    run([PY,'-B',str(WORK/'docs/e015_serial_recovery.py'),'audit-latest'],'production_resume_strict_audit',600)
    records=[json.loads(line) for line in (OUT/'logs/production_resume_strict_audit.log').read_text().splitlines() if line.startswith('{')]
    report=Path(next(r['report'] for r in reversed(records) if 'report' in r));proof=read(report/'recovery_audit.json')
    assert proof['selected']==pause_proof['checkpoint'],'Pause checkpoint changed; no blind resume'
    assert proof['next_updates']==pause_proof['next_updates'],'Deterministic next-operation mismatch'
    command=proof['command'];expected=[PY,'-B',str(WORK/'scripts/supervise_e015_training.py'),'--resume'];assert command==expected
    # Re-check existing supervisor source and launch directly. The older
    # recovery.launch helper copies a snapshot into the historical evaluation;
    # this assessment never adds files to either previous evaluation.
    source=(WORK/'scripts/supervise_e015_training.py').read_text();assert 'args.resume' in source and 'run_e015_integrated.py' in source
    resource('after_audit_before_production_launch',heavy=True)
    production_env=os.environ.copy()
    # Reproduce the existing recovery launcher cache/thread environment.
    # Runtime still sets the frozen scientific torch cpu_threads=2 itself.
    for k in ('TMPDIR','TMP','TEMP','XDG_CACHE_HOME','MPLCONFIGDIR','TORCH_HOME','HF_HOME','NUMBA_CACHE_DIR','CUDA_CACHE_PATH','TRITON_CACHE_DIR'):
        production_env[k]=str(ROOT/'runtime'/k.lower())
    with (OUT/'logs/production_supervisor_resume.log').open('a') as log:
        p=subprocess.Popen(command,cwd=WORK,env=production_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    launch=dict(supervisor_pid=p.pid,command=command,timestamp_unix=time.time())
    write(report/'launch.json',launch);write(OUT/'production_resume_launch.json',launch)
    progress('PRODUCTION_RESUMED_VERIFYING',supervisor_pid=p.pid)
    recovery.observe(report,1020)
    first=read(report/'first_resumed_heartbeat.json');latest=read(ROOT/'COMMITTED.json')
    assert latest==read(ROOT/'production/checkpoints/LATEST.json')
    assert latest['position']['update']>proof['selected']['position']['update'],'No subsequent durable checkpoint yet'
    # Trainer atomic_checkpoint already verifies its own serialized checkpoint
    # before committing. Here only bounded metadata/file identity is inspected,
    # without a second model or full checkpoint load during active training.
    rolling=Path(latest['path']);assert rolling.exists() and rolling.stat().st_size>100*1024**2
    s=read(ROOT/'STATE.json');sup=read(ROOT/'SUPERVISOR.json')
    assert time.time()-s['timestamp_unix']<120 and s['phase']=='training_production'
    result=dict(resumed=True,checkpoint=proof['selected'],next_operation=proof['next_updates'][0],first_heartbeat=first,subsequent_durable_checkpoint=latest,subsequent_durability_basis='Production atomic_checkpoint strict serialized verification and hash before durable COMMITTED/LATEST publication; no concurrent external checkpoint model load',supervisor_pid=sup['pid'],trainer_pid=s['pid'],command=command,no_skipped_or_duplicate_durable_exposure=True,observation=read(report/'training_observation.json'),audit=str(report),production_scientific_configuration_changed=False)
    write(OUT/'production_resume_proof.json',result);progress('PRODUCTION_RUNNING_VERIFIED',trainer_pid=s['pid'],supervisor_pid=sup['pid']);print(json.dumps(dict(resumed=True,trainer_pid=s['pid'],supervisor_pid=sup['pid'],first_computed_update=proof['next_updates'][0]['update'],first_computed_exposures=proof['next_updates'][0]['exposures'],new_durable_update=latest['position']['update'])))

def switch_eta():
    import statistics
    jobs=read(OUT/'colabfold/jobs.json');done=[j for j in jobs.values() if j['status']=='COMPLETE'];durations=[j['runtime_seconds'] for j in done if j['sequence']['source_category']=='real_heldout']
    remaining=82-len(done);median=statistics.median(durations)
    ordered=sorted(durations);lo=ordered[int(.1*(len(ordered)-1))];hi=ordered[int(.9*(len(ordered)-1))]
    x=dict(timestamp_unix=time.time(),evaluation_root=str(OUT),evaluation_type='External ProteinMPNN/ColabFold designability',phase=read(OUT/'progress.json')['phase'],current_sequence=read(OUT/'progress.json')['current_sequence'],controller_pid=1189240,completed_units=len(done),total_screen_units=82,remaining_screen_units=remaining,eta_screen_minutes=remaining*median/60,plausible_screen_range_minutes=[remaining*lo/60,remaining*hi/60],method='Median completed single-sequence screening durations; real-control jobs only, excluding all3 preflight predictions',observations=len(durations),median_unit_seconds=median,uncertainty='New-length compilation, final alignment and conditional confirmation/novelty stages are additional; this is a supported screening ETA and a lower bound on the complete evaluation',resumable=True,decision='SUSPEND_AFTER_ATOMIC_UNIT',maximum_wait_minutes=30,continuation_command=PY+' -B '+str(WORK/'docs/e015_external_designability.py')+' screen')
    write(ROOT/'EVALUATION_SWITCHOVER_STATUS.json',x);print(json.dumps(x,indent=2))

def suspend():
    import signal
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from e015_serial_recovery import processes
    status=read(ROOT/'EVALUATION_SWITCHOVER_STATUS.json');pid=status['controller_pid'];proc=Path(f'/proc/{pid}/cmdline');cmd=proc.read_bytes() if proc.exists() else None
    if cmd is not None:assert b'e015_external_designability.py' in cmd and b'screen' in cmd
    current=read(OUT/'progress.json')['current_sequence'];deadline=time.monotonic()+300
    write(OUT/'HOLD_E015.json',dict(timestamp_unix=time.time(),reason='GPU reassigned to matched E007 control',auto_resume=False))
    while time.monotonic()<deadline:
        jobs=read(OUT/'colabfold/jobs.json')
        if jobs.get(current,{}).get('status') in ('COMPLETE','FAILED'):break
        time.sleep(1)
    else:raise RuntimeError('Current atomic unit has not finished; no unsafe cancellation')
    # Documented standard Python controller interrupt AFTER durable unit commit.
    # Any worker dispatched in the narrow boundary race is allowed to finish;
    # workers receive no signal and their valid output is reconciled below.
    if cmd is None:
        mechanism='Existing exclusive-controller ownership gate stopped dispatch at the next durable unit boundary; no signal delivered'
    elif hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'):
        assert Path(f'/proc/{pid}/cmdline').read_bytes()==cmd
        fd=os.pidfd_open(pid);signal.pidfd_send_signal(fd,signal.SIGINT);os.close(fd)
    else:
        import ctypes
        assert os.uname().machine=='x86_64'
        libc=ctypes.CDLL(None,use_errno=True);fd=libc.syscall(434,pid,0)
        assert fd>=0
        try:
            assert Path(f'/proc/{pid}/cmdline').read_bytes()==cmd
            assert libc.syscall(424,fd,int(signal.SIGINT),0,0)==0
        finally:os.close(fd)
    if cmd is not None:mechanism='Standard Python SIGINT at durable unit boundary, controller only; no worker/trainer signal and no SIGKILL'
    write(OUT/'logs/controller_suspend_request.json',dict(controller_pid=pid,atomic_unit_completed=current,mechanism=mechanism,timestamp_unix=time.time()))
    deadline=time.monotonic()+600
    while time.monotonic()<deadline:
        workers=[r for r in processes() if 'colabfold_batch' in ' '.join(r['command']) and str(OUT) in ' '.join(r['command'])]
        if not workers:break
        print(json.dumps(dict(waiting_for_clean_worker_exit=[r['pid'] for r in workers])),flush=True);time.sleep(5)
    else:raise RuntimeError('Worker remains active; E007 launch forbidden')
    resource('evaluator_fully_suspended',heavy=True)
    jobs=read(OUT/'colabfold/jobs.json');selected=pq.read_table(OUT/'proteinmpnn_selected_sequences.parquet',use_threads=False).to_pylist();panel={b['backbone_id']:b for b in read(OUT/'backbone_panel.json')['backbones']}
    for r in selected:
        sid=r['sequence_id'];od=OUT/'colabfold/screening'/sid
        if jobs.get(sid,{}).get('status')=='COMPLETE':continue
        ps=list(od.glob('*unrelaxed_rank_001*.pdb'));ss=list(od.glob('*scores_rank_001*.json'))
        if len(ps)!=1 or len(ss)!=1:continue
        xyz=np.array([[float(l[a:b]) for a,b in ((30,38),(38,46),(46,54))] for l in ps[0].read_text().splitlines() if l.startswith('ATOM') and l[12:16].strip()=='CA']);assert xyz.shape==(r['length'],3) and np.isfinite(xyz).all()
        config=read(od/'config.json');assert config['model_type']=='alphafold2_ptm' and config['num_recycles']==3 and config['msa_mode']=='single_sequence'
        fa=OUT/'sequences'/(sid+'.fasta');assert fa.read_text().splitlines()[1]==r['sequence']
        bd=panel[r['backbone_id']];command=[str(OUT/'tools/colabfold_env/bin/colabfold_batch'),str(fa),str(od),'--msa-mode','single_sequence','--model-type','alphafold2_ptm','--num-models','1','--model-order','1','--num-recycle','3','--recycle-early-stop-tolerance','0.0','--num-seeds','1','--random-seed','151502','--max-msa','1:1','--recompile-padding','0','--disable-unified-memory','--data',str(OUT/'tools/alphafold_weights')]
        fingerprint=hashlib.sha256(json.dumps(dict(sequence=r['sequence_sha256'],target=bd['backbone_sha256'],command=command,tool_commit=read(OUT/'toolchain_manifest.json')['ColabFold']['commit']),sort_keys=True).encode()).hexdigest();sc=read(ss[0])
        jobs[sid]=dict(status='COMPLETE',fingerprint=fingerprint,sequence=r,target_backbone=bd['sanitized_pdb'],pdb=str(ps[0]),score_json=str(ss[0]),mean_plddt=float(np.mean(sc['plddt'])),ptm=float(sc['ptm']),runtime_seconds=None,command=command,files={str(p):sha(p) for p in od.rglob('*') if p.is_file()},reconciled_after_controller_interrupt=True)
    write(OUT/'colabfold/jobs.json',jobs)
    rows=[dict(sequence_id=k,backbone_id=v['sequence']['backbone_id'],length=v['sequence']['length'],status=v['status'],mean_plddt=v.get('mean_plddt'),ptm=v.get('ptm'),pdb=v.get('pdb'),runtime_seconds=v.get('runtime_seconds'),error=v.get('error')) for k,v in jobs.items()]
    pq.write_table(pa.Table.from_pylist(rows),OUT/'colabfold_screening.parquet')
    completed=[k for k,v in jobs.items() if v['status']=='COMPLETE'];status.update(status='SUSPENDED',completed_units=len(completed),remaining_screen_units=82-len(completed),suspended_at_unix=time.time(),completed_sequence_ids=completed,controller_exited=True,workers_exited=True,gpu_released=True,input_hashes=dict(panel=sha(OUT/'backbone_panel.json'),selected_sequences=sha(OUT/'proteinmpnn_selected_sequences.parquet'),sanitized_backbones=sha(OUT/'sanitized_backbone_manifest.json'),checkpoint=read(OUT/'backbone_panel.json')['checkpoint']['sha256']),hold_file=str(OUT/'HOLD_E015.json'))
    write(ROOT/'EVALUATION_SWITCHOVER_STATUS.json',status);progress('SUSPENDED_FOR_E007',completed_refolds=len(completed),remaining_refolds=82-len(completed),automatic_resume=False)
    manifest=read(OUT/'MANIFEST.json');manifest.update(status='SUSPENDED',completed_mpnn_sequences=544,completed_refolds=len(completed),jobs_manifest_sha256=sha(OUT/'colabfold/jobs.json'),hold_file=str(OUT/'HOLD_E015.json'),continuation=status['continuation_command']);write(OUT/'MANIFEST.json',manifest)
    print(json.dumps(dict(status='SUSPENDED',completed_refolds=len(completed),remaining=82-len(completed),gpu_released=True)))

if __name__=='__main__':
    arg=argparse.ArgumentParser();arg.add_argument('stage',choices=['init','audit-tools','pause','install','mpnn','sequences','fold-preflight','screen','align','monitor','novelty-audit','cleanup-installer','protocol','aggregate','cards','report','confirm','confirm-align','novelty','runtime-audit','resume','switch-eta','suspend']);arg.add_argument('--release-hold',action='store_true');a=arg.parse_args()
    if (OUT/'HOLD_E015.json').exists() and a.stage not in ('monitor','switch-eta','suspend','report'):
        if not a.release_hold:raise SystemExit('E015 intentionally held for E007; explicit --release-hold required for later authorized continuation')
        resource('explicit_hold_release',heavy=True)
        (OUT/'HOLD_E015.json').unlink()
    # Shared production locks prevent a competing training launch during external GPU work.
    leases=[]
    if a.stage in ('mpnn','fold-preflight','screen','confirm'):
        import fcntl
        for lockpath in (ROOT/'GPU_WORKER.lock',Path('/mnt/d/Simone/proteinGen/.e014_worktree/e007_matched_fullscale_v1/OWNER.lock')):
            lease=lockpath.open('a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);leases.append(lease)
    if a.stage=='init':init()
    elif a.stage=='audit-tools':audit_tools()
    elif a.stage=='pause':pause()
    elif a.stage=='install':install()
    elif a.stage=='mpnn':mpnn()
    elif a.stage=='sequences':sequences()
    elif a.stage=='fold-preflight':folding()
    elif a.stage=='screen':folding(screen=True)
    elif a.stage=='align':align()
    elif a.stage=='novelty-audit':novelty_audit()
    elif a.stage=='cleanup-installer':cleanup_installer()
    elif a.stage=='protocol':protocol()
    elif a.stage=='confirm':confirm()
    elif a.stage=='confirm-align':confirm_align()
    elif a.stage=='novelty':novelty()
    elif a.stage=='runtime-audit':runtime_audit()
    elif a.stage=='resume':resume()
    elif a.stage=='switch-eta':switch_eta()
    elif a.stage=='suspend':suspend()
    elif a.stage in ('aggregate','cards','report'):
        import e015_external_reporting as reporting
        getattr(reporting,a.stage)()
        if a.stage=='cards':reporting.neighbor_cards()
    elif a.stage=='monitor':
        x=resource('external_setup_monitor');print(json.dumps({k:x[k] for k in ('timestamp_unix','ram_available_bytes','swap_used_bytes','d_free_bytes','gpu_csv','gpu_compute_apps','load_average')}))
