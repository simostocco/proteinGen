"""Two native-sequence calibration units; never launches or resumes training."""
import os
from pathlib import Path

WORK = Path('/mnt/d/Simone/proteinGen/.e015_worktree')
ROOT = Path('/mnt/d/Simone/proteinGen/e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1')
EXT = ROOT / 'external_designability_loop_v1'
OUT = EXT / 'native_sequence_calibration_v1'
PY = '/home/simostocco/miniforge3/envs/proteingen/bin/python'
CFPY = str(EXT / 'tools/colabfold_env/bin/python')
CACHE_KEYS = ('TMPDIR','TMP','TEMP','XDG_CACHE_HOME','MPLCONFIGDIR','TORCH_HOME','HF_HOME',
              'CUDA_CACHE_PATH','TRITON_CACHE_DIR','PIP_CACHE_DIR','JAX_COMPILATION_CACHE_DIR',
              'COLABFOLD_CACHE_DIR','NUMBA_CACHE_DIR')
for key in CACHE_KEYS:
    os.environ[key] = str(OUT / 'cache' / key.lower())
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.80'
os.environ['TF_FORCE_UNIFIED_MEMORY'] = '0'

import collections, fcntl, hashlib, importlib.util, json, re, shutil, signal, subprocess, sys, time
import numpy as np
import pyarrow.parquet as pq


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        while b := f.read(1024 * 1024):
            h.update(b)
    return h.hexdigest()


def write(p, x):
    p = Path(p)
    assert p.is_relative_to('/mnt/d')
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.partial')
    with q.open('w') as f:
        json.dump(x, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    q.replace(p)


def ca(p):
    return np.array([[float(l[a:b]) for a,b in ((30,38),(38,46),(46,54))]
                     for l in Path(p).read_text().splitlines()
                     if l.startswith('ATOM') and l[12:16].strip() == 'CA'])


def alignment(text, target_length, prediction_length):
    scores = re.findall(r'TM-score=\s*([0-9.]+).*normalized by length of (?:Structure|Chain)_([12])', text)
    ts = {int(k):float(v) for v,k in scores}
    lengths = re.findall(r'Length of Structure_([12]):\s*(\d+)', text)
    lengths = {int(k):int(v) for k,v in lengths}
    assert lengths == {1:target_length, 2:prediction_length}, lengths
    m = re.search(r'Aligned length=\s*(\d+),\s*RMSD=\s*([0-9.]+)', text)
    assert m and set(ts) >= {1,2}
    coverage = int(m[1]) / target_length
    rmsd = float(m[2])
    return dict(target_tm_score=ts[1], prediction_tm_score=ts[2], aligned_residues=int(m[1]),
                aligned_rmsd_angstrom=rmsd, target_coverage=coverage,
                prediction_coverage=int(m[1])/prediction_length,
                strong_recovery=ts[1]>=.7 and coverage>=.8 and rmsd<=3)


CIF_READER = r'''
import gzip,io,json,sys
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
d=MMCIF2Dict(io.StringIO(gzip.open(sys.argv[1],'rt').read()))
out={k:v for k,v in d.items() if k.startswith(('_entity_poly.','_pdbx_struct_assembly.',
 '_struct_ref_seq_dif.','_pdbx_unobs_or_zero_occ_residues.','_struct_conn.','_chem_comp.','_struct_ref_seq.'))}
ks=['group_PDB','label_atom_id','auth_asym_id','pdbx_PDB_model_num','auth_seq_id','label_seq_id',
 'pdbx_PDB_ins_code','label_comp_id','label_alt_id','occupancy','Cartn_x','Cartn_y','Cartn_z']
atoms=[]
for i,name in enumerate(d['_atom_site.label_atom_id']):
 if name=='CA' and d['_atom_site.group_PDB'][i]=='ATOM' and d['_atom_site.auth_asym_id'][i]=='A' and d['_atom_site.pdbx_PDB_model_num'][i]=='1':
  atoms.append({k:d['_atom_site.'+k][i] for k in ks})
out['chain_A_CA_atoms']=atoms
print(json.dumps(out))
'''


def audit():
    """Read-only: original mmCIF, two sidecar rows, frozen targets, existing predictions."""
    e7 = read('/mnt/d/Simone/proteinGen/.e014_worktree/e007_matched_fullscale_v1/STATE.json')
    e15 = read(ROOT / 'production/STATE.json')
    assert e7['phase'] == 'paused_epoch1_comparative_evaluation' and e15['phase'].startswith('paused')
    panel = {b['backbone_id']:b for b in read(EXT/'backbone_panel.json')['backbones']}
    oldrows = pq.read_table(EXT/'structural_alignment_results.parquet',use_threads=False).to_pylist()
    oldjobs = read(EXT/'colabfold/jobs.json')
    result = dict(timestamp_unix=time.time(), stage='TARGET_INTEGRITY_ESTABLISHED', controls={},
                  training_paused=True, original_protocol_sha256=sha(EXT/'assessment_protocol.json'),
                  original_jobs_sha256=sha(EXT/'colabfold/jobs.json'),
                  alignment_source_sha256=sha(WORK/'docs/e015_external_designability.py'),
                  native_test_role='Predictor diagnostic, not replacement for designed-sequence five-control gate')
    cols = ['sample_id','sequence','residue_ids','insertion_codes','ca_coordinates','chain_break_mask',
            'selected_calpha_conformer','source_sha256','source_to_npz_calpha_coordinate_rmse_angstrom']
    aa = dict(zip(('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL').split(), 'ARNDCQEGHILKMFPSTWYV'))
    for pdb,shard,rowidx in [('3fm1','000145',362),('2qmi','000141',43)]:
        sidecar = Path('/mnt/d/Users/Simone Stocco/proteinGen_audits/e006_rich_geometry_sidecars_v2/validation') / ('part-'+shard+'.parquet')
        row = pq.read_table(sidecar,columns=cols,use_threads=False).slice(rowidx,1).to_pylist()[0]
        assert row['sample_id']==pdb+'_A'
        source = Path('/mnt/d/Simone/proteinGen/data/full/raw/mmcif')/(pdb+'.cif.gz')
        assert sha(source)==row['source_sha256']
        d = json.loads(subprocess.check_output([CFPY,'-B','-c',CIF_READER,str(source)],text=True,timeout=30))
        atoms = d.pop('chain_A_CA_atoms')
        selected = []
        for rid,icode,alt in zip(row['residue_ids'],row['insertion_codes'],row['selected_calpha_conformer']):
            candidates = [a for a in atoms if a['auth_seq_id']==rid and a['pdbx_PDB_ins_code'] in (icode or '?', '.')]
            if alt:
                candidates = [a for a in candidates if a['label_alt_id']==alt]
            else:
                candidates = [a for a in candidates if a['label_alt_id'] in ('.','?')]
            assert len(candidates)==1,(pdb,rid,candidates)
            selected.append(candidates[0])
        observed_sequence = ''.join(aa[a['label_comp_id']] for a in selected)
        assert observed_sequence==row['sequence']
        full_sequence = ''.join(d['_entity_poly.pdbx_seq_one_letter_code_can'][0].split())
        label_ids = [int(a['label_seq_id']) for a in selected]
        assert ''.join(full_sequence[i-1] for i in label_ids)==observed_sequence
        assert set(full_sequence)<=set('ACDEFGHIKLMNPQRSTVWY')
        xyz = np.array([[float(a[k]) for k in ('Cartn_x','Cartn_y','Cartn_z')] for a in selected])
        b = panel['real_'+pdb+'_A']
        target = np.load(b['source_coordinates'])['coordinates_angstrom']
        targetpdb = ca(b['sanitized_pdb'])
        assert sha(b['sanitized_pdb'])==b['backbone_sha256']
        centered_delta = (xyz-xyz.mean(0))-(target-target.mean(0))
        mapping_rmse = float(np.sqrt(np.mean(np.sum(centered_delta**2,axis=1))))
        assert mapping_rmse<.001 and np.max(np.abs(targetpdb-target))<=.000501
        assert not any(row['chain_break_mask'])
        gaps = [dict(after=label_ids[i],distance=float(dist)) for i,dist in enumerate(np.linalg.norm(np.diff(xyz,axis=0),axis=1)) if dist>4.2]
        assert not gaps
        old = [r for r in oldrows if r['backbone_id']==b['backbone_id']]
        assert len(old)==4
        for r in old:
            job=oldjobs[r['sequence_id']]
            assert job['status']=='COMPLETE'
            for filename,digest in job['files'].items():
                assert sha(filename)==digest
            assert len(ca(r['pdb']))==b['length']
            text=(EXT/'logs'/(r['sequence_id']+'_align.log')).read_text()
            parsed=alignment(text,b['length'],b['length'])
            for k in ('target_tm_score','aligned_ca_rmsd_angstrom','target_coverage'):
                parsed_key='aligned_rmsd_angstrom' if k=='aligned_ca_rmsd_angstrom' else k
                assert abs(parsed[parsed_key]-r[k])<1e-8
            verified_text=subprocess.check_output([str(EXT/'tools/USalign/USalign'),b['sanitized_pdb'],r['pdb'],'-mol','prot'],text=True,timeout=30)
            assert alignment(verified_text,b['length'],b['length'])==parsed
            designed=job['sequence']['sequence']
            r['native_sequence_identity']=sum(x==y for x,y in zip(designed,observed_sequence))/len(observed_sequence)
            sc=read(r['score_json'])
            pae=np.array(sc.get('pae',[]))
            r['pae_mean_angstrom']=float(pae.mean()) if pae.size else None
        missing = [i for i in range(1,len(full_sequence)+1) if i not in label_ids]
        self_text=subprocess.check_output([str(EXT/'tools/USalign/USalign'),b['sanitized_pdb'],b['sanitized_pdb'],'-mol','prot'],text=True,timeout=30)
        self_metrics=alignment(self_text,b['length'],b['length'])
        assert self_metrics['target_tm_score']==1 and self_metrics['aligned_rmsd_angstrom']==0 and self_metrics['target_coverage']==1
        result['controls'][pdb+'_A'] = dict(source=str(source),source_sha256=sha(source),
            deposited_sequence=full_sequence,observed_sequence=observed_sequence,
            deposited_length=len(full_sequence),target_length=len(target),target_pdb=b['sanitized_pdb'],
            target_sha256=b['backbone_sha256'],residue_mapping=label_ids,
            selected_alternate_conformers=[dict(residue=rid,alt=alt) for rid,alt in zip(row['residue_ids'],row['selected_calpha_conformer']) if alt],
            missing_label_residues=missing,internal_chain_breaks=gaps,insertion_codes=sorted(set(row['insertion_codes'])),
            target_source_centered_coordinate_rmse_angstrom=mapping_rmse,
            source_to_npz_reported_rmse_angstrom=row['source_to_npz_calpha_coordinate_rmse_angstrom'],
            target_pdb_rounding_max_angstrom=float(np.max(np.abs(targetpdb-target))),
            alignment_self_check=self_metrics,existing_design_alignment_recomputed=True,
            mmcif_metadata=d,previous_designs=old,
            biological_context=('Monomer; heme, calcium and other ions; four disulfide links. Their causal contribution is not established.' if pdb=='3fm1' else 'Octameric A8; isolated-chain predictor omits interfaces. Lutetium chelate is a crystallographic phasing reagent, not established physiological cofactor.'))
    return result


def prepare():
    a=audit()
    OUT.mkdir(parents=True,exist_ok=True)
    for key in CACHE_KEYS:
        Path(os.environ[key]).mkdir(parents=True,exist_ok=True)
    for name in ('logs','sequences','predictions','images','alignments'):
        (OUT/name).mkdir(exist_ok=True)
    write(OUT/'TARGET_AUDIT.json',a)
    for identity,c in a['controls'].items():
        (OUT/'sequences'/(identity+'.fasta')).write_text('>native_'+identity+'\n'+c['deposited_sequence']+'\n')
        (OUT/'sequences'/(identity+'_observed.fasta')).write_text('>observed_'+identity+'\n'+c['observed_sequence']+'\n')
    write(OUT/'PROTOCOL.json',dict(version='native_sequence_calibration_v1',maximum_predictions=2,
        one_per_control=True,batch_size=1,serialized=True,models=1,model_order=[1],recycles=3,
        random_seed=151502,msa_mode='single_sequence',max_msa='1:1',templates=False,relax=False,
        native_input='Deposited full chain sequence; unresolved target residues omitted only from structural reference',
        coverage_denominator='Number of C-alpha residues in frozen target; separate prediction denominator',
        thresholds_unchanged=True,original_panel_unchanged=True,training_paused=True,
        correction_policy='Any revised prediction protocol requires all five controls evaluated with the unchanged 4/5 validation gate before candidate claims'))
    return a


def host():
    assert read('/mnt/d/Simone/proteinGen/.e014_worktree/e007_matched_fullscale_v1/STATE.json')['phase']=='paused_epoch1_comparative_evaluation'
    assert read(ROOT/'production/STATE.json')['phase'].startswith('paused')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True,timeout=10).strip()
    assert not apps, ('Existing GPU owner',apps)
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        assert not any(n in cmd for n in ('colabfold_batch','protein_mpnn_run.py','run_e015_integrated.py','run_e007_fullscale.py')), ('Heavy worker exists',p.name)
    assert shutil.disk_usage('/mnt/d').free>=30*1024**3


def resource_module():
    sys.path.insert(0,str(WORK/'docs'))
    spec=importlib.util.spec_from_file_location('pinned_external',WORK/'docs/e015_external_designability.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.OUT=OUT
    # Preserve the existing resource monitor and admission thresholds.
    return m


def plot(identity,c,pdb,scores,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=ca(c['target_pdb']);full=ca(pdb);y=full[np.array(c['residue_mapping'])-1]
    xc=x-x.mean(0);yc=y-y.mean(0);u,s,vt=np.linalg.svd(yc.T@xc)
    rot=u@np.diag([1,1,np.linalg.det(u@vt)])@vt
    yp=yc@rot
    metrics['index_mapped_kabsch_rmsd_angstrom']=float(np.sqrt(np.mean(np.sum((xc-yp)**2,axis=1))))
    fig=plt.figure(figsize=(13,8));gs=fig.add_gridspec(2,3)
    extent=float(max(np.max(np.abs(xc)),np.max(np.abs(yp))))
    for j,title in enumerate(('Frozen C-alpha target','Native refold; observed residues','Residue-mapped Kabsch overlay')):
        ax=fig.add_subplot(gs[0,j],projection='3d')
        if j!=1:ax.plot(*xc.T,color='steelblue',lw=1,label='target')
        if j!=0:ax.plot(*yp.T,color='tomato',lw=1,label='native')
        ax.set(title=title,xlim=(-extent,extent),ylim=(-extent,extent),zlim=(-extent,extent));ax.legend(fontsize=7)
    ax=fig.add_subplot(gs[1,0]);pae=np.array(scores['pae']);im=ax.imshow(pae,vmin=0,vmax=30,cmap='magma');fig.colorbar(im,ax=ax,label='PAE Å');ax.set(title='Native full-chain PAE',xlabel='Residue',ylabel='Residue')
    ax=fig.add_subplot(gs[1,1]);ax.plot(scores['plddt']);ax.set(title='Native full-chain pLDDT',ylim=(0,100),xlabel='Residue')
    ax=fig.add_subplot(gs[1,2]);ax.axis('off');ax.text(0,1,f"{identity}: predictor diagnostic\nTM(target)={metrics['target_tm_score']:.5f}\nUS-align RMSD={metrics['aligned_rmsd_angstrom']:.2f} Å\nCoverage={metrics['target_coverage']:.3f}\npLDDT={metrics['mean_plddt']:.2f}; pTM={metrics['ptm']:.2f}\nPAE mean={metrics['pae_mean_angstrom']:.2f} Å\n\nOverlay uses known residue correspondence.\nUS-align metrics allow structural alignment.\nNo experimental or designability claim.",va='top',fontsize=10)
    fig.suptitle('Pinned single-sequence native calibration: '+identity);fig.tight_layout()
    path=OUT/'images'/(identity+'_comparison.png');fig.savefig(path,dpi=160);plt.close(fig)
    metrics['comparison_image']=str(path)


def summarize(a,jobs):
    lines=['Long-control native-sequence calibration v1','',
        'Both E007 and E015 training remain paused. Original 82/62 ledgers, predictions and gates are unchanged.',
        'Inputs are deposited full native sequences: 3FM1 331 aa (target 319; unresolved tail 320-331), 2QMI chain A 447 aa.',
        'Targets match original source chains with verified residue and alternate-conformer correspondence; no internal breaks.',
        '3FM1 has a deposited G191Q database-sequence discrepancy; deposited Q was used, not a reconstructed UniProt substitution.',
        '3FM1 is a monomeric heme/calcium peroxidase with four disulfide links. 2QMI is octameric; lutetium chelate is a phasing reagent.',
        'US-align target normalization and coverage parsing reproduce the existing design results. Native coverage uses target length, not input length.', '']
    diagnoses=[]
    for identity,c in a['controls'].items():
        j=jobs.get(identity,{})
        if j.get('status')!='COMPLETE':
            lines.append(identity+': native result unavailable; diagnosis inconclusive.');continue
        m=j['metrics'];previous=c['previous_designs']
        lines.append(f"{identity}: native TM={m['target_tm_score']:.5f}; RMSD={m['aligned_rmsd_angstrom']:.2f} Å; target coverage={m['target_coverage']:.4f}; pLDDT={m['mean_plddt']:.2f}; pTM={m['ptm']:.2f}; PAE mean/median/p90={m['pae_mean_angstrom']:.2f}/{m['pae_median_angstrom']:.2f}/{m['pae_p90_angstrom']:.2f} Å; strong recovery={m['strong_recovery']}.")
        lines.append('Existing four designs (TM / RMSD Å / coverage / pLDDT / pTM / mean PAE Å):')
        for r in previous:
            lines.append(f"  {r['sequence_id']}: {r['target_tm_score']:.5f} / {r['aligned_ca_rmsd_angstrom']:.2f} / {r['target_coverage']:.4f} / {r['mean_plddt']:.2f} / {r['ptm']:.2f} / {r['pae_mean_angstrom']:.2f}")
        if m['strong_recovery']:
            diagnosis='Evidence favors an inverse-design/sequence-context limitation: the pinned predictor can recover the native fold while all four designed sequences failed. It does not establish that every designed sequence is intrinsically non-designable.'
            if identity=='3fm1_A':diagnosis+=' Full-native input includes a missing 12-residue tail absent from the designs, so sequence truncation/context remains a confound.'
        else:
            diagnosis='Evidence favors a predictor/biological-context limitation or mixed causes: even native sequence fails the unchanged structural recovery criteria. This does not separately establish an inverse-folding defect.'
        diagnoses.append(dict(control=identity,interpretation=diagnosis))
        lines.extend([diagnosis,'Image: '+m['comparison_image'],''])
    complete=len([j for j in jobs.values() if j.get('status')=='COMPLETE'])==2
    next_experiment=('Do not rerun candidate predictions yet. A protocol correction must be preregistered as v2, keep TM>=0.7, coverage>=0.8, RMSD<=3 Å (short-chain extras unchanged), and evaluate all five existing real controls under one common corrected predictor protocol before candidate claims. The minimum gate is four of five passing; passing natives alone is not the ProteinMPNN-to-refolding gate. Reuse the existing 20 designed control sequences, so no ProteinMPNN rerun is required. If native failure occurs, first diagnose only the failed native(s) with a sequence-only MSA-assisted monomer protocol (no target templates); assembly dependence remains untested. If both natives pass, test a preregistered independent four-design panel for each failed backbone before any broader redesign, with the complete five-control gate evaluated prospectively. No revised runs are authorized or launched by this calibration.')
    lines.extend(['Exact next experiment:',next_experiment,'',
                  'Causal limits: two native predictions cannot isolate MSA, cofactors, disulfides, assembly, design quality and unresolved-tail context. No thresholds were changed.',
                  'Original control gate remains TOOLCHAIN_NOT_VALIDATED; calibration does not retroactively validate candidate failures.'])
    path=OUT/'DIAGNOSIS.txt';path.write_text('\n'.join(lines)+'\n')
    write(OUT/'DIAGNOSIS.json',dict(status='COMPLETE' if complete else 'PARTIAL',controls=diagnoses,
        results={k:j.get('metrics') for k,j in jobs.items()},next_experiment=next_experiment,
        original_gate_unchanged=True,training_paused=True))
    write(OUT/'progress.json',dict(stage='COMPLETE' if complete else 'PARTIAL',completed_native_refolds=sum(j.get('status')=='COMPLETE' for j in jobs.values()),maximum_native_refolds=2,controller_pid=os.getpid(),training_paused=True))


def run():
    leases=[]
    for path in (OUT/'CONTROLLER.lock', ROOT/'GPU_WORKER.lock',
                 Path('/mnt/d/Simone/proteinGen/.e014_worktree/e007_matched_fullscale_v1/OWNER.lock'),
                 Path('/mnt/d/Simone/proteinGen/e007_e015_epoch1_comparison_v1/PIPELINE_OWNER.lock')):
        lease=path.open('a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);leases.append(lease)
    host()
    a=read(OUT/'TARGET_AUDIT.json');jobs=read(OUT/'jobs.json') if (OUT/'jobs.json').exists() else {}
    m=resource_module();stop=[False]
    def stopped(signum,frame):
        stop[0]=True
        write(OUT/'STOP_REQUEST.json',dict(signal=signum,controller_pid=os.getpid(),timestamp_unix=time.time()))
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    for identity,c in a['controls'].items():
        if stop[0] or (OUT/'STOP_REQUEST.json').exists():break
        old=jobs.get(identity,{})
        if old.get('status')=='COMPLETE':
            assert all(sha(p)==h for p,h in old['files'].items());continue
        # Never repeat an attempted unit automatically: recover valid output below or require review.
        od=OUT/'predictions'/identity;fa=OUT/'sequences'/(identity+'.fasta')
        ps=list(od.glob('*unrelaxed_rank_001*.pdb'));ss=list(od.glob('*scores_rank_001*.json'))
        if old and not (len(ps)==len(ss)==1 and (od/("native_"+identity+'.done.txt')).exists()):
            raise RuntimeError('Interrupted/failed native unit requires output audit; no automatic repeated prediction: '+identity)
        if not old:
            host();before=m.resource('native_'+identity,heavy=True)
            for key in CACHE_KEYS:os.environ[key]=str(OUT/'cache'/key.lower())
            exe=str(EXT/'tools/colabfold_env/bin/colabfold_batch')
            cmd=[exe,str(fa),str(od),'--msa-mode','single_sequence','--model-type','alphafold2_ptm','--num-models','1','--model-order','1','--num-recycle','3','--recycle-early-stop-tolerance','0.0','--num-seeds','1','--random-seed','151502','--max-msa','1:1','--recompile-padding','0','--disable-unified-memory','--data',str(EXT/'tools/alphafold_weights')]
            od.mkdir(parents=True,exist_ok=True)
            jobs[identity]=dict(status='RUNNING',command=cmd,sequence_sha256=sha(fa),started_unix=time.time(),attempts=1)
            write(OUT/'jobs.json',jobs)
            with (OUT/'logs'/(identity+'.log')).open('a') as log:
                cpus=sorted(os.sched_getaffinity(0))[:2]
                p=subprocess.Popen(cmd,cwd=WORK,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                   preexec_fn=lambda:os.sched_setaffinity(0,cpus))
                jobs[identity]['worker_pid']=p.pid;write(OUT/'jobs.json',jobs)
                write(OUT/'progress.json',dict(stage='NATIVE_REFOLD',control=identity,controller_pid=os.getpid(),worker_pid=p.pid,training_paused=True))
                last=0;started=time.monotonic();cancelled=False
                while p.poll() is None:
                    if stop[0] or (OUT/'STOP_REQUEST.json').exists():
                        if not cancelled:p.send_signal(signal.SIGINT);cancelled=True
                    if time.monotonic()-last>=30:
                        x=m.resource('native_'+identity);last=time.monotonic()
                        owners=[int(line.split(',')[0]) for line in x['gpu_compute_apps'].splitlines() if line.strip()]
                        pressure=x['ram_available_bytes']<3*1024**3 or x['d_free_bytes']<30*1024**3 or (shutil.which('powershell.exe') and x.get('windows_ram_available_bytes',0)<3*1024**3) or any(pid!=p.pid for pid in owners)
                        if pressure and not cancelled:p.send_signal(signal.SIGINT);cancelled=True
                    time.sleep(2)
                rc=p.wait()
            jobs[identity].update(returncode=rc,runtime_seconds=time.monotonic()-started)
            if rc:
                jobs[identity]['status']='INTERRUPTED' if cancelled else 'FAILED';write(OUT/'jobs.json',jobs);break
            ps=list(od.glob('*unrelaxed_rank_001*.pdb'));ss=list(od.glob('*scores_rank_001*.json'))
        assert len(ps)==len(ss)==1
        pred=ca(ps[0]);assert pred.shape==(c['deposited_length'],3) and np.isfinite(pred).all()
        scores=read(ss[0]);assert len(scores['plddt'])==c['deposited_length']
        cfg=read(od/'config.json')
        for key,val in dict(msa_mode='single_sequence',model_type='alphafold2_ptm',num_models=1,num_recycles=3,random_seed=151502,num_seeds=1,use_templates=False,num_relax=0,max_seq=1,max_extra_seq=1,recycle_early_stop_tolerance=0.0).items():assert cfg[key]==val,(key,cfg[key])
        host()
        text=subprocess.check_output([str(EXT/'tools/USalign/USalign'),c['target_pdb'],str(ps[0]),'-mol','prot'],text=True,timeout=60)
        (OUT/'alignments'/(identity+'.log')).write_text(text)
        metrics=alignment(text,c['target_length'],len(pred));pae=np.array(scores['pae'])
        assert pae.shape==(len(pred),len(pred)) and np.isfinite(pae).all()
        metrics.update(mean_plddt=float(np.mean(scores['plddt'])),ptm=float(scores['ptm']),
            pae_mean_angstrom=float(pae.mean()),pae_median_angstrom=float(np.median(pae)),
            pae_p90_angstrom=float(np.quantile(pae,.9)),pae_max_angstrom=float(pae.max()),
            pdb=str(ps[0]),score_json=str(ss[0]),native_length=len(pred),target_length=c['target_length'])
        plot(identity,c,ps[0],scores,metrics)
        files={str(p):sha(p) for p in od.rglob('*') if p.is_file()}
        files[metrics['comparison_image']]=sha(metrics['comparison_image'])
        files[str(OUT/'alignments'/(identity+'.log'))]=sha(OUT/'alignments'/(identity+'.log'))
        for filename in files:
            with open(filename,'rb') as f:os.fsync(f.fileno())
        jobs[identity].update(status='COMPLETE',metrics=metrics,files=files,completed_unix=time.time())
        write(OUT/'jobs.json',jobs)
        print(json.dumps(dict(control=identity,status='COMPLETE',metrics=metrics)),flush=True)
    assert sha(EXT/'colabfold/jobs.json')==a['original_jobs_sha256']
    summarize(a,jobs)


def launch():
    prepare()
    if (OUT/'LAUNCH.json').exists():
        pid=read(OUT/'LAUNCH.json')['controller_pid']
        if Path(f'/proc/{pid}/cmdline').exists():
            print(json.dumps(dict(adopted=True,controller_pid=pid)));return
    if (OUT/'DIAGNOSIS.json').exists() and read(OUT/'DIAGNOSIS.json')['status']=='COMPLETE':
        print('Calibration already complete; no prediction launched.');return
    host()
    with (OUT/'logs/controller.log').open('a') as log:
        p=subprocess.Popen([PY,'-B',str(Path(__file__).resolve()),'run'],cwd=WORK,env=os.environ.copy(),
                           stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(OUT/'LAUNCH.json',dict(controller_pid=p.pid,command=[PY,'-B',str(Path(__file__).resolve()),'run'],
        detached=True,maximum_native_predictions=2,training_paused=True,timestamp_unix=time.time()))
    print(json.dumps(dict(controller_pid=p.pid,root=str(OUT))))


def finalize():
    """CPU-only closeout and prospective protocol; no new predictions."""
    a=read(OUT/'TARGET_AUDIT.json');jobs=read(OUT/'jobs.json')
    assert len(jobs)==2 and all(j['status']=='COMPLETE' and j['attempts']==1 for j in jobs.values())
    for j in jobs.values():
        assert all(sha(p)==h for p,h in j['files'].items())
    assert sha(EXT/'colabfold/jobs.json')==a['original_jobs_sha256']
    assert sha(EXT/'assessment_protocol.json')==a['original_protocol_sha256']
    host()
    proposal=dict(status='PROPOSED_NOT_EXECUTED',version='native_msa_calibration_v2',
        rationale='Both pinned single-sequence native controls failed; test evolutionary sequence information before attributing failures to inverse design or generator quality.',
        stage_1=dict(predictions=2,inputs=['Deposited native 3fm1_A,331 residues','Deposited native 2qmi_A,447 residues'],
            change=['--msa-mode mmseqs2_uniref_env','--max-msa 128:256'],
            unchanged=dict(pinned_colabfold=True,pinned_weights=True,model_type='alphafold2_ptm',
                model_order=[1],models=1,recycles=3,recycle_early_stop_tolerance=0.0,
                num_seeds=1,random_seed=151502,templates=False,relax=False,batch_size=1,
                cuda_workers=1,disable_unified_memory=True,recompile_padding=0),
            provenance='Archive returned MSAs and their hashes, server/database identifiers and retrieval time. No target structure, template, initial guess or ligand coordinates supplied.',
            success='Each native passes unchanged strong target recovery criteria; if either fails, stop this proposal and diagnose that case before running designed-control gate.'),
        stage_2=dict(condition='Both native sentinels recover under the revised protocol',predictions=20,
            inputs='Reuse all existing four ProteinMPNN designs for each of the original five controls; no new sequence design.',
            purpose='Validate the complete inverse-design/refolding pipeline under the common v2 predictor protocol; native success alone does not satisfy this gate.',
            positive_control_gate='At least four of five real backbones reach best target TM>=0.7; report D0.7 and full coverage/RMSD checks as before.',
            strong_recovery='TM>=0.7, coverage>=0.8, RMSD<=3 Å; below80 aa coverage>=0.9 and RMSD<=2.5 Å.',
            candidate_policy='No candidate designability claims before the full five-control evaluation passes. V1 candidate failures remain inconclusive; gate success does not reinterpret their old outputs as v2.',
            sequence_msa_caveat='Record MSA depth for natives, designed controls and eventual candidates; homolog-rich native success does not by itself demonstrate sensitivity for de novo sequences.'),
        if_msa_fails='Assembly-aware 2QMI and cofactor/disulfide-aware 3FM1 remain possible later diagnostic protocols, not causal conclusions or authorized runs.',
        minimum_validated_change='No change is validated by this experiment. MSA addition is the smallest proposed predictor intervention; only a passing complete five-control gate can validate it for subsequent candidate interpretation.')
    write(OUT/'NEXT_EXPERIMENT.json',proposal)
    summary=read(OUT/'DIAGNOSIS.json')
    summary['exact_next_experiment']=proposal;write(OUT/'DIAGNOSIS.json',summary)
    with (OUT/'DIAGNOSIS.txt').open('a') as f:
        f.write('\nProspective v2 specification (not executed):\n'
          'First, one MSA-assisted prediction for each deposited native sequence, changing only MSA mode to mmseqs2_uniref_env and max-msa to128:256; model1, three recycles, seed151502, pinned weights, no templates or relaxation remain fixed. Archive MSAs and provenance.\n'
          'If both natives pass the unchanged strong-recovery criteria, evaluate all20 existing designed control sequences under the same v2 settings, serialized/batch1. Require the original four-of-five gate; no threshold adjustment, replacement controls or candidate reruns. If either native still fails, stop before that20-prediction gate and diagnose context separately.\n'
          'Native MSA recovery may benefit from natural homologs; record depth and do not assume equivalent sensitivity for designed or de novo sequences.\n'
          'No validated corrective change currently exists. Exact prospective settings and stop rules are in NEXT_EXPERIMENT.json.\n')
        f.flush();os.fsync(f.fileno())
    files={str(p):sha(p) for p in [OUT/'TARGET_AUDIT.json',OUT/'PROTOCOL.json',OUT/'jobs.json',OUT/'DIAGNOSIS.json',OUT/'DIAGNOSIS.txt',OUT/'NEXT_EXPERIMENT.json']}
    for j in jobs.values():files.update(j['files'])
    write(OUT/'MANIFEST.json',dict(status='COMPLETE',native_predictions=2,attempts_per_control=1,
        files=files,original_jobs_sha256=a['original_jobs_sha256'],original_protocol_sha256=a['original_protocol_sha256'],
        all_artifacts_on_d=True,training_paused=True,corrected_protocol_not_launched=True))
    print(json.dumps(dict(status='COMPLETE',verified_files=len(files),diagnosis=str(OUT/'DIAGNOSIS.txt'),next_experiment=str(OUT/'NEXT_EXPERIMENT.json'))))


if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='audit':print(json.dumps(audit(),indent=2))
    elif mode=='launch':launch()
    elif mode=='run':run()
    elif mode=='finalize':finalize()
    else:raise ValueError(mode)
