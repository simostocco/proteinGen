"""Versioned local ColabFold AF2-pTM integration. Construction/dry-run never launches inference."""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json,subprocess,os,math
from ..contracts import AA,OracleResult,OracleStatus,digest,d_path

@dataclass(frozen=True)
class AF2Backend:
    python:str
    source_dir:str
    code_revision:str
    batch_sha256:str
    weights_dir:str
    weights_sha256:str
    installed_source_seal:str
    installed_source_seal_sha256:str
    recycles:int=3
    model:int=1
    max_msa:str='1:1'
    templates:bool=False
    relaxation:bool=False
    msa_mode:str='single_sequence'
    precision_contract:str='PINNED_COLABFOLD_DEFAULTS_NOT_RUNTIME_VALIDATED'
    def __post_init__(self):
        for k in ['python','source_dir','weights_dir','installed_source_seal']:d_path(getattr(self,k))
        if self.model!=1 or self.recycles<0 or self.max_msa!='1:1' or self.templates or self.relaxation or self.msa_mode!='single_sequence':raise ValueError('unreviewed AF2 protocol change')
    def protocol(self,seed=None):return digest({'backend':'LOCAL_COLABFOLD_AF2_PTM_V1','config':asdict(self),'network':'DENIED','template_input':False})


@dataclass(frozen=True)
class BoundProtocol:
    original:object
    backend:AF2Backend
    @property
    def scope(self):return self.original.scope
    @property
    def seeds(self):return self.original.seeds
    @property
    def protocol_sha256(self):return digest({'declared_protocol':self.original.protocol_sha256,'complete_backend':self.backend.protocol()})


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


class RealAlphaFoldAdapter:
    def __init__(self,protocol,backend=None):
        if backend is None or protocol.backend=='MOCK_UNIT_TEST' or protocol.scope not in ['PLANNED_UNCALIBRATED','CALIBRATED_EXTERNAL']:raise ValueError('explicit real backend/version/weights required; no hidden inference')
        if protocol.weights_sha256!=(backend.weights_sha256,) or protocol.recycles!=backend.recycles or protocol.msa_mode!=backend.msa_mode or protocol.templates_mode!='none' or protocol.code_revision!=backend.code_revision:raise ValueError('backend/protocol disagreement')
        self.protocol=protocol;self.backend=backend
        self.bound_protocol=BoundProtocol(protocol,backend)
    def preflight(self,verify_weights=False):
        b=self.backend;source=Path(b.source_dir)
        actual=subprocess.check_output(['git','--no-optional-locks','-C',str(source),'rev-parse','HEAD'],text=True).strip()
        if actual!=b.code_revision or file_hash(source/'colabfold/batch.py')!=b.batch_sha256:raise ValueError('backend source changed')
        if file_hash(b.installed_source_seal)!=b.installed_source_seal_sha256:raise ValueError('installed-source seal changed')
        seal=json.load(open(b.installed_source_seal));checked=0
        def walk(x):
            nonlocal checked
            if isinstance(x,dict):
                key='installed_path' if 'installed_path' in x else 'path'
                if key in x and 'sha256' in x:
                    if not Path(str(x[key])).is_file() or file_hash(x[key])!=x['sha256']:raise ValueError('installed backend source changed')
                    checked+=1
                else:
                    for v in x.values():walk(v)
            elif isinstance(x,list):
                for v in x:walk(v)
        walk(seal)
        weight=Path(b.weights_dir)/'params/params_model_1_ptm.npz'
        if not Path(b.python).is_file() or not weight.is_file() or not (weight.parent/'download_finished.txt').is_file():raise ValueError('local backend/weights missing; downloads prohibited')
        if verify_weights and file_hash(weight)!=b.weights_sha256:raise ValueError('weights changed')
        return {'source_revision':actual,'installed_source_files_checked':checked,'weights_bytes':weight.stat().st_size,'weights_sha_checked':verify_weights,'inference_executed':False,'protocol_runtime_validated':False}
    def prepare(self,sequence,seed,output):
        if not 20<=len(sequence)<=500 or set(sequence)-set(AA) or seed not in self.protocol.seeds:raise ValueError('complete canonical sequence/seed')
        out=d_path(output);b=self.backend
        argv=[b.python,'-c',"import socket; socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('network disabled')); import colabfold.download; colabfold.download.download_alphafold_params=lambda *a,**k: None; from colabfold.batch import main; main()",str(out/'query.fasta'),str(out/'prediction'),'--model-type','alphafold2_ptm','--model-order','1','--num-models','1','--num-recycle',str(b.recycles),'--recycle-early-stop-tolerance','0','--num-seeds','1','--random-seed',str(seed),'--msa-mode','single_sequence','--max-msa',b.max_msa,'--num-relax','0','--recompile-padding','0','--data',b.weights_dir]
        return {'sequence_sha256':hashlib.sha256(sequence.encode()).hexdigest(),'protocol_sha256':self.bound_protocol.protocol_sha256,'seed':seed,'argv':argv,'output':str(out),'scaffold_as_template':False,'execution_authorized':False}
    def evaluate(self,protein,seed,output,approval=None,*,queue=None,episode=None):
        from ..resources.disk_guard import require_budget
        import fcntl
        plan=self.prepare(protein.sequence,seed,output)
        if not approval or approval.get('protocol_sha256')!=plan['protocol_sha256'] or not approval.get('resource_preflight_approved') or not approval.get('precision_reviewed') or approval.get('approved_calls',0)<1 or len(protein.sequence)>approval.get('length_ceiling',0):raise PermissionError('separately approved bounded AF2 protocol/resource budget required')
        if queue is None or not episode or queue.max_calls>approval['approved_calls']:raise PermissionError('durable queue/call budget required')
        timeout=approval.get('timeout_seconds',0)
        if not 0<timeout<=1800:raise ValueError('bounded oracle timeout')
        row=queue.row(episode)
        if (row['sequence_sha'],row['geometry_sha'],row['protocol_sha'],row['seed'])!=(protein.sequence_sha256,protein.geometry_sha256,plan['protocol_sha256'],seed):raise ValueError('queued identity mismatch')
        if row['status']=='SUCCEEDED':
            data=json.loads(row['result']);data['status']=OracleStatus(data['status']);return OracleResult(**data)
        if Path(output).exists():raise ValueError('immutable oracle output already exists')
        self.preflight(verify_weights=True)
        out=d_path(output);require_budget(out.parent,2*2**30)
        lock_path=d_path(approval['exclusive_gpu_lock']);lock_path.parent.mkdir(parents=True,exist_ok=True)
        with lock_path.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if not queue.claim(episode):
                data=json.loads(queue.row(episode)['result']);data['status']=OracleStatus(data['status']);return OracleResult(**data)
            out.mkdir(exist_ok=False);(out/'query.fasta').write_text('>candidate\n'+protein.sequence+'\n')
            env=os.environ.copy()
            for k in ['TMPDIR','TMP','TEMP','SQLITE_TMPDIR','JOBLIB_TEMP_FOLDER','XDG_CACHE_HOME','HF_HOME','TORCH_HOME','JAX_COMPILATION_CACHE_DIR','TRITON_CACHE_DIR','PYTHONPYCACHEPREFIX']:
                q=out/'runtime'/k;q.mkdir(parents=True,exist_ok=True);env[k]=str(q)
            env['PYTHONDONTWRITEBYTECODE']='1'
            try:
                with (out/'backend.log').open('x') as log:
                    subprocess.run(plan['argv'],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,check=True)
                pdbs=list((out/'prediction').glob('*_unrelaxed_rank_001_*.pdb'));scores=list((out/'prediction').glob('*_scores_rank_001_*.json'))
                if len(pdbs)!=1 or len(scores)!=1:raise ValueError('unambiguous ranked output pair required')
                result=import_saved_result(protein.sequence,pdbs[0],scores[0],plan['protocol_sha256'],seed)
            except subprocess.TimeoutExpired:
                result=OracleResult(protein.sequence_sha256,plan['protocol_sha256'],OracleStatus.TIMEOUT,seed,'EXTERNAL_PREDICTION',failure_reason='bounded backend timeout')
            except Exception as exc:
                result=OracleResult(protein.sequence_sha256,plan['protocol_sha256'],OracleStatus.FAILED,seed,'EXTERNAL_PREDICTION',failure_reason=type(exc).__name__+': '+str(exc))
            queue.finish(episode,result)
            (out/'RESULT.json').write_text(json.dumps(asdict(result),indent=2,allow_nan=False))
            return result


def import_saved_result(sequence,pdb_path,scores_path,protocol_sha,seed):
    """Independent saved-output parsing; validates full residue correspondence, never folds."""
    import gemmi
    if not 20<=len(sequence)<=500 or set(sequence)-set(AA):raise ValueError('canonical sequence')
    structure=gemmi.read_structure(str(d_path(pdb_path)))
    if len(structure)!=1 or len(structure[0])!=1:raise ValueError('one full monomer model/chain required')
    residues=list(structure[0][0]);coords=[];observed=''
    for pos,res in enumerate(residues,1):
        if res.seqid.num!=pos or res.seqid.icode.strip():raise ValueError('full ordered residue mapping')
        observed+=gemmi.find_tabulated_residue(res.name).one_letter_code
        ca=[a for a in res if a.name=='CA']
        if len(ca)!=1:raise ValueError('missing/ambiguous predicted CA')
        coords.append((ca[0].pos.x,ca[0].pos.y,ca[0].pos.z))
    if observed!=sequence:raise ValueError('predictor sequence mismatch')
    scores=json.load(open(d_path(scores_path)));plddt=scores.get('plddt');ptm=scores.get('ptm');pae=scores.get('pae');n=len(sequence)
    if pae is not None and (len(pae)!=n or any(len(row)!=n or any(not math.isfinite(x) or x<0 for x in row) for row in pae)):raise ValueError('PAE mapping/finite values')
    h=digest({'pdb_sha':file_hash(pdb_path),'scores_sha':file_hash(scores_path)})
    return OracleResult(hashlib.sha256(sequence.encode()).hexdigest(),protocol_sha,OracleStatus.SUCCEEDED,seed,'EXTERNAL_PREDICTION',h,tuple(coords),tuple(plddt) if plddt is not None else None,ptm,tuple(tuple(row) for row in pae) if pae is not None else None)


def process_queued_saved(queue,episode,protein,result):
    from .fold_agreement import compare_completed
    if result.sequence_sha256!=protein.sequence_sha256:raise ValueError('saved result sequence identity')
    row=queue.row(episode)
    if row['geometry_sha']!=protein.geometry_sha256:raise ValueError('proposal identity')
    # Queue.finish verifies protocol/seed/evidence and keeps immutable prediction cache.
    agreement=compare_completed(protein,result)
    queue.finish(episode,result)
    return agreement
