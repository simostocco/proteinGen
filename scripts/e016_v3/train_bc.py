"""Bounded supervised engineering/pilot entry point, not authorization for a production study."""
import argparse,json,os,time,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--dataset-ref');p.add_argument('--engineering-fixtures',action='store_true');p.add_argument('--resume');p.add_argument('--updates',type=int,default=5)
a=p.parse_args();out=Path(a.output).resolve()
if not str(out).startswith('/mnt/d/'):raise ValueError('D-only output')
if not 1<=a.updates<=100:raise ValueError('bounded pilot max100 updates')
if shutil.disk_usage(out.parent).free<11*2**30:raise RuntimeError('10GiB reserve +1GiB pilot budget')
out.mkdir(exist_ok=False)
for k in ['TMPDIR','TMP','TEMP','SQLITE_TMPDIR','JOBLIB_TEMP_FOLDER','XDG_CACHE_HOME','HF_HOME','TORCH_HOME','PYTHONPYCACHEPREFIX','JAX_COMPILATION_CACHE_DIR','TRITON_CACHE_DIR']:
    d=out/'runtime'/k;d.mkdir(parents=True);os.environ[k]=str(d)
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import torch
from protein_joint_sac.rl.bc_training import BCConfig,Population,BCTrainer
from protein_joint_sac.contracts import DatasetRef
from protein_joint_sac.data.engineering_fixtures import records
from protein_joint_sac.evaluation.bc_metrics import generate,sequence_diversity
c=BCConfig(**json.load(open(a.config)));torch.set_num_threads(1)
if a.engineering_fixtures:
    if c.scope!='ENGINEERING_FIXTURE' or a.dataset_ref:raise ValueError('fixture scope')
    pop=Population(records(),'ENGINEERING_FIXTURE')
else:
    if c.scope!='CERTIFIED_PILOT' or not a.dataset_ref:raise ValueError('certified membership required')
    ref=json.load(open(a.dataset_ref));ref.pop('version',None);pop=Population.certified(DatasetRef(**ref),c.mode,c.population_view)
t=BCTrainer(c,pop)
if a.resume:t.resume(a.resume)
start=time.monotonic();history=[];validations=[];checkpoint_bytes=sum(p.numel() for p in t.model.parameters())*32+4*2**20
# A durable initial state makes even a first-step interruption recoverable.
t.checkpoint(out/f'start_update_{t.update:06d}.pt')
best_path=a.resume;best_score=t.best_validation
try:
    for index in range(a.updates):
        if time.monotonic()-start>=1800:break
        if shutil.disk_usage(out).free<10*2**30+checkpoint_bytes:raise RuntimeError('disk reserve')
        history.append(t.step())
        if (index+1)%5==0 or index+1==a.updates:
            validation=t.validate();validations.append({'update':t.update,**validation})
            cp=out/f'update_{t.update:06d}.pt'
            used=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
            if used+checkpoint_bytes>2**30:raise RuntimeError('persistent pilot artifact budget')
            t.checkpoint(cp)
            if validation['selection_loss']<best_score:best_score=validation['selection_loss'];best_path=str(cp)
except BaseException as exc:
    (out/'FAILURE.json').write_text(json.dumps({'error':repr(exc),'last_completed_update':t.update,'recover_from':'last verified checkpoint; no failed-step state is certified'}))
    raise
validation=t.validate()
samples=[generate(t.model,20,c.mode)] if c.scope=='ENGINEERING_FIXTURE' else []
report={'scope':c.scope,'config':t.lineage['config'],'lineage':t.lineage,'updates':t.update,'history':history,'validation':validation,'validation_history':validations,'selected_checkpoint':best_path,'selection_role':'VALIDATION_ONLY_NO_TEST','generation':samples,'diversity':sequence_diversity([s['sequence'] for s in samples]),'elapsed_seconds':time.monotonic()-start,'scientific_convergence_claim':False,'TEST_used':False}
(out/'RESULT.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps({'updates':t.update,'scope':c.scope,'output':str(out)}))
