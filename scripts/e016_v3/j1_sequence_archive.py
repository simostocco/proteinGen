"""Bounded CPU-only milestone weight preservation; never edits primary checkpoints."""
import json,shutil,time
from pathlib import Path
from protein_joint_sac.rl.checkpointing import load_checked,save_atomic
from protein_joint_sac.data.certified_unified import file_sha
R=Path('/mnt/d/Simone/proteinGen_data/e016_j1_sequence_first_20261018_v1')
def archive_model(source,dest):
 source=Path(source);dest=Path(dest)
 if dest.exists():
  return file_sha(dest)
 state=load_checked(source)
 if shutil.disk_usage(dest.parent).free<10*2**30+500*2**20:raise RuntimeError('reserve plus concurrent staging')
 payload={'model':state['model'],'lineage':state['lineage'],'additional_update':state['additional_update'],'source_full_checkpoint_sha256':file_sha(source),'kind':'MODEL_ONLY_MILESTONE_NOT_FULL_OPTIMIZER_RESTART'}
 return save_atomic(dest,payload)
def main():
 d=R/'candidates';d.mkdir(exist_ok=True)
 with (R/'logs/milestone_archiver.jsonl').open('a',buffering=1) as log:
  while True:
   for u in json.load(open(R/'configs/FROZEN.json'))['milestones']:
    source=R/'runs/J1'/f'checkpoint_{u:05}.pt';dest=d/f'model_{u:05}.pt';existing=R/'runs/J1'/f'stage_model_{u:05}.pt'
    if dest.exists() or existing.exists() or not source.exists():continue
    try:
     sha=archive_model(source,dest);log.write(json.dumps({'update':u,'model_sha256':sha,'source':str(source),'source_sha256':file_sha(source),'path':str(dest),'scope':'model-only; exact latest/best/numerical-boundary optimizer states retained by main controller'})+'\n')
    except FileNotFoundError:log.write(json.dumps({'update':u,'status':'source retired before copy; keep receipt; no fabricated snapshot'})+'\n')
   if (R/'runs/J1/COMPLETED.json').exists() or (R/'ARCHIVER_STOP').exists():return
   time.sleep(15)
if __name__=='__main__':main()
