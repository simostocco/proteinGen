"""Authorized storage-only controller amendment after a durable technical pause."""
import ast,copy,hashlib,json,os,time
from pathlib import Path
from protein_joint_sac.rl.checkpointing import load_checked
from protein_joint_sac.data.certified_unified import file_sha
R=Path('/mnt/d/Simone/proteinGen_data/e016_j1_clip_precision_pilot_20261018_v1')
def main():
 done=json.load(open(R/'runs/J1/COMPLETED.json'))
 if done['reason']!='signal/watchdog' or done['pilot_updates']!=1000:raise RuntimeError('only approved technical pause may resume; never override scientific stop')
 pid=json.load(open(R/'STORAGE_STAGING_PAUSE_REQUEST.json'))['pid']
 if Path(f'/proc/{pid}/cmdline').exists():raise RuntimeError('old worker must exit before amendment')
 cp=Path(done['latest_checkpoint']);st=load_checked(cp)
 if st['pilot_update']!=1000 or st['scheduler']['last_epoch']!=1000 or st['sampler']['cursor']!=6000 or {float(v['step']) for v in st['optimizer']['state'].values()}!={36000.}:raise ValueError('pause state counters')
 src=R/'source/scripts/e016_v3/j1_clip_pilot.py';old=src.read_text();new=old
 new=new.replace('def guard():','def guard(staging=210*2**20):',1).replace("if free<10*2**30+210*2**20:raise RuntimeError('10GiB reserve plus210MiB checkpoint staging unavailable')", "if free<10*2**30+staging:raise RuntimeError('10GiB reserve plus requested payload staging unavailable')",1)
 new=new.replace("guard();payload={'lineage':self.lineage", "guard(80*2**20 if model_only else 210*2**20);payload={'lineage':self.lineage",1)
 oldgate="if st['lineage']!=self.lineage:raise ValueError('pilot lineage/config/source mismatch')"
 newgate="""if st['lineage']!=self.lineage:
   receipt=read(R/'STORAGE_STAGING_AMENDMENT.json');amended=copy.deepcopy(st['lineage'])
   if sha(path)!=receipt['checkpoint_sha256'] or amended['entrypoint_sha256']!=receipt['old_entrypoint_sha256'] or self.lineage['entrypoint_sha256']!=receipt['new_entrypoint_sha256'] or self.lineage['source_commit']!=receipt['successor_source_commit']:raise ValueError('unapproved storage migration')
   for key in ['entrypoint_sha256','source_commit']:amended[key]=self.lineage[key]
   if amended!=self.lineage:raise ValueError('storage amendment changed numerical/data/config lineage')"""
 if oldgate not in new:raise ValueError('expected original recovery gate')
 new=new.replace(oldgate,newgate,1)
 oldrun="if existing:e.resume(existing[-1])"
 newrun="""if existing:
  e.resume(existing[-1])
  if (R/'STORAGE_STAGING_AMENDMENT.json').exists() and not (R/'STORAGE_STAGING_RESUME_VERIFICATION.json').exists():
   saved=s.load_checked(existing[-1]);checks={'model':same(saved['model'],e.model.state_dict()),'optimizer':same(saved['optimizer'],e.opt.state_dict()),'scheduler':same(saved['scheduler'],e.scheduler.state_dict()),'sampler':same(saved['sampler'],e.sampler.state_dict()),'RNG':same(saved['rng'],s.rng_state()),'rollout_RNG':same(saved['rollout_rng'],e.rollrng.getstate())}
   if not all(checks.values()):raise ValueError('storage-only no-step restore mismatch')
   e.elapsed=max(e.elapsed,read(R/'STORAGE_PAUSE_COMPLETED.json')['elapsed_seconds_including_evaluation'])
   write(R/'STORAGE_STAGING_RESUME_VERIFICATION.json',{'status':'PASS','checks':checks,'resumed_pilot_update':e.update,'optimizer_updates_for_verification':0,'checkpoint_sha256':sha(existing[-1]),'package_unchanged':saved['lineage']['package_sha256']==e.lineage['package_sha256'],'time':time.time()});del saved"""
 if oldrun not in new:raise ValueError('expected resume call')
 new=new.replace(oldrun,newrun,1)
 new=new.replace("status('PILOT_STOPPED',update=e.update,reason=reason or 'authorized maximum')", "status('PILOT_STOPPED',update=e.update,reason=reason or ('signal/watchdog' if STOP.is_set() else 'authorized maximum'))",1)
 def bodies(code):
  tree=ast.parse(code);result={}
  for node in ast.walk(tree):
   if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in ['__init__','step','backward','generate','assess','calibrate','recovery_test']:result[node.name]=ast.dump(node,include_attributes=False)
  return result
 if bodies(old)!=bodies(new):raise ValueError('mathematical/optimizer function changed')
 receipt={'version':'STORAGE_STAGING_ONLY_V1','time':time.time(),'checkpoint':str(cp),'checkpoint_sha256':file_sha(cp),'old_entrypoint_sha256':file_sha(src),'old_source_commit':st['lineage']['source_commit'],'new_entrypoint_sha256':hashlib.sha256(new.encode()).hexdigest(),'package_sha256_unchanged':st['lineage']['package_sha256'],'config_sha256_unchanged':file_sha(R/'configs/FROZEN.json'),'guard_full_bytes':210*2**20,'guard_model_only_bytes':80*2**20,'hard_reserve_bytes':10*2**30,'AST_unchanged_functions':sorted(bodies(old)),'successor_source_commit':'SET_AFTER_COMMIT_BEFORE_RESUME','training_budget':2500,'scientific_thresholds_changed':False}
 src.write_text(new);(R/'STORAGE_STAGING_AMENDMENT.json').write_text(json.dumps(receipt,indent=2)+'\n')
 os.replace(R/'runs/J1/COMPLETED.json',R/'STORAGE_PAUSE_COMPLETED.json')
 with (R/'EXPERIMENT_DECISIONS.md').open('a') as f:f.write('\nTechnical pause at+1000: measured model-only snapshot52.9MB; unconditional210MiB guard would falsely block later snapshots. New snapshot staging80MiB, full-state staging210MiB;10GiB reserve unchanged. Backward/step/initialization/generation/assessment/calibration/recovery-test ASTs identical. Resume from verified+1000 with exact model/moments/scheduler/sampler/RNG; no additional training budget, objective or scientific threshold changes. Historical checkpoints untouched.\n')
 print('STORAGE_ONLY_AMENDMENT_READY_FOR_COMMIT',flush=True)
if __name__=='__main__':main()
