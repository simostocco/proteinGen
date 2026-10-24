"""Retrospective TRAIN-only alternatives immediately before observed failures."""
import runpy,time,torch
from pathlib import Path
R=Path('/mnt/d/Simone/proteinGen_data/e017_v92_prefix_feasibility_20261024_v1');d=runpy.run_path(str(R/'source/scripts/e017_v92/study.py'));globals().update({k:d[k] for k in ['OLD','V9','c','m','read','save','load_engine','verify_engine','prefix_status','step_trials']})
def main():
 start=time.perf_counter();goals=read(OLD/'manifests/TRAIN_GOALS.json')['goals'];out=[];completion={}
 for pol in ['A','B','C','D100']:
  e,_=load_engine(pol)
  for gi,goal in enumerate(goals[:6]):
   path=R/f'results/TRAIN_{pol}_U_{gi:03}.json'
   if not path.exists():path=V9/f'results/TRAIN_{"A_J1" if pol=="A" else "B_V3"}_{gi:03}.json'
   accepted=0
   for ci,row in enumerate(read(path)['records']):
    state=c['BuildState'].empty(64)
    for aa,geom in zip(row['sequence'],row['geometry']):
     after=m['append'](state,aa,geom);status=prefix_status(after)
     if not status['valid']:
      result=step_trials(e,pol,goal,state,'retrospective_before_first_failure',16);result.update(source=str(path),candidate_index=ci,original_failed_AA=aa,original_failed_geometry=geom,original_failure_causes=status['irreversible_failures'],selection='retrospective: first2 failing candidates/task; not an unbiased hazard or held-out selector estimate');out.append(result);accepted+=1;break
     state=after
    if accepted==2:break
  verify_engine(e);completion[pol]=dict(calls=dict(e.calls),actor_fingerprint=e.audit_actor_fingerprint,J1_fingerprint=e.jfp,all_frozen=True);del e;torch.cuda.empty_cache()
 save('FAILURE_POINT_ALTERNATIVES.json',dict(records=out,completed=completion,seconds=time.perf_counter()-start,TRAIN_only=True,exploratory_retrospective=True));print('FAILURE_POINTS',len(out),sum(sum(q['valid'] for q in r['trials']) for r in out),len(out)*16,flush=True)
if __name__=='__main__':main()
