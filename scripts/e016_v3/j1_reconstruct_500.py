"""Exactly500 historical optimizer updates solely to reconstruct selected state."""
import json,time,gc,math,hashlib,subprocess
from pathlib import Path
import torch
import historical_clip_replay as h
R=Path(__file__).resolve().parents[3];O=R.parent/'e016_j1_clip_precision_pilot_20261018_v1'
h.R=R;h.s.R=R;h.s.j.R=R;h.s.j.core.R=R

def run():
 if (R/'replay/RECONSTRUCTION.json').exists():raise RuntimeError('preserve completed replay; no automatic repeat')
 receipt=json.load(open(R/'APPROVED_RETIREMENT_MANIFEST.json'))
 if receipt['status']!='COMPLETE':raise ValueError('retirement incomplete')
 h.s.j.core.setup();h.guard();h.status('REPLAY_DATA_LOADING');data=h.s.PartialData();h.write(R/'REPLAY_DATA_RECEIPT.json',data.receipt);c=json.load(open(O/'configs/FROZEN.json'))
 assert h.sha(Path(h.__file__))=='42fe5fd3d2389ac85f250dc7d61756e13b094d0cb02f87732acfba1347e364ba'
 e=h.Engine(data,c);old=[]
 for p in sorted((O/'runs/J1').glob('updates_*.jsonl')):
  old.extend(json.loads(x) for x in open(p) if x.strip())
 old={v['pilot_update']:v for v in old};comparisons=[];start=time.monotonic()
 with (R/'replay/UPDATES.jsonl').open('x') as log:
  for k in range(500):
   v=e.step();ref=old[e.update];checks={'sample_order':v['sample_order_sha256']==ref['sample_order_sha256']};errors={key:abs(v[key]-ref[key]) for key in ['AA_NLL_stochastic','local_geometry_NLL','AA_shared_L2','geometry_shared_scale']};comparisons.append({'update':e.update,'checks':checks,'errors':errors});log.write(json.dumps(v)+'\n')
   if e.update%25==0:log.flush();h.status('HISTORICAL_REPLAY',update=e.update,seconds=time.monotonic()-start);print(e.update,flush=True)
   if e.update==250:e.checkpoint(R/'replay/checkpoint_00250.pt')
 selected=h.s.load_checked(O/'runs/J1/model_00500.pt');expected=selected['model'];actual=e.model.state_dict();exact=expected.keys()==actual.keys() and all(torch.equal(expected[n],actual[n].detach().cpu()) for n in expected)
 maxerr=max(float((expected[n].double()-actual[n].detach().cpu().double()).abs().max()) for n in expected)
 statechecks={'optimizer_step':{int(float(v['step'])) for v in e.opt.state.values()}=={35500},'scheduler_epoch':e.scheduler.last_epoch==500,'sampler_cursor':e.sampler.cursor==5500,'all_history_sample_orders_match':all(v['checks']['sample_order'] for v in comparisons),'all_history_numeric_diagnostics_match':all(max(v['errors'].values())==0 for v in comparisons)}
 out={'status':'EXACT_RECONSTRUCTION' if exact and all(statechecks.values()) else 'REPLAY_MISMATCH_NEW_OPTIMIZER_REQUIRED','model_tensors_bitwise_equal':exact,'maximum_parameter_absolute_difference':maxerr,'state_checks':statechecks,'historical_optimizer_updates_replayed':500,'new_continuation_updates':0,'source_full_checkpoint':str(h.SOURCE),'source_sha256':h.SOURCE_SHA,'selected_model':str(O/'runs/J1/model_00500.pt'),'selected_sha256':h.sha(O/'runs/J1/model_00500.pt'),'historical_entrypoint_sha256':h.sha(Path(h.__file__)),'historical_revision':'2de6e21e81f1f34b912fca6bd1b2e2efe5b9fb6f','elapsed_seconds':time.monotonic()-start,'numeric_comparison':comparisons,'optimizer_sampler_rng_equality_limitation':'no historical full500 exists to compare these tensors directly; exact pinned trajectory, matching recorded updates and final model establish reconstruction; completeness/counters verified'}
 if out['status']=='EXACT_RECONSTRUCTION':
  e.history=selected['history'];e.best=500;e.checkpoint(R/'replay/checkpoint_00500_full.pt');loaded=h.s.load_checked(R/'replay/checkpoint_00500_full.pt');checks={'model':h.same(loaded['model'],e.model.state_dict()),'optimizer':h.same(loaded['optimizer'],e.opt.state_dict()),'scheduler':h.same(loaded['scheduler'],e.scheduler.state_dict()),'sampler':h.same(loaded['sampler'],e.sampler.state_dict()),'RNG':h.same(loaded['rng'],h.s.rng_state())}
  if not all(checks.values()):raise ValueError('reconstructed checkpoint load equality failed')
  out.update(checkpoint=str(R/'replay/checkpoint_00500_full.pt'),checkpoint_sha256=h.sha(R/'replay/checkpoint_00500_full.pt'),atomic_load_checks=checks)
 h.write(R/'replay/RECONSTRUCTION.json',out)
 (R/'OPTIMIZER_STATE_RECONSTRUCTION_REPORT.md').write_text('# Selected+500 reconstruction\n\n'+out['status']+f".500 historical updates replayed,0 new continuation updates. Model maximum error{maxerr}; exact{exact}. Completeness/counter checks{statechecks}. Parent, exact historical code, all update comparisons and full checkpoint hash in replay/RECONSTRUCTION.json. No later optimizer moments mixed with selected weights.\n")
 print(out['status'],flush=True)
if __name__=='__main__':run()
