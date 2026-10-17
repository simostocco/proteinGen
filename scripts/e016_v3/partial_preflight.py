"""Freeze view/configuration and test actual sampled continuation recovery before main runs."""
import copy,json,math,resource,shutil,subprocess,time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import scaled_bc as core
from partial_train import Engine,R,OLD,write,health
from protein_joint_sac.data.partial_continuation import PartialData
from protein_joint_sac.data.certified_unified import file_sha
from protein_joint_sac.rl.bc_training import rng_state,restore_rng
from scaled_train import cpu_copy,maxdiff,rng_digest
from protein_joint_sac.contracts import digest

def main():
 core.setup();start=time.monotonic();census=json.load(open(R/'PARTIAL_PDB_ELIGIBILITY_COUNTS.json'))
 if not census['splits']['TRAIN']['additional_to_complete_identities'] or not census['splits']['VALIDATION']['identities']:raise ValueError('no meaningful expanded population')
 if json.load(open(R/'tests/GPU_MASK_PROOF.json'))['status']!='PASS':raise ValueError('mandatory GPU mask proof')
 # Enrich reference-only view; preserve the original recovery table.
 p=R/'view/PARTIAL_OBSERVATIONS.parquet';raw=p.with_name('RECOVERED_PARTIAL_OBSERVATIONS.parquet')
 if not raw.exists():p.replace(raw)
 rows=pq.read_table(raw).to_pylist()
 for r in rows:
  r['polymer_position']=list(range(1,r['polymer_length']+1));r['source_model_id']=r['model'];r['coordinate_quality_flags']=['VERIFIED_FULL_POLYMER_MAPPING','ONLY_POSITIVELY_VERIFIED_C_N_LINKS_ADMITTED','MISSING_CA_DISTINCT_FROM_UNVERIFIED_LINK','OBSERVED_OUTLIERS_NOT_SILENTLY_MISSING'];r['altloc_provenance']=r['coordinate_path']+'#/row_or_observation/'+str(r['coordinate_row'])+'/selected_conformer';r['coordinate_frame_scope']='same source chain model; all observed coordinates retain common experimental frame'
 pq.write_table(pa.Table.from_pylist(rows),p,compression='zstd');del rows
 view={'version':'partial-supervised-view-v1','parent_split_manifest_sha256':census['split_manifest_sha256'],'source_table_sha256':census['intake_table_sha256'],'artifacts':{str(p.relative_to(R)):file_sha(p) for p in [R/'view/PARTIAL_OBSERVATIONS.parquet',R/'view/SOURCE_HASHES.json',R/'view/QUARANTINE.json',R/'PARTIAL_PDB_ELIGIBILITY_COUNTS.json']},'policy':'known canonical full parent; inherited split; unambiguous mapped CA/conformer; original method-resolution policy; C-N continuity reused from exact source/model/author/sequence/coordinate matching or authoritative v3 mapping; no raw reparse','TEST':'excluded from admission, selection and evaluation; bulk source batches can contain unselected TEST rows','coordinates_duplicated':False,'biology':'deposited polymer, natural gene termini UNKNOWN'}
 write(R/'view/VIEW_FROZEN.json',view);data=PartialData();write(R/'view/DATA_LOADER_RECEIPT.json',data.receipt)
 train_geometry_rate=float(np.mean([np.mean([data.cache[o]['mask'][:,0].mean() for o in data.owners['TRAIN'][h]]) for h in data.ids['TRAIN']]))
 panels={'complete':data.panel(False),'partial':data.panel(True)};write(R/'configs/VALIDATION_PANELS.json',panels)
 cfg={}
 for arm in ['J1-FULL','J1-PARTIAL','J2-FULL','J2-PARTIAL']:
  name=arm[:2];cfg[arm]={'version':'partial-continuation-v1','name':name,'arm':arm,'width':352,'layers':8,'heads':8,'ffn_mult':4,'parameters':12351746,'effective_batch':64,'live_batch':16,'precision':'BF16','geometry_projection_fp32':True,'activation_checkpoint':False,'max_updates':20000,'warmup_updates':100,'learning_rate':1e-4,'terminal_learning_rate':1e-5,'weight_decay':.01,'clip_norm':1.,'geometry_weight':1.,'geometry_normalization':'per-decision valid-component mean; zero neutral geometry contribution for empty mask; average over64 decisions','optimizer_moments':'restore original10k AdamW; restart scheduler only','sampler':'continue parent complete identity stream; PARTIAL exactly32complete+32partial decisions; uniform partial identity then observation','partial_geometry_oversample':bool(arm.endswith('PARTIAL') and train_geometry_rate<.5),'partial_position_policy':'50percent uniform geometric positions +50percent all positions' if train_geometry_rate<.5 else 'uniform all polymer positions','TRAIN_identity_weighted_partial_distance_target_rate':train_geometry_rate,'seed':42,'parent_sha256':{'J1':'b2c6aae57fa3cd0cf71abf07423b8480b6e216a0f33b56e1c67b79d18c692af2','J2':'06f29a590228ef75e9a5a0ad6274c4ead0dc91c725e9f5c94137ffac2faff7a4'}[name],'selection':'minimum common complete-structure VALIDATION AA+geometry loss, including original initialization; never TEST','early_stopping':'three consecutive milestones worse than earlier best by>1e-4','TEST_used':False,'checkpoint_retention':'new-task best+latest only, preserve all milestone metrics/SHA and deletion receipts; historical roots never touched'}
 write(R/'configs/CONTINUATION_FROZEN.json',cfg)
 # Cost all milestone states plus atomic staging, the actual existing engineering checkpoint,
 # logs and final generation; no historical deletion/cleanup is a resource strategy.
 budget={'D_free_before_continuations':shutil.disk_usage(R).free,'planned_milestone_states':8,'upper_bytes_per_state':170*2**20,'atomic_staging_bytes':340*2**20,'remaining_logs_reports_and_generation_bytes':200*2**20,'hard_reserve_bytes':10*2**30};budget['projected_remaining_peak_bytes']=8*budget['upper_bytes_per_state']+budget['atomic_staging_bytes']+budget['remaining_logs_reports_and_generation_bytes']
 write(R/'RESOURCE_BUDGET_FROZEN.json',budget)
 if budget['D_free_before_continuations']<budget['hard_reserve_bytes']+budget['projected_remaining_peak_bytes']:raise RuntimeError('continuation peak does not fit measured disk reserve')
 native_quality=[]
 for oid,r in data.cache.items():
  bad=np.flatnonzero(r['mask'][:,0]&((r['g'][:,0]<2)|(r['g'][:,0]>10)))
  if len(bad):native_quality.append({'observation_id':oid,'sequence_hash':data.rows[oid]['sequence_hash'],'split':data.rows[oid]['split'],'polymer_positions':(bad+1).tolist(),'CA_distances_A':r['g'][bad,0].tolist(),'action':'RETAINED_ACTUALLY_OBSERVED_NOT_MASKED_AS_MISSING'})
 write(R/'view/OBSERVED_GEOMETRY_DIAGNOSTICS.json',native_quality)
 profile=[];eng=Engine('J2-PARTIAL',data,cfg['J2-PARTIAL'])
 for N in [64,128,256,384,500]:
  choices=[(h,o,N-1) for h in data.ids['TRAIN'] if len(data.seqs[h])>=N for o in data.owners['TRAIN'][h][:1]]
  full=[(h,data.base.owners[h][0],N-1) for h in data.base.jointids['TRAIN'] if len(data.seqs[h])>=N]
  if not choices or not full:raise ValueError('representative real prefix missing at length'+str(N))
  ex=[full[i%len(full)] for i in range(8)]+[choices[i%len(choices)] for i in range(8)];b,y,g=data.batch(ex);torch.cuda.reset_peak_memory_stats();times=[]
  for rep in range(3):
   eng.opt.zero_grad(set_to_none=True);q=time.monotonic();v,a,geo,parts=core.loss_call(eng.model,b,y,g,eng.c);v.mean().backward();torch.cuda.synchronize();times.append(time.monotonic()-q)
   if not all(torch.isfinite(p.grad).all() for p in eng.model.parameters() if p.grad is not None):raise FloatingPointError('profile gradient nonfinite')
  profile.append({'target_length_min':N,'literal_prefix_length':N-1,'real_examples':16,'timed_forward_backward_seconds':times[1:],'valid_geometry_components':b['geometry_target_valid'].sum(0).tolist(),'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()});del b,y,g,v,a,geo,parts
 write(R/'tests/REAL_PARTIAL_GPU_PROFILE.json',profile)
 first=eng.step();cp=R/'tests/CONTINUATION_PREFLIGHT.pt';eng.checkpoint(cp);expected=eng.step();weights=cpu_copy(eng.model.state_dict());opt=cpu_copy(eng.opt.state_dict());sched=copy.deepcopy(eng.scheduler.state_dict());sampler=copy.deepcopy(eng.sampler.state_dict());rng=rng_digest();eng.resume(cp);actual=eng.step();pe=maxdiff(weights,eng.model.state_dict());oe=maxdiff(opt,eng.opt.state_dict());same=expected['sample_order_sha256']==actual['sample_order_sha256'] and sampler==eng.sampler.state_dict() and sched==eng.scheduler.state_dict() and rng==rng_digest()
 receipt={'status':'PASS' if same and pe<2e-5 and oe<1e-4 else 'FAIL','model_parameter_max_abs_error':pe,'optimizer_max_abs_error':oe,'sampler_scheduler_rng_and_examples_equal':same,'AA_loss_abs_error':abs(expected['AA_NLL']-actual['AA_NLL']),'geometry_loss_abs_error':abs(expected['geometry_NLL']-actual['geometry_NLL']),'scope':'ACTUAL_PAIRED_PARTIAL_SAMPLER_12M_GPU_CONTINUATION_ENGINEERING_ONLY','resources':health()};write(R/'tests/CONTINUATION_RESTART.json',receipt)
 if receipt['status']!='PASS':raise RuntimeError('actual continuation restart failed')
 write(R/'PREFLIGHT_PASS.json',{'status':'PASS','config_sha256':file_sha(R/'configs/CONTINUATION_FROZEN.json'),'view_sha256':file_sha(R/'view/VIEW_FROZEN.json'),'CPU_mask_results':'tests/MASK_CPU_FINAL.xml','GPU_mask_results':'tests/GPU_MASK_PROOF.json','actual_sampler_restart':'tests/CONTINUATION_RESTART.json','validation_complete_identities':len({h for h,o,t in panels['complete']}),'validation_partial_identities':len({h for h,o,t in panels['partial']}),'TEST_used':False,'seconds':time.monotonic()-start});print('PREFLIGHT_PASS',flush=True)
if __name__=='__main__':main()
