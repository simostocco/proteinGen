"""Validate amended actual-engine restart; no main updates or overwrite."""
import copy,gc,json,time
import j1_sequence_first as s
from j1_sequence_first import *
def main():
 j.core.setup();data=PartialData();c=json.load(open(R/'configs/FROZEN.json'));p=R/'runs/J1/checkpoint_03482.pt';e=Engine(data,c);e.resume(p)
 assert torch.are_deterministic_algorithms_enabled() and torch.backends.cudnn.deterministic
 expected=[]
 for i in range(16):expected.append(e.step(force_stage=3,force_horizon=8) if i==15 else e.step())
 mv=cpu_copy(e.model.state_dict());ov=cpu_copy(e.opt.state_dict());sam=e.sampler.state_dict();sch=copy.deepcopy(e.scheduler.state_dict());rng=rng_digest();roll=e.rollrng.getstate()
 del e;gc.collect();torch.cuda.empty_cache();e=Engine(data,c);e.resume(p);actual=[]
 for i in range(16):actual.append(e.step(force_stage=3,force_horizon=8) if i==15 else e.step())
 pe=maxdiff(mv,e.model.state_dict());oe=maxdiff(ov,e.opt.state_dict());same=sam==e.sampler.state_dict() and sch==e.scheduler.state_dict() and rng==rng_digest() and roll==e.rollrng.getstate()
 errors={k:max(abs(a.get(k,0)-b.get(k,0)) for a,b in zip(expected,actual)) for k in ['AA_NLL_stochastic_batch','local_geometry_NLL','nonlocal_energy_score','rollout_rg','rollout_clash','rollout_pair_energy']}
 assert same and pe==0 and oe==0 and max(errors.values())==0
 out={'status':'PASS','updates_per_path':16,'engineering_optimizer_calls':32,'parameter_max_error':pe,'optimizer_max_error':oe,'loss_errors':errors,'sampler_scheduler_RNG_equal':same,'entrypoint_sha256':file_sha(s.__file__),'starting_checkpoint_sha256':file_sha(p),'runtime_deterministic':torch.are_deterministic_algorithms_enabled()}
 write(R/'AMENDED_ENGINE_RESTART.json',out);print('AMENDED_ENGINE_RESTART_PASS',flush=True)
 # Benchmark full two-path global objective under the actual deterministic execution policy.
 bench=[]
 for H in [64,128]:
  h=next(h for h in data.base.jointids['TRAIN'] if len(data.seqs[h])>=H+32);o=data.base.owners[h][0];t=16
  st=load_checked(p);e.model.load_state_dict(st['model'],strict=True);del st
  ref=torch.tensor(data.base.ca[o][:t+H],device='cuda');ids=torch.tensor([j.AA.index(a) for a in data.seqs[h][:t+H]],device='cuda')
  e.model.train();e.opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();q=time.monotonic()
  r=generated_bundle(e.model,ref[:t],ids,len(data.seqs[h]),H,ref);loss=c['lambda_Rg']*r['Rg']+c['lambda_clash']*r['clash']+c['lambda_pair']*r['pair'];loss.backward();torch.cuda.synchronize()
  row={'H':H,'full_gradient_paths':2,'truncated':False,'seconds':time.monotonic()-q,'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved(),'geometry_grad':float(e.model.geom[-1].weight.grad.norm()),'Transformer_grad':float(e.model.encoder.blocks[0].qkv.weight.grad.norm()),'finite':all(p.grad.isfinite().all() for p in e.model.parameters() if p.grad is not None)}
  assert row['finite'];bench.append(row);write(R/'DETERMINISTIC_LONG_ROLLOUT_BENCHMARKS.json',bench)
  del r,loss,ref,ids;gc.collect()
 e.resume(p);real=e.step(force_stage=3,force_horizon=64)
 assert real['geometry_to_AA_shared_ratio']<=.500001
 write(R/'REAL_H64_AA_GRADIENT_CAP_UPDATE.json',{'status':'PASS','engineering_optimizer_calls':1,'main_calls':0,'metrics':real,'geometry_head_grad_finite':bool(e.model.geom[-1].weight.grad.isfinite().all())})
 gate=json.load(open(R/'PREFLIGHT_PASS.json'));write(R/'PREFLIGHT_PASS_SERIALIZATION_REPAIR.json',gate)
 gate['entrypoint_sha256']=file_sha(s.__file__);gate['numerical_amendment_sha256']=file_sha(R/'NUMERICAL_EXECUTION_AMENDMENT.json');gate['deterministic_restart_sha256']=file_sha(R/'AMENDED_ENGINE_RESTART.json');write(R/'PREFLIGHT_PASS.json',gate)
 print('DETERMINISTIC_ACTUAL_ENGINE_AND_LONG_ROLLOUT_PASS',flush=True)
if __name__=='__main__':main()
