from benchlib import *
receipt=json.load(open(R/'results/engineering_recovery/EXPECTED_RESTART.json'));choice=receipt['selected'];c=receipt['config'];path=Path(receipt['checkpoint']);assert file_sha(path)==receipt['sha256'];set_loader(choice['loader']);X.QUALIFIED.setup(pop);prior.shared_prefix_features=choice['share_prefix_features'];X.enable_compiled_rope(choice['compile_actor_rope']);restore(path,system,learner,samplers,c)
with torch.autocast('cuda',dtype=torch.bfloat16):out=X.initialization_update(system,learner,prior,pop,samplers['AA'],samplers['geometry'],samplers['goals'],c)
torch.cuda.synchronize();actual=dict(system=tensor_sha(system),optimizer=state_digest({k:v.state_dict() for k,v in learner.optimizers.items()}),scheduler=state_digest({k:v.state_dict() for k,v in learner.schedulers.items()}),samplers=state_digest({k:v.state_dict() for k,v in samplers.items()}),RNG=state_digest(rng_state()),steps=dict(learner.steps));assert actual==receipt['expected'],(actual,receipt['expected']);prior.verify(learner.optimizers.values());assert inactive==({k:tensor_sha(m) for k,m in system.models.items() if k!='actor'}|{'targets':tensor_sha(system.targets)})
(R/'results/FRESH_GPU_PROCESS_RESTART_TEST.json').write_text(json.dumps(dict(status='PASS',full_state_next_update_bitwise_equal=True,fields=list(actual),actual=actual,transient_checkpoint_sha256=receipt['sha256'],restarted_update=374,scientific_updates=0,loss=out['loss'],J1_and_inactive_tensors_unchanged=True),indent=2));print('FRESH_PROCESS_RESTART_PASS',flush=True)

# One discarded complete update: no attestation/hashing inside profiler region.
try:
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False) as prof:
  with torch.autocast('cuda',dtype=torch.bfloat16):X.initialization_update(system,learner,prior,pop,samplers['AA'],samplers['geometry'],samplers['goals'],c)
  torch.cuda.synchronize()
 (R/'results/OPTIMIZED_TORCH_PROFILE.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=35))
 events=[dict(key=e.key,count=e.count,self_cpu_us=e.self_cpu_time_total,cpu_total_us=e.cpu_time_total,self_device_us=getattr(e,'self_device_time_total',0)) for e in prof.key_averages()]
 (R/'results/OPTIMIZED_TORCH_PROFILE.json').write_text(json.dumps(dict(events=events,complete_discarded_update=True,profile_overhead_excluded_from_promotion_timings=True),indent=2))
except Exception as e:(R/'results/OPTIMIZED_TORCH_PROFILE_STATUS.json').write_text(json.dumps(dict(error=repr(e))))
