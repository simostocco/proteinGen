"""Exact new-sampler replay check; frozen model, no scientific new sample."""
from pathlib import Path
import runpy,time,torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v93_regional_controllability_20261024_v1');S=runpy.run_path(str(R/'source/scripts/e017_v93/study.py'));assert S['read'](R/'COMPLETED_validation.json')['PASS'];e,_=S['load_engine']('C');source=S['read'](R/'results/TRAIN_C_000.json');before=dict(e.calls);start=time.perf_counter();rows=S['generate'](e,source['task'],'C',16)
for a,b in zip(source['records'],rows):
 for key in ['sequence','geometry','proposals','seed','metrics','trace_logprob','reference_trace_logprob','termination']:assert a[key]==b[key],key
S['verify_engine'](e);S['save']('SAMPLING_REPLAY_VERIFICATION.json',dict(PASS=True,exact_candidates=16,independent_scientific_candidates=0,sequence_geometry_latent_logdensity_branch_and_reward_bit_exact=True,calls={k:e.calls[k]-before.get(k,0) for k in e.calls},proposals=sum(len(r['proposals']) for r in rows),seconds=time.perf_counter()-start,actor_unchanged=True,J1_unchanged=True,peak_GPU_bytes=torch.cuda.max_memory_allocated(),protein_updates=0));print('SAMPLING_BIT_EXACT',flush=True)
