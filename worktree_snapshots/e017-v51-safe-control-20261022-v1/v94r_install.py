"""Install isolated corrected controller, preserving historical code unchanged."""
from pathlib import Path
import json,shutil
R=Path('/mnt/d/Simone/proteinGen_data/e017_v94r_packing_safety_20261025_v1');O=R.parent/'e017_v94_fixed_sequence_geometry_rl_20261024_v1';V=R.parent/'e017_v93_regional_controllability_20261024_v1';D=R/'source/scripts/e017_v94r'
D.mkdir(exist_ok=True)
for src,name in [('v94r_packing_staged.py','packing.py'),('v94r_tests_staged.py','tests.py')]:shutil.copyfile(src,D/name)
common='''import collections,copy,math,runpy,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v94r_packing_safety_20261025_v1')
V=R.parent/'e017_v93_regional_controllability_20261024_v1'
B=runpy.run_path(str(V/'source/scripts/e017_v93/study.py'))
globals().update({k:B[k] for k in ['c','p','T','AA','SPEC','read','sha','ns','world','append','load_engine','reconstruct','prefix_status']})
F=runpy.run_path(str(R/'source/scripts/e017_v94r/packing.py'));extend_checked=F['extend_checked']
def guard():
 assert __import__('shutil').disk_usage(R).free>2*2**30+2*224*2**20+128*2**20
 assert not (R/'control/STOP').exists()
 if torch.cuda.is_initialized():assert torch.cuda.max_memory_allocated()<6*2**30
START=time.perf_counter();COUNTS=collections.Counter()
'''
old=(V/'source/scripts/e017_v93/study.py').read_text();sampler=old[old.index('@torch.no_grad()'):old.index('\ndef group(')]
sampler=sampler.replace('geometry_KL=[],repairs=0','geometry_KL=[],repairs=0,packing_prefix=[]')
sampler=sampler.replace("w.state=after if after is not None else append(w.state,seq[t],geo);rec['geometry'].append(geo);accepted=True;break", "w.state=after if after is not None else append(w.state,seq[t],geo);rec['geometry'].append(geo);accepted=True;packing=F['prefix_budget'](w.state);rec['packing_prefix'].append([t,packing['contacts'],packing['slack_pairs']]);break")
(D/'sampling.py').write_text(common+sampler)
old=(V/'source/scripts/e017_v93/campaign.py').read_text();grad=old[old.index('def gradients('):old.index('\ndef cpu(')]
grad=grad.replace('maxerr=0.;probes=[]','maxerr=0.;probes=[];loss_value=0.;entropy_sum=0.;entropy_dims=0')
grad=grad.replace("value=T['base_logprob']", "value=T['base_logprob']")
grad=grad.replace('torch.stack(loss).sum().backward()',"node_loss=torch.stack(loss).sum();loss_value+=float(node_loss.detach());node_loss.backward()\n   with torch.no_grad():\n    for j,i in enumerate(ids):\n     mask=torch.tensor(c['mask_for_step'](t),device='cuda');a=AA.index(task['fixed_sequence'][t]);entropy_sum+=float(((ls[j,a].double().clamp(-5,2)+.5*math.log(2*math.pi*math.e))*mask).sum());entropy_dims+=int(mask.sum())")
grad=grad.replace('return rewards,maxerr,probes',"e.last_gradient_diagnostics=dict(policy_gradient_surrogate=loss_value,augmented_advantage_variance=float(adv.var()),reward_variance=float(rewards.var()),latent_Gaussian_entropy_per_active_dimension=entropy_sum/max(1,entropy_dims),entropy_is_not_filtered_output_entropy=True,group_size=len(rows),trajectory_logprob_reduction='sum all sampled raw-latent proposal scores, mean trajectories only',AA_score_terms=0)\n return rewards,maxerr,probes")
(D/'gradients.py').write_text(common+'\n'+grad)
s=(O/'source/scripts/e017_v94/production.py').read_text().replace(str(O),str(R)).replace('scripts/e017_v94','scripts/e017_v94r')
s=s.replace("CPU=K['cpu']; SPEC=read(V93/'SUBSTANTIAL_CAMPAIGN_SPEC.json')", "CPU=K['cpu']; SPEC=read(ROOT/'TRAINING_CONFIGURATION_AND_SEEDS.json')\nS=runpy.run_path(str(ROOT/'source/scripts/e017_v94r/sampling.py'))\nK['gradients']=runpy.run_path(str(ROOT/'source/scripts/e017_v94r/gradients.py'))['gradients']")
s=s.replace("schema='V94_FULL_TRACE_RECOVERY_1'","schema='V94R_PACKING_FULL_TRACE_RECOVERY_1'")
s=s.replace("step_KL=maxstep,replay_logprob_maxerror=error,lr=", "step_KL=maxstep,replay_logprob_maxerror=error,diagnostics=self.e.last_gradient_diagnostics,lr=")
s=s.replace("trace_log_ratio=float(np.mean([z['trace_logprob']-z['reference_trace_logprob'] for z in rows])))", "trace_log_ratio=float(np.mean([z['trace_logprob']-z['reference_trace_logprob'] for z in rows])),global_contact_violations=sum(z['metrics']['complete'] and z['metrics']['geometry']['longrange_contact_degree']>S['SPEC']['packing_limits']['64'][1] for z in rows),packing_proposals_rejected=sum('provably_exceeds_final_contact_budget' in q[8] for z in rows for q in z['proposals']))")
a=s.index("        if fault=='partial':",s.index('    def publish('));b=s.index('        record=atomic_checkpoint',a)
s=s[:a]+'''        if fault=='partial':
            # Interrupt the actual full-state Torch writer, not a mock header.
            import builtins
            original=builtins.open;target=str(incoming)+'.part'
            compact(self.root/'INTERRUPTED_WRITE.json',dict(partial=target,previous_index_SHA256=sha(self.root/'RECOVERY_INDEX.json')))
            class Interrupt:
                def __init__(self,f):self.f=f;self.n=0
                def __getattr__(self,n):return getattr(self.f,n)
                def __enter__(self):return self
                def __exit__(self,*a):return self.f.__exit__(*a)
                def write(self,b):
                    n=self.f.write(b);self.n+=n
                    if self.n>=8*2**20:self.f.flush();os.fsync(self.f.fileno());os._exit(75)
                    return n
            def opened(file,*a,**kw):
                f=original(file,*a,**kw)
                return Interrupt(f) if str(file)==target and a and a[0]=='xb' else f
            builtins.open=opened
''' +s[b:]
s=s.replace("compact(self.root/'RECOVERY_INDEX.json',dict(schema='V94_DUAL_SLOT_1',latest=latest))", "compact(self.root/'RECOVERY_INDEX.json',dict(schema='V94R_DUAL_SLOT_1',latest=latest))\n        compact(self.root/'LIVE_STATUS.json',dict(state='STOPPED' if 'STOP' in reason or 'PAUSE' in reason else 'COMPLETED' if 'COMPLETED' in reason else 'RUNNING',PID=os.getpid(),update=self.update,counters=dict(self.counters),latest_checkpoint=latest,reason=reason))")
s=s.replace("rt=Runtime(Path(args.run)); rt.replay()", "rt=Runtime(Path(args.run)); rt.replay()\n    if args.fault=='partial' and rt.update==1:rt.publish('FAULT_PARTIAL',fault='partial')")
s=s.replace("release=read(ROOT/'recovery_qualification_20261025_v1/RELEASE.json')", "authorization=read(Path(args.authorization)) if args.authorization else {}\n    assert authorization.get('explicit_review_authorized') is True and authorization.get('configuration_SHA256')==sha(ROOT/'TRAINING_CONFIGURATION_AND_SEEDS.json'),'FULL PRODUCTION REQUIRES SEPARATE USER REVIEW; NOT AUTHORIZED BY V94R ENGINEERING'\n    release=read(ROOT/'qualification/RELEASE.json')")
# The original hard/backbone/diversity floors remain binding, even if a new
# filtered baseline is worse; a changed controller cannot lower its own floor.
s=s.replace("a['hard_valid']>=b['hard_valid']-.05", "a['hard_valid']>=max(b['hard_valid'],.96875)-.05")
s=s.replace("a['backbone_valid']>=b['backbone_valid']-.05", "a['backbone_valid']>=max(b['backbone_valid'],1.)-.05")
s=s.replace("a['structural_pair_RMSD']>=.8*b['structural_pair_RMSD']", "a['structural_pair_RMSD']>=.8*max(b['structural_pair_RMSD'],15.561398437372143)")
s=s.replace("if not safe:\n                    rt.publish", "if not safe:\n                    evaluator['curves'](rt.root)\n                    rt.publish")
s=s.replace("ap.add_argument('--run');", "ap.add_argument('--authorization'); ap.add_argument('--run');")
(D/'production.py').write_text(s)
e=(O/'source/scripts/e017_v94/evaluation.py').read_text().replace(str(O),str(R)).replace('scripts/e017_v94','scripts/e017_v94r')
e=e.replace("rejections=sum(not q[6] for r in rows for q in r['proposals']),aborts=", "global_contact_violations=sum(r['metrics']['complete'] and r['metrics']['geometry']['longrange_contact_degree']>S['SPEC']['packing_limits']['64'][1] for r in rows),rejections=sum(not q[6] for r in rows for q in r['proposals']),aborts=")
e=e.replace("result[label]=summarize(groups)","result[label]=summarize(groups)\n        if policy=='B':result[label]['invalid_proposal_checks']=result[label]['rejections'];result[label]['rejections']=0")
e=e.replace('w.writerow(row)',"w.writerow({k:row[k] for k in fields})")
(D/'evaluation.py').write_text(e)
shutil.copyfile(O/'INPUT_VERIFICATION.json',R/'INPUT_VERIFICATION.json')
cfg=json.loads((O/'TRAINING_CONFIGURATION_AND_SEEDS.json').read_text())
cfg.update(experiment='E017_V94R_PACKING_EXACT_FILTER',authorized=False,engineering_authorized=True,production_authorized=False,
    changed_controller='EXACT_FINAL_N_CONTACT_BUDGET_1',initializer='verified original preV94 V3 actor plus repaired goal adapter; fresh optimizer; never update500',
    original_scientific_reward_unchanged=True,neural_architecture_unchanged=True,loss_unchanged=True,
    conservative_peak_full_images=14,qualification_checkpoint_images_included=4,production_peak_full_images=10,
    minimum_free_bytes=14*224*2**20+4*2**30+512*2**20,
    original_absolute_safety_floors=dict(hard_valid=.91875,backbone_valid=.95,structural_pair_RMSD=12.449118749897715),
    main_new_intervention='add exact necessary append-only final contact budget to finite8 acceptance; no risk penalty',
    raw_actor_and_filtered_system_evaluated_separately=True,reference_filter='identical corrected deterministic acceptance/fallback for trace ratios',
    scientific_interpretation='new arm; not exact continuation of stopped V94; primary frozen success criteria unchanged')
(R/'TRAINING_CONFIGURATION_AND_SEEDS.json').write_text(json.dumps(cfg,indent=2)+'\n')
for file in D.glob('*.py'):compile(file.read_text(),str(file),'exec')
print('V94R_INSTALLED',cfg['minimum_free_bytes'],shutil.disk_usage(R).free)
