import pathlib,sys,json
import torch
R=pathlib.Path('/mnt/d/Simone/proteinGen_data/e017_v5_goal_observability_20261022_v1');sys.path.insert(0,str(R/'source/scripts/e017_v5'));import study as s
from protein_joint_sac.crisp_v3.storage import rng_state,set_rng
from protein_joint_sac.crisp_v3.recovery import state_digest
ref=json.load(open(R/'results/RESTART_EXPECTED.json'));s.CONFIG.update(ref['config']);e=s.Engine();e.arm='C';st=torch.load(R/'checkpoints/engineering_recovery.pt',map_location='cpu',weights_only=False);e.actor.load_state_dict(st['actor']);opt=s.optimizer(e);sched=s.scheduler(opt);opt.load_state_dict(st['optimizer']);sched.load_state_dict(st['scheduler']);set_rng(st['rng']);task=next(t for t in s.tasks('TRAIN') if t['task_id']==ref['task_id']);assert s.tasks('TRAIN',2)[st['sampler_order'][st['sampler_cursor']]]['task_id']==ref['task_id'];e.actor.train();opt.zero_grad(set_to_none=True)
with torch.autocast('cuda',dtype=torch.bfloat16):loss,_,_,_=e.rollout(task,ref['noise_seed'])
loss.backward();torch.nn.utils.clip_grad_norm_(e.actor.parameters(),s.CONFIG['clip']);opt.step();sched.step();e.verify([opt]);actual=dict(actor=state_digest(e.actor.state_dict()),optimizer=state_digest(opt.state_dict()),scheduler=state_digest(sched.state_dict()),rng=state_digest(rng_state()),loss=float(loss));assert actual==ref['expected'],(actual,ref['expected']);s.save('results/FRESH_GPU_RESTART.json',dict(PASS=True,expected=ref['expected'],actual=actual,J1_frozen=True,main_updates=0,engineering_optimizer_executions=1));print('FRESH_GPU_RESTART_PASS')
