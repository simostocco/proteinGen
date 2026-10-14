"""Prepared profiling entrypoint; full-length/GPU execution requires another gate."""
import argparse,json,os,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--approval-card',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--lengths',type=int,nargs='+',default=[20,64,128,256,500]);a=p.parse_args();root=Path(os.environ['E016_ARTIFACT_ROOT']);card_path=Path(a.approval_card).resolve()
 if not str(card_path).startswith('/mnt/d/'):raise SystemExit('D-only approval card')
 card=json.loads(card_path.read_text())
 if card.get('stage')!='G4' or card.get('approval')!='EXPLICIT_USER_APPROVAL':raise SystemExit('BLOCKED_G4_APPROVAL')
 if a.device=='cuda':raise SystemExit('BLOCKED_GPU_TENSOR_CONTRACT: CPU reference only; device-aware integration and exclusive GPU guard required before CUDA profiling')
 import torch
 from protein_joint_sac.contracts import BuildState,BuildAction
 from protein_joint_sac.geometry.ca_builder import append,mask_for_step
 from protein_joint_sac.models.joint_actor import JointActor
 from protein_joint_sac.models.build_critic import TwinCritics
 from protein_joint_sac.models.dense_geometry_attention import ModelConfig
 from protein_joint_sac.resources.disk_guard import require_budget
 require_budget(root,1024**3);torch.set_num_threads(1);rows=[]
 for N in a.lengths:
  if not 20<=N<=500:raise ValueError('supported N20..500')
  state=BuildState.empty(N)
  for i in range(N-1):
   mask=mask_for_step(i);g=torch.tensor([3.8 if mask[0] else 0.,1.9 if mask[1] else 0.,.6 if mask[2] else 0.],dtype=torch.float64);state=append(state,BuildAction('A',g,mask))
  actor=JointActor();q=TwinCritics();start=time.monotonic();action,la,lg=actor.sample(state);loss=q.q1(state,action)+la+lg;loss.backward();rows.append({'N':N,'t':state.t,'device':'cpu','forward_backward_seconds':time.monotonic()-start,'complete_sampling_speed':'NOT_MEASURED','synthetic_geometry':'benchmark only, not TRAIN prior'})
 (root/'reports/approved_cpu_profile.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
