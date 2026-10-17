"""Bounded12M CUDA engineering verification on two known TRAIN observations."""
import copy,json,math,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
import scaled_bc as core
from protein_joint_sac.contracts import digest
from protein_joint_sac.data.pdb_split_proposal import PDBSplitProposal
from protein_joint_sac.data.partial_geometry import batch_targets,derive
from protein_joint_sac.data.certified_unified import file_sha
from protein_joint_sac.models.batched_bc import losses
from protein_joint_sac.rl.checkpointing import save_atomic,load_checked
from protein_joint_sac.rl.bc_training import rng_state,restore_rng
from scaled_train import cpu_copy,maxdiff
R=Path('/mnt/d/Simone/proteinGen_data/e016_partial_supervision_20261016_v1');core.R=R

def main():
 core.setup();start=time.monotonic();proposal=PDBSplitProposal(core.P);root=Path('/mnt/d/Users/Simone Stocco/proteinGen_audits');path=root/'sequence_data_readiness_raw_full_compact_v1/tables/protein_chain_models/part-000000.parquet';side=root/'e006_rich_geometry_sidecars_v2/train/part-000000.parquet'
 for p in [path,side]:
  if file_sha(p)!=proposal.external[str(p)]:raise ValueError('source checksum mismatch')
 raw=pq.ParquetFile(path).read_row_group(0).slice(0,2).to_pylist();side_rows=pq.ParquetFile(side).read_row_group(0).slice(0,2).to_pylist();records={};refs=[]
 for i,r in enumerate(raw):
  seq=''.join(json.loads(r['residue_tokens']));h=__import__('hashlib').sha256(seq.encode()).hexdigest()
  if proposal.sequences[h]['split']!='TRAIN':raise ValueError('TRAIN-only engineering sources')
  x=json.loads(r['selected_calpha_coordinates']);seen=np.array([v is not None for v in x]);x=np.array([v if v is not None else [0,0,0] for v in x],dtype=np.float64);author={(str(a),b):k for k,(a,b) in enumerate(zip(json.loads(r['auth_sequence_ids']),json.loads(r['insertion_codes'])))};author.update({(a+ins,ins):k for (a,ins),k in list(author.items()) if ins and (a+ins,ins) not in author});v=side_rows[i];pos=[author[(str(a),b)] for a,b in zip(v['residue_ids'],v['insertion_codes'])]
  if ''.join(seq[k] for k in pos)!=v['sequence'] or any(not np.array_equal(x[k].astype(np.float32),np.array(v['ca_coordinates'][j],dtype=np.float32)) for j,k in enumerate(pos)):raise ValueError('sidecar alignment mismatch')
  links=np.zeros(len(seq)-1,bool)
  for j,ok in enumerate(v['chain_continuity_mask']):
   if ok and pos[j+1]==pos[j]+1:links[pos[j]]=True
  records[str(i)]={'sequence':seq,'x':x,'seen':seen,'links':links};refs.append({'entry':r['pdb_id'],'sequence_hash':h,'source_path':str(path),'row':i,'atom_sidecar':str(side),'sidecar_row':i})
 # Include native missing-coordinate AA targets plus synthetic internal gaps, explicitly labelled.
 gapped=copy.deepcopy(records['0']);gapped['seen'][4:9]=False;records['injected_gap_engineering_only']=gapped
 ex=[('0',0),('0',1),('0',2),('0',3),('0',100),('1',100),('1',163),('injected_gap_engineering_only',12)]
 b,y,g=batch_targets(records,ex,'cuda');core.guard();model=core.model('J2');parent=load_checked(core.R.parent/'e016_scaled10m_training_20261016_v1/runs/J2/checkpoint_10000.pt');model.load_state_dict(parent['model']);del parent
 torch.cuda.reset_peak_memory_stats();aa,ge,_=losses(model,b,y,g,geometry_fp32=True);loss=(aa+ge).mean();loss.backward();grad=torch.cat([p.grad.detach().float().flatten().cpu() for p in model.parameters() if p.grad is not None]);fp=float(loss.detach());model.zero_grad(set_to_none=True)
 with core.autocast('BF16'):aa,ge,_=losses(model,b,y,g,geometry_fp32=True);loss=(aa+ge).mean()
 loss.backward();mixed=torch.cat([p.grad.detach().float().flatten().cpu() for p in model.parameters() if p.grad is not None]);relative=float(torch.linalg.vector_norm(grad-mixed)/torch.linalg.vector_norm(grad));error=abs(float(loss)-fp)
 core.write(R/'tests/GPU_PRECISION_FP32_HEAD.json',{'loss_abs_error':error,'relative_gradient_L2':relative,'FP32_loss':fp,'BF16_loss':float(loss.detach()),'predeclared_limits':{'loss':.05,'gradient_relative_L2':.05}})
 if error>.05 or relative>.05:raise RuntimeError('predeclared FP32/BF16 .05 parity failed')
 # Masked native placeholders have no effect even when set to NaN.
 changed=copy.deepcopy(b);changed['x'][~changed['valid']]=float('nan')
 with core.autocast('BF16'):a0,g0,_=losses(model,b,y,g,geometry_fp32=True);a1,g1,_=losses(model,changed,y,g,geometry_fp32=True)
 torch.testing.assert_close(a0,a1,atol=0,rtol=0);torch.testing.assert_close(g0,g1,atol=0,rtol=0)
 opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01,fused=True)
 def step():
  opt.zero_grad(set_to_none=True)
  with core.autocast('BF16'):a,c,_=losses(model,b,y,g,geometry_fp32=True);v=(a+c).mean()
  v.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();return float(v),float(norm)
 step();cp=R/'tests/GPU_MASK_ENGINEERING_CHECKPOINT.pt';save_atomic(cp,{'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(),'examples':ex});expected=step();weights=cpu_copy(model.state_dict());optimizer=cpu_copy(opt.state_dict());state=load_checked(cp);model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);restore_rng(state['rng']);actual=step();pe=maxdiff(weights,model.state_dict());oe=maxdiff(optimizer,opt.state_dict())
 if pe>2e-5 or oe>1e-4 or abs(expected[0]-actual[0])>1e-5:raise RuntimeError('mixed-mask restart failed')
 out={'status':'PASS','scope':'BOUNDED_ENGINEERING_ONLY_NOT_CONTINUATION_TRAINING','real_TRAIN_sources':refs,'controlled_gap_example':True,'TEST_used':False,'parameters':sum(p.numel() for p in model.parameters()),'geometry_projection_precision':'FP32','FP32_BF16_loss_abs_error':error,'FP32_BF16_gradient_relative_L2':relative,'masked_NaN_placeholder_invariance':True,'geometry_component_counts':b['geometry_target_valid'].sum(0).tolist(),'restart_parameter_max_abs_error':pe,'restart_optimizer_max_abs_error':oe,'restart_loss_abs_error':abs(expected[0]-actual[0]),'seconds':time.monotonic()-start,'resources':core.guard()}
 core.write(R/'tests/GPU_MASK_PROOF.json',out);print(json.dumps(out),flush=True)
if __name__=='__main__':main()
