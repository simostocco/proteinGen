from r1e_common import *
import sys,time
sys.path.insert(0,str(R/'vendor/fair-esm'))
import torch,esm
prior=read('ESM2_SEQUENCE_PLAUSIBILITY.json'); panel=read('FROZEN_J1_GENERATION_PANEL.json')['rows'];selected=[r for r in panel if r['index']<2]
write('configs/ESM_SENSITIVITY_LOCK.json',dict(selected_ids=[r['id'] for r in selected],reason='initial10 short-sequence full checks showed sampled mean bias -0.102nats/MAE0.136; expand same first2 rule across all lengths/groups to quantify approximation, not rescore/select the best-looking sequences',max_remaining_seconds=2700-prior['seconds'],no_primary_metric_change=True))
torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
model,a=esm.pretrained.load_model_and_alphabet_core('esm2_t12_35M_UR50D',torch.load(R/'weights/esm2_t12_35M_UR50D.pt',map_location='cpu',weights_only=False),None);model.eval().cuda().requires_grad_(False);converter=a.get_batch_converter();out=[];start=time.monotonic()
with torch.inference_mode():
 for r in selected:
  if time.monotonic()-start+prior['seconds']>2600:raise RuntimeError('total evaluator cap')
  _,_,t=converter([('s',r['sequence'])]);t=t.cuda();lp=[]
  for k in range(0,r['N'],16):
   ix=torch.arange(k+1,min(k+16,r['N'])+1,device='cuda');v=t.repeat(len(ix),1);truth=v[torch.arange(len(ix),device='cuda'),ix].clone();v[torch.arange(len(ix),device='cuda'),ix]=a.mask_idx
   z=model(v,return_contacts=False)['logits'].float().log_softmax(-1);lp.extend(z[torch.arange(len(ix),device='cuda'),ix,truth].cpu().tolist())
  old=next(x for x in prior['scores'] if x['id']==r['id']);ps=np.array(old['positions']);assert np.allclose(np.array(lp)[ps],old['masked_logp'],atol=2e-5)
  out.append(dict(id=r['id'],group=r['group'],N=r['N'],full_mean_logp=float(np.mean(lp)),sampled_mean_logp=old['mean_logp'],sampled_minus_full=old['mean_logp']-float(np.mean(lp))))
  if len(out)%10==0:write('results/ESM_FULL_POSITION_PARTIAL.json',out);print('full',len(out),len(selected),round(time.monotonic()-start,1),flush=True)
# One-mask leakage test: changing original identity at the hidden position yields identical actual model input.
_,_,x=converter([('s','ACDEFGHIKLMNPQRSTVWY')]);_,_,y=converter([('s','ACDEFGHIKLMAPQRSTVWY')]);different=torch.nonzero(x!=y,as_tuple=False);assert different.shape==(1,2);site=int(different[0,1]);x[:,site]=a.mask_idx;y[:,site]=a.mask_idx
# Identify the differing residue after BOS explicitly; both actual hidden inputs must match.
assert torch.equal(x,y)
write('ESM_SAMPLING_SENSITIVITY.json',dict(rows=out,seconds=time.monotonic()-start,allocated_peak_bytes=torch.cuda.max_memory_allocated(),summary={str(N):desc([r['sampled_minus_full'] for r in out if r['N']==N]) for N in [64,128,256,384,500]},masked_truth_leakage_test='PASS identical masked input despite changed original hidden target',primary_scoring_rule_unchanged=True,no_trainable_parameters=True))
print('sensitivity complete',time.monotonic()-start,flush=True)
