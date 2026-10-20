from r1e_common import *
import sys,time,os,subprocess
sys.path.insert(0,str(R/'vendor/fair-esm'))
import torch,esm,pandas as pd
# No network is used by the loader; contact-regression parameters are unnecessary for token scoring.
torch.set_num_threads(6);torch.manual_seed(20261020);torch.backends.cuda.matmul.allow_tf32=False
weight=R/'weights/esm2_t12_35M_UR50D.pt';assert sha(weight)=='7f21e80e61d16a71735163ef555d3009afb0c98da74c48e29df08606973cc55e'
model,alphabet=esm.pretrained.load_model_and_alphabet_core('esm2_t12_35M_UR50D',torch.load(weight,map_location='cpu',weights_only=False),regression_data=None)
model=model.eval().cuda();model.requires_grad_(False);convert=alphabet.get_batch_converter();start=time.monotonic();rows=read('FROZEN_J1_GENERATION_PANEL.json')['rows'];scores=[]
df=pd.read_csv(R/'data/Garcia_DataS1.csv')
for i,r in df.iterrows():rows.append(dict(id=f'Garcia_{i}',group='Garcia_designs',N=len(r['sequence']),index=int(i),sequence=r['sequence'],article=r['article'],success=bool(r['Experimental Success'])))
write('configs/ESM_PANEL_LOCK.json',dict(total_sequences=len(rows),selection='all757 frozen J1/control sequences and all614 verified Garcia sequences; no score-based selection',positions=16,direction='higher mean masked log probability is more compatible',full_subset='first2 eachgroup N64, selected by existing frozen order',max_seconds=2700))
def positions(n,k):return np.unique(np.rint(np.linspace(0,n-1,min(k,n))).astype(int))
@torch.inference_mode()
def score(s,ps):
 _,_,t=convert([('s',s)]);t=t.cuda();out=[]
 for off in range(0,len(ps),16):
  ix=torch.as_tensor(ps[off:off+16]+1,device='cuda');v=t.repeat(len(ix),1);target=v[torch.arange(len(ix),device='cuda'),ix].clone();v[torch.arange(len(ix),device='cuda'),ix]=alphabet.mask_idx
  logits=model(v,return_contacts=False)['logits'];lp=logits.float().log_softmax(-1);out.extend(lp[torch.arange(len(ix),device='cuda'),ix,target].cpu().tolist())
 return out
bench=[]
for N in [64,128,256,384,500]:
 s=next(r['sequence'] for r in rows if r['N']==N);ps=positions(N,16);torch.cuda.synchronize();t=time.monotonic();a=score(s,ps);torch.cuda.synchronize();bench.append(dict(N=N,seconds=time.monotonic()-t,masked_positions=len(ps),finite=bool(np.isfinite(a).all()),allocated_peak_bytes=torch.cuda.max_memory_allocated()))
projected=sum(next(b['seconds'] for b in bench if b['N']==min([64,128,256,384,500],key=lambda N:abs(N-r['N']))) for r in rows)
write('ESM_BENCHMARK.json',dict(benchmarks=bench,projected_seconds=projected,model_trainable_parameters=0,torch=torch.__version__,weight_sha256=sha(weight),full_precision='FP32; TF32 disabled',device=torch.cuda.get_device_name(),budget_seconds=2700))
print('BENCHMARK',bench,'projected',projected,flush=True)
if projected>2400:raise RuntimeError('prospective complete panel exceeds budget; freeze balanced reduction before scoring')
for i,r in enumerate(rows):
 if time.monotonic()-start>2650:write('ESM_PARTIAL_RESULTS.json',scores);raise RuntimeError('hard evaluator time cap')
 s=r['sequence']
 if not s or set(s)-set(AA):scores.append(dict(id=r['id'],status='INVALID_CANONICAL'));continue
 ps=positions(len(s),16);v=score(s,ps);z={k:r[k] for k in ['id','group','N','index']};z.update(positions=ps.tolist(),masked_logp=v,mean_logp=float(np.mean(v)),sampled_masked_pseudo_perplexity=float(np.exp(-np.mean(v))))
 if r['N']==64 and r['index']<2 and r['group']!='Garcia_designs':
  full=score(s,np.arange(64));z['full_mean_logp']=float(np.mean(full));z['sampled_minus_full']=z['mean_logp']-z['full_mean_logp']
 scores.append(z)
 if i%50==0:write('results/ESM_HEARTBEAT.json',dict(done=i+1,total=len(rows),seconds=time.monotonic()-start));print(i+1,'/',len(rows),round(time.monotonic()-start,1),flush=True)
summary={}
for g in sorted({r['group'] for r in scores if 'mean_logp'in r}):
 summary[g]={str(N):desc([v['mean_logp'] for v in scores if v.get('group')==g and v['N']==N]) for N in sorted({r['N'] for r in scores if r.get('group')==g})}
paired={}
for N in [64,128,256,384,500]:
 a=sorted([r for r in scores if r.get('group')=='J1_7500' and r['N']==N],key=lambda r:r['index']);b=sorted([r for r in scores if r.get('group')=='J1_500' and r['N']==N],key=lambda r:r['index']);paired[str(N)]=desc([x['mean_logp']-y['mean_logp'] for x,y in zip(a,b)])
write('ESM2_SEQUENCE_PLAUSIBILITY.json',dict(model='esm2_t12_35M_UR50D',weight_sha256=sha(weight),definition='mean over16 stratified one-at-a-time masked true-residue log probabilities in full uncorrupted surrounding sequence; natural logs; higher better; not autoregressive likelihood or stability',scores=scores,summary=summary,paired_J1_7500_minus_500=paired,seconds=time.monotonic()-start,allocated_peak_bytes=torch.cuda.max_memory_allocated(),reserved_peak_bytes=torch.cuda.max_memory_reserved(),new_optimizer_updates=0,training_corpus='UniRef50/UR50D; overlap with natural and published sequences possible; not independent experimental ground truth'))
print('ESM COMPLETE',time.monotonic()-start,flush=True)
