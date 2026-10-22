"""Freeze generated-prefix inputs before prospective validation scoring; no outcomes or updates."""
import pathlib,sys,json,copy
import numpy as np
import torch
R=pathlib.Path('/mnt/d/Simone/proteinGen_data/e017_v5_goal_observability_20261022_v1');sys.path.insert(0,str(R/'source/scripts/e017_v5'));import study as s
s.seed(s.CONFIG['seed']);e=s.Engine();e.arm='A'
for split,nident in [('DEV',24),('CONFIRM',32)]:
 for H in [2,4,8]:
  pool=s.panel(split,H,nident);shifted=[];cache={}
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   for t in pool:
    key=(t['identity'],t['t'])
    if key not in cache:
     base=next(z for z in pool if z['identity']==t['identity'] and z['t']==t['t'] and z['variant']=='native');x=torch.tensor(base['native_prefix'][:-2],dtype=torch.float64,device='cuda');eps=torch.randn(2,3,dtype=torch.float64,generator=torch.Generator().manual_seed(1101+int(t['identity'][:6],16))).cuda()
     for k in range(2):
      mu,ls,*_=e.predict(base,x);x=torch.cat([x,s.next_ca(x,s.physical(mu.double()+ls.double().exp()*eps[k]))[None]],0)
     cache[key]=x.cpu().tolist()
    new=copy.deepcopy(t);new['native_prefix']=cache[key];d=torch.linalg.vector_norm(torch.tensor(cache[key])[1:]-torch.tensor(cache[key])[:-1],dim=-1);new['prefix_spacing_valid']=bool(d.min()>=s.v.LIMITS['adjacent_low_A'] and d.max()<=s.v.LIMITS['adjacent_high_A']);new['necessary_contour_bound']=bool(np.linalg.norm(np.array(new['target'])-np.array(cache[key][-1]))<=H*s.v.LIMITS['adjacent_high_A']+8);new['shift_note']='Two frozen-A-generated prefix actions reused for all goals and arms; target retained; feasibility not guaranteed';shifted.append(new)
  s.save(f'manifests/SHIFTED_{split}_H{H}.json',dict(tasks=shifted,no_prospective_outcomes_evaluated=True))
e.verify();s.save('results/CONTROL_MANIFEST_PREFLIGHT.json',dict(J1_frozen=True,no_updates=True,no_terminal_validation_scores=True));print('CONTROL_MANIFESTS_FROZEN')
