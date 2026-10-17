"""Partial deposited-polymer geometry: no interpolation, local adjacency requires evidence."""
import numpy as np
import torch
from ..contracts import AA,BuildState
from ..models.batched_bc import collate
VERSION='partial_prefix_mask_v1'

def derive(x,observed,connected):
 x=np.asarray(x,dtype=np.float64);seen=np.asarray(observed,dtype=bool);link=np.asarray(connected,dtype=bool)
 N=len(x)
 if x.shape!=(N,3) or seen.shape!=(N,) or link.shape!=(N-1,):raise ValueError('polymer-aligned coordinates/masks')
 if not np.isfinite(x[seen]).all():raise ValueError('nonfinite observed atom')
 x=np.where(seen[:,None],x,0.);link=link&seen[:-1]&seen[1:]
 g=np.zeros((N,3));mask=np.zeros((N,3),dtype=bool);segments=np.full(N,-1,dtype=np.int32);segment=-1
 for i in range(N):
  if seen[i]:
   if i==0 or not link[i-1]:segment+=1
   segments[i]=segment
 b=np.diff(x,axis=0);d=np.linalg.norm(b,axis=1)
 mask[1:,0]=link
 g[1:,0]=np.where(link,d,0.)
 if N>=3:
  cross=np.cross(b[:-1],b[1:]);a=np.arctan2(np.linalg.norm(cross,axis=1),-(b[:-1]*b[1:]).sum(1));am=link[:-1]&link[1:]
  mask[2:,1]=am;g[2:,1]=np.where(am,a,0.)
 if N>=4:
  u=b[1:-1]/np.maximum(d[1:-1,None],1e-300);v=-b[:-2];v=v-(v*u).sum(1)[:,None]*u;w=b[2:]-(b[2:]*u).sum(1)[:,None]*u
  v=v/np.maximum(np.linalg.norm(v,axis=1)[:,None],1e-300);w=w/np.maximum(np.linalg.norm(w,axis=1)[:,None],1e-300)
  t=np.arctan2((np.cross(u,v)*w).sum(1),(v*w).sum(1));tm=link[:-2]&link[1:-1]&link[2:]
  mask[3:,2]=tm;g[3:,2]=np.where(tm,t,0.)
 # Observed but invalid action support is a source-quality exception, never silently missing.
 bad=(mask[:,0]&(g[:,0]<=0))|(mask[:,1]&((g[:,1]<=0)|(g[:,1]>=np.pi)))|(mask[:,2]&(np.abs(g[:,2])>=np.pi))
 if bad.any() or not np.isfinite(g).all():raise ValueError('OBSERVED_GEOMETRY_OUTSIDE_UNCHANGED_ACTION_SUPPORT')
 return g,mask,segments

def target(sequence,x,seen,links,t):
 if set(sequence)-set(AA) or not 0<=t<len(sequence):raise ValueError('known complete canonical sequence and prefix boundary required')
 # Derive target separately; no target/suffix information enters state or input links.
 g,mask,seg=derive(np.asarray(x)[:t+1],np.asarray(seen)[:t+1],np.asarray(links)[:t])
 prefix=np.where(np.asarray(seen[:t])[:,None],np.asarray(x[:t],dtype=np.float64),0.)
 state=BuildState(tuple(sequence[:t]),torch.from_numpy(prefix.copy()),len(sequence),torch.tensor(seen[:t],dtype=torch.bool))
 return state,AA.index(sequence[t]),torch.from_numpy(g[t].copy()),mask[t].copy()

def batch_targets(records,examples,device):
 states=[];ys=[];gs=[];ms=[];links=[]
 for oid,t in examples:
  r=records[oid];state,y,g,m=target(r['sequence'],r['x'],r['seen'],r['links'],t)
  states.append(state);ys.append(y);gs.append(g);ms.append(m);links.append(r['links'][:max(0,t-1)])
 b=collate(states,device);bv=torch.zeros_like(b['valid'])
 for i,(s,v) in enumerate(zip(states,links)):
  if s.t>1:bv[i,1:s.t]=torch.as_tensor(v,device=device)&b['valid'][i,:s.t-1]&b['valid'][i,1:s.t]
 b['bond_valid']=bv;b['geometry_target_valid']=torch.as_tensor(np.asarray(ms),dtype=torch.bool,device=device)
 return b,torch.tensor(ys,device=device),torch.stack(gs).to(device)
