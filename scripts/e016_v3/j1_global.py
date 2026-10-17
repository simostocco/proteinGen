"""Single J1 global-geometry continuation. No TEST, oracle, RL, or second model training."""
import collections,copy,hashlib,json,math,os,random,resource,shutil,time
from pathlib import Path
import numpy as np
import torch
import scaled_bc as core
from protein_joint_sac.contracts import AA,digest
from protein_joint_sac.data.scaled_supervised import Data,Sampler
from protein_joint_sac.data.certified_unified import file_sha
from protein_joint_sac.models.batched_bc import encode_batch
from protein_joint_sac.rl.checkpointing import load_checked
R=Path('/mnt/d/Simone/proteinGen_data/e016_j1_global_pretraining_20261017_v1')
OLD=Path('/mnt/d/Simone/proteinGen_data/e016_partial_supervision_20261016_v1')
PARENT=OLD/'runs/J1-FULL/checkpoint_20000.pt'
PARENT_SHA='9e9215d3720ffc73edb0c3e382db25332f91cb26b0c8d38d01668e383b1e3810'
BANDS=[(20,64),(65,128),(129,256),(257,384),(385,500)]
CFG={'precision':'BF16','geometry_projection_fp32':True,'activation_checkpoint':False,'live_batch':16}
core.R=R
write=core.write

def band(n):return next(i for i,(lo,hi) in enumerate(BANDS) if lo<=n<=hi)
def guard():
 out=core.guard(450*2**20)
 mem={v.split(':')[0]:int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith(('MemAvailable:','MemTotal:','SwapFree:','SwapTotal:'))}
 if mem['MemAvailable']<.1*mem['MemTotal']:raise RuntimeError('actual host RAM headroom below10percent')
 out.update(mem);return out
def parent_model():
 if file_sha(PARENT)!=PARENT_SHA:raise ValueError('immutable parent checkpoint mismatch')
 s=load_checked(PARENT);m=core.model('J1');m.load_state_dict(s['model'],strict=True)
 if sum(p.numel() for p in m.parameters() if p.requires_grad)!=12351746:raise ValueError('architecture changed')
 return m,s

def panels(data):
 val=json.load(open(OLD/'configs/VALIDATION_PANELS.json'))['complete'];counts=collections.Counter(band(len(data.seqs[h])) for h in {h for h,o,t in val});rng=random.Random(20261017);chosen=[]
 for k,n in sorted(counts.items()):
  ids=[h for h in data.jointids['TRAIN'] if band(len(data.seqs[h]))==k];rng.shuffle(ids)
  if len(ids)<n:raise ValueError('cannot match validation strata')
  chosen+=ids[:n]
 train=[]
 for h in sorted(chosen):
  N=len(data.seqs[h]);o=data.owners[h][0]
  train.extend((h,o,int(t)) for t in np.rint(np.linspace(0,N-1,8)).astype(int))
 if {h for h,o,t in train}&{h for h,o,t in val}:raise ValueError('TRAIN VALIDATION overlap')
 result={'TRAIN':train,'VALIDATION':val,'seed':20261017,'selection':'uniform identities within matched full-length strata; first sorted observation; 8 rounded linspace prefix targets including BOS; same VALIDATION definition','TEST':'SEALED_NOT_EVALUATED'}
 path=R/'configs/EVALUATION_PANELS.json'
 if path.exists() and json.load(open(path))!=json.loads(json.dumps(result)):raise ValueError('frozen panel changed')
 write(path,result);return result

def fit_sequences(data):
 result={};alpha=.5
 for name,ids in [('matched_joint',data.jointids['TRAIN']),('full_pdb',data.ids['TRAIN'])]:
  raw=np.zeros(20);ident=np.zeros(20);trans=np.zeros((21,20));itrans=np.zeros((21,20))
  for h in ids:
   s=[AA.index(a) for a in data.seqs[h]];c=np.bincount(s,minlength=20);raw+=c;ident+=c/len(s);previous=20
   for a in s:trans[previous,a]+=1;itrans[previous,a]+=1/len(s);previous=a
  # Jeffreys pseudocount .5/residue or transition cell, identity-first effective identity counts.
  result[name]={'identities':len(ids),'residues':int(raw.sum()),'membership_sha256':digest(sorted(ids)),'AA_order':AA,'smoothing_pseudocount':alpha,'unigram_residue_weighted':((raw+alpha)/(raw.sum()+20*alpha)).tolist(),'unigram_identity_first':((ident+alpha)/(ident.sum()+20*alpha)).tolist(),'markov_residue_weighted':((trans+alpha)/(trans.sum(1,keepdims=True)+20*alpha)).tolist(),'markov_identity_first':((itrans+alpha)/(itrans.sum(1,keepdims=True)+20*alpha)).tolist(),'BOS_row':20}
 return result

@torch.no_grad()
def evaluate(model,data,panel,role,statistics=None):
 start=time.monotonic();model.eval();sums=np.zeros(7);components=np.zeros(3);counts=np.zeros(3,dtype=int);by=collections.defaultdict(lambda:np.zeros(7));lens=collections.defaultdict(lambda:np.zeros(7));prefix=collections.defaultdict(lambda:np.zeros(7));base=collections.defaultdict(list)
 ordered=sorted(panel,key=lambda v:(v[2],v[0],v[1]))
 for i in range(0,len(panel),16):
  ex=ordered[i:i+16]
  if any(data.proposal.sequences[h]['split']!=role for h,o,t in ex):raise ValueError('evaluation role mismatch')
  b,y,g=core.make(data,ex,True);_,aa,ge,parts=core.loss_call(model,b,y,g,CFG)
  with core.autocast('BF16'):hstate=encode_batch(model,b);logits=model.aa_head(hstate).float()
  correct=logits.topk(3,dim=-1).indices.eq(y[:,None]);mask=b['t'][:,None]>=torch.arange(1,4,device='cuda')[None];counts+=mask.sum(0).cpu().numpy();components+=parts.sum(0).cpu().numpy()
  for (h,o,t),a,z,c in zip(ex,aa.cpu().tolist(),ge.cpu().tolist(),correct.cpu().numpy()):
   v=np.array([a,z,float(c[0]),float(c.any()),1.,t,len(data.seqs[h])]);sums+=v;by[h]+=v;lens[str(BANDS[band(len(data.seqs[h]))])]+=v;prefix['BOS' if t==0 else ('1-15' if t<16 else '16-63' if t<64 else '64-127' if t<128 else '128+')]+=v
   if statistics:
    target=AA.index(data.seqs[h][t]);prev=AA.index(data.seqs[h][t-1]) if t else 20
    for population,st in statistics.items():
     for key in ['unigram_residue_weighted','unigram_identity_first','markov_residue_weighted','markov_identity_first']:
      prob=st[key][prev][target] if key.startswith('markov') else st[key][target];base[population+'/'+key].append(-math.log(prob))
  if i%2048==0:print(json.dumps({'evaluation_role':role,'decisions':i,'total':len(panel),'seconds':time.monotonic()-start}),flush=True)
 def summary(v):
  n=v[4];a=v[0]/n
  return {'decisions':int(n),'AA_NLL':float(a),'perplexity':math.exp(a),'bits_per_residue':float(a/math.log(2)),'top1':float(v[2]/n),'top3':float(v[3]/n),'geometry_NLL':float(v[1]/n),'historic_combined':float((v[0]+v[1])/n)}
 out=summary(sums);out.update({'role':role,'identities':len(by),'component_NLL':(components/counts).tolist(),'component_counts':counts.tolist(),'per_identity':{h:summary(v) for h,v in by.items()},'length_bands':{k:summary(v) for k,v in lens.items()},'prefix_length':{k:summary(v) for k,v in prefix.items()},'panel_sha256':digest(panel),'seconds':time.monotonic()-start,'probability_baselines':{k:{'NLL':float(np.mean(v)),'perplexity':math.exp(float(np.mean(v)))} for k,v in base.items()}})
 rng=np.random.default_rng(742);vals=np.array([v[0]/v[4] for v in by.values()]);means=[vals[rng.integers(len(vals),size=len(vals))].mean() for _ in range(500)];out['identity_bootstrap_AA_NLL_95pct']=np.quantile(means,[.025,.975]).tolist();return out

def baseline():
 core.setup();guard();receipt={};data=Data(True,receipt);p=panels(data);stats=fit_sequences(data);m,parent=parent_model();del parent;torch.cuda.reset_peak_memory_stats();result={}
 for role in ['TRAIN','VALIDATION']:
  result[role]=evaluate(m,data,p[role],role,stats);write(R/('BASELINE_'+role+'.json'),result[role])
 out={'uniform':{'AA_NLL':math.log(20),'perplexity':20.,'bits_per_residue':math.log2(20)},'fit':stats,'J1':result,'TRAIN_VALIDATION_NLL_gap':result['VALIDATION']['AA_NLL']-result['TRAIN']['AA_NLL'],'parent_sha256':PARENT_SHA,'data':receipt,'resources':guard()};write(R/'SEQUENCE_BASELINES.json',out)
 text='# Frozen native-prefix sequence baselines\n\n'
 for role in ['TRAIN','VALIDATION']:
  v=result[role];text+=f"{role}: {v['identities']} identities, {v['decisions']} fixed decisions; J1 NLL {v['AA_NLL']:.8f}, perplexity {v['perplexity']:.6f}, top1 {v['top1']:.4f}, top3 {v['top3']:.4f}.\n\n"
 text+=f"Uniform NLL {math.log(20):.9f}, perplexity exactly 20. TRAIN–VALIDATION NLL gap {out['TRAIN_VALIDATION_NLL_gap']:.8f}. Complete per-length/prefix and probability-baseline values in SEQUENCE_BASELINES.json. Identity-bootstrap intervals do not account for all family correlations. TRAIN evaluation is frozen and distinct from stochastic optimization loss. TEST is not evaluated.\n"
 (R/'BASELINE_TRAIN_VALIDATION_PERPLEXITY.md').write_text(text);print('BASELINE_COMPLETE',flush=True)

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['baseline']);a=ap.parse_args();baseline()
