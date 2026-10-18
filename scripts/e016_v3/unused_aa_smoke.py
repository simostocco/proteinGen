"""Bounded inference only. Never constructs an optimizer."""
import json,hashlib,sys,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
ROOT=Path('/mnt/d/Simone/proteinGen_data/e016_unused_train_census_20261018_v1');sys.path.insert(0,str(ROOT/'source/src'));sys.path.insert(0,str(ROOT/'source/scripts/e016_v3'))
from scaled_bc import model
from protein_joint_sac.contracts import BuildState,AA
from protein_joint_sac.models.batched_bc import collate,features_batch,encode_batch,losses
D=ROOT.parent;cp=D/'e016_j1_clip_precision_pilot_20261018_v1/runs/J1/model_00500.pt'
h=hashlib.sha256(cp.read_bytes()).hexdigest();assert h=='9303f0d82a8df044302488340f3ad81535f2fc321360a00bf2892f3a0407aa91'
torch.set_num_threads(4);torch.manual_seed(42);m=model('J1',device='cpu');v=torch.load(cp,map_location='cpu',weights_only=False);m.load_state_dict(v['model'],strict=True);m.eval()
rows=pq.read_table(D/'pdb_supervised_split_50id80cov_20261016_v1/SEQUENCE_ASSIGNMENTS.parquet').to_pylist();p=pq.read_table(D/'e016_partial_supervision_20261016_v1/view/PARTIAL_OBSERVATIONS.parquet',columns=['sequence_hash','split']).to_pylist();partial={r['sequence_hash'] for r in p if r['split']=='TRAIN'};u=sorted([r for r in rows if r['split']=='TRAIN' and not r['joint_eligible'] and r['sequence_id'] not in partial],key=lambda r:r['sequence_id'])
checks=[];start=time.monotonic()
with torch.inference_mode():
 for lo,hi in [(20,64),(65,128),(129,256),(257,384),(385,500)]:
  r=next(r for r in u if lo<=r['length']<=hi);seq=r['canonical_full_sequence'];N=len(seq)
  for t in sorted(set([0,1,3,N//2,N-1])):
   state=BuildState(tuple(seq[:t]),torch.zeros(t,3,dtype=torch.float64),N,torch.zeros(t,dtype=torch.bool));b=collate([state],'cpu');b['geometry_target_valid']=torch.zeros(1,3,dtype=torch.bool);y=torch.tensor([AA.index(seq[t])]);feat,pair,present=features_batch(b,False,False);pred=m.aa_head(encode_batch(m,b));aa,g,components=losses(m,b,y,None)
   c={k:v.clone() for k,v in b.items()};c['x'].fill_(float('nan'));pred2=m.aa_head(encode_batch(m,c));err=float((pred-pred2).abs().max());assert err==0 and torch.isfinite(pred).all() and torch.isfinite(aa).all();assert g.item()==0 and not components.any()
   assert int(present.sum())==t and not b['valid'].any();assert pair is None or not pair[1].any()
   # All-missing geometry is positionally honest; geometry channels are zero, sequence tokens remain present.
   checks.append(dict(sequence_id=r['sequence_id'],N=N,t=t,AA_NLL=float(aa),masked_NaN_placeholder_logit_error=err,valid_sequence_tokens=int(present.sum()),valid_CA=0,geometry_loss=0.,pair_physical_valid_count=int(pair[1].sum()) if pair else 0))
result=dict(status='PASS',checkpoint=str(cp),checkpoint_sha256=h,parameter_count=sum(p.numel() for p in m.parameters()),checks=checks,elapsed_seconds=time.monotonic()-start,optimizer_constructed=False,optimizer_updates=0,device='CPU',future_suffix='Never supplied to collate or encoder; only requested full length N is visible',limitations='Existing supervised Data.target(joint=False) semantics tested directly through identical BuildState construction. Current J1 PartialData.target needs explicit routing for no-observation AA-only examples; production was not changed.')
(ROOT/'AA_ONLY_FORWARD_SMOKE_TEST.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',len(checks),'CPU forward checks')
