from pathlib import Path
import json,hashlib,shutil,time,collections,subprocess
import torch,pandas as pd,numpy as np
R=Path(__file__).resolve().parents[3];D=R.parent
AA='ACDEFGHIKLMNPQRSTVWY'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(n,x):(R/n).write_text(json.dumps(x,indent=2,allow_nan=False))
paths=[D/'e016_j1_bidirectional_multitask_20261019_v1/runs/J1/checkpoint_07500.pt',D/'e016_j1_torsion_stabilization_20261018_v1/replay/checkpoint_00500_full.pt']
expected=['881457ff0396975c1b20dd0a8e2caac8521e65b7e1cec76a7bb531dbc437b412','9208263300fb9e88519ff0d40e6d9d29a9744b709d9a6181819cd0894b593000']
verified=[]
for p,e in zip(paths,expected):
 h=sha(p);assert h==e;verified.append(dict(path=str(p),sha256=h));print('verified',p.name,flush=True)
modelonly=D/'e016_j1_clip_precision_pilot_20261018_v1/runs/J1/model_00500.pt'
a=torch.load(paths[1],map_location='cpu',weights_only=False)['model'];b=torch.load(modelonly,map_location='cpu',weights_only=False)['model'];assert set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a);del a,b
cache=[D/'e016_j1_bidirectional_multitask_20261019_v1/generation/new07500_ca_fp64_interior_v1.json',D/'e016_j1_clip_precision_pilot_20261018_v1/generation/final00500_ca_fp64_interior_v1.json']
rows=[];sources=[]
for group,p,cp in zip(['J1_7500','J1_500'],cache,verified):
 x=json.loads(p.read_text());assert len(x)==160
 for v in x:
  assert v['protocol']=='ca_fp64_interior_v1' and v['live_batch']==4
  assert v['checkpoint_sha256']==(expected[0] if group=='J1_7500' else sha(modelonly))
  assert v['seed']==2026101700+100*v['N']+v['sample_index']
  rows.append(dict(id=f"{group}_N{v['N']}_{v['sample_index']:02d}",group=group,N=v['N'],index=v['sample_index'],sequence=v.get('sequence'),status=v['status'],source=str(p),source_index=x.index(v),seed=v['seed'],draws_sha256=v['draws_sha256'],geometry_metrics=v.get('metrics')))
 sources.append(dict(path=str(p),sha256=sha(p)))
for N in [64,128,256,384,500]:
 a=[x for x in rows if x['group']=='J1_7500' and x['N']==N];b=[x for x in rows if x['group']=='J1_500' and x['N']==N];assert len(a)==len(b)==32
 assert all(x['draws_sha256']==y['draws_sha256'] for x,y in zip(a,b))
p=D/'pdb_supervised_split_50id80cov_20261016_v1/SEQUENCE_ASSIGNMENTS.parquet';df=pd.read_parquet(p);rng=np.random.default_rng(20261020)
for split,group in [('TRAIN','natural_TRAIN'),('VALIDATION','natural_VALIDATION')]:
 for N in [64,128,256,384,500]:
  ds=df[(df['split']==split)&(df['length']==N)].copy();ds['order']=ds['sequence_id'].map(lambda h:hashlib.sha256((h+'r1e').encode()).hexdigest());ds=ds.sort_values('order').head(32)
  for i,z in enumerate(ds.to_dict('records')):
   s=z['canonical_full_sequence'];assert len(s)==N and set(s)<=set(AA)
   row=dict(id=f'{group}_N{N}_{i:02d}',group=group,N=N,index=i,sequence=s,status='VALID',sequence_id=z['sequence_id'],protected_group_id=z['protected_group_id']);rows.append(row)
   if split=='TRAIN':
    t=list(s);rng.shuffle(t);rows.append(dict(id=f'shuffled_N{N}_{i:02d}',group='shuffled',N=N,index=i,sequence=''.join(t),status='STATISTICAL_CONTROL',source_identity=z['sequence_id']))
for split in ['TRAIN','VALIDATION']:
 out=R/'data'/f'{split}.fasta'
 with out.open('w') as f:
  for z in df[df['split']==split].itertuples():f.write(f'>{z.sequence_id}\n{z.canonical_full_sequence}\n')
write('FROZEN_J1_GENERATION_PANEL.json',dict(checkpoints=verified,parent_model_only_bitwise_equal=True,cache_sources=sources,sequence_manifest_sha256=sha(p),generation_attempts=320,new_generations=0,controls='SHA-ordered exact requested full length; no fragments; up to32/split/length; shuffle seed20261020',rows=rows))
write('configs/SCORING_PROTOCOL.json',dict(sealed_before_scores=True,esm_model='esm2_t12_35M_UR50D',masked_positions=16,position_policy='rounded linspace 0..N-1; each forward masks exactly one position',gpu_seconds_cap=2700,full_check='first2 per group N64; all positions',novelty='MMseqs2 bounded local search TRAIN and separate VALIDATION; report identity and both coverages; no exhaustive remote homology claim',clusters='connected components >=50%/80% coverage and >=80%/80% coverage',experimental='published individual scores only; higher confidence/lower PAE and MPNN; full614 and paper exclusion NMR44 separately; study bootstrap1000; no fitted models'))
print('panel',collections.Counter(x['group'] for x in rows),flush=True)
