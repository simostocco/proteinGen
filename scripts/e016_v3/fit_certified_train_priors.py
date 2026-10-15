"""Fit small, global residue-conditioned C-alpha priors, certified TRAIN only."""
import os,json,hashlib,collections,gzip,math
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from protein_joint_sac.contracts import DatasetRef,AA
from protein_joint_sac.data.certified_unified import verify_release,file_sha
from protein_joint_sac.chemistry.reference_adapter import load,ref
R=Path(os.environ['E016_ARTIFACT_ROOT'])
def assert_train(rows):
 if any(r['split']!='TRAIN' or r['role']!='TRAINING' or not r['eligible'] for r in rows):raise ValueError('heldout/unqualified example cannot fit statistics')
def internal(x):
 x=np.asarray(x,dtype=np.float64);b=np.diff(x,axis=0);d=np.linalg.norm(b,axis=1);theta=np.arccos(np.clip(-np.sum(b[:-1]*b[1:],axis=1)/(d[:-1]*d[1:]),-1,1));u=b[1:-1]/d[1:-1,None];v=-b[:-2];v-=np.sum(v*u,axis=1)[:,None]*u;w=b[2:]-np.sum(b[2:]*u,axis=1)[:,None]*u;v/=np.linalg.norm(v,axis=1)[:,None];w/=np.linalg.norm(w,axis=1)[:,None];tau=np.arctan2(np.sum(np.cross(u,v)*w,axis=1),np.sum(v*w,axis=1));return d,theta,tau
def main():
 raw=json.load(open(R/'DATASET_REF.json'));raw.pop('version',None);reference=DatasetRef(**raw);manifest=verify_release(reference);root=Path(reference.root)
 rows=[]
 for batch in pq.ParquetFile(root/'joint_bc_eligibility.parquet').iter_batches(batch_size=512,use_threads=False):
  rows.extend(r for r in batch.to_pylist() if r['eligible'] and r['split']=='TRAIN' and r['role']=='TRAINING')
 assert_train(rows);assert rows
 D=R/'priors/train_ca_geometry_v1';D.mkdir(parents=True,exist_ok=False)
 nobs=collections.Counter(r['sequence_hash'] for r in rows);grouped=collections.defaultdict(list)
 for row in rows:grouped[row['coordinate_path']].append(row)
 stats={a:{k:np.zeros(4) for k in ['d_A','log_d','theta_rad','logit_theta','sin_tau','cos_tau']} for a in AA};ledger=(D/'TRAIN_FIT_INPUTS.jsonl').open('w');processed=0
 def consume(row,x):
  nonlocal processed
  seq=row['sequence'];assert len(x)==len(seq);d,theta,tau=internal(x);assert np.isfinite(d).all() and np.all(d>0) and np.all(theta>0) and np.all(theta<np.pi) and np.all(np.abs(tau)<np.pi)
  for key,values,offset in [('d_A',d,1),('log_d',np.log(d),1),('theta_rad',theta,2),('logit_theta',np.log(theta/(np.pi-theta)),2),('sin_tau',np.sin(tau),3),('cos_tau',np.cos(tau),3)]:
   weight=1/(nobs[row['sequence_hash']]*len(values))
   for a in AA:
    z=values[np.fromiter((r==a for r in seq[offset:]),dtype=bool)];n=len(z)
    stats[a][key]+=np.asarray([n*weight,float(z.sum())*weight,float((z*z).sum())*weight,n])
  ledger.write(json.dumps({k:row[k] for k in ['observation_id','sequence_hash','source_sha','split','role','coordinate_path','coordinate_row']})+'\n');processed+=1
 for path,items in grouped.items():
  if items[0]['coordinate_kind']=='legacy_cache':
   wanted={r['coordinate_row']:r for r in items};offset=0
   for batch in pq.ParquetFile(path).iter_batches(batch_size=64,columns=['selected_calpha_coordinates'],use_threads=False):
    for i in range(len(batch)):
     if offset+i in wanted:consume(wanted[offset+i],json.loads(batch.column(0)[i].as_py()))
    offset+=len(batch)
  elif items[0]['coordinate_kind'] in ['priority_v3','regression_v3']:
   data=json.load(gzip.open(path,'rt'))
   for row in items:consume(row,[r['ca'] for r in data['observations'][row['coordinate_row']]['residue_mapping']])
  else:raise ValueError('unrecognized coordinate source')
 ledger.close();assert processed==len(rows)
 out={}
 for a,features in stats.items():
  out[a]={}
  for name,(weight,total,total2,count) in features.items():
   mean=total/weight if weight else None;variance=max(0,total2/weight-mean**2) if weight else None
   out[a][name]={'valid':bool(weight),'weighted_mass':float(weight),'observed_positions':int(count),'mean':mean,'variance':variance}
 m=load();payload={'status':'CERTIFIED_TRAIN_ONLY_DESCRIPTIVE_PRIORS','dataset_ref':raw,'prior_version':'train_ca_geometry_v1','chemistry_registry':ref().__dict__,'approved_online_features':list(m.APPROVED_NAMES),'fitted_observations':processed,'unique_TRAIN_sequences':len(nobs),'fit_input_sha256':file_sha(D/'TRAIN_FIT_INPUTS.jsonl'),'weighting':'one vote per exact full sequence; uniform eligible observations within sequence; each geometry component normalized over its valid positions within observation; residue-conditioned statistics then normalized by observed weighted mass','units':'distance Angstrom, theta/tau radians; transformed quantities dimensionless','distance_support':'positive, empirical moments are not universal physical limits','torsion':'signed periodic; circular sine/cosine moments, not rotamers or atomistic chemistry','deferred_features_activated':False,'native_future_retrieval':False,'production_model_changed':False,'statistics':out}
 (D/'PRIORS.json').write_text(json.dumps(payload,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in payload.items() if k!='statistics'}),flush=True)
if __name__=='__main__':main()
