"""Freeze the 32-control panel and independently audit existing coordinates, no network/GPU."""
import json,hashlib,os,subprocess,math,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from protein_joint_sac.data.scaled_supervised import Data
from protein_joint_sac.data.accepted_supervised import CoordinateReader
from protein_joint_sac.data.certified_unified import file_sha
from protein_joint_sac.rl.sequence_rewards import mapped_structure_scores
R=Path(__file__).resolve().parents[3];PREV=R.parent/'e016_sequence_first_rl_readiness_20261019_v1';TOOL=Path('/mnt/d/Simone/proteinGen/e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1/external_designability_loop_v1/tools/USalign/USalign')
AA3=dict(zip('ACDEFGHIKLMNPQRSTVWY','ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR'.split()))
def write(n,x):
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.part');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n' if not isinstance(x,str) else x);os.replace(tmp,p)
def pdb(path,x,seq):
 with open(path,'w') as f:
  for i,(p,a) in enumerate(zip(x,seq)):
   if p is None or not np.isfinite(p).all():continue
   f.write(f'ATOM  {i+1:5d}  CA  {AA3[a]} A{i+1:4d}    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00           C\n')
  f.write('TER\nEND\n')
def aligned_deviations(x,y):
 mask=np.isfinite(x).all(1)&np.isfinite(y).all(1);a=x[mask];b=y[mask];ca=a.mean(0);cb=b.mean(0);u,_,vh=np.linalg.svd((a-ca).T@(b-cb));rot=u@np.diag([1,1,np.linalg.det(u@vh)])@vh;return np.linalg.norm((a-ca)@rot+cb-b,axis=1)
def comparisons(control,result,tag):
 seq=control['sequence'];x=np.array([v if v is not None else [np.nan]*3 for v in control['native_ca']]);y=np.asarray(result,float);mask=np.isfinite(x).all(1)&np.isfinite(y).all(1);out=mapped_structure_scores(x,np.where(np.isfinite(y),y,0.),observed=mask);d=aligned_deviations(x,y);out.update({'Kabsch_per_observed_residue_deviation_A':d.tolist(),'observed_polymer_positions':(np.flatnonzero(mask)+1).tolist(),'Kabsch_deviation_quantiles_A':np.quantile(d,[0,.25,.5,.75,.9,.99,1]).tolist()});a=R/'alignment'/f'{tag}_native.pdb';b=R/'alignment'/f'{tag}_predicted.pdb';pdb(a,x,seq);pdb(b,y,seq)
 run=subprocess.run([str(TOOL),str(a),str(b),'-mol','prot'],capture_output=True,text=True,timeout=45);write(f'alignment/{tag}_USalign.txt',run.stdout+run.stderr);import re
 scores=re.findall(r'TM-score=\s*([\d.]+).*?normalized by length of (\w+)',run.stdout);out['USalign']={'returncode':run.returncode,'scores':[{'value':float(v),'normalization':n} for v,n in scores],'output_path':str(R/'alignment'/f'{tag}_USalign.txt'),'native_observed_length':int(np.isfinite(x).all(1).sum()),'predicted_observed_length':int(np.isfinite(y).all(1).sum()),'full_target_length':len(y),'optimized_residue_mapping_is_not_authoritative_map':True}
 out['regional_windows']=[]
 for lo in range(0,len(seq)-31,64):
  hi=min(lo+64,len(seq));m=np.isfinite(x[lo:hi]).all(1)&np.isfinite(y[lo:hi]).all(1)
  if m.sum()>=4:out['regional_windows'].append({'start':lo+1,'end':hi,'fixed_map_window':mapped_structure_scores(x[lo:hi],np.where(np.isfinite(y[lo:hi]),y[lo:hi],0.),observed=m),'interpretation':'fixed sequence windows, not annotated biological domains'})
 return out

def main():
 if (R/'configs/NATIVE_PANEL_32.json').exists():print('Panel already frozen; resume alignment only.');audit(json.load(open(R/'configs/NATIVE_PANEL_32.json'))['controls']);return
 data=Data(False);reader=CoordinateReader(data.proposal.external);old=json.load(open(PREV/'configs/NATIVE_CONTROL_PANEL.json'));controls=[c for c in old['controls'] if c['kind']=='native'];used={c['sequence_id'] for c in controls};groups={data.proposal.sequences[h]['protected_group_id'] for h in used}
 for role,count in [('TRAIN',24),('VALIDATION',8)]:
  for k in range(count-sum(c['split']==role for c in controls)):
   lo,hi=[(20,64),(65,128),(129,256),(257,384),(385,500)][k%5];pool=[h for h in data.jointids[role] if h not in used and lo<=len(data.seqs[h])<=hi and data.proposal.sequences[h]['protected_group_id'] not in groups];h=min(pool,key=lambda h:hashlib.sha256(('R1b20261019'+h).encode()).hexdigest());o=data.owners[h][0];row=data.rows[o];x=reader.read(row);controls.append({'id':role.lower()+f'_r1b_{k+1:02d}','kind':'native','split':role,'sequence_id':h,'sequence':data.seqs[h],'N':len(data.seqs[h]),'native_ca':x.tolist(),'observed_positions':list(range(1,len(x)+1)),'observation_id':o,'source_sha256':row['source_sha'],'entry_id':row['entry_id'],'historical_case':False,'seed':151502,'selection_rationale':'stable SHA ranking within rotating fixed E016 length strata, unique protected group; no new AFDB scores inspected'});used.add(h);groups.add(data.proposal.sequences[h]['protected_group_id'])
 wanted={c['observation_id'] for c in controls if 'observation_id' in c};full={}
 for role in ['TRAIN','VALIDATION']:
  for row in data.proposal.iter_joint(role):
   if row['observation_id'] in wanted:full[row['observation_id']]=row
 meta={};mapping=[]
 for c in controls:
  assert data.proposal.sequences[c['sequence_id']]['canonical_full_sequence']==c['sequence'] and data.proposal.sequences[c['sequence_id']]['split']==c['split']
  if c.get('observation_id'):
   row=full[c['observation_id']];c['observation_metadata']=row;c['entry_id']=str(row['entry_id']).lower();x=reader.read(row);oldx=np.array(c['native_ca']);err=aligned_deviations(x,oldx);c['stored_vs_authoritative_CA_Kabsch_max_A']=float(err.max());assert err.max()<1e-7
   if row['coordinate_kind']=='legacy_cache':
    path=Path(row['coordinate_path']);f=pq.ParquetFile(path);idx=int(row['coordinate_row']);off=0
    for rg in range(f.num_row_groups):
     n=f.metadata.row_group(rg).num_rows
     if idx<off+n:
      cols=[n for n in f.schema.names if n not in ['selected_calpha_coordinates']];detail=f.read_row_group(rg,columns=cols,use_threads=False).slice(idx-off,1).to_pylist()[0];break
     off+=n
    meta[c['id']]=detail
   else:meta[c['id']]=reader.small(row['coordinate_path'])['observations'][int(row['coordinate_row'])]
  else:c['entry_id']=c['id'].split('historical_')[-1].split('_')[0]
  mapping.append({'id':c['id'],'N':c['N'],'observed_count':len(c['observed_positions']),'canonical_sequence_sha_verified':True,'stored_CA_provenance_verified':True,'source_metadata_keys':list(meta.get(c['id'],{}))})
 assert len(controls)==32 and sum(c['split']=='TRAIN' for c in controls)==24 and len({c['sequence_id'] for c in controls})==32
 write('configs/NATIVE_PANEL_32.json',{'controls':controls,'selected_before_new_AFDB_scores':True,'TRAIN':24,'VALIDATION':8,'selection_seed':'R1b20261019','old_panel_sha256':file_sha(PREV/'configs/NATIVE_CONTROL_PANEL.json'),'split_manifest_sha256':file_sha(data.proposal.root/'PROPOSAL_MANIFEST.json'),'model_download_limit':40,'download_bytes_limit':750*2**20,'fit_vs_holdout_rule':'TRAIN ids sorted SHA, first16 fit, remaining8 heldout; VALIDATION8 audit only','MSA_first_six':[c['id'] for c in controls[:8] if c['split']=='TRAIN'],'MSA_conditional_audit':['validation_96','validation_256'],'no_TEST':True})
 write('native/SOURCE_METADATA.json',meta);write('native/MAPPING_PRECHECK.json',mapping)
 audit(controls)
def audit(controls):
 rows=[]
 for c in controls[:8]:
  result=json.load(open(PREV/'oracle'/('v3_'+c['id'])/'RESULT.json'));z=comparisons(c,result['predicted_ca'],'single_'+c['id']);previous=next(v for v in json.load(open(PREV/'ALPHAFOLD_NATIVE_CONTROL_RESULTS.json'))['rows'] if v['id']==c['id'] and v['status']=='SUCCEEDED');z['original_mapped_score_delta']=z['robust_fixed_map_TM_similarity_fullN']-previous['recovery']['robust_fixed_map_TM_similarity_fullN'];assert abs(z['original_mapped_score_delta'])<1e-10;rows.append({'id':c['id'],'split':c['split'],'N':c['N'],'diagnostics':z})
 write('STRUCTURE_ALIGNMENT_METHOD_COMPARISON.json',{'scope':'independent recalculation of prior eight native controls, historical scores preserved','USalign_binary_path':str(TOOL),'USalign_binary_sha256':file_sha(TOOL),'rows':rows});print(json.dumps({'panel':32,'native_precheck':'native/MAPPING_PRECHECK.json','alignment_recomputed':8}),flush=True)
if __name__=='__main__':main()
