"""Independent traceback accounting, small scalar SW spotchecks, read-only BC checks."""
import collections,hashlib,json,math,time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from practical_split import *
from practical_allocate import experimental_methods

# Deliberately independent scalar implementation / scoring, not MMseqs bindings.
# Not an exhaustive search and not an estimate of evolutionary homology.
def scalar_sw(a,b):
 scores=np.zeros((len(a)+1,len(b)+1),dtype=np.int32);trace=np.zeros_like(scores,dtype=np.int8);best=(0,0,0)
 for i,x in enumerate(a,1):
  for j,y in enumerate(b,1):
   opts=(0,int(scores[i-1,j-1])+(2 if x==y else -1),int(scores[i-1,j])-2,int(scores[i,j-1])-2)
   k=max(range(4),key=opts.__getitem__);v=opts[k];scores[i,j]=v;trace[i,j]=k
   if v>best[0]:best=(v,i,j)
 score,i,j=best;qe,te=i,j;ni=al=0
 while i and j and trace[i,j]:
  k=trace[i,j];al+=1
  if k==1:ni+=a[i-1]==b[j-1];i-=1;j-=1
  elif k==2:i-=1
  else:j-=1
 return {'nident':int(ni),'alnlen':al,'qstart':i+1,'qend':qe,'tstart':j+1,'tend':te,'qlen':len(a),'tlen':len(b),'score':score}

def traceback_check(root):
 p,by,active=load()
 if (root/'traceback_spotcheck/RESULT.json').exists():
  d=json.load(open(root/'traceback_spotcheck/RESULT.json'));assert sha(root/'traceback_spotcheck/TRACEBACK.tsv')==d['traceback_tsv_sha256'];return d['actual_aligned_strings_checked']
 first=next((root/'alignment_coverage_v1').glob('batch-*/DONE.json')).parent
 native=first/'result1';reader=Reader(native);out=root/'traceback_spotcheck';out.mkdir(exist_ok=True);w=Writer(out/'selected',kind=5);num=0
 for q,body in reader.entries():
  lines=body.splitlines(keepends=True)
  if not lines:continue
  w.add(q,b''.join(lines[:3]));num+=min(len(lines),3)
  if num>=90:break
 reader.close();w.close()
 fmt='query,target,nident,alnlen,qstart,qend,tstart,tend,qlen,tlen,qaln,taln'
 run(out,'convert',['/usr/bin/mmseqs','convertalis',str(H/'all_db'),str(H/'all_db'),str(out/'selected'),str(out/'TRACEBACK.tsv'),'--format-output',fmt,'--threads','1'])
 checked=0
 for line in (out/'TRACEBACK.tsv').open():
  v=line.rstrip().split('\t');a,b,ni,al,qs,qe,ts,te,ql,tl=map(int,v[:10]);qa,ta=v[10:]
  assert len(qa)==len(ta)==al
  assert qa.replace('-','')==p['sequences'][a][qs-1:qe] and ta.replace('-','')==p['sequences'][b][ts-1:te]
  assert sum(x==y and x!='-' for x,y in zip(qa,ta))==ni
  checked+=1
 save(out/'RESULT.json',{'status':'PASS','traceback_tsv_sha256':sha(out/'TRACEBACK.tsv'),'actual_aligned_strings_checked':checked,'checks':'nident, alignment column count, inclusive full sequence coordinate span and exact source subsequence; includes gaps'})
 return checked

def independent_check(root):
 audit=json.load(open(root/'CROSS_SPLIT_AUDIT.json'));p,by,active=load();pairs=[];seen=set()
 for a,b in audit['independent_spotcheck_pairs']:
  if (a,b) not in seen:pairs.append((a,b));seen.add((a,b))
 pairs=pairs[:20];out=[];bad=[]
 for a,b in pairs:
  v=scalar_sw(by[a]['canonical_full_sequence'],by[b]['canonical_full_sequence']);flags=match(*(v[k] for k in ('nident','alnlen','qstart','qend','tstart','tend','qlen','tlen')));row={'query':a,'target':b,**v,'protected_under_alternative_alignment':any(flags)};out.append(row)
  if any(flags):bad.append(row)
 traceback_count=traceback_check(root)
 save(root/'INDEPENDENT_SPOTCHECK.json',{'status':'PASS' if not bad else 'FAIL_KNOWN_ADDITIONAL_CONFLICT','scalar_SW_scoring':'match+2 mismatch-1 linear gap-2, local, no masking/composition correction; independent implementation; non-equivalent scoring to MMseqs','real_cross_split_pairs_checked':len(out),'traceback_accounting_pairs_checked':traceback_count,'additional_conflicts':bad,'pairs':out,'limitations':'small deterministic spotcheck, not all-pair proof or remote homology test'})
 assert not bad,'independent detected conflict must be repaired'


def loader_check(root):
 from protein_joint_sac.data.pdb_split_proposal import PDBSplitProposal
 from protein_joint_sac.data.certified_unified import native_coordinates,canonical_native_prefix
 from protein_joint_sac.geometry.ca_builder import append
 from protein_joint_sac.geometry.local_frames import features
 from protein_joint_sac.chemistry.reference_adapter import reference_features,load as chemical_load
 import torch
 from protein_joint_sac.rl.bc_training import Population
 from types import SimpleNamespace
 torch.set_num_threads(1);proposal=PDBSplitProposal(root);out=[];maxerr=0;chemchecks=0
 def same(a,b):
  if isinstance(a,dict):assert a.keys()==b.keys();[same(a[k],b[k]) for k in a]
  elif isinstance(a,(tuple,list)):assert len(a)==len(b);[same(x,y) for x,y in zip(a,b)]
  elif isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=True)
  elif isinstance(a,np.ndarray):np.testing.assert_equal(a,b)
  else:assert a==b
 selected=[]
 for split in SPLITS:
  counts=collections.Counter();method_counts=collections.Counter();seen_ids=set()
  for row in proposal.iter_bc_rows(split,mode='joint',evaluation_only=True):
   b=row['length_band']
   method='+'.join(experimental_methods(json.loads(row['metadata'])))
   if counts[b]<1 or (method in ('X-RAY DIFFRACTION','ELECTRON MICROSCOPY','SOLUTION NMR') and method_counts[method]<1):
    if row['observation_id'] not in seen_ids:selected.append(row);seen_ids.add(row['observation_id'])
    counts[b]+=1;method_counts[method]+=1
   if len(counts)==5 and all(v>=1 for v in counts.values()) and len(method_counts)>=3:break
  assert len(counts)==5
 for row in selected:
  parent=proposal.sequences[row['sequence_hash']];seq=parent['canonical_full_sequence'];x=native_coordinates(row);N=len(seq)
  for t in sorted({0,1,2,3,min(7,N-1),N-1}):
   target=proposal.teacher_prefix(row,t)
   legacy_row=row
   assert legacy_row['sequence']==seq and legacy_row['label_positions']==list(range(1,N+1))
   # Read-only preprocessing method only: no Population admission or BCTrainer.
   view=SimpleNamespace(rows=(legacy_row,),coordinates=lambda index:x)
   joint_state,joint_label,joint_geometry=Population.target(view,0,t,'joint','cpu')
   assert joint_label=='ACDEFGHIKLMNPQRSTVWY'.index(seq[t]);torch.testing.assert_close(joint_state.ca,target.state.ca,atol=0,rtol=0);torch.testing.assert_close(joint_geometry,target.next_action.geometry,atol=0,rtol=0)
   def forbidden_geometry(index):raise AssertionError('sequence-only loader accessed coordinates')
   sequence_view=SimpleNamespace(rows=(legacy_row,),coordinates=forbidden_geometry)
   sequence_state,sequence_label,no_geometry=Population.target(sequence_view,0,t,'sequence','cpu')
   assert sequence_state.residues==target.state.residues and sequence_label==joint_label and no_geometry is None and not sequence_state.valid.any()
   nextstate=append(target.state,target.next_action);expected=canonical_native_prefix(seq,x,t+1);err=float((nextstate.ca-expected.ca).abs().max());maxerr=max(maxerr,err);assert err<1e-8
   y=x.copy();y[t:]=np.nan;swap=seq[:t]+'W'*(N-t);other=canonical_native_prefix(swap,y,t)
   assert other.residues==target.state.residues;torch.testing.assert_close(other.ca,target.state.ca,atol=0,rtol=0)
   same(features(other),features(target.state));same(reference_features(other),reference_features(target.state));chemchecks+=1
  # Independent direct cache field checks for sampled legacy mappings.
  if row['coordinate_kind']=='legacy_cache':
   pf=pq.ParquetFile(row['coordinate_path']);offset=0;source=None
   columns=['pdb_id','auth_asym_id','model_id','scheme_sequence','label_sequence_ids','missing_calpha_mask','missing_residue_mask']
   for batch in pf.iter_batches(batch_size=64,columns=columns,use_threads=False):
    if offset+len(batch)>row['coordinate_row']:source=batch.to_pylist()[int(row['coordinate_row'])-offset];break
    offset+=len(batch)
   assert source['pdb_id'].upper()==row['entry_id'].upper() and source['auth_asym_id']==row['auth_chain']
   assert source['scheme_sequence']==seq
   assert list(map(int,json.loads(source['label_sequence_ids'])))==list(range(1,N+1))
   assert not any(json.loads(source['missing_calpha_mask'])) and not any(json.loads(source['missing_residue_mask']))
  out.append({'observation_id':row['observation_id'],'sequence_id':row['sequence_hash'],'split':row['split'],'length_band':row['length_band'],'length':N,'prefixes_checked':6})
 matched_rows=pq.read_table(root/'MATCHED_JOINT_SEQUENCE_VIEW.parquet',columns=['sequence_id','split']).to_pylist()
 assert len({r['sequence_id'] for r in matched_rows})==len(matched_rows)
 assert {(r['sequence_id'],r['split']) for r in matched_rows}=={(h,r['split']) for h,r in proposal.sequences.items() if r['joint_eligible']}
 for s in SPLITS:
  seqids={r['sequence_id'] for r in proposal.iter_sequences(s,matched_joint=True)};jointids={r['sequence_hash'] for r in proposal.iter_joint(s)};assert seqids==jointids
 try:proposal.training_population();raise AssertionError('optimizer admission unexpectedly permitted')
 except ValueError:pass
 result={'status':'PASS','observations':len(out),'prefix_checks':chemchecks,'roundtrip_max_absolute_error_A':maxerr,'approved_chemical_features':list(chemical_load().APPROVED_NAMES),'coordinate_sources_fresh_hash_checked':len(proposal.checked),'all_split_matched_identity_ownership_pass':True,'training_authorized':False,'existing_bc_Population_target_compatibility_checks':chemchecks,'sequence_geometry_isolation':True,'optimizer_or_BCTrainer_constructed':False,'samples':out};save(root/'LOADER_INTEGRATION_TEST.json',result);print('LOADER',result['observations'],result['prefix_checks'],'maxerr',maxerr,flush=True)
