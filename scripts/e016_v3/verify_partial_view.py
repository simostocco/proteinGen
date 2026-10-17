"""Independent reference-view checks; no alignments, coordinate imputation or TEST evaluation."""
import collections,json,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from protein_joint_sac.data.pdb_split_proposal import PDBSplitProposal
from protein_joint_sac.data.certified_unified import file_sha
from prepare_partial_view import R,P,U,write
BANDS=[(20,64),(65,128),(129,256),(257,384),(385,500)]
def main():
 start=time.monotonic();p=PDBSplitProposal(P);v=R/'view/PARTIAL_OBSERVATIONS.parquet';rows=pq.read_table(v).to_pylist();index={r['observation_id']:r for r in rows}
 if len(index)!=len(rows):raise ValueError('duplicated observation identity')
 counts={s:collections.Counter() for s in ['TRAIN','VALIDATION']};lengths={s:collections.Counter() for s in counts};methods={s:collections.Counter() for s in counts};candidate_ids={s:set() for s in counts};admitted_ids={s:set() for s in counts};unique={s:{} for s in counts};distance_union={};observed_ids=set()
 for r in rows:
  h=r['sequence_hash'];parent=p.sequences[h];s=r['split'];N=parent['length']
  if s not in counts or parent['split']!=s or r['polymer_length']!=N:raise ValueError('observation ownership contradiction')
  seen=np.array(r['ca_observed'],bool);link=np.array(r['continuity'],bool)
  if seen.shape!=(N,) or link.shape!=(N-1,) or all(seen) or len(r['aa_known'])!=N or not all(r['aa_known']):raise ValueError('CA-partial canonical view/mask contradiction')
  d=link&seen[:-1]&seen[1:];masks=[np.r_[False,d],np.r_[False,False,d[:-1]&d[1:]],np.r_[False,False,False,d[:-2]&d[1:-1]&d[2:]]]
  if [int(x.sum()) for x in masks]!=[r['distance_targets'],r['angle_targets'],r['torsion_targets']]:raise ValueError('independent stencil counts disagree')
  if N-int(seen.sum())!=r['AA_targets_at_missing_CA'] or N-int(d.sum())!=r['AA_only_targets']:raise ValueError('AA missingness/target denominator')
  heads=seen&np.r_[True,~d];segments=np.where(seen,np.cumsum(heads)-1,-1)
  if segments.tolist()!=r['coordinate_segment_id']:raise ValueError('independent segment identifiers disagree')
  sizes=np.bincount(segments[seen]);c=counts[s];c['observations']+=1;c['AA_targets']+=N;c['AA_at_missing_CA']+=int((~seen).sum());c['AA_only_targets']+=r['AA_only_targets'];c['coordinate_segments']+=len(sizes)
  for k,name in enumerate(['distance','angle','torsion']):c[name+'_targets']+=int(masks[k].sum());c['segments_at_least_'+str(k+2)]+=int((sizes>=k+2).sum())
  flags=unique[s].setdefault(h,[0]*4)
  for k,a in enumerate(masks+[~seen]):flags[k]|=int.from_bytes(np.packbits(a,bitorder='little').tobytes(),'little')
  admitted_ids[s].add(h);lengths[s][next(str(lo)+'-'+str(hi) for lo,hi in BANDS if lo<=N<=hi)]+=1;methods[s][r['method']]+=1
  if not any(x in r['method'] for x in ['X-RAY DIFFRACTION','ELECTRON MICROSCOPY','SOLUTION NMR']):raise ValueError('unsupported experimental-method policy: '+r['method'])
 # Prove journal replay reproduces every admitted legacy observation link.
 for path in sorted((R/'view/continuity_recovery_journal').glob('*.json')):
  j=json.loads(path.read_text())
  if j['version']!='continuity-recovery-v1':raise ValueError('journal version')
  for oid,links,reference in j['links']:
   if oid in index:
    for k in links:distance_union[oid]=distance_union.get(oid,0)|(1<<k)
 for oid,r in index.items():
  if r['coordinate_kind']=='legacy_cache':
   actual=sum((1<<i) for i,x in enumerate(r['continuity']) if x)
   if actual!=distance_union.get(oid,0):raise ValueError('durable continuity replay differs from published view')
 # Input ownership and exclusive exclusion denominators, without decoding coordinates.
 ip=U/'release/intake_v1/chain_observations.parquet';candidate_counts=collections.Counter()
 cols=['observation_id','sequence_hash','complete_CA','source_sha','coordinate_path','coordinate_row','model','mapping_status']
 for b in pq.ParquetFile(ip).iter_batches(batch_size=4096,columns=cols,use_threads=False):
  for x in b.to_pylist():
   parent=p.sequences.get(x['sequence_hash'])
   if parent is None or parent['split'] not in counts or x['complete_CA']:continue
   s=parent['split'];candidate_counts[s]+=1;candidate_ids[s].add(x['sequence_hash']);r=index.get(x['observation_id'])
   if r:
    if x['observation_id'] in observed_ids:raise ValueError('duplicate input observation')
    observed_ids.add(x['observation_id'])
    for k in ['sequence_hash','source_sha','coordinate_path','coordinate_row','model']:
     if x[k]!=r[k]:raise ValueError('intake reference changed: '+k)
    if x['mapping_status']!='VERIFIED_LABEL_CORRESPONDENCE':raise ValueError('unverified mapping admitted')
 if observed_ids!=set(index):raise ValueError('admitted observation absent from source population')
 out={'status':'PASS','view_sha256':file_sha(v),'split_manifest_sha256':file_sha(P/'PROPOSAL_MANIFEST.json'),'independent_checks':['owner/source/model match','observed-only contiguous 2/3/4-anchor stencils','segment recomputation','journal replay union equality'],'splits':{},'scope':'at least one missing CA; all-CA observations failing previous full-chain qualification remain outside this view; no TEST admission or evaluation','seconds':time.monotonic()-start}
 for s in counts:
  full={h for h,r in p.sequences.items() if r['split']==s and r['joint_eligible']}
  out['splits'][s]={'observations_and_targets':dict(counts[s]),'partial_candidate_observations':candidate_counts[s],'excluded_or_unresolved_partial_observations':candidate_counts[s]-counts[s]['observations'],'candidate_identities':len(candidate_ids[s]),'admitted_partial_identities':len(admitted_ids[s]),'candidate_identities_without_admitted_partial':len(candidate_ids[s]-admitted_ids[s]),'additional_to_complete_identities':len(admitted_ids[s]-full),'length_band_observations':dict(lengths[s]),'method_observations':dict(methods[s]),'unique_identity_position_distance_angle_torsion_missing_AA':[sum(a[k].bit_count() for a in unique[s].values()) for k in range(4)]}
 write(R/'view/PARTIAL_VIEW_QA.json',out);print(json.dumps(out),flush=True)
if __name__=='__main__':main()
