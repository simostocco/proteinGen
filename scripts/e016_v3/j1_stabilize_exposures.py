"""Metadata-only sampler replay: verify every actual batch hash and exact target coverage."""
import collections,gc,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pyarrow.parquet as pq
import torch
import j1_stabilize_train as t
from protein_joint_sac.data.scaled_supervised import Data
R=t.R

def main():
 done=t.read(R/'runs/J1/COMPLETED.json');n=done['new_optimizer_updates'];origin=t.c.s.load_checked(R/'replay/checkpoint_00500_full.pt');end=t.c.s.load_checked(done['latest_checkpoint']);base=Data(False);owners=collections.defaultdict(list);cache={}
 p=R.parent/'e016_partial_supervision_20261016_v1/view/PARTIAL_OBSERVATIONS.parquet'
 for row in pq.read_table(p,columns=['observation_id','sequence_hash','ca_observed','split'],filters=[('split','=','TRAIN')]).to_pylist():
  owners[row['sequence_hash']].append(row['observation_id']);cache[row['observation_id']]={'seen':np.asarray(row['ca_observed'],bool)}
 for v in owners.values():v.sort()
 data=SimpleNamespace(base=base,seqs=base.seqs,owners={'TRAIN':owners},cache=cache,ids={'TRAIN':sorted(owners)});sampler=t.c.s.Sampler(data);sampler.load_state_dict(origin['sampler']);traces={}
 for p in (R/'runs/J1').glob('updates_*.jsonl'):
  for line in open(p):
   v=json.loads(line);traces[v['new_update']]=v
 aa_bits=collections.defaultdict(int);geo_bits=collections.defaultdict(int);obs_bits=collections.defaultdict(int);bands=collections.Counter();deciles=collections.Counter();sources=collections.Counter();missing=0
 for u in range(1,n+1):
  aa,geo=sampler.draw()
  if t.c.s.j.digest([aa,geo])!=traces[u]['sample_order_sha256']:raise ValueError('sampler replay differs at'+str(u))
  for h,o,pos in aa:
   aa_bits[h]|=1<<pos;obs_bits[(h,o)]|=1<<pos;kind='partial' if o in cache else 'complete';sources[kind]+=1;bands[(kind,t.c.s.j.band(len(base.seqs[h])))]+=1;deciles[min(9,int(10*pos/len(base.seqs[h])))]+=1
   if kind=='partial':missing+=int(not cache[o]['seen'][pos])
  for h,o,pos in geo:geo_bits[h]|=1<<pos
 if not t.c.same(sampler.state_dict(),end['sampler']):raise ValueError('terminal sampler state differs')
 out={'status':'PASS','metadata_only_no_coordinates_or_optimizer_updates':True,'all_actual_batch_hashes_matched':n,'terminal_sampler_state_exact':True,'AA_total':64*n,'geometry_only_total':32*n,'partial_AA':sources['partial'],'complete_AA':sources['complete'],'partial_AA_fraction':sources['partial']/(64*n),'partial_AA_at_missing_CA':missing,'unique_AA_identities':len(aa_bits),'distinct_AA_identity_position_targets':sum(v.bit_count() for v in aa_bits.values()),'repeated_AA_targets_within_new_lineage':64*n-sum(v.bit_count() for v in aa_bits.values()),'distinct_AA_observation_position_contexts':sum(v.bit_count() for v in obs_bits.values()),'distinct_geometry_identity_positions':sum(v.bit_count() for v in geo_bits.values()),'geometry_identities':len(geo_bits),'AA_length_band_counts':{str(k):v for k,v in bands.items()},'AA_position_deciles':dict(deciles),'source_and_end_sampler_cursors':[origin['sampler']['cursor'],end['sampler']['cursor']],'exposure_count_deltas':{k:v-origin['sampler']['counts'].get(k,0) for k,v in end['sampler']['counts'].items()}}
 out['new_AA_identity_positions_beyond_source_coverage']=sum((v&~origin['sampler']['coverage']['visited'].get(h,0)).bit_count() for h,v in aa_bits.items());out['cumulative_end_AA_coverage']=sampler.cover.report()
 if missing!=out['exposure_count_deltas'].get('partial_AA_missing_CA',0):raise ValueError('missing CA exposure count contradiction')
 t.write(R/'SAMPLING_MIXTURE_AND_EXPOSURES.json',out);print('EXPOSURE_AND_COVERAGE_VERIFIED',n)
if __name__=='__main__':main()
