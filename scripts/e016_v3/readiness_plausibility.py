"""CPU-only bounded TRAIN nearest-sequence diagnostic; no large homology index."""
import hashlib,json,heapq,difflib,time
from pathlib import Path
import pyarrow.parquet as pq
R=Path(__file__).resolve().parents[3]
def main():
 start=time.monotonic();p=R.parent/'pdb_supervised_split_50id80cov_20261016_v1/SEQUENCE_ASSIGNMENTS.parquet';rows=pq.read_table(p,columns=['sequence_id','canonical_full_sequence','split'],filters=[('split','==','TRAIN')]).to_pylist();assert len(rows)==118219
 exact={r['sequence_id'] for r in rows};queries={}
 for n in ['J1','fallback']:queries[n]=json.load(open(R/f'generated/{n}_initial64.json'))['sequence']
 for q in json.load(open(R/'configs/NATIVE_CONTROL_PANEL.json'))['controls']:
  if q['kind']=='variant':queries[q['id']]=q['sequence']
 kmers=lambda s:set(s[i:i+4] for i in range(len(s)-3));queryk={k:kmers(v) for k,v in queries.items()};heaps={k:[] for k in queries}
 for row in rows:
  sk=kmers(row['canonical_full_sequence'])
  for k,q in queryk.items():
   score=len(sk&q)/max(1,len(sk|q));item=(score,row['sequence_id'],row['canonical_full_sequence'])
   if len(heaps[k])<16:heapq.heappush(heaps[k],item)
   elif item>heaps[k][0]:heapq.heapreplace(heaps[k],item)
 out={}
 for k,sequence in queries.items():
  evaluated=[{'TRAIN_sequence_id':h,'fourmer_Jaccard':score,'matching_block_ratio':difflib.SequenceMatcher(None,sequence,s,autojunk=False).ratio(),'length':len(s)} for score,h,s in heaps[k]];out[k]={'sequence_sha256':hashlib.sha256(sequence.encode()).hexdigest(),'exact_TRAIN_sequence_overlap':hashlib.sha256(sequence.encode()).hexdigest() in exact,'candidate_scan_TRAIN_identities':118219,'shortlist_size':16,'shortlist':sorted(evaluated,key=lambda z:z['matching_block_ratio'],reverse=True),'nearest_homology_claim':False}
 (R/'GENERATED_SEQUENCE_PLAUSIBILITY.json').write_text(json.dumps({'scope':'fourmer shortlist then matching-block similarity; not exhaustive alignment identity or evolutionary novelty','queries':out,'seconds':time.monotonic()-start,'no_retired_prefilter_rebuilt':True,'no_new_dataset':True},indent=2)+'\n');print('Bounded TRAIN-only sequence plausibility scan complete.',flush=True)
if __name__=='__main__':main()
