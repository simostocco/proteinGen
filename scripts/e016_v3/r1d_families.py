"""CPU-only operational family graph; BioPython in installed AF environment."""
import json,sys,hashlib
from pathlib import Path
from Bio.Align import PairwiseAligner
R=Path(__file__).resolve().parents[3]
def main():
 ps=json.load(open(R/'analysis/PARENT_PRELIMINARY.json'))['parents'];names=sorted(ps);root={n:n for n in names};edges=[]
 def find(n):
  while root[n]!=n:root[n]=root[root[n]];n=root[n]
  return n
 def join(a,b,reason,detail=None):
  ra,rb=find(a),find(b)
  if ra!=rb:root[max(ra,rb)]=min(ra,rb)
  edges.append({'a':a,'b':b,'reason':reason,'detail':detail})
 for key in ['WT_cluster','PDB_accession']:
  seen={}
  for n in names:
   v=ps[n].get(key)
   if v and v not in ['nan','-','None']:
    if v in seen:join(n,seen[v],key)
    else:seen[v]=n
 seen={}
 for n in names:
  for g in ps[n].get('E016_protected_groups',[]):
   if g in seen:join(n,seen[g],'same accepted E016 protected sequence group')
   else:seen[g]=n
 aligner=PairwiseAligner();aligner.mode='global';aligner.match_score=2.;aligner.mismatch_score=-1.;aligner.open_gap_score=-8.;aligner.extend_gap_score=-1.
 for k,a in enumerate(names):
  sa=ps[a]['sequence']
  for b in names[k+1:]:
   sb=ps[b]['sequence']
   if not sa or not sb:continue
   q=aligner.align(sa,sb)[0];matches=0;paired=0
   for (a0,a1),(b0,b1) in zip(*q.aligned):
    x=sa[a0:a1];y=sb[b0:b1];assert len(x)==len(y);paired+=len(x);matches+=sum(c==d for c,d in zip(x,y))
   identity=matches/max(paired,1);coverage=paired/min(len(sa),len(sb))
   if identity>=.50 and coverage>=.80:join(a,b,'global sequence identity>=.50 and shorter coverage>=.80',{'matches':matches,'paired_columns':paired,'identity':identity,'shorter_coverage':coverage})
 groups={}
 for n in names:groups.setdefault(find(n),[]).append(n)
 family={n:hashlib.sha256('|'.join(sorted(ns)).encode()).hexdigest()[:16] for ns in groups.values() for n in ns}
 out={'method':{'author_WT_clusters':True,'PDB_accession_merge':True,'accepted_E016_protected_group_merge':True,'global_alignment':{'match':2,'mismatch':-1,'gap_open':-8,'gap_extend':-1,'identity_threshold':.5,'shorter_aligned_coverage':.8},'limitation':'operational family separation, remote homology below criteria and structural convergence are not excluded; short-sequence identity is not evolutionary independence'},'family_by_parent':family,'families':[{'id':family[ns[0]],'parents':ns} for ns in groups.values()],'edges':edges}
 p=R/'analysis/FAMILY_GRAPH.json';p.write_text(json.dumps(out,indent=2));print(json.dumps({'parents':len(names),'components':len(groups),'edges':len(edges)}))
if __name__=='__main__':main()
