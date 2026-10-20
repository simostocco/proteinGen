from r1e_common import *
import subprocess,time,threading,collections
start=time.monotonic();runs=[]
for target in ['TRAIN','VALIDATION','queries']:
 guard(400*2**20)
 out=R/'results'/f'align_{target}.tsv';tmp=R/'runtime'/f'mmseqs_{target}';log=R/'results'/f'mmseqs_{target}.log'
 cmd=['/usr/bin/mmseqs','easy-search',str(R/'data/queries.fasta'),str(R/'data'/f'{target}.fasta'),str(out),str(tmp),'--threads','6','--split-memory-limit','2G','-s','5.7','--max-seqs','100' if target=='queries' else '20','-e','0.001','--alignment-mode','3','--format-output','query,target,pident,alnlen,qstart,qend,tstart,tend,qlen,tlen,qcov,tcov,evalue,bits','--remove-tmp-files','1']
 with log.open('w') as f:
  st=time.monotonic();p=subprocess.Popen(cmd,stdout=f,stderr=f)
  while p.poll() is None:
   if shutil.disk_usage(R).free<10*2**30+128*2**20 or time.monotonic()-st>600:p.terminate();raise RuntimeError('bounded search resource gate')
   time.sleep(1)
  runs.append(dict(command=cmd,returncode=p.returncode,seconds=time.monotonic()-st));assert p.returncode==0
 print(target,'done',flush=True)
rows=read('FROZEN_J1_GENERATION_PANEL.json')['rows'];by={r['id']:r for r in rows};hits={};clusters={}
for target in ['TRAIN','VALIDATION']:
 best={}
 for line in (R/'results'/f'align_{target}.tsv').read_text().splitlines():
  z=line.split('\t');k=z[0]
  r=dict(target=z[1],identity_percent=float(z[2]),alignment_length=int(z[3]),qstart=int(z[4]),qend=int(z[5]),tstart=int(z[6]),tend=int(z[7]),qlen=int(z[8]),tlen=int(z[9]),query_coverage=float(z[10]),reference_coverage=float(z[11]),evalue=float(z[12]),bits=float(z[13]))
  if k not in best or r['bits']>best[k]['bits']:best[k]=r
 hits[target]=[dict(id=k,group=by[k]['group'],N=by[k]['N'],best_local_hit=best.get(k),no_hit_interpretation='no significant hit under this bounded search, not absence of homology') for k in by]
# exact full sequence matches independent of heuristic search
for target in ['TRAIN','VALIDATION']:
 lines=(R/'data'/f'{target}.fasta').read_text().splitlines();exact={lines[i+1]:lines[i][1:] for i in range(0,len(lines),2)}
 for r in hits[target]:r['exact_full_sequence_match']=exact.get(by[r['id']]['sequence'])
edges=[]
for line in (R/'results/align_queries.tsv').read_text().splitlines():
 z=line.split('\t');a,b=z[:2]
 if a!=b and by[a]['group']==by[b]['group'] and by[a]['N']==by[b]['N']:edges.append((a,b,float(z[2])/100,float(z[10]),float(z[11])))
for group in sorted({r['group'] for r in rows}):
 clusters[group]={}
 for N in [64,128,256,384,500]:
  ids=[r['id'] for r in rows if r['group']==group and r['N']==N];result={}
  for threshold in [.5,.8]:
   parents={k:k for k in ids}
   def find(k):
    while parents[k]!=k:parents[k]=parents[parents[k]];k=parents[k]
    return k
   for a,b,i,q,t in edges:
    if a in parents and b in parents and i>=threshold and min(q,t)>=.8:parents[find(a)]=find(b)
   result[str(threshold)]=len({find(k) for k in ids})
  clusters[group][str(N)]=result
write('SEQUENCE_NOVELTY_AND_HOMOLOGY.json',dict(search_version=subprocess.check_output(['/usr/bin/mmseqs','version'],text=True).strip(),runs=runs,rows=hits,clusters=clusters,cluster_note='connected components of detected >=identity threshold and both coverages>=0.8; heuristic sensitivity; no exhaustive remote homology certification',seconds=time.monotonic()-start))
