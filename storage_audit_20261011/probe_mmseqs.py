import subprocess,os,json,hashlib,resource
from pathlib import Path
O=Path(__file__).resolve().parent
P=O/'mmseqs_probe';P.mkdir(exist_ok=True)
sample=Path('/mnt/d/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization/scientific_corrected_protein_v1/clusters/diversity_sample_1000000.fasta')
with sample.open('rb') as f: raw=b''.join(f.readline() for _ in range(40))
results=[]
def limit():resource.setrlimit(resource.RLIMIT_FSIZE,(16*1024**2,16*1024**2))
for ending in ['lf','crlf']:
 for mode in [1,0]:
  d=P/f'{ending}_mode{mode}';d.mkdir(exist_ok=True)
  fasta=d/'input.fasta';fasta.write_bytes(raw.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n') if ending=='crlf' else raw.replace(b'\r\n',b'\n'))
  digest=hashlib.sha256(fasta.read_bytes()).hexdigest()
  cmd=['mmseqs','easy-linclust',str(fasta),str(d/'out'),str(d/'tmp'),'--min-seq-id','0.5','-c','0.80','--cov-mode','0','--cluster-mode','0','--threads','1','--remove-tmp-files','0','--createdb-mode',str(mode),'--shuffle','0','--dbtype','1']
  env=os.environ.copy();env.update({k:str(d) for k in ['TMPDIR','TMP','TEMP','XDG_CACHE_HOME']})
  with (d/'command.log').open('w') as h:
   try:r=subprocess.run(cmd,stdout=h,stderr=h,env=env,timeout=40,preexec_fn=limit);code=r.returncode
   except subprocess.TimeoutExpired:code='timeout'
  outputs={str(p.relative_to(d)):p.stat().st_size for p in d.rglob('*') if p.is_file() and not p.is_symlink()}
  rep=d/'out_rep_seq.fasta';data=rep.read_bytes() if rep.exists() and rep.stat().st_size<16*1024**2 else b''
  row={'ending':ending,'mode':mode,'returncode':code,'input_bytes':len(raw),'source_unchanged':digest==hashlib.sha256(fasta.read_bytes()).hexdigest(),'outputs':outputs,'representative_nuls':data.count(b'\0'),'representative_bytes':len(data),'command':cmd};results.append(row)
(O/'mmseqs_probe_results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
