"""Measured complete-pool MMseqs screen; no representative-only certification."""
import os,sys,json,time,subprocess,resource,shutil,signal,pickle,hashlib,collections
from pathlib import Path
R=Path(os.environ['E016_ARTIFACT_ROOT']);D=R/'homology';freeze=json.load(open(D/'POOL_FREEZE.json'))
if (D/'GLOBAL_SCREEN_COMPLETE.json').exists():
 completed=json.load(open(D/'GLOBAL_SCREEN_COMPLETE.json'));assert completed['pool_sha256']==freeze['FASTA_sha256'];print(json.dumps({'status':'ALREADY_COMPLETE_USE_component_screen_FOR_VERIFIED_RESUME','queries':completed['query_records']}),flush=True);sys.exit(0)
assert freeze['full_source_population_complete'];N=freeze['node_sequences'];threads=min(4,max(1,(os.cpu_count() or 2)//2));RESERVE=10*2**30

from homology_tools import run,mem,diskbytes

mm=mem();limit=int(mm['MemAvailable']/2);binary='/usr/bin/mmseqs'
version=subprocess.check_output([binary,'version'],text=True).strip();assert version==freeze['tool_version']
maxlen=max(65535,freeze['max_length']+1)
common=['--min-seq-id','0.3','--seq-id-mode','0','--alignment-mode','3','--min-aln-len','20','-c','0','--cov-mode','0','-e','0.001','-s','7.5','--mask','1','--max-seqs',str(N),'--max-accept','2147483647','--max-rejected','2147483647','--threads',str(threads),'--split-memory-limit',str(limit//2**20)+'M','--max-seq-len',str(maxlen),'--search-type','1','--compressed','1']
run('createdb',[binary,'createdb',str(D/'all_full_sequences.fasta'),str(D/'all_db'),'--dbtype','1','--shuffle','0'])
with (D/'initial_graph.pkl').open('rb') as f:pool=pickle.load(f)
# Stratified actual-length pilot, plus heavily repeated identities for density stress.
bands=[(20,64),(65,128),(129,256),(257,384),(385,500),(501,freeze['max_length'])];selected=set()
for low,high in bands:
 ids=[i for i,s in enumerate(pool['sequences']) if low<=len(s)<=high]
 ids.sort(key=lambda i:hashlib.sha256(('42|pilot|'+pool['hashes'][i]).encode()).hexdigest());selected.update(ids[:128])
selected.update(sorted(range(N),key=lambda i:(-pool['multiplicity'][i],i))[:128])
with (D/'pilot.fasta').open('w') as f:
 for i in sorted(selected):f.write('>'+str(i)+'\n'+pool['sequences'][i]+'\n')
run('pilot_createdb',[binary,'createdb',str(D/'pilot.fasta'),str(D/'pilot_db'),'--dbtype','1','--shuffle','0'])
run('pilot_search',[binary,'search',str(D/'pilot_db'),str(D/'all_db'),str(D/'pilot_result'),str(D/'pilot_scratch'),*common])
fmt='query,target,fident,alnlen,qcov,tcov,evalue,qstart,qend,tstart,tend,qlen,tlen,bits'
run('pilot_convert',[binary,'convertalis',str(D/'pilot_db'),str(D/'all_db'),str(D/'pilot_result'),str(D/'pilot.tsv'),'--format-output',fmt,'--threads',str(threads)])
# Pilot scratch is preserved, and therefore included in the full-search budget.
pilot_result=sum(p.stat().st_size for p in D.glob('pilot_result*') if p.is_file());scratch=diskbytes(D/'pilot_scratch');fasta_bytes=(D/'all_full_sequences.fasta').stat().st_size
projected=int((pilot_result+scratch)*N/max(1,len(selected))*2 +fasta_bytes*4)
free=shutil.disk_usage(R).free
budget={'pilot_queries':len(selected),'pool_sequences':N,'pool_residues':freeze['residues'],'pilot_result_bytes':pilot_result,'pilot_scratch_bytes':scratch,'full_search_projected_peak_additional_bytes':projected,'projection_status':'INFERRED_2xPilotScaling_not_a_guarantee','free_bytes_before_full':free,'reserve_bytes':RESERVE,'persistent_export_and_independent_screen_allowance_bytes':4*2**30,'MMseqs_split_memory_limit_bytes':limit,'threads':threads,'command':common,'max_seqs_is_entire_target_count':True,'domain_relation':'identity30%+Evalue<=1e-3+local_alignment>=20; coverage annotations distinguish both80/shorter80/localdomain'}
(D/'MEASURED_SEARCH_BUDGET.json').write_text(json.dumps(budget,indent=2))
if free<RESERVE+4*2**30+projected:raise RuntimeError('BLOCKED_RESOURCE_MEASURED_FULL_SEARCH_FORECAST; component-aware/sharded exact procedure needed')
run('full_search',[binary,'search',str(D/'all_db'),str(D/'all_db'),str(D/'full_result'),str(D/'full_scratch'),*common])
run('full_convert',[binary,'convertalis',str(D/'all_db'),str(D/'all_db'),str(D/'full_result'),str(D/'full_edges.tsv'),'--format-output',fmt,'--threads',str(threads)])
(D/'GLOBAL_SCREEN_COMPLETE.json').write_text(json.dumps({'status':'COMPLETE_DECLARED_HEURISTIC_SCREEN','tool_version':version,'parameters':common,'pool_sha256':freeze['FASTA_sha256'],'coverage_rule':'both80 plus shorter80 and significantlocaldomain safeguards; not all remote homology','edge_format':fmt},indent=2))
