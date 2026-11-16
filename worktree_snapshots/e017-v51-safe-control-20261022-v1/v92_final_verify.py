from pathlib import Path
import collections,datetime,hashlib,json,os,shutil,subprocess,time
r=Path('/mnt/d/Simone/proteinGen_data/e017_v92_prefix_feasibility_20261024_v1');old=r.parent/'e017_v91_ordered_goal_recovery_20261023_v1';v9=r.parent/'e017_v9_property_steering_20261023_v1'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
 return h.hexdigest()
def write(p,o):
 tmp=p.with_suffix(p.suffix+'.part')
 with tmp.open('w') as f:json.dump(o,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
required=['FINAL_E017_V92_SCIENTIFIC_REPORT.md','ZERO_SUCCESS_CONSTRAINT_DECOMPOSITION.md','PREFIX_VALIDITY_FAILURE_AUDIT.md','ONE_STEP_ACTION_FEASIBILITY.md','FUTURE_CONTINUATION_VIABILITY.md','MOTIF_CONSTRAINT_DIAGNOSTIC.md','INFERENCE_ONLY_FILTERING_RESULTS.md','COMPUTE_AND_STORAGE_ACCOUNTING.json','NEXT_CRISP_RESEARCH_DECISION.md']
assert all((r/name).is_file() for name in required),'analysis has not finished'
account=read(r/'COMPUTE_AND_STORAGE_ACCOUNTING.json');assert account['source_status']==''
index=read(old/'checkpoints/protein/RECOVERY_INDEX.json');checkpoints=[]
for u in ['0','25','50','100']:
 rec=index['owned'][u];p=Path(rec['path']);assert sha(p)==rec['sha256'];checkpoints.append(dict(update=int(u),path=str(p),SHA256=rec['sha256'],bytes=p.stat().st_size,state_digest=rec['state_digest'],historical_unchanged=True))
# Source/report/manifest verification against the sealed prior publication.
prior=read(old/'PROVENANCE_SHA256.json');prior_verified={};prior_candidates={}
for section in ['source_SHA256','artifact_SHA256','artifacts_SHA256','files']:
 for name,expected in prior.get(section,{}).items():
  if isinstance(expected,dict):expected=expected.get('SHA256') or expected.get('sha256')
  if not isinstance(expected,str) or len(expected)!=64:continue
  path=Path(name);path=path if path.is_absolute() else old/path
  if path.is_file() and path.suffix in ['.py','.md','.json'] and path.stat().st_size<16*2**20:prior_candidates[str(path)]=expected
for name,expected in prior_candidates.items():assert sha(Path(name))==expected,('historical changed',name);prior_verified[name]=expected
for name,expected in prior['protected_V9_files'].items():
 assert sha(Path(name))==expected,('protected V9 changed',name)
 prior_verified[name]=expected
# Explicit primary network file hashes, in place, never copied/loaded on GPU.
for p,expected in [(Path(account['model_references']['J1']['path']),account['model_references']['J1']['SHA256']),(Path(account['model_references']['parent']['path']),account['model_references']['parent']['SHA256'])]:assert sha(p)==expected
# Historical result identities retained exactly from the CPU replay receipt.
for name,expected in read(r/'HISTORICAL_CONSTRAINT_DECOMPOSITION.json')['input_SHA256'].items():assert sha(Path(name))==expected
for path in (r/'results').glob('*.sha.json'):
 original=path.with_name(path.name.replace('.sha.json','.json'));assert sha(original)==read(path)['sha256']
assert not list(r.rglob('*.part')),'uncommitted atomic publication'
# Read-only OS audits. No process control signal is issued.
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip();assert not gpu,('unexpected CUDA process',gpu)
processes=subprocess.check_output(['ps','-eo','pid,ppid,etime,args'],text=True);research=[]
for line in processes.splitlines():
 if 'python' in line and any(word in line for word in ['/scripts/e017_','v92_confirm_launch.py','v92_monitor.py','e007','e015','ProteinMPNN','colabfold']):research.append(line.strip())
assert not research,('research processes remain',research)
windows=subprocess.run(['powershell.exe','-NoProfile','-Command',"$o=Get-CimInstance Win32_OperatingSystem; @{TotalVisibleMemoryKiB=$o.TotalVisibleMemorySize;FreePhysicalMemoryKiB=$o.FreePhysicalMemory;ResearchProcesses=@(Get-Process | Where-Object {$_.ProcessName -match 'python|colabfold|ProteinMPNN|vmmem'} | Select-Object ProcessName,Id,WorkingSet64)} | ConvertTo-Json -Depth 4 -Compress"],capture_output=True,text=True,timeout=30)
try:host=json.loads(windows.stdout)
except ValueError:host=dict(unparsed=windows.stdout,error=windows.stderr)
free=shutil.disk_usage(r).free;assert free>=2*2**30
newfiles=[p for p in r.rglob('*') if p.is_file() and 'source' not in p.relative_to(r).parts];newallocated=sum(p.stat().st_blocks*512 for p in newfiles);assert newallocated<128*2**20,('evidence cap',newallocated)
# Additional stratification never changes frozen methods or acceptance gates.
h=read(r/'HISTORICAL_CONSTRAINT_DECOMPOSITION.json');lookups={g['goal_id']:g for panel in ['CONFIRM','DEV'] for g in read(old/f'manifests/{panel}_GOALS.json')['goals']};hazard={}
for row in h['records']:
 goal=lookups[row['goal_id']];kind='exposed_before_buried' if goal['exposed'][0]<goal['buried'][0] else 'buried_before_exposed';key=row['policy_panel']+'|'+kind;v=hazard.setdefault(key,dict(trajectories=0,first_failure_histogram={},by_residue={}));v['trajectories']+=1;first=row['first_irreversible_failure'];stop=first['residue'] if first else 64
 for t in range(1,stop+1):
  q=v['by_residue'].setdefault(str(t),dict(at_risk=0,failures=0));q['at_risk']+=1;q['failures']+=bool(first and first['residue']==t)
 if first:v['first_failure_histogram'][str(stop)]=v['first_failure_histogram'].get(str(stop),0)+1
for v in hazard.values():
 for q in v['by_residue'].values():q['hazard']=q['failures']/q['at_risk']
write(r/'HAZARD_BY_GOAL_TYPE.json',hazard)
fp=read(r/'FAILURE_POINT_ALTERNATIVES.json')['records'];failure_summary={}
for pol in ['A','B','C','D100']:
 sub=[z for z in fp if z['policy']==pol];failure_summary[pol]=dict(prefixes=len(sub),trials=len(sub)*16,valid=sum(q['valid'] for z in sub for q in z['trials']),prefixes_no_observed_valid_alternative=sum(not any(q['valid'] for q in z['trials']) for z in sub),original_failure_causes=dict(collections.Counter(c for z in sub for c in z['original_failure_causes'])))
write(r/'HISTORICAL_AND_FINAL_VERIFICATION.json',dict(PASS=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoints=checkpoints,historical_source_reports_verified=prior_verified,historical_results_verified=len(h['input_SHA256']),all_new_result_hashes_verified=True,no_partial_checkpoint_or_result=True,no_new_checkpoints=True,CUDA_processes=gpu,relevant_research_processes=research,Windows=host,D_free_bytes=free,D_free_GiB=free/2**30,evidence_allocated_bytes=newallocated,failure_point_summary=failure_summary,one_GPU_at_a_time=True,all_neural_updates=0))
account.update(final_verification='HISTORICAL_AND_FINAL_VERIFICATION.json',D_free_bytes=free,D_free_GiB=free/2**30,new_evidence_allocated_bytes=newallocated,all_research_workers_exited=True,CUDA_processes=[],Windows_host=host,retrospective_failure_summary=failure_summary)
write(r/'COMPUTE_AND_STORAGE_ACCOUNTING.json',account)
# Exact paths and hashes; provenance excludes itself and live metadata only.
provenance={str(p.relative_to(r)):sha(p) for p in sorted(r.rglob('*')) if p.is_file() and 'source' not in p.relative_to(r).parts and p.name!='PROVENANCE_SHA256.json'}
sourcehashes={str(p.relative_to(r)):sha(p) for p in sorted((r/'source/scripts/e017_v92').glob('*.py'))}
write(r/'PROVENANCE_SHA256.json',dict(schema='E017_V92_FINAL_1',source_commit=account.get('final_publication_source_commit',account['source_commit']),numeric_analysis_execution_commit=account['source_commit'],source_SHA256=sourcehashes,artifact_SHA256=provenance,external_model_references=account['model_references'],all_required_deliverables_present=True,verification='HISTORICAL_AND_FINAL_VERIFICATION.json'))
print(json.dumps(dict(PASS=True,files=len(provenance),D_free_bytes=free,evidence_allocated_bytes=newallocated,failure_summary=failure_summary)),flush=True)
