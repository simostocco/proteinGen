from pathlib import Path
import csv,datetime,hashlib,json,os,shutil,subprocess
R=Path('/mnt/d/Simone/proteinGen_data/e017_v94_fixed_sequence_geometry_rl_20261024_v1');read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 t=p.with_name(p.name+'.part')
 with t.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p);fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
p=R/'TRAINING_CONFIGURATION_AND_SEEDS.json';v=read(p);v['authorized']=True;v['authorization_meaning']='current user has authorized the bounded5000-update campaign conditional on storage and recovery; no additional training authorization is requested';v['storage_scope_E_authorized']=False;v['launch_admitted']=False;write(p,v)
required=['FINAL_E017_V94_SCIENTIFIC_REPORT.md','PRODUCTION_STORAGE_ADMISSION.md','OPTIMIZER_JOURNAL_RECOVERY_QUALIFICATION.md','TRAINING_CONFIGURATION_AND_SEEDS.json','RL_TRAINING_CURVES.csv','TRAINING_STABILITY_AND_VALIDITY.md','FILTERED_POLICY_DIAGNOSTICS.md','PROSPECTIVE_RL_EVALUATION.md','RL_VS_SEARCH_COMPARISON.md','CHECKPOINT_RECOVERY_MANIFEST.json','NEXT_CRISP_RESEARCH_DECISION.md','PROVENANCE_SHA256.json'];assert all((R/p).is_file() for p in required)
with (R/'RL_TRAINING_CURVES.csv').open() as f:assert len(list(csv.reader(f)))==1
assert not list(R.rglob('*.pt')) and not list(R.rglob('*.part'))
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
status=read(R/'FINAL_IDLE_STATUS.json');status['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();status['D_free_bytes']=shutil.disk_usage(R).free;assert status['D_free_bytes']>2*2**30;write(R/'FINAL_IDLE_STATUS.json',status)
shutil.copyfile(Path(__file__),R/'audit_source/v94_finalize_metadata.py');prov=read(R/'PROVENANCE_SHA256.json');prov['artifact_SHA256']={str(p.relative_to(R)):sha(p) for p in sorted(R.rglob('*')) if p.is_file() and p.name!='PROVENANCE_SHA256.json'};prov['required_deliverables']=required;prov['all_required_present']=True;write(R/'PROVENANCE_SHA256.json',prov)
for name,expected in read(R/'PROVENANCE_SHA256.json')['artifact_SHA256'].items():assert sha(R/name)==expected
print(json.dumps(dict(PASS=True,status='INFRASTRUCTURE_BLOCKED',D_free_bytes=status['D_free_bytes'],E_free_bytes=status['E_free_bytes'],all12deliverables=True,provenance_files=len(prov['artifact_SHA256']),training_authorized_conditionally=True,launch_admitted=False,optimizer_updates=0,complete_trajectories=0,no_deletions=True)),flush=True)
