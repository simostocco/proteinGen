"""Correct list/dict reconstruction interface; resume read-only phases only."""
from pathlib import Path
import ast,hashlib,json,os,shutil,subprocess,time
R=Path('/mnt/d/Simone/proteinGen_data/e017_v72_counterfactual_ranking_20261023_v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,data):
 p=R/name;q=p.with_name(p.name+'.part')
 with q.open('w') as f:json.dump(data,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
assert json.loads((R/'EXECUTION.json').read_text())['failed_stage']=='predict'
assert not (R/'PREDICTION_COMPLETED.json').exists()
assert all((R/(p+'_COMPLETED.json')).exists() for p in ['FIT','DEV','CONFIRM','GENERATED'])
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
data={str(p):sha(p) for p in (R/'results').glob('*_[0-9][0-9][0-9].json')};selection=sha(R/'SELECTION_SEAL.json');protocol=json.loads((R/'PROTOCOL.json').read_text());assert not (R/'PROTOCOL_INITIAL.json').exists();shutil.copyfile(R/'PROTOCOL.json',R/'PROTOCOL_INITIAL.json')
p=R/'source/scripts/e017_v72/study.py';old=p.read_text();needle="state(t,item['anchor_geometry'],rows[0]['stage'])";assert old.count(needle)==1;new=old.replace(needle,"state(t,dict(geometry=item['anchor_geometry']),rows[0]['stage'])");ast.parse(new);p.write_text(new)
subprocess.run(['git','-C',str(R/'source'),'add','scripts/e017_v72/study.py'],check=True);subprocess.run(['git','-C',str(R/'source'),'commit','-m','Fix read-only counterfactual critic prefix reconstruction interface'],check=True)
source=subprocess.check_output(['git','-C',str(R/'source'),'rev-parse','HEAD'],text=True).strip();before=protocol['source'];protocol['source']=source;protocol['input_SHA256'][str(p)]=sha(p);protocol['implementation_correction']='001: only wrap saved anchor action list in expected geometry dictionary for read-only prediction; scientific config and model selection unchanged'
save('PROTOCOL.json',protocol);assert all(sha(p)==h for p,h in data.items());assert sha(R/'SELECTION_SEAL.json')==selection
save('IMPLEMENTATION_CORRECTION_001.json',dict(UTC=time.time(),before_source_commit=before,after_source_commit=source,failure='prediction reader list/dict anchor mismatch',fix='state(t,dict(geometry=item[anchor_geometry]),stage)',scientific_config_unchanged=True,selection_seal_SHA256=selection,collected_task_SHA256=data,sampling_repeated=False,actor_updates=0,resume_stages=['predict','analyze'],original_protocol_SHA256=sha(R/'PROTOCOL_INITIAL.json')))
procs={}
for label,args in [('campaign_resume',[str(R/'source/scripts/e017_v72/study.py'),'campaign']),('resource_watch_resume',[str(R/'control/v72_watch.py')])]:
 log=R/f'logs/{label}_{time.time_ns()}.log'
 with log.open('x') as f:child=subprocess.Popen([str(R/'run.sh'),*args],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 procs[label]=dict(pid=child.pid,log=str(log))
print(json.dumps(procs))
