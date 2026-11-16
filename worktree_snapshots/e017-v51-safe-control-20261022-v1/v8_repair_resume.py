from pathlib import Path
import ast,json,os,shutil,subprocess,time,hashlib
R=Path('/mnt/d/Simone/proteinGen_data/e017_v8_constrained_planning_20261023_v1');here=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):
 p=R/n;q=p.with_name(p.name+'.part')
 with q.open('w') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
assert json.loads((R/'EXECUTION.json').read_text())['failed_stage']=='run_TRAIN';assert not list((R/'results').glob('TRAIN_[0-9][0-9][0-9]_*.json'));assert not (R/'VALIDATION_COMPLETED.json').exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
assert not (R/'PROTOCOL_INITIAL.json').exists();shutil.copyfile(R/'PROTOCOL.json',R/'PROTOCOL_INITIAL.json');shutil.copyfile(R/'PREFLIGHT.json',R/'PREFLIGHT_INITIAL.json');p=json.loads((R/'PROTOCOL.json').read_text());old=p['source'];target=R/'source/scripts/e017_v8'
for src,name in [('v8_study_staged.py','study.py'),('v8_analyze_staged.py','analyze.py')]:
 path=here/src;ast.parse(path.read_text());shutil.copyfile(path,target/name);p['input_SHA256'][str(target/name)]=sha(target/name)
subprocess.run(['git','-C',str(R/'source'),'add','scripts/e017_v8'],check=True);subprocess.run(['git','-C',str(R/'source'),'commit','-m','Fix V8 panel path and distinguish CPU CUDA-event and wall timing before validation'],check=True);p['source']=subprocess.check_output(['git','-C',str(R/'source'),'rev-parse','HEAD'],text=True).strip();p['implementation_correction']='001: panel-path parentheses and explicit CPU/CUDA-event instrumentation; no method/config/threshold/sampling change; zero durable planning tasks before repair';save('PROTOCOL.json',p)
save('IMPLEMENTATION_CORRECTION_001.json',dict(UTC=time.time(),initial_source=old,repaired_source=p['source'],initial_protocol_SHA256=sha(R/'PROTOCOL_INITIAL.json'),initial_preflight_SHA256=sha(R/'PREFLIGHT_INITIAL.json'),completed_planning_queries_before_fix=0,scientific_config_unchanged=True,changes=['path concatenation parentheses','host process CPU and CUDA-event timing spans, distinguished from synchronized neural wall'],sampling_repeated=False))
subprocess.run([str(R/'run.sh'),str(target/'study.py'),'preflight'],check=True)
procs={}
for label,args in [('campaign_resume',[str(target/'study.py'),'campaign']),('resource_watch_resume',[str(R/'control/v8_watch.py')])]:
 log=R/f'logs/{label}_{time.time_ns()}.log'
 with log.open('x') as f:child=subprocess.Popen([str(R/'run.sh'),*args],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 procs[label]=dict(pid=child.pid,log=str(log))
print(json.dumps(procs))
