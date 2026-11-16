from pathlib import Path
import ast,hashlib,json,os,shutil,subprocess,time
R=Path('/mnt/d/Simone/proteinGen_data/e017_v8_constrained_planning_20261023_v1');here=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):
 p=R/n;q=p.with_name(p.name+'.part')
 with q.open('w') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
assert json.loads((R/'EXECUTION.json').read_text())['failed_stage']=='analyze';assert json.loads((R/'VALIDATION_COMPLETED.json').read_text())['complete'];files=sorted(f for f in (R/'results').glob('VALIDATION_[0-9][0-9][0-9]_*.json') if not f.name.endswith('.sha.json'));assert len(files)==128;data={str(f):sha(f) for f in files};assert not (R/'PROTOCOL_BEFORE_ANALYSIS_FIX.json').exists();shutil.copyfile(R/'PROTOCOL.json',R/'PROTOCOL_BEFORE_ANALYSIS_FIX.json');p=json.loads((R/'PROTOCOL.json').read_text());old=p['source'];src=here/'v8_analyze_staged.py';ast.parse(src.read_text());target=R/'source/scripts/e017_v8/analyze.py';shutil.copyfile(src,target)
subprocess.run(['git','-C',str(R/'source'),'add','scripts/e017_v8/analyze.py'],check=True);subprocess.run(['git','-C',str(R/'source'),'commit','-m','Exclude task checksum sidecars from V8 CPU analysis record reader'],check=True);p['source']=subprocess.check_output(['git','-C',str(R/'source'),'rev-parse','HEAD'],text=True).strip();p['input_SHA256'][str(target)]=sha(target);p['analysis_correction']='002: exclude.sha.jsonsidecars fromtaskrecordglob; no algorithm/statistical/config change';save('PROTOCOL.json',p);assert all(sha(f)==h for f,h in data.items());save('IMPLEMENTATION_CORRECTION_002.json',dict(UTC=time.time(),collection_source_commit=old,final_analysis_source_commit=p['source'],task_block_SHA256=data,scientific_protocol_unchanged=True,planning_repeated=False,scope='CPU-only filename filter',collection_protocol_SHA256=sha(R/'PROTOCOL_BEFORE_ANALYSIS_FIX.json')))
shutil.copyfile(here/'v8_final_audit.py',R/'control/v8_final_audit.py');log=R/f'logs/campaign_analysis_repair_{time.time_ns()}.log'
with log.open('x') as f:child=subprocess.Popen([str(R/'run.sh'),str(R/'source/scripts/e017_v8/study.py'),'campaign'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=child.pid,log=str(log),scope='CPU analysis only; all GPU planning markers complete')))
