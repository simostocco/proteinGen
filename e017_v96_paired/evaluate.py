"""Frozen paired panel, serial model evaluation, immutable candidate commitments."""
from paired import *

METHODS=[('J1','A','A0'),('A0_raw','B','A0'),('A0_filtered','C','A0'),('B0_raw','B','B0'),('B0_filtered','C','B0'),('Af_raw','B','Af'),('Af_filtered','C','Af'),('Bf_raw','B','Bf'),('Bf_filtered','C','Bf')]
def specs(panel,method):
    for i,task in enumerate(panel['tasks']):yield i,task,panel['seeds'][task['goal']['goal_id']],'primary'
    if method=='J1':return
    index=len(panel['tasks'])
    for group in panel['diagnostics']:
        for tag,task in [('base',group['base_task']),('burial',group['alternatives']['burial']),('exposure',group['alternatives']['exposure'])]:
            yield index,task,group['seeds'],'diagnostic';index+=1
def validate(obj,task,seeds,method,fp,policy,index):
    assert obj['task']==task and obj['seeds']==seeds and obj['method']==method and obj['actor_fingerprint']==fp and obj['policy']==policy and obj['index']==index
    assert len(obj['records'])==8
    for i,row in enumerate(obj['records']):
        assert row['candidate_index']==i and row['seed']==seeds[i] and row['goal_id']==task['goal']['goal_id'] and row['AA_decisions']==0
        assert len(row['sequence'])==len(row['geometry'])<=64 and row['sequence']==task['fixed_sequence'][:len(row['sequence'])]
        assert row['metrics']['complete']==(len(row['sequence'])==64)
        assert row['termination'] is None or not row['metrics']['success']
def load_model(P,F,e,model):
    arm=model[0]
    if model.endswith('0'):
        meta=F['checkpoint'](e,5000)
    else:
        outcome=read(ARMS[arm]/'TRAINING_OUTCOME.json');v=outcome['checkpoint'];path=ARMS[arm]/'production'/v['relative'];assert sha(path)==v['SHA256']
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert P['state_digest'](state)==v['state_digest'] and state['binding']==P['bindings']()
        e.actor.load_state_dict(state['actor'],strict=True);meta=dict(path=str(path),SHA256=v['SHA256'],update=state['update'])
    e.actor.eval().requires_grad_(False);meta['actor_fingerprint']=P['c']['state_digest'](e.actor.state_dict());return meta
def main():
    verify();execution();no_gpu_worker();lock=(PAIR/'control/EVALUATION.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (PAIR/'evaluation/EVALUATION_COMPLETED.json').exists()
    root=PAIR/'evaluation';(root/'results').mkdir(parents=True,exist_ok=True);(root/'commits').mkdir(exist_ok=True)
    panel=read(PAIR/'VALIDATION_PANEL_MANIFEST.json');attempts=0;total=sum(len(list(specs(panel,m))) for m,p,u in METHODS)*8
    assert total==89088
    modelmeta={};current_arm=None;e=None;P=None;F=None;t0=time.monotonic();last=0.
    for method,policy,model in METHODS:
        arm=model[0]
        if current_arm!=arm:
            if e is not None:del e,P,F;__import__('gc').collect();torch.cuda.empty_cache()
            P,F=factory(arm);e,_=P['K']['load_engine']('C');current_arm=arm
            def evaluation_guard():
                assert not (PAIR/'control/STOP').exists(),'EVALUATION_COOPERATIVE_STOP'
                assert shutil.disk_usage(PAIR).free>2*2**30+256*2**20,'STORAGE_RESERVE'
                assert torch.cuda.max_memory_allocated()<6*2**30,'VRAM_STOP'
            # A training safety stop never permits more optimizer steps. Frozen
            # post-stop assessment is separately authorized and honors global STOP.
            P['Runtime'].__init__.__globals__['guard']=evaluation_guard
        meta=load_model(P,F,e,model);fp=meta['actor_fingerprint'];modelmeta[method]=meta
        for index,task,seeds,scope in specs(panel,method):
            assert shutil.disk_usage(PAIR).free>2*2**30+256*2**20,'STORAGE_RESERVE'
            key=f'{method}_{index:05d}';path=root/'results'/(key+'.json.gz');cp=root/'commits'/(key+'.json')
            if cp.exists():
                seal=read(cp);assert seal['path']==str(path.relative_to(root)) and sha(path)==seal['SHA256'];obj=read(path)
            elif path.exists():obj=read(path)
            else:
                start=time.monotonic();rows=P['sample'](e,task,policy,8,seeds)
                obj=dict(method=method,policy=policy,model=model,actor_fingerprint=fp,index=index,scope=scope,task=task,seeds=seeds,records=rows,seconds=time.monotonic()-start)
                validate(obj,task,seeds,method,fp,policy,index);atomic(path,obj,True)
            validate(obj,task,seeds,method,fp,policy,index)
            if not cp.exists():atomic(cp,dict(path=str(path.relative_to(root)),SHA256=sha(path),method=method,index=index,actor_fingerprint=fp,UTC=now()),True)
            attempts+=8
            if time.monotonic()-last>=60:
                last=time.monotonic();atomic(root/'LATEST_STATUS.json',dict(UTC=now(),status='RUNNING',PID=os.getpid(),method=method,index=index,attempts=attempts,target=total,seconds=time.monotonic()-t0));print('EVALUATION',method,index,attempts,'/',total,flush=True)
        e.prior.verify();assert P['c']['state_digest'](e.parent.state_dict())==e.fp
    assert attempts==total
    expected={f'{m}_{i:05d}.json' for m,p,u in METHODS for i,t,s,scope in specs(panel,m)}
    assert {p.name for p in (root/'commits').glob('*.json')}==expected
    assert {p.name for p in (root/'results').iterdir()}=={name[:-5]+'.json.gz' for name in expected}
    for p in (root/'commits').glob('*.json'):
        v=read(p);assert sha(root/v['path'])==v['SHA256']
    atomic(root/'EVALUATION_COMPLETED.json',dict(UTC=now(),attempts=attempts,batches=attempts//8,seconds=time.monotonic()-t0,models=modelmeta,manifest_SHA256=sha(PAIR/'VALIDATION_PANEL_MANIFEST.json'),all_committed=True,TEST=False),True)
    print('PAIRED_EVALUATION_COMPLETED',attempts,flush=True)
if __name__=='__main__':main()
