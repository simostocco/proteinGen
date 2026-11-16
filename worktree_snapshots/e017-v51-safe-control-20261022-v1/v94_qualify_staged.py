"""Sequential controlled crash/restart qualification, same production update."""
import json, os, runpy, shutil, subprocess, sys, time
from pathlib import Path
ROOT=Path('/mnt/d/Simone/proteinGen_data/e017_v94_fixed_sequence_geometry_rl_20261024_v1')
Q=ROOT/'recovery_qualification_20261025_v1'
P=runpy.run_path(str(ROOT/'source/scripts/e017_v94/production.py'))
read=P['read'];compact=P['compact'];sha=P['sha']

def child(name,fault=None):
    run=Q/name; log=Q/(name+('_fault' if fault else '_resume')+'.log')
    assert not log.exists(),'never overwrite qualification evidence'
    args=[str(ROOT/'run.sh'),str(ROOT/'source/scripts/e017_v94/production.py'),'qualify-child','--run',str(run)]
    if fault: args+=['--fault',fault]
    with log.open('xb') as out: code=subprocess.call(args,stdout=out,stderr=subprocess.STDOUT,cwd=ROOT/'source')
    compact(Q/(log.stem+'_EXIT.json'),dict(command=args,exitcode=code,log_SHA256=sha(log)))
    assert code==({'rollout':73,'commit':74,'partial':75}[fault] if fault else 0),('qualification subprocess failure',name,fault,code)

def main():
    assert not (Q/'RELEASE.json').exists()
    assert shutil.disk_usage(ROOT).free>=9*2**30
    verify=P['verify_inputs']();compact(Q/'INPUT_REVERIFICATION.json',verify)
    binding=P['bindings']();compact(Q/'PROTOCOL.json',dict(binding=binding,updates_per_case=2,group_size=16,
        cases=['uninterrupted','after_rollout_before_step','after_commit_before_checkpoint','partial_checkpoint_write'],
        equality='full actor/Adam/scheduler/RNG/cursor/counters/config semantic state and next RNG draws; timing/file-content hashes excluded only from cross-case semantic comparison',
        all_physical_optimizer_executions_engineering=True,production_optimizer_updates=0,failed_gate_requires_stop=True))
    try:
        child('uninterrupted'); baseline=read(Q/'uninterrupted/QUALIFICATION_FINAL.json')
        results=[]
        for name,fault in [('after_rollout','rollout'),('after_commit','commit'),('partial_write','partial')]:
            child(name,fault)
            if fault=='partial':
                interrupted=read(Q/name/'INTERRUPTED_WRITE.json')
                assert sha(Q/name/'RECOVERY_INDEX.json')==interrupted['previous_index_SHA256']
                assert Path(interrupted['partial']).stat().st_size<1024
            child(name); value=read(Q/name/'QUALIFICATION_FINAL.json')
            assert value['semantic_state']==baseline['semantic_state'],('semantic replay mismatch',name)
            assert value['next_RNG_draws']==baseline['next_RNG_draws'],('RNG replay mismatch',name)
            assert value['update']==2 and value['counters']==baseline['counters']
            results.append(dict(case=name,PASS=True,checkpoint=value['checkpoint']))
        # Corruption check uses a tiny newly-owned fixture, no canonical copy.
        bad=Q/'CORRUPTION_NEGATIVE_FIXTURE.bin';bad.write_bytes(b'intentionally invalid qualification fixture')
        assert sha(bad)!=baseline['checkpoint']['SHA256']
        release=dict(PASS=True,binding=binding,cases=results,uninterrupted=baseline,
            corruption_hash_guard=True,full_production_optimizer_sampler_RNG_replay=True,
            production_optimizer_updates=0,qualification_log_files=[str(p) for p in Q.glob('*.log')],
            checkpoint_bytes=baseline['checkpoint']['bytes'],D_free_bytes=shutil.disk_usage(ROOT).free)
        compact(Q/'RELEASE.json',release); print('RECOVERY_QUALIFIED',flush=True)
    except BaseException as error:
        compact(Q/'FAILED.json',dict(PASS=False,error=repr(error),production_optimizer_updates=0,production_launch_authorized=False,
            outcome='INFRASTRUCTURE_BLOCKED',binding=binding,UTC=time.time()))
        raise
if __name__=='__main__':main()
