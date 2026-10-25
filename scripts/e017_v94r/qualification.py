"""Same production runtime; four retained images cover sequential crash cases."""
import json,runpy,shutil,subprocess,sys,time
from pathlib import Path
R=Path('/mnt/d/Simone/proteinGen_data/e017_v94r3_one_step_feasibility_20261025_v1');Q=R/'qualification'
P=runpy.run_path(str(R/'source/scripts/e017_v94r/production.py'));read=P['read'];save=P['compact'];sha=P['sha']

def child(name,fault=None,label=None):
    log=Q/(label or name+('_'+fault if fault else '_finish'))
    log=log.with_suffix('.log');assert not log.exists()
    args=[str(R/'run.sh'),str(R/'source/scripts/e017_v94r/production.py'),'qualify-child','--run',str(Q/name)]
    if fault:args+=['--fault',fault]
    with log.open('xb') as out:rc=subprocess.call(args,stdout=out,stderr=subprocess.STDOUT,cwd=R/'source')
    save(Q/(log.stem+'_EXIT.json'),dict(command=args,exitcode=rc,log_SHA256=sha(log)))
    assert rc==({'rollout':73,'commit':74,'partial':75}[fault] if fault else 0),(name,fault,rc)

def main():
    assert read(R/'GPU_ENGINEERING_COMPLETE.json')['PASS'] and read(R/'FILTER_AND_GRADIENT_CPU_TESTS.json')['PASS']
    assert not (Q/'RELEASE.json').exists();Q.mkdir(exist_ok=True)
    cfg=read(R/'TRAINING_CONFIGURATION_AND_SEEDS.json');admission=read(R/'ENGINEERING_STORAGE_ADMISSION.json'); assert admission['PASS']; assert shutil.disk_usage(R).free>=admission['engineering_peak_remaining_bytes_including_2GiB_reserve']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    save(Q/'INPUT_REVERIFICATION.json',P['verify_inputs']())
    binding=P['bindings']();save(Q/'PROTOCOL.json',dict(binding=binding,updates_per_case=2,group_size=16,
        reference='uninterrupted2updates',recovery_sequence=['after_rollout_before_step','after_commit_before_checkpoint','real_mid_Torch_archive_write','restart_replay_and_update2'],
        retained_full_images=4,physical_engineering_optimizer_executions_expected=6,production_optimizer_updates=0,
        no_qualification_checkpoint_deletion=True,full_production_launch_authorized=False))
    try:
        child('reference');reference=read(Q/'reference/QUALIFICATION_FINAL.json')
        child('recovery','rollout');child('recovery','commit');child('recovery','partial')
        intent=read(Q/'recovery/INTERRUPTED_WRITE.json');part=Path(intent['partial'])
        assert sha(Q/'recovery/RECOVERY_INDEX.json')==intent['previous_index_SHA256']
        assert part.exists() and part.stat().st_size>=8*2**20
        rejected=False
        try:P['torch'].load(part,map_location='cpu',weights_only=False)
        except (RuntimeError,EOFError):rejected=True
        assert rejected
        child('recovery',label='recovery_final_restart');result=read(Q/'recovery/QUALIFICATION_FINAL.json')
        assert result['semantic_state']==reference['semantic_state'],'full semantic state mismatch'
        assert result['next_RNG_draws']==reference['next_RNG_draws'],'RNG mismatch'
        assert result['counters']==reference['counters'] and result['update']==2
        d=Q/'corruption';(d/'checkpoints').mkdir(parents=True,exist_ok=False)
        (d/'checkpoints/bad.pt').write_bytes(b'NEW_OWNED_CORRUPTION_NEGATIVE_FIXTURE')
        save(d/'RECOVERY_INDEX.json',dict(latest=dict(relative='checkpoints/bad.pt',SHA256=result['checkpoint']['SHA256'])))
        rt=P['Runtime'].__new__(P['Runtime']);rt.root=d;rejected=False
        try:rt.restore()
        except AssertionError as e:assert str(e)=='checkpoint hash mismatch';rejected=True
        assert rejected
        launch_block=False
        from types import SimpleNamespace
        try:P['production'](SimpleNamespace(authorization=None))
        except AssertionError as e:assert 'FULL PRODUCTION REQUIRES' in str(e);launch_block=True
        assert launch_block
        save(Q/'RELEASE.json',dict(PASS=True,binding=binding,reference=reference,recovery=result,full_production_optimizer_sampler_RNG_replay=True,
            four_fault_boundaries_qualified=True,real_interrupted_archive_bytes=part.stat().st_size,
            real_corruption_guard=True,unreviewed_production_launch_blocked=True,checkpoint_bytes=result['checkpoint']['bytes'],
            engineering_physical_optimizer_executions=6,production_optimizer_updates=0,D_free_bytes=shutil.disk_usage(R).free,
            exact_configuration='RTX5060,FP32NN/FP64rawscore,deterministic CUDA,batch16,noAMP',no_automatic_production=True))
        print('V94R_FULL_RECOVERY_PASS',flush=True)
    except BaseException as error:
        save(Q/'FAILED.json',dict(PASS=False,error=repr(error),binding=binding,production_optimizer_updates=0))
        raise
if __name__=='__main__':main()
