"""Explicit stopped-run recovery only; replay accepted work, no new updates.

Repairs an interrupted reserved-TRAIN milestone callback before the documented
core resume command. It neither starts training nor clears scientific stops.
"""
import copy,gc,json,os,runpy,sys
from pathlib import Path
R=Path('/mnt/d/Simone/proteinGen_data/e017_v94_fixed_sequence_geometry_rl_20261024_v1')
MILESTONES=[250,500,1000,2000,3500,5000]

def reconcile(rt,evaluate):
    if rt.update not in MILESTONES:return 'NO_PENDING_MILESTONE'
    report=evaluate(rt,rt.update);a=report['filtered'];b=rt.baseline['filtered']
    safe=(a['hard_valid']>=b['hard_valid']-.05 and a['backbone_valid']>=b['backbone_valid']-.05
          and a['geometry_KL_per_active']<=1 and a['structural_pair_RMSD']>=.8*b['structural_pair_RMSD'])
    if not safe:return 'SCIENTIFIC_STABILITY_STOP_DO_NOT_RESUME'
    if (a['K1'],a['K8'],a['hard_valid'])>(rt.best['K1'],rt.best['K8'],rt.best['hard_valid']):
        rt.best=dict(update=rt.update,**a);rt.publish('DEVELOPMENT_BEST')
    else:rt.publish('RECONCILED_DEVELOPMENT_BOUNDARY')
    return 'BOUNDARY_COMMITTED'

def test():
    from types import SimpleNamespace
    base=dict(K1=.1,K8=.4,hard_valid=.9,backbone_valid=.95,geometry_KL_per_active=.1,structural_pair_RMSD=5.)
    calls=[];rt=SimpleNamespace(update=250,baseline={'filtered':copy.deepcopy(base)},best=dict(update=0,**base),publish=calls.append)
    a=dict(base,K1=.2)
    assert reconcile(rt,lambda *_:{'filtered':a})=='BOUNDARY_COMMITTED' and rt.best['update']==250 and calls==['DEVELOPMENT_BEST']
    assert reconcile(rt,lambda *_:{'filtered':a})=='BOUNDARY_COMMITTED' and calls[-1]=='RECONCILED_DEVELOPMENT_BOUNDARY'
    bad=dict(a,hard_valid=.8);n=len(calls)
    assert reconcile(rt,lambda *_:{'filtered':bad})=='SCIENTIFIC_STABILITY_STOP_DO_NOT_RESUME' and len(calls)==n
    rt.update=251;assert reconcile(rt,lambda *_:None)=='NO_PENDING_MILESTONE'
    print('CALLBACK_RECONCILIATION_CPU_TESTS_PASS')

def main():
    if '--test' in sys.argv:test();return
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:args=(p/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if args and b'python' in args[0] and any(b'e017_v94/' in x and b'production.py' in x for x in args[1:]):
            raise SystemExit('Active V94 trainer; no recovery mutation allowed')
    assert not (R/'production/STOP_RECORD.json').exists(),'scientific stop requires explicit review'
    assert not (R/'control/STOP').exists(),'cooperative stop remains active'
    assert not (R/'production/EXECUTION_FAILURE.json').exists(),'execution failure requires explicit review before recovery'
    P=runpy.run_path(str(R/'source/scripts/e017_v94/production.py'))
    rel=P['read'](R/'recovery_qualification_20261025_v1/PRODUCTION_RECOVERY_RELEASE.json')
    assert rel['PASS'] and rel['binding']==P['bindings']()
    assert not P['subprocess'].check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    P['torch'].set_num_threads(1);P['torch'].use_deterministic_algorithms(True)
    P['torch'].backends.cuda.matmul.allow_tf32=False;P['torch'].backends.cudnn.allow_tf32=False
    rt=P['Runtime'](R/'production');rt.replay()
    ev=runpy.run_path(str(R/'source/scripts/e017_v94/evaluation.py'))['development']
    result=reconcile(rt,ev)
    if result.startswith('SCIENTIFIC'):raise SystemExit(result)
    rt.publish('EXPLICIT_RECOVERY_READY_NO_NEW_UPDATES')
    print(json.dumps(dict(status='RESUMABLE_NO_NEW_OPTIMIZER_UPDATES',accepted_update=rt.update,reconciliation=result,
        resume_command=f'{R}/run.sh {R}/source/scripts/e017_v94/production.py production')))
if __name__=='__main__':main()
