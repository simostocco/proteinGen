"""Prepared campaign. Production requires a future configuration-bound receipt."""
import copy, fcntl, math, runpy, sys, time
from pathlib import Path
import torch
from common96 import *
from algorithm import install, VERSION


def reconcile_schedule(current,end,step,callback):
    if current:callback(current)
    for u in range(current+1,end+1):step(u);callback(u)


def runtime_factory():
    P,F,compat=load()
    original_loader=P['K']['load_engine']
    def loader(policy):
        e,meta=original_loader(policy)
        F['checkpoint'](e,5000)
        install(e.actor)
        return e,meta
    P['K']['load_engine']=loader
    config=read(ROOT/'TRAINING_CONFIGURATION_AND_SEEDS.json')
    def binding():
        old=read(ROOT/'HISTORICAL_BINDINGS.json')
        return dict(schema='E017_V96_BINDING_1',historical_scientific_binding=old['original_scientific_binding'],
             token_schema=VERSION,source96={p.name:sha(p) for p in sorted((ROOT/'source96').glob('*.py'))},
             configuration_SHA256=sha(ROOT/'TRAINING_CONFIGURATION_AND_SEEDS.json'),
             initializer_SHA256=old['models']['milestone_05000']['SHA256'],
             TRAIN_SHA256=old['original_scientific_binding']['TRAIN_SHA256'])
    g=P['Runtime'].__init__.__globals__
    g['ROOT']=ROOT;g['SPEC']=config;g['bindings']=binding
    P['ROOT']=ROOT;P['SPEC']=config;P['bindings']=binding
    # Runtime still uses the unchanged full-trace gradients and exact sampler.
    assert g['K']['gradients'] is P['K']['gradients']
    return P, F


def baseline(P,rt):
    E=runpy.run_path(str(R3/'source/scripts/e017_v94r/evaluation.py'))
    path=rt.root/f'diagnostics/DEVELOPMENT_{rt.update:05}.json'
    fp=P['c']['state_digest'](rt.e.actor.state_dict())
    if path.exists():
        obj=read(path);assert obj['actor_fingerprint']==fp
        return obj['summary']
    tasks=read(P['V93']/'manifests/TRAIN_TASKS.json')['tasks'];report={}
    for policy,label in [('C','filtered'),('B','underlying_unfiltered')]:
        groups=[]
        for i,task in enumerate(tasks):
            seeds=[P['S']['ns']('CANDIDATE',task['goal']['goal_id'],j) for j in range(8)]
            gp=rt.root/f'diagnostics/DEV_{rt.update:05}_{label}_{i:02}.json'
            if gp.exists():
                obj=read(gp);assert obj['actor_fingerprint']==fp and obj['task']==task and obj['seeds']==seeds
            else:
                rows=P['sample'](rt.e,task,policy,8,seeds)
                obj=dict(task=task,records=rows,actor_fingerprint=fp,seeds=seeds)
                P['compact'](gp,obj)
            groups.append(obj)
        report[label]=E['summarize'](groups)
        if policy=='B':
            report[label]['invalid_proposal_checks']=report[label]['rejections']
            report[label]['rejections']=0
    P['compact'](path,dict(summary=report,actor_fingerprint=fp,TRAIN_development_only=True,attempts=16*len(tasks)))
    return report


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--authorization',required=True);a=ap.parse_args()
    receipt=read(a.authorization)
    assert receipt.get('explicit_review_authorized') is True, 'PRODUCTION_NOT_AUTHORIZED'
    assert receipt.get('configuration_SHA256')==sha(ROOT/'TRAINING_CONFIGURATION_AND_SEEDS.json')
    assert receipt.get('source96_SHA256')=={p.name:sha(p) for p in sorted((ROOT/'source96').glob('*.py'))}
    release=read(ROOT/'qualification/RELEASE.json');assert release['PASS'] and release['production_started'] is False
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    lock=(ROOT/'control/PRODUCTION.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    P,F=runtime_factory();rt=P['Runtime'](ROOT/'production')
    if rt.baseline is None:
        rt.baseline=baseline(P,rt);rt.best=dict(update=0,**rt.baseline['filtered']);rt.publish('DEVELOPMENT_BEST')
    initial=rt.baseline['filtered']
    assert initial['hard_valid']>=.91875 and initial['backbone_valid']>=.95 and initial['geometry_KL_per_active']<=1 and initial['structural_pair_RMSD']>=12.449118749897715,'INITIAL_SAFETY_ADMISSION'
    def after_commit(u):
        if u in P['SPEC']['development_updates']:
            report=baseline(P,rt);v=report['filtered'];b=rt.baseline['filtered']
            safe=v['hard_valid']>=max(b['hard_valid'],.96875)-.05 and v['backbone_valid']>=max(b['backbone_valid'],1.)-.05 and v['geometry_KL_per_active']<=1 and v['structural_pair_RMSD']>=.8*max(b['structural_pair_RMSD'],15.561398437372143)
            if not safe:
                P['compact'](rt.root/'STOP_RECORD.json',dict(update=u,report=report));raise RuntimeError('TRAINING_STABILITY_MONITOR_STOP')
            if (v['K1'],v['K8'],v['hard_valid'])>(rt.best['K1'],rt.best['K8'],rt.best['hard_valid']):
                rt.best=dict(update=u,**v);rt.publish('DEVELOPMENT_BEST')
        if u%50==0:rt.publish('ROLLING')
    try:
        # Reconcile callbacks at a restored frontier, and between every replayed
        # journal step; otherwise a missed best-state callback can change replay.
        commits=sorted((rt.root/'commits').glob('u*.json'))
        end=int(commits[-1].stem[1:]) if commits else 0
        assert len(commits)==end,'JOURNAL_GAP'
        reconcile_schedule(rt.update,end,rt.step,after_commit)
        for u in range(rt.update+1,5001):
            if (ROOT/'control/STOP').exists():rt.publish('COOPERATIVE_PAUSE');return
            rt.step(u)
            after_commit(u)
        rt.publish('COMPLETED_DECLARED_BUDGET_PAUSED')
        P['compact'](rt.root/'TRAINING_COMPLETED.json',dict(update=rt.update,counters=dict(rt.counters),best=rt.best))
    except BaseException as error:
        P['compact'](rt.root/'EXECUTION_FAILURE.json',dict(error=repr(error),committed_update=rt.update,resume_automatic=False));raise


if __name__=='__main__':main()
