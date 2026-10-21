from pathlib import Path
R=Path('/mnt/d/Simone/proteinGen_data/e017_crisp_v3_microbatch_20261021_v1');S=R/'source';BASE=R.parent/'e017_crisp_v3_training_admission_20261021_v1/source';f=S/'src/protein_joint_sac/crisp_v3/admission.py';v=(BASE/'src/protein_joint_sac/crisp_v3/admission.py').read_text()
v=v.replace("geometry=nll[has].mean() if has.any() else (mu.sum()+ls.sum())*0", "geometry=(nll[has].mean() if has.any() else (mu.sum()+ls.sum())*0) if len(aa)==1 else nll.mean() # retain sealed scalar dtype/path; batch rows include unavailable zero targets")
v=v.replace("geometry_kl=(kg/count.clamp_min(1))[has].mean() if has.any() else (mu.sum()+ls.sum())*0", "geometry_kl=((kg/count.clamp_min(1))[has].mean() if has.any() else (mu.sum()+ls.sum())*0) if len(aa)==1 else (kg/count.clamp_min(1)).mean()")
f.write_text(v)
# Runtime-only continuation migration: preserve all scientific state, expose a safe future pause request.
f=S/'src/protein_joint_sac/crisp_v3/training.py';v=(BASE/'src/protein_joint_sac/crisp_v3/training.py').read_text()
anchor=" if config.get('admission_protocol') and counters.get('scientific_stop',{}).get('stop'):"
insert=""" if config.get('execution_parent') and not resume:
  from .execution import migrate_execution
  counters=migrate_execution(config,system,learner,samplers,effective_config,replay,run)
  replay_rng.setstate(counters['replay_rng']);replay.close();replay=Replay(run/'replay.sqlite')
"""
v=v.replace(anchor,insert+anchor)
v=v.replace("prior=FrozenPrior('cuda');system=", "from .execution import enable_loader\n enable_loader(config)\n prior=FrozenPrior('cuda');system=")
v=v.replace("if manager.current()['latest'] is None and monitor is None:checkpoint()","if manager.current()['latest'] is None and (monitor is None or config.get('execution_parent')):checkpoint(config.get('execution_parent',{}).get('validation'))")
v=v.replace("scientific_stop=False;monitor=None", "scientific_stop=False;monitor=None;operator_paused=False")
v=v.replace("manager.runway(replay);at_boundary=False", """request=run/'PAUSE_REQUEST.json'
   if request.exists():
    ticket=json.load(open(request))
    if ticket.get('schema')!='crisp_operator_pause_v1' or ticket.get('config_sha256')!=digest(effective_config):raise ValueError('invalid operator pause request')
    latest=manager.current()['latest']
    if not latest or latest['update']!=counters['updates']:checkpoint()
    manager._bundle_valid(manager.current()['latest']);operator_paused=True
    atomic_json(run/'PAUSE_APPLIED.json',dict(update=counters['updates'],checkpoint=manager.current()['latest']['checkpoint'],request=ticket))
    request.rename(run/('APPLIED_PAUSE_REQUEST_'+str(counters['updates'])+'_'+str(time.time_ns())+'.json'))
    from .recovery import fsync_dir
    fsync_dir(run);break
   manager.runway(replay);at_boundary=False""")
v=v.replace("else 'STOPPED_SCIENTIFIC' if scientific_stop else 'COMPLETED'", "else 'PAUSED_OPERATOR' if operator_paused else 'STOPPED_SCIENTIFIC' if scientific_stop else 'COMPLETED'")
f.write_text(v)
# All original likelihood/validation/reward definitions stay in b629783. Only the comparison baseline path
# follows the original execution segment rather than requiring a large copied history.
f=S/'src/protein_joint_sac/crisp_v3/admission.py';v=f.read_text().replace("baseline_file=Path(run)/'validation_00000.json'", "baseline_file=Path(config.get('execution_baseline_validation',Path(run)/'validation_00000.json'))");f.write_text(v)
