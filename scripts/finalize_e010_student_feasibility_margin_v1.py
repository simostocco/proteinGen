"""Freeze the completed offline pilot and its reproducible D: artifacts."""
import json
import shutil
import e010_student_feasibility_margin_v1 as e


def main():
    result=json.loads((e.ROOT/'results_AB.json').read_text())
    tests=json.loads((e.ROOT/'tests.json').read_text())
    assert tests['passed'] and not result['phase_C_gate']['supports_margin_target_training']
    ss=result['summaries']['test'];g=result['geometry']['test']
    report=(e.ROOT/'REPORT.md').read_text()
    report+='\n## Selected test strata (retention / unsafe-state rate)\n\n'
    report+='Full results for all nine alphas are archived in results_AB.json; the following columns summarize the curve.\n\n'
    report+='| Stratum | Alpha 1.0 | Alpha 0.5 | Alpha 0.2 |\n|---|---:|---:|---:|\n'
    lookup={(r['category'],r['label'],r['alpha']):r for r in result['strata'] if r['split']=='test'}
    for category,labels in [('length',e.old.STRATA),('condition',['50','250','450']),('step',['1','2','4','8','12','16'])]:
        for label in labels:
            values=[lookup[category,label,alpha] for alpha in ('1.0','0.5','0.2')]
            cells=[f"{100*v['rollout_retention']:.2f}% / {100*v['roll_failed_rate']:.2f}%" for v in values]
            report+=f"| {category} {label} | {' | '.join(cells)} |\n"
    report+='\nStep-row retention is cumulative through that step; its unsafe rate is for that individual step, not the prefix.\n'
    report+='\n## Verification and scope\n\n'
    report+='The directional risk ratio is equivalently the action-error L2 norm divided by the approximate first-order distance to a boundary along that error direction. Nonpositive directional derivatives cannot predict a crossing along this direction and contribute zero risk. The 1e-12 slack floor bounds the ratio numerically; native exact margins remain archived without that floor. This diagnostic incorporates error magnitude and direction and does not establish exact nonlinear boundary distance.\n\n'
    report+=f"Verified unchanged: {tests['frozen_student_files_verified']} student artifacts, {tests['frozen_teacher_files_verified']} teacher artifacts, and {tests['frozen_geometry_sources_verified']} geometry source files. Alpha 1.0 exactly reproduces archived raw metrics; maximum absolute metric discrepancy {tests['maximum_metric_error']:.3g}. Independently reconstructed {tests['independently_reconstructed_rollout_states']} states across both splits, all length bins and all alphas. No new model or marginized dataset was generated.\n"
    report+='\nThis closed gate does not prove marginized-target training cannot help. Alphas below 0.2 were not tested. The privileged X-based shield remains undeployable and is not used anywhere in the new inference path.\n'
    elapsed=json.loads((e.ROOT/'PHASE_AB_COMPLETED.json').read_text())['wall_seconds']
    report+=f"\nCPU-only evaluation elapsed wall time: {elapsed/60:.2f} minutes with eight single-threaded workers; no GPU computation or training. Archived alpha-1 actions were reused.\n"
    (e.ROOT/'REPORT.md').write_text(report)
    code=e.ROOT/'code';code.mkdir(exist_ok=True)
    sources={}
    for name in ('e010_student_feasibility_margin_v1.py','report_e010_student_feasibility_margin_v1.py',
                 'verify_e010_student_feasibility_margin_v1.py','finalize_e010_student_feasibility_margin_v1.py'):
        source=e.REPO/'scripts'/name;shutil.copyfile(source,code/name);sources[str(source)]=e.old.sha(source)
    doc="""# E010 student feasibility margin v1

Completed offline negative feasibility pilot, separate from frozen student v1
commit fc76211. Teacher/corpus 6d1ed30 and projected method 73f3330 remain unchanged.
All generated artifacts reside at:
`/mnt/d/Simone/proteinGen/outputs/e010_student_feasibility_margin_v1`.

The original 300 validation and 300 test identities (all project TRAIN) retain
their frozen identity/group-isolated splits. Three conditions and sixteen logical
states/steps give 14,400 one-step and recursive states per split per alpha.
The fixed global alpha ladder is 1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2. No X is
used in network inputs, global scaling, or an inference shield. X scores results
offline. No retraining, new teacher data, capacity increase, or SLSQP solve.

| Test alpha | One-step repair retention | K16 repair retention | Unsafe K16 states |
|---:|---:|---:|---:|
"""
    for alpha,s in ss.items():
        doc+=f"| {alpha} | {100*s['one_step_retention']:.2f}% | {100*s['rollout_retention']:.2f}% | {s['roll_failed']}/{s['roll_states']} |\n"
    gate=result['phase_C_gate']
    doc+=f"""
Retention compares summed absolute mean-local-RMSE repair with the archived
original privileged teacher. Persistent inversions are counted per state, and
constraint-family failure counts overlap. Exact slacks, action errors, and
error-direction finite differences are archived per transition.

Teacher actions within 1e-5 of any native normalized hard boundary:
{100*g['teacher_near_any_native_boundary']['1e-05']:.2f}% of nonzero test teacher
states. Error-magnitude failure AUC: {g['error_magnitude_failure_AUC']:.3f};
directional error-to-margin failure AUC: {g['directional_margin_risk_failure_AUC']:.3f}.
These are mechanistic associations, not causal proof or exact boundary distances.

The Phase C gate was frozen before new test scoring: validation alpha 0.5 must
halve the failed-state rate and retain at least 25% original-teacher repair.
Observed validation failure reduction: {100*gate['validation_failure_reduction']:.2f}%;
retention: {100*gate['validation_half_retention']:.2f}%. The gate failed, so no
marginized targets or retraining were performed. The preregistered unused positive
buffer ladder is 0.001/0.01/0.05 in native normalized inequality units.

Global shrinking does not produce a deployable repair/safety point on the tested
ladder. This does not establish that marginized training cannot help; alphas below
0.2 are untested. The X-based shield from student v1 is explicitly undeployable.

Validation: all {tests['frozen_student_files_verified']} frozen student artifacts,
{tests['frozen_teacher_files_verified']} teacher artifacts, and 18 geometry sources
remain unchanged; alpha-1 archived baseline parity; independent reconstruction
of {tests['independently_reconstructed_rollout_states']} states across both splits,
all length bins and all alphas; source lint passed.

`REPORT.md` contains the full Pareto table, family margins, failure discrimination,
selected strata, verification and scope. `results_AB.json` contains all alpha,
length, condition and step aggregates. `records/` contains identity results;
`geometry/` contains exact inequality snapshots and directional derivatives;
`actions/` contains reconstructible recursive actions. `protocol.json` and
`parent_freeze.json` pin the protocol, model and split hashes. `COMPLETED.json`
hashes the generated scientific artifacts and frozen source snapshots.
"""
    (e.REPO/'docs/e010_student_feasibility_margin_v1.md').write_text(doc)
    e.write(e.ROOT/'provenance.json',dict(source_sha256=sources,
        protocol_sha256=e.old.sha(e.ROOT/'protocol.json'),parent_commit='fc76211',
        model_sha256=e.old.sha(e.PARENT/'models/imitation_best.pt'),
        split_identity_hashes=json.loads((e.PARENT/'splits.json').read_text())['identity_list_sha256'],
        phase_C_run=False,no_privileged_inference_shield=True))
    files={str(p):e.old.sha(p) for p in sorted(e.ROOT.rglob('*')) if p.is_file() and p.name!='COMPLETED.json' and
        not any(x in p.relative_to(e.ROOT).parts for x in ('logs','tmp','cache'))}
    e.write(e.ROOT/'COMPLETED.json',dict(status='complete',classification='negative deployable-feasibility pilot',
        phase_A_complete=True,phase_B_complete=True,phase_C_run=False,phase_C_gate=gate,
        files=files,source_sha256=sources,all_artifacts_on_D=True,baseline_unchanged=True))
    print(json.dumps(dict(completed=True,files=len(files),report=str(e.ROOT/'REPORT.md'),phase_C_run=False)),flush=True)


if __name__=='__main__':main()
