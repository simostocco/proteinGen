"""Freeze the completed observability diagnostic and gated controls on D:."""
import json
import shutil
import time
import e010_student_direction_observability_v1 as e


def main():
    root=e.ROOT
    result=json.loads((root/'results_AB.json').read_text())
    phase_C=result['phase_C_gate']['run'];controls=None
    if phase_C:
        assert (root/'phase_C/RUNS_COMPLETED.json').exists()
        assert json.loads((root/'phase_C/tests.json').read_text())['passed']
        assert (root/'phase_C/latency_benchmark.json').exists()
        controls=json.loads((root/'phase_C/results.json').read_text())
    tests=json.loads((root/'tests.json').read_text());assert tests['passed']
    assert (root/'ANALYSIS_COMPLETED.json').exists()
    lineage_source_counts={}
    for label,parent_root in [('feasibility_margin',e.PARENT),('student',e.old.ROOT),('teacher',e.old.PARENT)]:
        completion=json.loads((parent_root/'COMPLETED.json').read_text())
        for path,digest in completion.get('source_sha256',{}).items():assert e.old.sha(path)==digest,path
        lineage_source_counts[label]=len(completion.get('source_sha256',{}))
    sources=[e.REPO/'scripts'/name for name in (
        'e010_student_direction_observability_v1.py',
        'analyze_e010_student_direction_observability_v1.py',
        'report_e010_student_direction_observability_v1.py',
        'verify_e010_student_direction_observability_v1.py',
        'report_e010_observability_action_norms.py',
        'freeze_e010_student_direction_observability_v1.py')]
    if phase_C:sources += [e.REPO/'scripts'/name for name in (
        'train_e010_student_direction_observability_v1.py','run_e010_observability_controls.py',
        'report_e010_observability_controls.py','verify_e010_observability_context_inputs.py',
        'verify_e010_observability_controls.py','benchmark_e010_observability_controls.py')]
    for path in sources:shutil.copyfile(path,root/'code'/path.name)
    parent=json.loads((root/'parent_freeze.json').read_text())
    stages={name:json.loads((root/name).read_text()) for name in (
        'EXTRACTION_COMPLETED.json','PANEL_COMPLETED.json','ANALYSIS_COMPLETED.json')}
    stages['geometry_search']=json.loads((root/'neighbors/geometry_SEARCH_COMPLETED.json').read_text())
    stages['latent_search']=json.loads((root/'neighbors/latent_SEARCH_COMPLETED.json').read_text())
    package=root/'third_party/faiss_cpu-1.15.1.dist-info/METADATA'
    assert package.exists()
    package_files={str(p):e.old.sha(p) for p in sorted((root/'third_party').rglob('*')) if p.is_file()}
    e.write(root/'dependency_freeze.json',dict(package='faiss-cpu',version='1.15.1',
        installed_on_D=True,dependencies_unchanged=True,files=package_files))
    details=['','## Action-norm sensitivity (descriptive follow-up)','',
        'Angular statistics give every defined direction equal weight. The following outcome-only stratification checks larger teacher vectors. It was added after Phase B, is explicitly descriptive, and did not change any matching rule, context gate, model, checkpoint or decision threshold.', '',
        '| Split | Representation | Both norms above (Å) | Defined-pair coverage | Median cosine | Opposition rate | Norm-weighted mean cosine |','|---|---|---:|---:|---:|---:|---:|']
    for value in json.loads((root/'action_norm_sensitivity.json').read_text())['summaries']:
        details.append(f"| {value['split']} | {value['representation']} | {value['both_action_norms_above_A']:g} | {100*value['fraction_of_defined_pairs']:.2f}% | {value['median_cosine']:.4f} | {100*value['opposition_rate']:.2f}% | {value['action_norm_weighted_mean_cosine']:.4f} |")
    details+=['','## Secondary latent-context comparison','',
        '| Split | Context | Median neighbor cosine | Matched-control cosine | Opposition rate | Explained directional variance |','|---|---|---:|---:|---:|---:|']
    for split in ('validation','test'):
        for name,value in result['summaries'][split]['latent'].items():
            if name=='baseline':continue
            details.append(f"| {split} | {name} | {value['cosine']['median']:.4f} | {value['control_cosine']['median']:.4f} | {100*value['opposition_rate']:.2f}% | {100*value['directional_variance_explained']:.2f}% |")
    details+=['','## Frozen splits and verification','',
        'Identity counts: TRAIN 1,400; validation 300; test 300. All are from the project structural TRAIN pilot. Original split/group/sequence-cluster assignments are preserved.', '',
        '| Split | Identity-list SHA256 |','|---|---|']
    details += [f'| {name} | `{digest}` |' for name,digest in parent['split_identity_sha256'].items()]
    details += ['', f"Exact representation archives: {tests['representation_archives_verified']:,} identities and {tests['eligible_residue_transitions']:,} eligible residue transitions. Frozen student/teacher files verified unchanged: {tests['frozen_student_files_verified']:,}/{tests['frozen_teacher_files_verified']:,}; geometry sources: {tests['frozen_geometry_sources_verified']}. All stored neighbor identity/group/cluster exclusions and reference TRAIN membership pass.", '',
        f"Frozen student-head and coarse P0 reconstruction are bitwise exact. Maximum teacher-frame Cartesian round-trip error: {tests['maximum_teacher_frame_roundtrip_error']:.3g} Å; numerical basis inverses required at {tests['numerical_frame_inverse_corrections']:,} eligible cells. Independent stored-distance/cosine checks pass.", '',
        '| Retrieval | Exact-audit queries | Recall@8 |','|---|---:|---:|']
    for name,info in result['ANN_audits'].items():details.append(f"| {name} | {info['queries']} | {100*info['ladder'][-1]['recall_at_8']:.3f}% |")
    details+=['', 'Approximate retrieval is checked against the exact reference panel; this does not establish that the finite reference panel contains genuinely identical deployable states.', '',
        '## Computation and scope','',
        'Extraction, frozen-model inference, indexing and context scoring ran on CPU. No new protein labels or SLSQP were used. Stage timers below exclude interrupted extraction retries, package installation and final archival checks.', '',
        '| Completed stage | Wall seconds |','|---|---:|']
    for name,stage in stages.items():
        if 'wall_seconds' in stage:details.append(f"| {name} | {stage['wall_seconds']:.2f} |")
    if phase_C:
        details+=['','| Student control | Training wall seconds | GPU training seconds |','|---|---:|---:|']
        for name,info in controls['training'].items():details.append(f"| {name} | {info['wall_seconds']:.2f} | {info['GPU_training_seconds']:.2f} |")
        details+=['','Raw evaluation latency includes current-state feature extraction, network forward and the inference-available action-cap guard. X-based metric scoring is excluded. Static sequence/condition encoding and source-denoiser feature preparation are outside the per-step timer. Raw state predictions are independently reproduced without an X-based shield; all 600 reproduced-baseline identity archives are bitwise equal to fc76211.']
        latency=json.loads((root/'phase_C/latency_benchmark.json').read_text())
        details+=['','An isolated CPU benchmark uses identical validation fixtures for every variant: two identities per length bin, all three conditions, steps1/8, two warmups and ten repeats (600 observations per variant), one Torch thread.', '', '| Control | Isolated median / p90 milliseconds |','|---|---:|']
        for name,value in latency['timings_seconds'].items():details.append(f"| {name} | {1000*value['median']:.3f} / {1000*value['p90']:.3f} |")
        details.append(f"\nStatic denoiser-feature preparation, once per trajectory: median {1000*latency['source_feature_preparation_once_per_trajectory_seconds']['median']:.3f} ms. It uses source/mask only and is excluded from the per-step timing.")
    details+=['', 'The frozen coarse model may have seen these project TRAIN proteins during upstream training; identity holdout here is for the student and neighbor reference bank. No upstream-pretraining holdout claim is made. Exact input extraction covers all archived eligible transitions, while neighbor statistics use the preregistered residue panel and local receptive field. Whole-protein correspondence and arbitrary richer global deployable representations were not tested.', '',
        'All generated archives, temporary files, package files, caches, logs and this report reside under this D:-resident experiment root. Source snapshots and dependency hashes are pinned. COMPLETED.json hashes every scientific output; transient caches and runtime logs are excluded.']
    report=root/'REPORT.md';text=report.read_text();marker='\n## Action-norm sensitivity (descriptive follow-up)\n'
    if marker in text:text=text.split(marker)[0]
    report.write_text(text+'\n'.join(details)+'\n')
    validation=result['summaries']['validation']['geometry']
    doc=e.REPO/'docs/e010_student_direction_observability_v1.md'
    lines=['# E010 student direction observability v1','',
        'Completed identity-held-out observability diagnostic. Feasibility-margin commit 1de4619, student fc76211 and teacher/corpus 6d1ed30 remain frozen. No marginized-target training, increased capacity, new teacher data or SLSQP.', '',
        f'All artifacts: `{root}`. Full report: `REPORT.md`; complete context/length/condition/step aggregates: `results_AB.json`; independent checks: `tests.json`.', '',
        'The exact frozen input, local frame, penultimate latent and teacher action are archived for all 2,000 identities. Neighbor matching uses eight deterministic query residues per each of 96,000 logical K16 states and four reference residues per TRAIN state. The reference bank contains only the original 1,400 student TRAIN identities. Every query excludes its identity, split group and sequence cluster. Frozen validation/test each contain 300 identities.', '',
        '| Representation | Test median teacher-direction cosine | Opposed teacher directions |','|---|---:|---:|']
    for name in ('geometry','latent'):
        s=result['summaries']['test'][name]['baseline']
        lines.append(f"| {name} | {s['cosine']['median']:.4f} | {100*s['opposition_rate']:.2f}% |")
    lines+=['', '| Added context | Validation median cosine gain | Directional variance explained | Gate |','|---|---:|---:|---|']
    for name in ('condition','residue_identity','sequence_local','sequence_condition','step','coarse_scalar_latent','initial_coarse_correction','denoiser_geometry','rollout_history'):
        s=validation[name];g=result['context_gates'][name]
        lines.append(f"| {name} | {s['median_cosine_gain']:.4f} | {100*s['directional_variance_explained']:.2f}% | {g['passes']} |")
    lines+=['', 'Context reranking uses inference-available features only and is compared with geometrically distance-matched controls. X and safety margins are offline outcomes, never matching features. Cached denoiser target-error metrics are excluded. Zero actions have undefined direction and are counted separately.', '',
        ('Denoiser geometry passed the validation-only observability gate, triggering the five authorized same-capacity controls. Their raw deployment evaluation uses no X-based shield. See the controlled comparison below.' if phase_C else 'No requested context passed the validation-only materiality/stability gate. Phase C was not run.')+' Results provide finite-resolution evidence about privileged-policy ambiguity in the tested representations, not proof of information-theoretic unobservability. Corpus density, neighbor metric, local receptive field and sampled spatial coverage remain limitations. Teacher-path history is assessed separately because student rollout history differs.', '',
        'No inference safety improvement is claimed from a neighbor diagnostic. The undeployable X-based shield was not used as a deployment solution. Frozen source hashes, identity-list hashes, exact model parity, all reference exclusions and ANN recall checks are documented in the D:-resident report and completion manifest.']
    if phase_C:
        lines+=['','| Student control | Test raw K16 repair retention | Raw unsafe-state rate | Mean teacher-action cosine |','|---|---:|---:|---:|']
        for name,values in controls['summaries'].items():
            value=values['test'];lines.append(f"| {name} | {100*value['raw_K16_retention']:.2f}% | {100*value['rollout']['hard_failure_rate']:.2f}% | {value['teacher_action_cosine']['mean']:.4f} |")
        lines+=['',f"Validation-selected minimal useful context: {controls['decision']['minimal_useful_context']}. Full raw identity/length/condition/step, directional-error AUC, latency and exact-gate metrics are in `phase_C/results.json`. No further model, teacher-data, full-corpus or final-training action followed."]
    doc.write_text('\n'.join(lines)+'\n')
    start=time.perf_counter();files={}
    for path in sorted(root.rglob('*')):
        relative=path.relative_to(root)
        if not path.is_file() or relative.parts[0] in ('tmp','cache','logs','third_party') or path.name=='COMPLETED.json':continue
        if relative.parts[0] in ('representations','neighbors') and path.suffix=='.npz' and path.name.startswith('identity_'):
            # tests.json was written only after independently verifying these
            # immutable archives against their per-identity completion hashes.
            files[str(path)]=json.loads(path.with_suffix('.json').read_text())['archive_sha256']
        elif relative.parts[0]=='phase_C' and path.suffix=='.npz' and path.name.startswith('identity_'):
            files[str(path)]=json.loads(path.with_suffix('.json').read_text())[0]['action_archive_sha256']
        else:files[str(path)]=e.old.sha(path)
        if len(files)%1000==0:print(json.dumps(dict(hashed_scientific_files=len(files),wall_seconds=time.perf_counter()-start)),flush=True)
    e.write(root/'COMPLETED.json',dict(status='complete',experiment=e.PROTOCOL['experiment'],
        phase_C_run=phase_C,phase_C_gate=result['phase_C_gate'],parent_commit='1de4619',
        files=files,source_sha256={str(path):e.old.sha(path) for path in sources+[doc]},
        frozen_parent=parent,tests=tests,frozen_source_counts=lineage_source_counts,scientific_artifacts_D_only=True,
        no_training=not phase_C,controlled_decision=controls['decision'] if phase_C else None,
        controlled_tests=json.loads((root/'phase_C/tests.json').read_text()) if phase_C else None,
        no_new_teacher_actions=True,no_SLSQP=True,no_full_corpus=True,
        large_identity_archive_hash_provenance='independent verification in tests.json; immutable completion markers reused without redundant reads',
        completed_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    print(json.dumps(dict(status='complete',files=len(files),report=str(report))),flush=True)


if __name__=='__main__':main()
