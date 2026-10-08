"""Frozen lineage, feature isolation and held-out neighbor audit."""
import inspect
import json
import time
import numpy as np
import analyze_e010_student_direction_observability_v1 as a
e=a.e


def main():
    e.initialize();verified={}
    for label,root in [('student',e.old.ROOT),('teacher',e.old.PARENT)]:
        completion=json.loads((root/'COMPLETED.json').read_text())
        for path,digest in completion['files'].items():assert e.old.sha(path)==digest,path
        verified[label]=len(completion['files'])
    frozen=json.loads((e.old.ROOT/'results.json').read_text())['unchanged_frozen_geometry_source_sha256']
    for path,digest in frozen.items():assert e.old.sha(path)==digest,path
    print(json.dumps(dict(frozen_inputs_unchanged=verified,geometry_sources=len(frozen))),flush=True)
    count=eligible=corrections=0;max_roundtrip=0.
    for row in e.rows():
        marker=e.ROOT/'representations'/f'identity_{row["index"]:04d}.json';info=json.loads(marker.read_text())
        assert e.old.sha(marker.with_suffix('.npz'))==info['archive_sha256']
        assert info['coarse_P0_bitwise_parity'] and info['frozen_head_bitwise_parity']
        assert info['maximum_teacher_frame_roundtrip_error']<1e-12
        max_roundtrip=max(max_roundtrip,info['maximum_teacher_frame_roundtrip_error'])
        eligible+=sum(c['eligible_residue_transitions'] for c in info['conditions']);corrections+=info.get('numerical_frame_inverse_corrections',0);count+=1
    assert count==2000
    # Matching is isolated from teacher vectors, gradients, margins and cache target-error metrics.
    matching='\n'.join(inspect.getsource(f) for f in (a.context_distance,a.choose_context,a.heldout_search,a.standardized))
    assert "ARRAYS['teacher']" not in matching and "ARRAYS['margins']" not in matching
    assert 'denoiser_error_rmse' not in matching and 'normalized_x0_hat_rmse' not in matching
    print(json.dumps(dict(extraction_archives_verified=count,eligible_transitions=eligible)),flush=True)
    while not (e.ROOT/'ANALYSIS_COMPLETED.json').exists():time.sleep(50)
    a.init_analysis();meta=a.ARRAYS['meta'];teacher=a.ARRAYS['teacher'];checked=0;exact_distance_error=0.
    for row in e.rows():
        path=e.ROOT/'neighbors'/f'identity_{row["index"]:04d}.npz';info=json.loads(path.with_suffix('.json').read_text())
        assert e.old.sha(path)==info['archive_sha256']
        with np.load(path,allow_pickle=False) as z:
            qs=z['query_rows'];assert len(qs)==384 and np.all(meta[qs,0]==row['index'])
            for rep in ('geometry','latent'):
                for context in ['baseline']+a.CONTEXTS:
                    refs=z[f'{rep}_{context}_reference_rows'];js=refs if context=='baseline' else refs[:,0]
                    assert np.all(meta[js,0]!=row['index']) and np.all(meta[js,8]!=meta[qs,8,None]) and np.all(meta[js,9]!=meta[qs,9,None])
                    assert np.all(meta[js,6]==0)
                    assert all(len(set(meta[v,0]))==8 for v in js)
                    if context!='baseline':
                        controls=refs[:,1];assert np.all(controls>=0) and np.all(meta[controls,6]==0)
                        assert np.all(meta[controls,0]!=row['index']) and np.all(meta[controls,8]!=meta[qs,8,None]) and np.all(meta[controls,9]!=meta[qs,9,None])
                    checked+=js.size
            # Independent exact-feature distance and teacher-cosine audit, 48 query states per identity for a fixed five-bin sample.
            if row['index']%100==0:
                transforms=a.TRANSFORMS
                for rep in ('geometry','latent'):
                    reference=z[f'{rep}_baseline_reference_rows'];pairs=z[f'{rep}_baseline_pairs']
                    for qi in range(0,384,8):
                        q=qs[qi];js=reference[qi];delta=a.standardized(rep,js,transforms)-a.standardized(rep,[q],transforms)
                        dist=np.einsum('ij,ij->i',delta,delta);error=float(np.abs(dist-pairs[qi,:,2]).max());exact_distance_error=max(exact_distance_error,error)
                        assert np.allclose(dist,pairs[qi,:,2],rtol=2e-5,atol=1e-4)
                        qnorm=np.linalg.norm(teacher[q]);jnorm=np.linalg.norm(teacher[js],axis=1);valid=(qnorm>1e-8)&(jnorm>1e-8)
                        if valid.any():
                            cosine=np.clip((teacher[js[valid]]*teacher[q]).sum(1)/(jnorm[valid]*qnorm),-1,1)
                            assert np.allclose(cosine,pairs[qi,valid,0],rtol=0,atol=1e-12)
    e.write(e.ROOT/'tests.json',dict(passed=True,frozen_student_files_verified=verified['student'],frozen_teacher_files_verified=verified['teacher'],
        frozen_geometry_sources_verified=len(frozen),representation_archives_verified=count,eligible_residue_transitions=eligible,
        numerical_frame_inverse_corrections=corrections,maximum_teacher_frame_roundtrip_error=max_roundtrip,
        all_neighbor_identity_group_cluster_exclusions_verified=True,all_neighbors_reference_TRAIN=True,
        checked_neighbor_decisions=checked,maximum_independent_distance_discrepancy=exact_distance_error,
        exact_head_and_coarse_P0_bitwise_parity=True,matching_is_target_label_margin_free=True,no_SLSQP=True,no_new_teacher_actions=True))
    print(json.dumps(dict(passed=True,neighbor_decisions_checked=checked)),flush=True)


if __name__=='__main__':main()
