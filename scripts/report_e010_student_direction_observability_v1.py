"""Identity-level context gate and complete observability diagnostic report."""
import collections
import json
import time
import numpy as np
import analyze_e010_student_direction_observability_v1 as a
e=a.e
EDGES=np.linspace(-1.,1.,1001)


def bucket():
    return dict(queries=0,pairs=0,defined_context=0,defined_both=0,matched=0,cos_sum=0.,control_cos_sum=0.,
        negative=0,control_negative=0,context_hist=np.zeros(1000,dtype=np.int64),control_hist=np.zeros(1000,dtype=np.int64),
        context_values=[],control_values=[],sse_context=0.,sse_control=0.,vector_queries=0,
        dispersion_context=0.,dispersion_control=0.,norm_difference_sum=0.,norm_count=0,
        zero_query=0,zero_neighbor=0,distance_mismatch_sum=0.,margin_global_sum=0.,margin_local_sum=0.,margin_local_count=0,
        margin_cap_sum=0.,margin_count=0,identity_effects={},distance_sum=0.,distance_count=0,baseline_distance_cosine=[])


def add(b,pairs,vector,identity,exact=False,baseline=False):
    b['queries']+=len(pairs);b['pairs']+=pairs.shape[0]*pairs.shape[1]
    context=np.isfinite(pairs[:,:,0]);both=context&np.isfinite(pairs[:,:,1]);good=both&(pairs[:,:,4]<=.1)
    if baseline:good=context
    b['defined_context']+=int(context.sum());b['defined_both']+=int(both.sum());b['matched']+=int(good.sum())
    x=pairs[:,:,0][good];y=pairs[:,:,1][good]
    b['cos_sum']+=float(x.sum());b['control_cos_sum']+=float(y.sum());b['negative']+=int((x<0).sum());b['control_negative']+=int((y<0).sum())
    b['context_hist']+=np.histogram(x,EDGES)[0];b['control_hist']+=np.histogram(y,EDGES)[0]
    if exact:b['context_values'].append(x.copy());b['control_values'].append(y.copy())
    distance=np.isfinite(pairs[:,:,2]);b['distance_sum']+=float(pairs[:,:,2][distance].sum());b['distance_count']+=int(distance.sum())
    if exact and baseline:b['baseline_distance_cosine'].append(np.stack([pairs[:,:,2][good],x],axis=1))
    if len(x):b['identity_effects'][identity]=float(np.mean(x-y))
    if vector is not None:
        valid=np.isfinite(vector).all(1);v=vector[valid]
        b['vector_queries']+=len(v)
        if len(v):
            b['sse_context']+=float(v[:,0].sum());b['sse_control']+=float(v[:,1].sum())
            b['dispersion_context']+=float(v[:,2].sum());b['dispersion_control']+=float(v[:,3].sum())
    normvalid=np.isfinite(pairs[:,:,5])&np.isfinite(pairs[:,:,7])
    b['norm_difference_sum']+=float(np.abs(pairs[:,:,5]-pairs[:,:,7])[normvalid].sum());b['norm_count']+=int(normvalid.sum())
    b['zero_query']+=int((pairs[:,0,7]<=1e-8).sum());b['zero_neighbor']+=int((pairs[:,:,5]<=1e-8).sum())
    b['distance_mismatch_sum']+=float(pairs[:,:,4][both].sum())
    margin=np.isfinite(pairs[:,:,8]);b['margin_count']+=int(margin.sum())
    b['margin_global_sum']+=float(pairs[:,:,8][margin].sum());b['margin_cap_sum']+=float(pairs[:,:,10][margin].sum())
    local=np.isfinite(pairs[:,:,9]);b['margin_local_count']+=int(local.sum());b['margin_local_sum']+=float(pairs[:,:,9][local].sum())


def quantile(hist,q):
    if not hist.sum():return None
    i=min(int(np.searchsorted(np.cumsum(hist),q*hist.sum())),999)
    return float((EDGES[i]+EDGES[i+1])/2)


def finalize(b,bootstrap=False):
    n=b['matched'];v=b['vector_queries']
    if b['context_values']:
        x=np.concatenate(b['context_values']);y=np.concatenate(b['control_values'])
        sx=e.old.frozen.old.statistics(x);sy=e.old.frozen.old.statistics(y);method='exact'
    else:
        sx=dict(count=n,p10=quantile(b['context_hist'],.1),median=quantile(b['context_hist'],.5),p90=quantile(b['context_hist'],.9),mean=b['cos_sum']/n if n else None)
        sy=dict(count=n,p10=quantile(b['control_hist'],.1),median=quantile(b['control_hist'],.5),p90=quantile(b['control_hist'],.9),mean=b['control_cos_sum']/n if n else None);method='histogram midpoint +/-0.001'
    d=dict(queries=b['queries'],scored_pairs=n,direction_defined_context_pairs=b['defined_context'],direction_defined_both_pairs=b['defined_both'],
        geometry_matched_coverage=n/b['defined_both'] if b['defined_both'] else 0.,cosine=sx,control_cosine=sy,quantile_method=method,
        opposition_rate=b['negative']/n if n else None,control_opposition_rate=b['control_negative']/n if n else None,
        paired_mean_cosine_gain=(b['cos_sum']-b['control_cos_sum'])/n if n else None,
        median_cosine_gain=sx['median']-sy['median'] if sx.get('median') is not None else None,
        directional_variance_explained=1-b['sse_context']/b['sse_control'] if b['sse_control']>0 else None,
        vector_queries=v,context_direction_dispersion=b['dispersion_context']/v if v else None,
        control_direction_dispersion=b['dispersion_control']/v if v else None,
        mean_absolute_action_norm_difference=b['norm_difference_sum']/b['norm_count'] if b['norm_count'] else None,
        zero_query_action_rate=b['zero_query']/b['queries'],zero_neighbor_action_rate=b['zero_neighbor']/b['pairs'],
        mean_relative_geometry_distance_mismatch=b['distance_mismatch_sum']/b['defined_both'] if b['defined_both'] else None,
        mean_squared_representation_distance=b['distance_sum']/b['distance_count'] if b['distance_count'] else None,
        mean_absolute_global_margin_difference=b['margin_global_sum']/b['margin_count'] if b['margin_count'] else None,
        mean_absolute_local_margin_difference=b['margin_local_sum']/b['margin_local_count'] if b['margin_local_count'] else None,
        mean_absolute_action_ball_margin_difference=b['margin_cap_sum']/b['margin_count'] if b['margin_count'] else None)
    if bootstrap:
        values=np.array(list(b['identity_effects'].values()));rng=np.random.default_rng(e.PROTOCOL['seed'])
        samples=values[rng.integers(len(values),size=(1000,len(values)))].mean(1)
        d['identity_bootstrap_mean_cosine_gain']=dict(identities=len(values),mean=float(values.mean()),
            lower95=float(np.quantile(samples,.025)),upper95=float(np.quantile(samples,.975)),replicates=1000)
    if b['baseline_distance_cosine']:
        data=np.concatenate(b['baseline_distance_cosine']);cuts=np.quantile(data[:,0],[0,.25,.5,.75,1.]);bins=[]
        for i,(lo,hi) in enumerate(zip(cuts[:-1],cuts[1:])):
            chosen=(data[:,0]>=lo)&(data[:,0]<=hi if i==3 else data[:,0]<hi);values=data[chosen,1]
            bins.append(dict(squared_distance_lo=float(lo),squared_distance_hi=float(hi),pairs=len(values),
                median_cosine=float(np.median(values)),opposition_rate=float(np.mean(values<0))))
        d['direction_agreement_by_neighbor_distance_quartile']=bins
    return d


def main():
    assert (e.ROOT/'ANALYSIS_COMPLETED.json').exists()
    start=time.perf_counter();groups=collections.defaultdict(bucket);meta=a.open_array('meta');identity_summaries=[]
    for row in e.rows():
        with np.load(e.ROOT/'neighbors'/f'identity_{row["index"]:04d}.npz',allow_pickle=False) as z:
            query=z['query_rows'];m=meta[query];split={0:'train',1:'validation',2:'test'}[int(m[0,6])]
            for representation in ('geometry','latent'):
                for context in ['baseline']+a.CONTEXTS:
                    pairs=z[f'{representation}_{context}_pairs'];vector=z[f'{representation}_{context}_vector']
                    own=bucket();add(own,pairs,vector,row['index'],baseline=context=='baseline')
                    identity_summaries.append(dict(identity=row['sample_id'],index=row['index'],split=split,length=row['length'],
                        length_bin=row['stratum'],representation=representation,context=context,**finalize(own)))
                    categories=[('all','all',np.ones(len(query),bool)),('length',row['stratum'],np.ones(len(query),bool))]
                    categories += [('condition',str(c),m[:,1]==i) for i,c in enumerate((50,250,450))]
                    categories += [('step',str(i+1),m[:,2]==i) for i in range(16)]
                    for dimension,label,mask in categories:
                        add(groups[(split,representation,context,dimension,label)],pairs[mask],vector[mask] if vector is not None else None,
                            row['index'],exact=dimension=='all' and split!='train',baseline=context=='baseline')
        if (row['index']+1)%100==0:print(json.dumps(dict(aggregated_identities=row['index']+1,wall_seconds=time.perf_counter()-start)),flush=True)
    e.write(e.ROOT/'identity_summaries.json',identity_summaries)
    summaries={split:{representation:{context:finalize(groups[(split,representation,context,'all','all')],bootstrap=context!='baseline' and split=='validation')
        for context in ['baseline']+a.CONTEXTS} for representation in ('geometry','latent')} for split in ('train','validation','test')}
    strata=[dict(split=s,representation=r,context=c,dimension=d,label=l,**finalize(b)) for (s,r,c,d,l),b in groups.items() if d!='all']
    gates={}
    for context in a.CONTEXTS:
        x=summaries['validation']['geometry'][context];boot=x['identity_bootstrap_mean_cosine_gain']
        neg=1-x['opposition_rate']/x['control_opposition_rate'] if x['control_opposition_rate'] else 0.
        length=[next(r for r in strata if r['split']=='validation' and r['representation']=='geometry' and r['context']==context and r['dimension']=='length' and r['label']==label) for label in e.old.STRATA]
        condition=[next(r for r in strata if r['split']=='validation' and r['representation']=='geometry' and r['context']==context and r['dimension']=='condition' and r['label']==str(c)) for c in (50,250,450)]
        checks=dict(median_gain=x['median_cosine_gain']>=.05,opposition_reduction=neg>=.20,
            variance_explained=x['directional_variance_explained'] is not None and x['directional_variance_explained']>=.03,
            coverage=x['geometry_matched_coverage']>=.75,positive_identity_CI=boot['lower95']>0,
            length_stability=sum(r['paired_mean_cosine_gain']>0 for r in length)>=4,
            condition_stability=all(r['paired_mean_cosine_gain']>0 for r in condition))
        gates[context]=dict(passes=all(checks.values()),checks=checks,opposition_reduction=neg,test_used=False)
    eligible=['condition','residue_identity','sequence_local','sequence_condition','coarse_scalar_latent','initial_coarse_correction','denoiser_geometry']
    passing=[c for c in eligible if gates[c]['passes']]
    winner=max(passing,key=lambda c:summaries['validation']['geometry'][c]['directional_variance_explained']) if passing else None
    result=dict(protocol_sha256=e.old.sha(e.ROOT/'protocol.json'),summaries=summaries,strata=strata,
        context_gates=gates,phase_C_gate=dict(run=bool(passing),passing_contexts=passing,winning_context=winner,selection_split='validation',test_used=False),
        ANN_audits={n:json.loads((e.ROOT/'neighbors'/f'{n}_audit.json').read_text()) for n in ('geometry','latent')},
        coverage=json.loads((e.ROOT/'PANEL_COMPLETED.json').read_text()),
        ambiguity_limits='Finite-resolution nearest-neighbor disagreement is not proof of exact input collisions or fundamental unobservability; full global current-state information and arbitrary inference-available representations are not exhaustively tested.')
    e.write(e.ROOT/'results_AB.json',result)
    lines=['# E010 student direction observability v1','',
        'Frozen baseline commits: feasibility-margin 1de4619; student fc76211; teacher/corpus 6d1ed30. No new teacher actions, SLSQP solves, marginized targets or capacity increase.',
        '', '## Representation and cohort','',
        'All eligible archived teacher transitions are extracted for the frozen 2,000-identity TRAIN pilot. Logical K16 zero tails are explicitly mapped to exact repeated states. Inputs are the exact 41 float32 geometric channels and float64 local frames; penultimate hidden states have 64 channels. Matching uses the complete two-block ±6-residue receptive field (13×41), not only central-residue geometry. Teacher Cartesian vectors are expressed in the same frozen local bases.',
        '', 'Neighbors are residue-centered decisions, summarized over states/proteins. A deterministic eight-residue query panel covers all 96,000 logical protein-condition-step states (768,000 decisions). Four residues per TRAIN state form the 268,800-point reference bank. Every search excludes the query identity, split group and sequence cluster, including TRAIN leave-group-out queries. Validation/test identities never enter the bank. Sampling uses no teacher actions. Unsampled eligible residues are fully archived but are not individually queried; conclusions have finite spatial and cohort coverage.',
        '', '## Phase A: directional ambiguity','', '| Split | Representation | Median neighbor teacher cosine | Opposition rate | Zero neighbor actions |', '|---|---|---:|---:|---:|']
    for split in ('validation','test'):
        for rep in ('geometry','latent'):
            s=summaries[split][rep]['baseline']
            lines.append(f"| {split} | {rep} | {s['cosine']['median']:.4f} | {100*s['opposition_rate']:.2f}% | {100*s['zero_neighbor_action_rate']:.2f}% |")
    lines+=['','| Split | Representation | Cosine p10 / p90 | Mean absolute action-norm difference (Å) | Global / local / action-ball margin difference |', '|---|---|---:|---:|---:|']
    for split in ('validation','test'):
        for rep in ('geometry','latent'):
            s=summaries[split][rep]['baseline']
            lines.append(f"| {split} | {rep} | {s['cosine']['p10']:.4f} / {s['cosine']['p90']:.4f} | {s['mean_absolute_action_norm_difference']:.5f} | {s['mean_absolute_global_margin_difference']:.5g} / {s['mean_absolute_local_margin_difference']:.5g} / {s['mean_absolute_action_ball_margin_difference']:.5g} |")
    lines+=['','Global/local margins use their frozen normalized constraint units; action-ball margins are Å. Their variation is descriptive and is never used to retrieve neighbors.', '', '| Split | Representation | Distance quartile (nearest first) | Median cosine | Opposition rate |', '|---|---|---:|---:|---:|']
    for split in ('validation','test'):
        for rep in ('geometry','latent'):
            for i,s in enumerate(summaries[split][rep]['baseline']['direction_agreement_by_neighbor_distance_quartile'],1):
                lines.append(f"| {split} | {rep} | {i} | {s['median_cosine']:.4f} | {100*s['opposition_rate']:.2f}% |")
    lines+=['','Norms ≤1e-8 have undefined direction and are excluded from angular statistics; their frequency is reported. Native safety margins are offline outcomes only and never enter matching. Frame conversion retains transpose coordinates when their Cartesian round trip is below 1e-12 Å and solves the actual numerical 3×3 basis at the few affected residues otherwise, without changing frames or teacher Cartesian actions.',
        '', '## Phase B: geometry-controlled observable context','',
        'Contexts rerank the same fixed candidate set without teacher labels. Each selected neighbor has a different-identity control at the closest geometric distance. Effects use only defined-direction pairs with ≤10% relative distance mismatch. Coverage, control distance mismatch and zero-action rates are archived. Directional variance explained compares unit-vector prediction SSE with distance-matched controls. Direction dispersion is a resultant-vector variance proxy, not Shannon entropy.',
        '', '| Context | Validation median cosine | Matched-control cosine | Median gain | Opposition rate | Explained directional variance | Matched coverage | Gate |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for context in a.CONTEXTS:
        s=summaries['validation']['geometry'][context]
        lines.append(f"| {context} | {s['cosine']['median']:.4f} | {s['control_cosine']['median']:.4f} | {s['median_cosine_gain']:.4f} | {100*s['opposition_rate']:.2f}% | {100*s['directional_variance_explained']:.2f}% | {100*s['geometry_matched_coverage']:.2f}% | {gates[context]['passes']} |")
    lines+=['','| Context | Test median cosine | Matched-control cosine | Opposition rate | Explained directional variance |', '|---|---:|---:|---:|---:|']
    for context in a.CONTEXTS:
        s=summaries['test']['geometry'][context]
        lines.append(f"| {context} | {s['cosine']['median']:.4f} | {s['control_cosine']['median']:.4f} | {100*s['opposition_rate']:.2f}% | {100*s['directional_variance_explained']:.2f}% |")
    lines+=['', '## Held-out strata', '', 'Geometric nearest-neighbor directional agreement by length, corruption condition and rollout step. Angular statistics exclude zero actions; zero query-action rates make late zero tails explicit.', '', '| Split | Stratum | Median cosine | Opposition rate | Zero query actions |', '|---|---|---:|---:|---:|']
    for r in strata:
        if r['split'] in ('validation','test') and r['representation']=='geometry' and r['context']=='baseline':
            cosine=r['cosine']['median'];opposition=r['opposition_rate']
            cos_text=f'{cosine:.4f}' if cosine is not None else 'undefined'
            opposition_text=f'{100*opposition:.2f}%' if opposition is not None else 'undefined'
            lines.append(f"| {r['split']} | {r['dimension']} {r['label']} | {cos_text} | {opposition_text} | {100*r['zero_query_action_rate']:.2f}% |")
    lines+=['','Sequence is primary-sequence identity and local ±3 context only. Condition is the known corruption/diffusion timestep; rollout step is controller metadata. Frozen coarse scalar features come from the existing global model applied to archived denoiser prediction and mask. Cached denoiser_error_rmse and normalized_x0_hat_rmse are X-derived and excluded. Coarse-output context is decomposed into initial correction, denoiser geometry and rollout history. Teacher-path history can overstate deployment utility because a student produces its own history.',
        '', '## Preregistered decision','',
        'The validation-only gate requires median cosine gain ≥0.05, ≥20% relative opposition reduction, ≥3% directional SSE explained, ≥75% geometric matched coverage, a positive identity-bootstrap 95% lower bound for mean cosine gain, and positive effects in ≥4/5 length bins and all three conditions. Test never selects context. Step/history-only and composite context effects do not automatically add unrequested training variants.',
        '', f"Phase C gate: **{bool(passing)}**. Passing requested context sources: {passing}. Winner: {winner}.",
        '', 'All context sources, representations, length/condition/step strata, norm variation, margin variation, zero-direction incidence, bootstrap bounds and gate checks are in results_AB.json. Per-protein aggregates are in identity_summaries.json. Per-decision neighbor identities, distances, labels and controls are reconstructible from neighbors/ and panel/. Complete exact features/latent/frame/teacher labels/eligibility are in representations/.']
    if not passing:
        lines+=['','## Interpretation and stop','',
            'No requested context source meets the preregistered materiality/stability gate. Stop without retraining. Nearest-neighbor disagreement is evidence of residual teacher-direction ambiguity in these deployable representations, with no demonstrated context remedy under the tested matching rule. It is not proof that every possible inference-available representation is insufficient: finite corpus density, the metric, approximate retrieval, sample coverage and the teacher’s nonlinear policy can all contribute. The experiment does not establish an information-theoretic impossibility.', '',
            'No student variant, larger model, additional teacher labels, full corpus, SLSQP solve or final training was launched. The X-based shield remains undeployable and is absent from this analysis’s inference path.']
    (e.ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(phase_C_gate=result['phase_C_gate'],context_gates=gates)),flush=True)


if __name__=='__main__':main()
