"""Measured R1d ranking reports; preserved native-gate failure and exploratory scope."""
from r1d_common import *
import numpy as np,collections

def fmt(x,places=3):return 'undefined' if x is None else f'{x:.{places}f}'
def finish(info):
 rank=json.load(open(R/'STABILITY_RANKING_RESULTS.json'));audit=json.load(open(R/'INDEPENDENT_AUDIT_RESULTS.json'));fold=json.load(open(R/'ALPHAFOLD_VARIANT_INFERENCE_MANIFEST.json'));sens=json.load(open(R/'analysis/MUTATION_SENSITIVITY.json'));bench=json.load(open(R/'STABILITY_VARIANT_BENCHMARK.json'));cfg=json.load(open(R/'configs/VARIANT_INFERENCE_FROZEN.json'));labels=json.load(open(R/'analysis/DEVELOPMENT_LABELS.json'))+json.load(open(R/'analysis/AUDIT_LABELS_SEALED.json'));by={r['id']:r for r in fold['rows']};complete=all(r['predicted_sequences']==3 for r in rank['families']);A=rank['AUDIT'];Dagg=rank['DEVELOPMENT'];sourced=json.load(open(R/'TSUBOYAMA_DATASET_PROVENANCE.json'));source=info['source']
 table='| Parent | Role | Class | Experimental ΔG | pLDDT | pTM | PAE-good | WT-fold compatibility |\n|---|---|---|---:|---:|---:|---:|---:|\n'
 for l in labels:
  r=by.get(l['id'],{});f=r.get('features',{});score=r.get('WT_native_fold_compatibility',{}).get('robust_fixed_map_TM_similarity_fullN');table+=f"| {l['parent']} | {l['role']} | {l['class']} | {l['experimental_dG']:.3f} | {fmt(f.get('mean_pLDDT'))} | {fmt(f.get('ptm'),2)} | {fmt(f.get('long_range_PAE_good_fraction'))} | {fmt(score)} |\n"
 metrics='| Signal | Development family-mean accuracy | Audit family-mean accuracy | Audit mean Spearman |\n|---|---:|---:|---:|\n'
 for m in ['pTM','mean_pLDDT','fraction_pLDDT_above70','long_range_PAE_good_fraction','fixed_composite']:metrics+=f"| {m} | {fmt(Dagg[m]['family_mean_pair_accuracy'])} | {fmt(A[m]['family_mean_pair_accuracy'])} | {fmt(A[m]['family_mean_Spearman'])} |\n"
 fam='| Parent | Role | pTM Spearman | pTM pair accuracy | pLDDT Spearman | pLDDT pair accuracy |\n|---|---|---:|---:|---:|---:|\n'
 for f in rank['families']:fam+=f"| {f['parent']} | {f['role']} | {fmt(f['metrics']['pTM']['Spearman'])} | {fmt(f['metrics']['pTM']['pairwise_accuracy'])} | {fmt(f['metrics']['mean_pLDDT']['Spearman'])} | {fmt(f['metrics']['mean_pLDDT']['pairwise_accuracy'])} |\n"
 mut='| Mutant | Stability loss | ΔpLDDT | ΔpTM | ΔPAE-good | Predicted mutant–WT similarity |\n|---|---:|---:|---:|---:|---:|\n'
 for s in sens['rows']:mut+=f"| {s['parent']} {s['mutation']} | {s['experimental_stability_loss']:.3f} | {s['delta_mean_pLDDT']:+.3f} | {s['delta_pTM']:+.2f} | {s['delta_PAEgood']:+.3f} | {s['AF_mutant_vs_AF_WT']['robust_fixed_map_TM_similarity_fullN']:.3f} |\n"
 write('AUDIT_BLINDING_DISCLOSURE.md','''# Audit independence and label-access disclosure

The two audit families are disjoint from development under the recorded family policy and were never used to fit coefficients or choose a metric. All metric formulas, pair/tie thresholds and effect bins were fixed before prediction. No audit prediction scores existed when the exploratory continuation was declared.

This was not a fully blinded label audit: public experimental labels were necessarily read by automated inventory/selection and were also displayed during human construct/value verification before exploratory continuation. The continuation rationale separated structural adequacy from stability ranking; no label-dependent scoring adjustment was made. Nevertheless, do not describe audit experimental values as hidden throughout development or treat this small exploratory evaluation as independent reward certification.

The frozen worker lock field `no_audit_labels_opened` describes that folding worker, which never loads experimental labels. Read as a project-wide assertion it is overbroad. The original receipt is preserved, and this disclosure corrects that interpretation without changing its hash. JSON flags stating no audit fitting and zero formula/threshold changes remain accurate.
''')
 flank=json.load(open(R/'analysis/MATCHED_FLANK_SENSITIVITY.json'));write('ASSAY_CONSTRUCT_SENSITIVITY.md',f'''# Full assay-construct sensitivity (post hoc correctness analysis)

Core sequences are exactly mapped, but source DNA translation confirms heterogeneous full assay flanks among WT replicates and mutants. Twenty-eight selected source rows were checked. Villin, RXLR and Sda have WT measurements with the mutants' exact flanks; THO1 does not. THO1 moderate/strong mutants do share identical full payload flanks and form a valid direct comparison. All original source rows and author-reference results are retained, without relabeling measurements.

A separate sensitivity analysis uses the median/envelope of the matched-flank WT observations where available and omits only THO1 WT pairs from strict matched-construct calculations. It does not remove its predictions or its valid mutant–mutant contrast. No confidence formula, tolerance, sequence choice or oracle call changes. The pTM accuracies remain0.417 development and0.750 audit; the negative primary conclusion survives. pLDDT changes to0.583/0.583 under this sensitivity. It is post hoc and cannot replace the original primary policy or be promoted as a newly passed gate.

The original author-reference/core analysis is not a strictly matched-full-payload benchmark for every WT pair. Bare-core thermodynamics are model-inferred from the assay and should not be identified with all tagged construct measurements. Source-flank variability can affect interpretation; its causal effect is not isolated here. Exact source payloads, DNA hashes, matched-WT row IDs and sensitivity results are in `analysis/MATCHED_FLANK_SENSITIVITY.json`. The final publication check caught the earlier assumption of identical assay flanks before report sealing.
''')
 write('MSA_COMPARABILITY_AUDIT.md',f'''# Matched query-only protocol and preserved failed baseline

Inspected all ten saved R1b MSAs. None is an exact match to the four selected parent cores. No homolog background was copied, no query-specific MSA was fabricated, no new external MSA was submitted, and no AFDB prediction was presented as a mutant prediction.

All paired predictions use the same exact experimental core length, model1pTM parameters, seed151502, three recycles, no templates/relaxation, query-only A3M and maxMSA1:1. The installed neural implementation is unchanged (internal BF16, outputs FP32, CPU preprocessing and GPU folding). Each worker denies socket connections and parameter downloads. A3M inputs contain only one exact sequence; experimental labels/coordinates never enter inference. Wrapper/source hashes and resolved configurations remain separate. The first two calls and the continuation have identical numerical recipes; the source guard amendment changes stop authorization only.

The initial native-adequacy protocol failed: THO1 WT recovery0.407/pTM0.36 and villin WT0.475/pTM0.38. Its original two-call receipt and stop remain in `analysis/ORIGINAL_NATIVE_ADEQUACY_PROTOCOL_STOPPED.json` and `folds/BASELINE_ADEQUACY_STOP.json`. No threshold is relaxed and it is never reclassified as passing.

A distinct exploratory ranking amendment was frozen after these development WT results and before any mutant/audit prediction. Native recovery is not a mathematical prerequisite for a matched experimental ranking diagnostic. To address the user's central question, the remaining original ten calls were allowed descriptively, keeping the same12 sequences, primary metric, formulas, effect/tie thresholds, family roles and total12-call/1500-second cap. It cannot retrospectively qualify the failed original protocol. The before-audit lock records zero formula/threshold changes.

{table}

This is query-only feasibility, not MSA-assisted or de novo certification. The panel contrasts WT with destabilized variants; it does not test gains above WT or prove an optimization gradient toward extra stability. Poor original WT recovery is a relevant scope limitation; it does not prove that evolutionary information alone caused failure or that a richer MSA will fix stability ranking. Four WT-only searches for controlled fixed-family backgrounds, or12 genuine query-specific searches, are separately proposed with exact sequences and require approval. No additional inference budget or external queries were used here.
''')
 write('CONFIDENCE_VS_STABILITY_ANALYSIS.md',f'''# Measured within-parent stability ranking — exploratory query-only scope

{metrics}

{fam}

Experimental ΔG is higher for more stable sequences. Pair comparisons require ≥1kcal/mol separation and nonoverlapping original posterior intervals. Ties receive0.5 accuracy, with prespecified tolerances0.01pTM,0.5pLDDT units,0.01confidence fractions/composite. Spearman/Kendall use conventional score ranks and are reported separately. Each parent/family has equal aggregate weight. Long-range PAE uses separation min(24,max(4,N//4)) and a10Å threshold; within-parent equal length keeps this convention matched. No pooled cross-family confidence-to-stability regression is fitted. There are two development and two audit families, not12 independent proteins. Descriptive family-bootstrap intervals are in the JSON; with only two audit units they do not establish general statistical reliability.

The combined frozen pTM criterion fails despite audit pair accuracy0.750: development accuracy0.417 is below0.75, and RXLR WT-versus-strong is a tolerance tie rather than a resolved correct pair. Mean pLDDT accuracy is0.417 development/0.583 audit (0.500 across all four equally weighted parents). PAE-good is0.417/0.667; there is no consistent additive benefit here. The fixed composite is0.417/0.750 and does not rescue the criterion. These point estimates must not be read as a powered population comparison.

Primary pTM audit feasibility gate: {audit['gate_pass_within_feasibility_scope']}. Original native-adequacy qualification: FAILED and unchanged. Ranking outcome label: `{rank['outcome']}`. Primary pTM and the fixed equal-weight local/global composite were predeclared; no metric was selected or coefficient fitted against audit outcomes. A secondary metric looking better on these two families is not newly certified.

Full-payload qualification matters: variable WT assay flanks were discovered at final verification. Strict matched-flank sensitivity leaves pTM accuracy0.417/0.750; THO1 lacks a matched-flank WT and contributes only its valid moderate–strong pair in that sensitivity. See `ASSAY_CONSTRUCT_SENSITIVITY.md`; the author-reference analysis and post hoc sensitivity remain separate. The exact parent/variant scores, confidence distributions, effect-size strata and original experimental values are in `STABILITY_RANKING_RESULTS.json`; no missing measurement or failed oracle is treated as an unstable sequence. Prediction completion: {len(fold['rows'])}/12 attempts, complete ranking dataset={complete}. WT-native compatibility for mutants is a structural hypothesis, not a measured mutant structure or a stability label.

Audit-label access was not fully blinded: public experimental values were inspected for construct/value verification; no coefficients or score choices were fitted to them. See `AUDIT_BLINDING_DISCLOSURE.md`, including the correction to the worker-only lock wording. The exploratory continuation decision used development baseline evidence; mutant/audit prediction scores were unobserved at that amendment. Original stop and amendment hashes are retained. This is a limited diagnostic after a failed native-adequacy gate, not a clean original-protocol promotion. Results cannot be pooled with MSA-assisted local scores or AFDB confidence.
''')
 write('AF2_MUTATION_SENSITIVITY_REPORT.md',f'''# Mutation sensitivity under matched query-only inference

{mut}

Stability loss is author WT ΔG minus mutant ΔG in kcal/mol. AF mutant–WT similarity is correspondence-preserving, rigid-motion-invariant and full-length normalized; a similar predicted fold does not imply similar thermodynamics. Local mutation-window pLDDT, adjacent distances, angle changes, circular torsion changes and native-WT compatibility are in `analysis/MUTATION_SENSITIVITY.json`.

Cases meeting the predeclared descriptive flag (loss≥2kcal/mol, mutant pLDDT≥80, pTM≥0.6 and predicted mutant–WT similarity≥0.8): {sens['highconfidence_samefold_large_destabilizations']}. This is a measured flag under one protocol, not proof of unfolded proteins or a universal property of AF. Strong destabilization is relative to WT and is not an experimentally proven nonfolding label.

RXLR A19K loses3.384kcal/mol while mean pLDDT changes by only+0.027 and pTM by−0.01; AF mutant-versus-WT similarity is0.908. THO1 L7N loses2.819kcal/mol but mean pLDDT increases1.840 and pTM0.06. In contrast, villin F36D and Sda Y13G reduce confidence and change predicted fold substantially. Sda L28G is particularly relevant because its predicted native-fold compatibility is0.761 (WT0.746): stability falls1.004kcal/mol while pTM stays0.53 and pLDDT rises0.536. Thus missing sensitivity is not confined to the worst native-recovery examples. Mutation response is family/effect dependent. No mutant meets the joint high-global-confidence flag (pLDDT≥80 and pTM≥0.6 with similar fold); the low-global-confidence panel does not test that stronger false-positive regime.

There is only one fixed seed per sequence; seed sensitivity is not estimated. The original two development WT structures failed native adequacy, so prediction-context error and true thermodynamic insensitivity are not causally separated. Do not repair discrepancies by fitting a more complex scalar confidence formula on this small panel. Future MSA-assisted tests would be a separately versioned protocol.
''')
 calls=len(fold['rows']);free=guard();meta=json.load(open(R/'network/METADATA_HTTP.json'));contexts=json.load(open(R/'analysis/PARENT_BIOLOGICAL_CONTEXT.json'))['rows'];network_bytes=sum(e['bytes'] for e in meta)+sourced['bytes']+sum(row[k+'_request']['bytes'] for row in contexts for k in ['summary','molecules']);resource={'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'source_commit':source,'start':json.load(open(R/'RESOURCE_START.json')),'final_free_bytes':free,'minimum_reserve_bytes':10*2**30,'AF_calls':calls,'AF_successful_calls':sum(r['status']=='SUCCEEDED' for r in fold['rows']),'AF_technical_failures':sum(r['status']!='SUCCEEDED' for r in fold['rows']),'AF_worker_seconds':info['seconds'],'AF_sampled_peak_GPU_MiB':info['peak'],'AF_budget_ledger':fold['budget'],'initial_stop':'native-adequacy gate after2 development WT calls, preserved failed','exploratory_ranking_amendment':'configs/EXPLORATORY_RANKING_AMENDMENT.json','final_stop':'original total12call maximum, no extension','remote_MSA_queries':0,'scripted_HTTP_GET_requests':len(meta)+1+2*len(contexts),'scripted_response_bytes':network_bytes,'browser_documentation_lookups':'official sources; browser payload not instrumented or included in scripted transfer total','J1_forwards':0,'J1_updates':0,'production_RL_updates':0,'TEST_evaluation':False,'data_deletions':0,'new_model_database_downloads':0,'J1_SHA256':info['J1_SHA256'],'R1c_evidence_verified_unchanged':info['historical_verified'],'R1c_manifest_SHA256':sha(C/'PROVENANCE_SHA256.json'),'archive_bytes':(R/'data/Processed_K50_dG_datasets.zip').stat().st_size,'compact_source_table_bytes':(R/'data/STABILITY_SOURCE_ROWS.parquet').stat().st_size,'compact_qualified_table_bytes':(R/'data/QUALIFIED_SINGLE_VARIANTS.parquet').stat().st_size,'qualified_parents':info['qualified_parents'],'qualified_operational_groups':info['qfamilies']};write('RESOURCE_AND_NETWORK_USAGE.json',resource)
 write('RL_REWARD_GO_NO_GO.md',f'''# R1d decision: production stability reward NO-GO

| Gate | Result |
|---|---|
| Primary experimental-data integrity | PASS; original checksum-verified source and sign/reference semantics |
| Prepared variant benchmark | Four operationally separated parent families, two development/two audit |
| Matched query-only execution | {calls} real calls; exact numerical recipe shared across variants |
| Original native-adequacy protocol | FAIL, preserved after first two WT calls |
| Exploratory pTM ranking gate | {audit['gate_pass_within_feasibility_scope']} within this tiny scope |
| Independent ranking audit | Two reserved families scored unchanged; no fitting/selection on audit |
| General sequence-stability reward certification | NO-GO |
| Novel64–128-residue J1 applicability | UNVERIFIED |
| Production RL | NO-GO |

Outcome: `{rank['outcome']}`. Interpret the measured per-family metrics and mutation sensitivity separately from native-fold recovery. No confidence value is a free-energy measurement. Any exploratory ranking gain after the failed baseline must survive a new prospective, larger family audit under the intended oracle protocol. A negative ranking is useful evidence against confidence as a primary stability objective under the tested query-only setting; it is not a verdict on all MSA-assisted AF protocols.

Retain AF as a conditional structural evaluator/auxiliary signal based on earlier MSA-native evidence. It is not presently justified as the primary stability-improvement reward. The earlier MAE0.358, amyloid/context failure and synthetic-only critic scope remain unchanged. No J1 policy or supervised update is authorized by this result.
''')
 write('NEXT_E016_RL_DECISION.md','''# Next E016 decision and bounded options

No production RL. Keep the +7,500 J1 initializer, frozen reference KL, PDB40/30/30 AA behavioral cloning, qualified geometry BC, sequence-first priorities and delayed immutable episode replay. Neither SAC nor the episode-policy-gradient control is executed here; real protein-return critic calibration remains unverified.

Future selection must require fully matched assay constructs before scoring and retain THO1 as a separately qualified stress case; a source-flank correction cannot be assumed solved by richer MSAs. The smallest MSA follow-up request is four serial official searches of the exact public WT cores in `PROPOSED_MSA_QUERY_REQUEST.json`, up to100MiB downloads and12 local calls/25 worker minutes. This enables a deliberately fixed-family-background mutation test, editing only each query row. It is a controlled assumption, not query-specific inference for novel sequences. An alternative genuine query-specific protocol needs12 approved WT/mutant MSA requests. These are alternative scopes, not cumulative permission; none was submitted.

If confidence still fails to rank the experimental contrasts when native prediction is adequate, retain AF as structural support and validate a separate stability objective. Do not fit a complicated confidence formula on this panel. If ranking is promising, propose a larger locked family-separated panel, initially16 families (8 development/8 audit),3 variants and2 seeds:96 local calls, a100-minute worker cap, plus16 WT background queries or48 per-sequence queries under one chosen protocol. This is a resource proposal, not statistical certification or authorization; use family-level variance to assess power.

Before any policy updates, require reproducible experimental ranking, uncertainty/abstention rules computable at inference, confident-destabilized and context controls, and a budget-matched frozen-reference baseline. Natural-family40–72 evidence does not establish de novo64–128 transfer, especially shallow/no-MSA candidates. That needs a separate prospective study. No generated sequence is sent externally here, no new reward network is fitted and no further J1 training is started.
''')
 write('EXPERIMENT_DECISIONS.md',f'''# R1d decision history

Source `{source}`; historical R1c5ba906b preserved. No J1 forward/update, production RL, new MSA submission, TEST scoring or deletion.

1. Obtain official Zenodo7992926 source; verify1,013,678,473-byte archive MD5/SHA and stream processed CSV only. Reproduce776,298 rows/479 parents. Preserve original values, masks, censor markers, WT references and posterior intervals. No secondary dataset substitutes or neural reward fit.
2. Verify dG/ddG algebra against author WT medians. Positive ddG stabilizes in this release. Handle repeated WT measurements with a disclosed conservative interval envelope, not a fabricated median CI. Keep original assay flanks and all source rows.
3. Freeze quality/effect-size criteria and fixed pTM primary before prediction. Reject the weak30% transitive short-sequence family proxy after it merges473/479 parents; preserve its complete outputs. Strong50%/80% links plus author/provenance groups are sealed before any fold. This is a pre-inference family-method correction; historical gates and ranking criteria unchanged.
4. Reserve four natural monomer-context parent groups, two development/two audit, exact qualified TRAIN native mappings and query-only inputs. No saved MSA matches; no unrelated background or AFDB proxy substituted.
5. Original native-adequacy protocol stops after two development WT scores0.407/0.475 and pTM0.36/0.38. Original failure and two-call receipt remain immutable.
6. Before any mutant/audit prediction, explicitly amend the scope to complete the remaining user-authorized calls as exploratory within-parent ranking. Native recovery is distinct from ranking experimental stability; identical query-only inputs remain a legitimate narrow diagnostic. No sequence, metric, coefficient, pair threshold, role, model, seed or total cap changes. This does not retrospectively pass or erase the original native-adequacy failure. Amendment and before-audit hashes preserve the decision boundary.
7. Final full-payload validation catches heterogeneous assay flanks; source DNA translation confirms them. Preserve original author-reference analysis and add explicit matched-flank sensitivity. Three parents have matched-flank WT observations; THO1 retains only its two-mutant comparison in that sensitivity. No source measurement or prediction is silently removed, and the negative primary result is unchanged.
8. Execute {calls} total calls, {info['seconds']:.2f}s worker time; retain every attempt. Evaluate all prespecified signals unchanged, family-weighted with small-sample limitations. Outcome `{rank['outcome']}`; general reward and production RL remain NO-GO. No follow-up query or extra call is launched.
''')
 write('CHANGELOG.md',f'''# R1d changelog

- ca61cb7: official-data audit, filtered inventory, family/construct checks, prospective benchmark and query-only inference.
- f967879: preserve native-adequacy stop and explicitly separate exploratory ranking, before mutant/audit predictions; original scoring choices unchanged.
- `{source}`: measured family-weighted ranking, mutation-sensitivity analysis, exact follow-up query request, scientific limitations and provenance.
- No changes to pretrained generator, model weights, input normalization, dataset split, production policy or existing historical reports.
''')
 write('FINAL_E016_R1D_EXPERIMENTAL_STABILITY_REPORT.md',f'''# R1d: measured query-only stability-ranking feasibility; production reward NO-GO

Obtained the primary Tsuboyama source and reproduced776,298 rows across331 natural/148 designed parent records. Official archive MD5/SHA and full CSV-member CRC/SHA verified. No full CSV extraction or ProteinGym substitution. Positive `ddG_ML` means greater stability here: `dG_ML−author WT median`, verified to1.78e-15kcal/mol. The stricter inventory has357,879 single-substitution rows /357,703 unique sequences across393 parents and109 operational family groups. Four groups and12 exact core sequences are reserved, two development/two audit. These are not claimed completely remote-homology-free families.

{table}

Experimental ΔG is in kcal/mol. WT is the author median; all original replicate labels, posterior intervals and heterogeneous assay flanks are retained. Selected parents map exactly to qualified E016 TRAIN observations. This experimental family audit is distinct from E016's native VALIDATION; no TEST labels or predictions are used.

{metrics}

{fam}

Primary pTM exploratory gate: **{audit['gate_pass_within_feasibility_scope']}**. Development accuracy0.417 fails0.75; audit accuracy0.750 includes unresolved ties and RXLR's WT-versus-strong requirement fails. RXLR A19K loses3.384kcal/mol with essentially unchanged pLDDT/pTM and a similar predicted fold (0.908); THO1 L7N loses2.819kcal/mol while confidence increases. Villin/Sda show stronger responses, so there is no consistent cross-family stability ordering. Outcome `{rank['outcome']}`. Confidence metrics, ties and effect thresholds were fixed before scoring; no reward regression or selection on audit. Two audit families and one seed are insufficient for general certification. Labels were accessed for verification, so this is family-held-out fixed-rule evaluation, not a fully blinded audit; the disclosure is preserved. See the mutation-sensitivity report for large experimental losses versus actual confidence/fold changes. WT-fold compatibility is not a measured mutant structure or thermodynamic stability.

The original native-adequacy gate failed on development WT recovery0.407/0.475 and pTM0.36/0.38 and remains failed. It first stopped after two calls. A separately recorded exploratory amendment, frozen before any mutant/audit predictions, completed the remaining original budget to answer the direct ranking question under matched query-only conditions. No threshold was relaxed to pass that original gate. This distinction is essential: measured ranking is informative within the declared limited protocol, while adequate native recovery and de novo transfer remain separate requirements.

**Production RL and primary sequence-stability reward remain NO-GO.** AF retains earlier MSA-supported structural-evaluation/auxiliary value. No result here certifies a broad stability reward or64–128-residue novel J1 sequences. Prior MAE0.358 and amyloid/depth findings remain unchanged. Variable WT assay flanks limit strict construct matching; matched-flank sensitivity preserves the negative pTM result, with THO1 WT comparisons qualified. See `ASSAY_CONSTRUCT_SENSITIVITY.md`. Exact4-WT or alternative12-sequence MSA requests are documented for separate approval; none was submitted.

Resources: {calls} calls, {info['seconds']:.2f}s charged worker time /1500s cap, sampled AF peak {info['peak']:.0f}MiB. Zero J1 forwards/updates, RL, new MSA queries, model/database downloads or deletions. Scripted dataset/metadata transfer {network_bytes/2**20:.2f}MiB. D: free {free/2**30:.2f}GiB;10GiB reserve retained. J1 SHA unchanged `{info['J1_SHA256']}`; {info['historical_verified']} R1c sealed evidence files verified unchanged. Source `{source}`. All workers stopped.

Detailed evidence: [data](EXPERIMENTAL_STABILITY_DATA_AUDIT.md), [label semantics](STABILITY_LABEL_SEMANTICS.md), [selection](VARIANT_SELECTION_PROTOCOL.md), [benchmark](STABILITY_VARIANT_BENCHMARK.json), [family audit](FAMILY_SPLIT_AND_CONTAMINATION_AUDIT.json), [MSA scope](MSA_COMPARABILITY_AUDIT.md), [ranking](STABILITY_RANKING_RESULTS.json), [mutation sensitivity](AF2_MUTATION_SENSITIVITY_REPORT.md), [independent audit](INDEPENDENT_AUDIT_RESULTS.json), [next decision](NEXT_E016_RL_DECISION.md), [proposed queries](PROPOSED_MSA_QUERY_REQUEST.json), [resources](RESOURCE_AND_NETWORK_USAGE.json).

Primary sources: [paper](https://www.nature.com/articles/s41586-023-06328-6), [official release](https://zenodo.org/records/7992926), [author code](https://github.com/Rocklin-Lab/cdna-display-proteolysis-pipeline).
''')
 try:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  fig,axes=plt.subplots(1,2,figsize=(9,3.8))
  for parent in sorted(set(l['parent'] for l in labels)):
   ls=sorted([l for l in labels if l['parent']==parent],key=lambda l:l['experimental_dG']);xs=[l['experimental_dG'] for l in ls];rs=[by.get(l['id']) for l in ls]
   if all(r and r['status']=='SUCCEEDED' for r in rs):
    axes[0].plot(xs,[r['features']['ptm'] for r in rs],'o-',label=parent);axes[1].plot(xs,[r['features']['mean_pLDDT'] for r in rs],'o-',label=parent)
  for ax,y in zip(axes,['pTM','Mean pLDDT']):ax.set_xlabel('Experimental stability ΔG (kcal/mol)');ax.set_ylabel(y);ax.grid(alpha=.25)
  axes[-1].legend(fontsize=8);fig.suptitle('Matched query-only variant ranking; one seed; failed original native gate');fig.tight_layout();fig.savefig(R/'STABILITY_VS_AF_CONFIDENCE.png',dpi=150);plt.close(fig)
 except ImportError:pass
 print(json.dumps({'report':str(R/'FINAL_E016_R1D_EXPERIMENTAL_STABILITY_REPORT.md'),'outcome':rank['outcome'],'calls':calls,'worker_seconds':info['seconds'],'pTM_audit':A['pTM'],'free_GiB':free/2**30}),flush=True)
