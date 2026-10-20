from r1e_common import *
import collections,subprocess,time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'figure.dpi':140,'savefig.dpi':170})
panel=read('FROZEN_J1_GENERATION_PANEL.json');esm=read('ESM2_SEQUENCE_PLAUSIBILITY.json');bench=read('EXPERIMENTAL_DESIGN_SUCCESS_BENCHMARK.json');comp=read('SEQUENCE_COMPOSITION_AND_COMPLEXITY.json');div=read('GENERATION_VALIDITY_AND_DIVERSITY.json');nov=read('SEQUENCE_NOVELTY_AND_HOMOLOGY.json');aux=read('AUXILIARY_STRUCTURE_EVALUATION.json');tests=read('EVALUATION_CORRECTNESS_TESTS.json');avail=read('EVALUATOR_AVAILABILITY.json');lengths=[64,128,256,384,500];groups=['J1_7500','J1_500','natural_TRAIN','natural_VALIDATION','shuffled'];labels=['J1 +7,500','J1 +500','Natural TRAIN','Natural VALIDATION','TRAIN composition shuffled'];colors=['#147d92','#cc7832','#38843c','#7c55a5','#777777']
counts={g:{N:sum(r['group']==g and r['N']==N for r in panel['rows'])for N in lengths}for g in groups}
fig,ax=plt.subplots(figsize=(9,5))
for j,(g,label,col) in enumerate(zip(groups,labels,colors)):
 vals=[esm['summary'][g][str(N)]for N in lengths];means=np.array([v['mean']for v in vals]);cis=np.array([v['mean_bootstrap95']for v in vals]);x=np.arange(5)+(j-2)*.11;ax.errorbar(x,means,yerr=[means-cis[:,0],cis[:,1]-means],marker='o',capsize=3,label=label,color=col)
ax.set_xticks(range(5),lengths);ax.set_xlabel('Requested / exact native sequence length');ax.set_ylabel('Mean sampled masked log probability (nats / residue; higher = more compatible)');ax.set_title('Frozen ESM-2 35M: generated, natural and shuffled sequences');ax.legend(fontsize=8);fig.text(.5,.01,'16 single-mask positions / sequence; n=32/group/length except VALIDATION '+str([counts['natural_VALIDATION'][N]for N in lengths])+'; 95% sequence bootstrap CI.\nIntervals condition on fixed sampled positions; not folding/stability estimates.',ha='center',fontsize=8);fig.tight_layout(rect=(0,.08,1,1))
for ext in ['png','pdf']:fig.savefig(R/f'ESM2_SEQUENCE_COMPARISON.{ext}')
plt.close(fig)
main=bench['results']['paper_main570_excluding_solved_NMR_pdb'];names=list(main);fig,ax=plt.subplots(figsize=(9,5));m=np.array([main[n]['AUROC']for n in names]);ci=np.array([main[n]['AUROC_study_cluster_bootstrap95']for n in names]);ax.errorbar(m,np.arange(len(names)),xerr=[m-ci[:,0],ci[:,1]-m],fmt='o',capsize=4);ax.set_yticks(range(len(names)),[n.replace('ESM2_sampled_masked_mean_logp','ESM-2 sampled masked PLL')for n in names]);ax.axvline(.5,color='gray',ls='--');ax.set_xlim(.2,.9);ax.set_xlabel('AUROC (higher = better success/failure discrimination)');ax.set_title('Garcia published design benchmark: individual frozen evaluators');fig.text(.5,.01,'n=570 designs, 225 successes / 345 failures, 11 studies; author exclusion of44 solved-NMR designs.\n95% study-cluster bootstrap CI, 1,000 replicates. Correlated designs are not independent families. No predictor fitted.',ha='center',fontsize=8);fig.tight_layout(rect=(0,.08,1,1))
for ext in ['png','pdf']:fig.savefig(R/f'EXPERIMENTAL_SCORE_DISCRIMINATION.{ext}')
plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,4.5))
for ax,key,title in zip(axes,['Rg_A','clashes_per100_residues','long_contact_fraction'],['Radius of gyration (A)','Clashes /100 residues (lower)','Long-range contact fraction']):
 for g,label,col in zip(groups[:2],labels[:2],colors[:2]):
  v=[aux['metrics'][g][str(N)][key]for N in lengths];mean=np.array([x['mean']for x in v]);ci=np.array([x['mean_bootstrap95']for x in v]);ax.errorbar(lengths,mean,yerr=[mean-ci[:,0],ci[:,1]-mean],fmt='o-',capsize=3,label=label,color=col)
 ref=[aux['native_reference']['TRAIN'][str(N)]['metrics'][key]for N in lengths];ax.plot(lengths,[x['mean']for x in ref],color='green',ls='--',label='Frozen native TRAIN window');ax.set_title(title);ax.set_xlabel('Length');ax.legend(fontsize=7)
fig.suptitle('Auxiliary J1 CA geometry: n=32 generated attempts / model / length');fig.text(.5,.005,'95% sequence bootstrap CI for generated means. Native reference uses frozen +/-10% length windows, not exact-length controls.\nClash: distance<2A and sequence separation>=3; contact: distance<8A and separation>=24. Rg/contact are not monotonic quality scores.',ha='center',fontsize=8);fig.tight_layout(rect=(0,.1,1,.94))
for ext in ['png','pdf']:fig.savefig(R/f'AUXILIARY_GEOMETRY_COMPARISON.{ext}')
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4))
for j,g in enumerate(groups[:2]):
 diff=[]
 for N in lengths:
  def avg(group):return np.mean([r['composition']for r in comp['rows']if r['group']==group and r['N']==N],axis=0)
  diff.append(avg(g)-avg('natural_TRAIN'))
 im=axes[j].imshow(diff,aspect='auto',vmin=-.04,vmax=.04,cmap='coolwarm');axes[j].set_xticks(range(20),list(AA));axes[j].set_yticks(range(5),lengths);axes[j].set_title(labels[j]+' minus natural TRAIN');axes[j].set_ylabel('Sequence length')
fig.colorbar(im,ax=axes.ravel().tolist(),label='AA-frequency difference (descriptive; no good direction)');fig.suptitle('Composition: n=32 sequences per group and exact length');fig.subplots_adjust(top=.8,bottom=.2,right=.85);fig.text(.5,.04,'Full sequences, all canonical residues; per-length TRAIN reference. Heatmap shows point estimates without uncertainty.',ha='center',fontsize=8)
for ext in ['png','pdf']:fig.savefig(R/f'COMPOSITION_DIFFERENCES.{ext}')
plt.close(fig)
def md(name,s):(R/name).write_text(s)
def fmt(x):return f'{x:.3f}'
ESMtable='| Length | +7,500 | +500 | Natural TRAIN | Shuffled | Paired difference [95% CI] |\n|---:|---:|---:|---:|---:|---|\n'
for N in lengths:
 v=esm['paired_J1_7500_minus_500'][str(N)];ESMtable+=f"| {N} | "+' | '.join(fmt(esm['summary'][g][str(N)]['mean'])for g in ['J1_7500','J1_500','natural_TRAIN','shuffled'])+f" | {v['mean']:+.3f} [{v['mean_bootstrap95'][0]:+.3f}, {v['mean_bootstrap95'][1]:+.3f}] |\n"
exptable='| Evaluator (oriented toward success) | AUROC | Study macro | Study-bootstrap 95% CI | Average precision | High-score failures* |\n|---|---:|---:|---|---:|---:|\n'
for n,v in main.items():exptable+=f"| {n} | {v['AUROC']:.3f} | {v['study_macro_AUROC']:.3f} | [{v['AUROC_study_cluster_bootstrap95'][0]:.3f}, {v['AUROC_study_cluster_bootstrap95'][1]:.3f}] | {v['average_precision']:.3f} | {v['top_quartile_failed_count']} |\n"
md('LITERATURE_AND_BENCHMARK_SOURCES.md','''# Sources and adaptation boundaries

Accessed2026-10-20.

- [ProteinBench, ICLR2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/49b9d06c2dae5af46125490e0a6f9e32-Abstract-Conference.html), [official benchmark](https://proteinbench.github.io/), appendix B.1.3: retain separate quality, novelty, diversity and robustness dimensions. Published sequence naturalness uses ProGen2 autoregressive perplexity; R1e substitutes explicitly labeled ESM-2 sampled masked PLL. Its structural diversity/novelty require independent fold predictions; our sequence alignments and CA diversity are auxiliary alternatives, not the same leaderboard metrics. Website AF2 labels and appendix ESMFold description differ; no pooled numerical leaderboard comparison is claimed.
- [ProtGPT2, Nature Communications2022](https://www.nature.com/articles/s41467-022-32007-7): motivates composition, disorder, novelty and diverse sampling checks. Its BPE top-k950/repetition1.2 decoding is not applicable to J1's20-residue categorical head. We preserve J1's archived decoding rather than copy tokenizer-specific settings. No ProtGPT2 model or experimental claim is reproduced.
- [Garcia, Dixit & Rocklin2026](https://doi.org/10.1002/pro.70453), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12817478/): official XML and complete DataS1 preserved under data/. CSV alias PRO-35-e70453-s002.csv recovered from the official Europe PMC supplementaryFiles archive after publisher403/PMC challenge. CC BY4.0. We reproduce frozen individual metrics, not the authors' fitted logistic predictor.
- [ESM official inference source](https://github.com/facebookresearch/esm), [35M model card](https://huggingface.co/facebook/esm2_t12_35M_UR50D): official fair-esm2.0.0 MIT wheel, direct FAIR weights. UR50D pretraining overlaps may exist; no independence from all benchmark proteins is assumed.

This is a focused adaptation, not a complete ProteinBench installation or state-of-the-art leaderboard result. No full benchmark/proteome database downloads.
''')
md('EVALUATOR_INSTALLATION_AND_RESOURCE_AUDIT.md','''# Inference evaluator audit

| Tool | Availability and use | License / interpretation | Resource evidence |
|---|---|---|---|
| MMseqs2 | /usr/bin/mmseqs15-6f452+ds-2; bounded local TRAIN/VALIDATION searches | GPLv3; significant local similarity, not exhaustive remote homology | 2GiB RAM limit; successful searches ~16s total. Initial512MiB allocation failed before search and was corrected; no missing hits recoded as homology absence. |
| ESM-2 | official35M FAIR checkpoint, pinned hash; fair-esm2.0.0 source vendored without environment installation | MIT; sequence-statistics compatibility, not energy/stability |134,095,705byte weights; FP32, eval(), no gradients; measured0.03–0.17s/16masked positions after startup; GPU allocation<1GiB. |
| Legacy ESM-2 | 150M safetensors already present in external_models/e007_phase4b_v1 | MIT; not used | Broader project inventory located these after the small-checkpoint retrieval started; main environment has no Transformers. Explicitly authorized35M evaluator was used; no second model downloaded. |
| IUPred3 | Not found in inspected project/tool paths or PATH; omitted | No disorder prediction claim | No installation |
| ESMFold | No compatible installed folding weights verified; omitted | No fold predictions inferred from ESM-2 | No large download |
| AlphaFold2 | Existing ColabFold1.6.3 / AF2.3.20 installation and model1pTM weights | Parameter notice CC BY-NC-SA4.0; historical oracle restrictions retained | Zero new calls; R1d12short calls476.633s/peak5081MiB is historical, not this task's timing |
| ProteinMPNN | Existing source and legacy CA-only weights; no verified E016 compatible scoring adopted | MIT; published full-backbone scores reused for Garcia only | No fabricated N/C/O; zero J1 ProteinMPNN inference |
| US-align | Existing binary20261012 | Existing license notices govern use | No new structural-alignment campaign |

Machine-readable paths, hashes, limits and implementation receipts: EVALUATOR_AVAILABILITY.json, ESM_BENCHMARK.json, data/ESM_IMPLEMENTATION_PROVENANCE.json. Contact-regression weights are intentionally absent: ESM token scoring never requests contact predictions. No uncalibrated contact output is used.
''')
md('EXPERIMENTAL_METRIC_DISCRIMINATION.md','# Published experimental-design discrimination\n\nMain comparable population:570designs,225successes/345failures,11studies. Success prevalence39.47%. Scores are frozen; higher pLDDT/PLL, lower PAE/MPNN. No model or combination fitted.\n\n'+exptable+'\n*Failures among the top quartile of the oriented score, including ties; not a selected deployment threshold.\n\nAUROC differs substantially by study and topology. AF2 within-study AUC ranges0.518–1.000, including a16-design study; ESM-2 ranges0.362–1.000. These are small heterogeneous groups, not guaranteed independent families. Full per-study/per-topology denominators and undefined single-class handling are in the JSON. Study-cluster uncertainty is wide; formal superiority of ESMFold over AF2 is not established by comparing means. ESM-2 overall AUC0.424 versus macro0.599 reveals strong study/composition confounding. We do not reverse its direction or fit a formula after seeing labels.\n\nAll614 sensitivity: AF2.673, ESMFold.731, MPNN.658, PAE.698, AF3.626, ESM-2.418. The44solved-NMR records raise apparent performance, supporting separate reporting of the authors’570design comparison. Predicted Probability/Success columns are supplied fitted outputs and were not used as new rewards.\n\n![Discrimination](EXPERIMENTAL_SCORE_DISCRIMINATION.png)\n')
md('E016_SEQUENCE_GENERATION_SCORECARD.md','''# E016 sequence-generation scorecard

| Dimension | Measured finding | Interpretation |
|---|---|---|
| Validity |320/320 complete, exact length, canonical sequences; no duplicates | Reliable finite generation in this fixed panel |
| Statistical plausibility | Both J1 groups score above shuffles and below natural controls in mean ESM PLL | Learned sequence order is detectable; gap to natural statistics remains |
| Model comparison | Equal-length paired ESM delta−0.0032nats,95%CI[−0.0404,+0.0356] | No defensible independent plausibility improvement for +7,500 |
| Diversity | All32sequences unique per length/model; +7,500 has31clusters at50%identity/80%both coverage atN128, otherwise32 | No gross batch collapse; small panel and heuristic clusters |
| Novelty | No exact TRAIN/VALIDATION duplicates; significant TRAIN local hits11/160 versus17/160 | Low detected similarity is not proof of remote-homology novelty or quality |
| Composition | Near natural mean entropy and hydrophobic fractions; some length-dependent charge shifts | No universal ideal composition; detailed AA deviations and tails retained |
| Complexity | Low-complexity fraction generally low; +7,500 reduces repeat/low-complexity means in several lengths | Descriptive, not a disorder or stability prediction |
| External structure | No exact frozen-panel AF sequence/protocol match; no ESMFold | Sequence foldability remains untested on this panel |
| J1 geometry | Finite completion; mixed changes in clashes/contacts/Rg | Auxiliary CA proxies, not independent folding evidence |
| Experimental relevance | ESMFold/AF2/PAE moderately discriminate published designs; ESM-2 is weak/confounded | No tested evaluator certifies biological success |
| Robustness | Length strata, seed bootstrap, full-position PLL sensitivity and study/topology analyses | Fixed16-position PLL and small groups limit strong model ranking |

'''+ESMtable+'\nMean sampled masked log probabilities, higher means greater compatibility. This is neither autoregressive perplexity nor a stability score.\n\n![ESM comparison](ESM2_SEQUENCE_COMPARISON.png)\n\n![Composition differences](COMPOSITION_DIFFERENCES.png)\n\n![Auxiliary geometry](AUXILIARY_GEOMETRY_COMPARISON.png)\n')
md('RL_REWARD_CANDIDATE_ASSESSMENT.md','''# Future reward assessment

Production sequence-quality RL: **NO-GO on current evidence**. ESM-2 separates natural order from composition shuffles but does not rank experimental design success reliably. AlphaFold confidence failed earlier calibration/stability gates and remains only a conditional structural signal. Garcia success spans expression, solubility, monomericity and secondary structure; it is not measured thermodynamic stability.

| Candidate | Mathematical signal | Cost/domain | Status and exploitation checks |
|---|---|---|---|
| Frozen ESM-2 | ell(S)=mean_i log p_ESM(s_i | S with i masked) |16fixed or fully scored positions, short canonical chains; measured cheap | Evaluation-only by default; restricted statistical-naturalness engineering reward could be tested with separate approval. Repeats/composition/memorization can exploit it; no stability claim. |
| AF2/ESMFold confidence | Independently report pLDDT, global/PAE measures, not an unvalidated product | Protocol/MSA/context dependent; ESMFold not locally available | Structural evaluator/filter only. Confidence can be high for failed designs; no new reward certification. |
| ProteinMPNN | Published mean negative log probability conditional on AF2 full backbone | Full qualified backbone required | Garcia reproduction only; J1 CA scoring unavailable under this protocol. |
| Complexity/composition | TRAIN length-stratified descriptive envelopes | Cheap | Constraints/monitoring, not a universal quality reward. Disorder/protein classes require context. |
| CA validity | Finite construction and independent clash/contact/Rg diagnostics | Cheap | Auxiliary safety; never multiply away a promising sequence's entire external score because J1 backbone disagrees. |

A prospective exploratory terminal reward could be r_stat=clip((ell(S)−mu_TRAIN,N)/sigma_TRAIN,N,−2,0): it rewards reaching natural-statistics compatibility and stops paying above the natural mean. Fit mu/sigma only on preserved TRAIN controls; score allpositions for short64–128residues or random stratified positions from a secret independent stream at training time. Fixed public positions alone invite exploitation. Invalid sequences/construction are explicit failures; oracle missing/pending states are ineligible, not biological zero rewards. This is not a certified protein-quality/stability reward.

Essential controls: frozen J1 reference KL; PDB40/30/30 AA BC; qualified geometry BC separately; composition/complexity envelopes; duplication/identity monitoring; held-out lengths/seeds; budget-matched generation-and-selection baseline; stop on diversity collapse, reference KL excess, worsening independent geometry or held-out AA prediction. Do not optimize novelty without a bound. Do not infer folding from low homology.

Independent evaluation must retain scores not optimized: full versus sampled PLL on sealed positions, sequence novelty/diversity/composition, qualified CA geometry, native/geometry-free held-out AA NLL and separately budgeted blind external-fold/experimental controls. An ESM reward gain alone is not a success claim. A larger prospectively separated experimental study would be needed to establish biological relevance; no fitted neural reward model is proposed.
''')
md('NEXT_E016_RL_RESEARCH_DECISION.md','''# Next E016 decision

**NO-GO for production biological-quality RL.** +7,500 remains the user's RL initializer without retroactive promotion. R1e adds independent evidence of natural-like sequence order, not evidence that it generates biologically better proteins than +500. No additional supervised run is recommended automatically.

If the next authorized task is an engineering-only RL test, prefer KL-regularized episode-level policy gradient before SAC:64–128residue episodes, frozen ESM statistical-naturalness terminal reward capped at the TRAIN mean, leave-one-out return baseline within length, frozen reference KL and PDB BC. Suggested bounded study:2seeds,32completed training episodes/seed,32held-out evaluation episodes/seed, at most100actor minibatch updates total; zero automatic AF/MSA queries. Freeze reward normalization/envelopes using TRAIN, keep all controls unchanged. Shared LR1e−6, AA-head5e−6, geometry-head frozen initially; target categorical KL<=0.02nats/decision and monitor geometry KL<=0.01nats/valid component, calibrating BC/gradient scales on TRAIN before any policy update. These numbers are proposed, not validated or executed. Require bitwise restart and all real action-density tests again.

Primary optimization claim would be improved pretrained-language statistical compatibility only. Evaluation must show gains beyond equal-budget generation-and-selection, retain diversity, and avoid degradation of independent AA/geometry metrics. More confident folds or experimental success would require separately approved evidence. Stop if reward rises while these independent checks deteriorate.

Hybrid SAC remains an alternative for expensive delayed rewards because replay can improve sample efficiency, but its real-protein-return critics remain uncalibrated. Synthetic full-trajectory Transformer results from R1c stay preserved. Episode PG avoids critic overestimation and is the simpler first control; SAC could be compared later under identical actor/oracle budgets. No RL system rewrite, production policy updates or critic optimization occurred in R1e.
''')
# Resources keep failed fixtures/download attempts explicit rather than hiding them.
sensitivity=read('ESM_SAMPLING_SENSITIVITY.json') if (R/'ESM_SAMPLING_SENSITIVITY.json').exists() else None
cost=esm['seconds']+(sensitivity['seconds']if sensitivity else 0)+97.1
resources=dict(D_free_before_bytes=12545732608,D_free_after_bytes=shutil.disk_usage(R).free,reserve_bytes=10*2**30,new_pretrained_weight_bytes=134095705,new_weight_cap_bytes=250*2**20,primary_ESM_seconds=esm['seconds'],sensitivity_seconds=sensitivity['seconds']if sensitivity else None,failed_sensitivity_fixture_seconds=97.1,total_charged_ESM_inference_seconds=cost,evaluator_cap_seconds=2700,peak_GPU_allocated_bytes=max(esm['allocated_peak_bytes'],sensitivity['allocated_peak_bytes']if sensitivity else 0),new_J1_generations=0,cached_J1_attempts=320,new_AF_calls=0,new_external_MSA_queries=0,optimizer_updates_all_models=0,new_environment_installs=0,historical_deletions=0,search_seconds=sum(r['seconds']for r in nov['runs']),network_receipts='data/DOWNLOAD_RECEIPTS.json; data/ESM_IMPLEMENTATION_PROVENANCE.json; data/NETWORK_ATTEMPT_AUDIT.json',failed_attempts=['publisher supplement403 and PMCchallenge; EuropePMC succeeded','GitHub source archive8MiB read cap reached, discarded memory only; official93125byte wheel used','MMseqs512MiB allocation failed;2GiB RAM search passed','BOS index error in standalone hidden-token test after full inference; primary score unaffected, corrected test repeated with explicit differing token index'],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip())
write('RESOURCE_AND_STORAGE_REPORT.json',resources)
md('EXPERIMENT_DECISIONS.md','''# R1e decisions

1. Freeze two immutable production checkpoints, verify SHA and bitwise equivalence of accepted full/model-only parent; reuse320correctedFP64 cached attempts with matched saved draws. Preserve historical qualification.
2. Select natural TRAIN/VALIDATION exact requested lengths by deterministic SHA order; keep shortfalls, duplicates and failures visible. Use composition-preserving TRAIN shuffles as statistical controls, not known unfolded proteins. No TEST.
3. Use one expressly authorized small ESM-2 evaluator, eval mode/no gradients. Freeze16stratified single-mask positions; compare a predetermined full-position subset. Record sampling error; no unmasked-logit shortcut. Broader inventory found legacy150M weights after small retrieval; no second evaluator downloaded.
4. Reproduce614official designs and570author-exclusion subset; preserve raw labels/missingness. Only individual frozen scores; no fitted regression, neural reward or posthoc reversed ESM score. Bootstrap by study; report topology/study heterogeneity.
5. Keep sequence plausibility, novelty, diversity, experimental discrimination and auxiliary geometry separate. No certified independent fold exists for this panel. No ProteinMPNN score on fabricated full atoms.
6. Production RL remains NO-GO; only a separately approved, capped statistical-naturalness PG engineering study is proposed. No J1/critic optimizer updates, no folding/MSA campaign, no historical artifact deletion.
''')
md('CHANGELOG.md','''# R1e changelog

2026-10-20: added isolated inference-only frozen-panel preparation, sequence analysis, bounded MMseqs novelty/clustering, official Garcia reproduction, frozen ESM-2 masked PLL and sensitivity checks, cached structural analysis, plots, scorecard and provenance. Recorded failed resource/download/test attempts and corrected fixture without changing the primary scoring rule. Historical source/evidence and model weights preserved.
''')
# Main report explicitly answers the user's20scientific questions.
fulltext='Not complete' if sensitivity is None else '; '.join(f"N{N}: bias {v['mean']:+.3f}nats"for N,v in sensitivity['summary'].items())
md('FINAL_E016_PROTEIN_GENERATION_EVALUATION_REPORT.md',f'''# E016 R1e inference-only evaluation

**J1+7,500 does not show a clear ESM-2 plausibility advantage over+500. Production protein-quality RL remains NO-GO.** Both panels have detectable sequence order beyond composition shuffles, but remain below natural proteins. No metric here certifies folding, stability or experimental design success.

## Frozen populations and correctness

320cached J1 attempts,160/model,32/length at64/128/256/384/500; all320complete and canonical, zero duplicates. Parent full-state SHA and cached model-only tensors verified bitwise equal. Matched draws and corrected ca_fp64_interior_v1 protocol retained. No new J1 forward generation, training, TEST evaluation, folding or external MSA submissions.

Natural TRAIN160, composition-preserving shuffles160, natural VALIDATION117; exact full lengths without truncation, SHA-ordered selection. VALIDATION per-length counts: {counts['natural_VALIDATION']}. Natural controls are not experimentally labeled foldability negatives/positives for every possible outcome. Full manifests: [panel](FROZEN_J1_GENERATION_PANEL.json), [correctness](EVALUATION_CORRECTNESS_TESTS.json).

## Sequence-centric evidence

{ESMtable}

Equal-length paired mean difference {tests['overall_paired_ESM_delta']['mean']:+.4f}nats/residue,95%CI[{tests['overall_paired_ESM_delta']['bootstrap95'][0]:+.4f},{tests['overall_paired_ESM_delta']['bootstrap95'][1]:+.4f}]. Every length-specific interval spans zero. This does not establish equivalence; the panel/metric may miss real differences. Higher held-out J1 AA likelihood does not guarantee better independent generation scores.

ESM-2 uses16one-at-a-time masked true residues; surrounding sequence remains visible. It is not an autoregressive likelihood or free energy. Full-position approximation sensitivity on first2sequences/group/length: {fulltext}. These checks diagnose fixed-position error without replacing the frozen main metric or selecting favorable sequences. Bootstrap intervals condition on sampled positions; use them conservatively for small checkpoint differences. Official ESM contact-regression weights were not needed or used.

![Sequence plausibility](ESM2_SEQUENCE_COMPARISON.png)

## Experimental relevance

OfficialDataS1:614unique sequences,11studies,53topologies,269successes. The authors’ main subset excludes44pdb=True solved-NMR records, leaving570/225successes. NMR=True alone is not the exclusion rule. Missing component measurements remain missing. Experimental success combines expression, solubility, monomericity and intended secondary structure; no thermodynamic-label substitution.

{exptable}

*Top-quartile oriented-score failures, with ties. High confidence does occur among experimental failures. Study/topology heterogeneity is substantial; overall versus macro differences are not ignored. All614sensitivity, per-study/per-topology results, PR curves/prevalence, effect sizes and failure examples are in [benchmark](EXPERIMENTAL_DESIGN_SUCCESS_BENCHMARK.json). No supplied fitted Predicted Probability or newly fitted reward model was used. This is a reproduction, not a fresh independent biological validation.

![Experimental discrimination](EXPERIMENTAL_SCORE_DISCRIMINATION.png)

## Auxiliary structure and novelty

No exact TRAIN/VALIDATION duplicates in either generated panel. Significant bounded TRAIN local hits:11/160for+7,500,17/160for+500; identity, query/reference coverage, coordinates and E-values are retained individually. No-hit is not proof of remote-homology novelty. All32sequences/model/length are unique;+7,500 has31connected components at50%identity/80%both coverage atN128,otherwise32. Effective exact uniqueness does not imply evolutionary/fold diversity.

Both panels have generally natural-like gross composition and entropy. +7,500 has fewer low-complexity windows/repeated triplets in several lengths, but length-dependent charge/composition shifts remain. Low complexity and disorder are not synonymous with experimental instability. IUPred3 was not available in the inspected installation.

Geometry changes are mixed. +7,500 atN384has meanRg43.28A versus50.58A, paired difference−7.30A95%CI[−12.66,−2.35]; N256contacts increase0.00187[0.00013,0.00369]. Clash differences span zero at all lengths. SmallerRg is not automatically better. Multiple metric/length comparisons are exploratory, without multiplicity correction. Native geometry controls use preserved +/-10%length windows, not the exact-length sequence controls. There is no certified exact-sequence AF cache match for this frozen panel; no independent full-fold qualification is claimed.

![Auxiliary geometry](AUXILIARY_GEOMETRY_COMPARISON.png)

## Direct answers and next decision

1. Official benchmarks/papers consulted; Garcia XML/CSV retrieved and hashed;614design/11study counts verified. Focused ProteinBench adaptation, not full leaderboard reproduction.
2. Available tools: MMseqs2, frozen35M ESM-2, existing AF2/US-align and ProteinMPNN source; IUPred3/compatible ESMFold omitted. Standard J1 ProteinMPNN scoring unavailable under verified full-backbone semantics.
3. Primary ESM scoring1371sequences took{esm['seconds']:.2f}s;16masked positions per sequence, main GPU allocation{esm['allocated_peak_bytes']/2**20:.1f}MiB. Full sensitivity and one failed fixture run are separately charged; total{cost:.1f}s below45minutes. MMseqs successful searches~16s.
4–5. Gross composition is broadly plausible; both J1 means beat composition-shuffled ESM means at all lengths but trail natural controls. This is statistical compatibility, not biological proof.
6. Frozen ESM does not clearly favor+7,500; pooled paired interval and all length intervals include zero.
7–8. Exact known-sequence duplication is absent and detected local matches are limited. Neither greater novelty nor a diversity/plausibility tradeoff is conclusively established by the difference11versus17hits.
9. No gross repeat/low-complexity collapse; composition/charge tails remain in [composition](SEQUENCE_COMPOSITION_AND_COMPLEXITY.json). Do not penalize unusual valid protein classes automatically.
10–12. ESMFold pLDDT has best overall point estimate among these published scores; AF2/PAE/MPNN are moderate, vary across studies, and all have high-score failures. ESM-2 overall0.424/study-macro0.599 is strongly context-confounded, not a reliable success predictor.
13–14. No new independent folds qualify these exact sequences; J1 geometry contributes only auxiliary validity/compactness/contact/diversity evidence and never dominates the sequence conclusion.
15. +7,500 retains its historical held-out AA improvement and user-selected RL role, but R1e does not establish a biologically superior generator versus+500.
16–18. Keep ESM/AF/heuristics as separate evaluations; no primary biological reward is certified. A separately approved capped ESM statistical-naturalness reward could test optimization mechanics only, with frozen-reference KL, PDBBC, diversity/novelty/composition monitoring, geometry safety and independent evaluation.
19–20. Smallest useful alternative: KL-regularized episode-level PG for a plainly labeled sequence-statistics objective, with a matched generation-and-selection baseline. [Executable-scope recommendation](NEXT_E016_RL_RESEARCH_DECISION.md) proposes64–128residues,2seeds,64training episodes total and<=100actor updates, but executes none. SAC remains available with passed synthetic critics; real-return calibration is unverified. No production biological-quality RL until a relevant external objective passes prospective controls.

## Integrity and resources

Main checkpoint SHA881457ff0396975c1b20dd0a8e2caac8521e65b7e1cec76a7bb531dbc437b412; parentSHA9208263300fb9e88519ff0d40e6d9d29a9744b709d9a6181819cd0894b593000. Original files unchanged. No checkpoint copies created.

New pretrained weights134,095,705bytes; official fair-esm wheel93,125bytes; no environment expansion. Initial512MiB search allocation failed and was rerun at2GiB RAM. A test-only BOS-offset assertion failed after sensitivity inference; corrected fixture/repeated cost is explicitly recorded and primary scores were unaffected. Publisher retrieval failures and source-archive read cap are retained in network provenance.

D: free{resources['D_free_after_bytes']/2**30:.3f}GiB versus10GiB reserve. All critical scripts/evidence live onD. Source commit and verification hashes are in [provenance](PROVENANCE_SHA256.json). Workers stop after publication. No training/RL is authorized by these findings.

Detailed [scorecard](E016_SEQUENCE_GENERATION_SCORECARD.md), [evaluator audit](EVALUATOR_INSTALLATION_AND_RESOURCE_AUDIT.md), [experimental analysis](EXPERIMENTAL_METRIC_DISCRIMINATION.md), [reward assessment](RL_REWARD_CANDIDATE_ASSESSMENT.md), and [resources](RESOURCE_AND_STORAGE_REPORT.json).
''')
print('reports/plots published',flush=True)
