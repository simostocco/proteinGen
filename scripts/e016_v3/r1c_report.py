"""Publish R1c scientific reports from measured receipts; no training or network."""
from r1c_common import *
import numpy as np,subprocess,re
_raw_write=write
def write(p,x):
 if isinstance(x,str) and p.endswith('.md'):
  # Presentation only: separate explicitly listed words/units from adjacent numbers.
  parts=re.split(r'(`[^`]*`|https?://[^\s)]+)',x)
  for i in range(0,len(parts),2):
   v=parts[i]
   v=re.sub(r'\b(Completed|all|All|of|in|by|for|only|below|under|above|after|before|both|Both|exactly|selected|the|The|uses|use|with|and|Another|Frozen|reaches|despite|including|on|not|rows|from|gain|gains|case|TRAIN|IID|OOD|amyloid|fragment|residues|errors|error|limit|fraction|MAE|RMSE)(?=[−-]?\d)',r'\1 ',v)
   v=re.sub(r'(?<=\d)(?=(MiB|GiB|ms\b|s\b|nats\b))',' ',v)
   v=v.replace('Neff80','Neff80').replace('gains are0','gains are 0').replace('remains0','remains 0').replace('beta0','beta 0')
   parts[i]=v
  x=''.join(parts)
 _raw_write(p,x)

def main():
 h=json.load(open(R/'HISTORICAL_PER_CONTROL_DIAGNOSIS.json'));d=json.load(open(R/'CONFIDENCE_VS_NATIVE_RECOVERY.json'));fold=json.load(open(R/'MSA_DEPTH_ABLATION.json'));critic=json.load(open(R/'FULL_TRAJECTORY_CRITIC_COMPARISON.json'));rows=d['depth_rows'];source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip()
 assert len(fold['rows'])==12 and all(r['status']=='SUCCEEDED' for r in fold['rows'])
 assert json.load(open(R/'critic/CPU_PREFLIGHT.json'))['full_episode_file_sha256']==sha(R/'critic/FULL_EPISODES.json');assert len(critic['rows'])==2 and all(v['restart']['model_max_abs_difference']==0 for v in critic['rows'])
 parent=R.parent/'e016_j1_bidirectional_multitask_20261019_v1/runs/J1/checkpoint_07500.pt';assert sha(parent)=='881457ff0396975c1b20dd0a8e2caac8521e65b7e1cec76a7bb531dbc437b412'
 free=guard();start=json.load(open(R/'RESOURCE_START.json'));seconds=sum(v['worker_wall_seconds'] for v in fold['rows']);assert seconds<1500
 peak=max(float(v.split(',')[1]) for r in fold['rows'] for v in r['GPU_samples']);cseconds=sum(v['seconds'] for v in critic['rows'])
 # Verify the historical evidence manifest rather than merely assuming old reports were unchanged.
 old=json.load(open(B/'PROVENANCE_SHA256.json'));bad=[]
 for p,z in old['evidence_files'].items():
  if not (B/p).exists() or sha(B/p)!=z:bad.append(p)
 for p,z in old['source_files'].items():
  if sha(B/'source'/p)!=z:bad.append('source/'+p)
 assert not bad,bad
 resources={'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'source_commit':source,'source_branch':'e016-r1c-reward-reliability-20261019-v1','initial_free_bytes':start['free_bytes'],'final_free_bytes':free,'reserve_bytes':10*2**30,'new_AF_calls':len(fold['rows']),'successful_folds':sum(v['status']=='SUCCEEDED' for v in fold['rows']),'folding_worker_seconds':seconds,'max_charged_seconds':1500,'ledger':fold['budget'],'AF_sampled_peak_GPU_MiB':peak,'critic_seconds':cseconds,'critic_GPU_peak_allocated_bytes':max(v['GPU_peak_allocated_bytes'] for v in critic['rows']),'remote_queries':0,'downloads':0,'data_deletions':0,'J1_optimizer_updates':0,'production_RL_updates':0,'J1_checkpoint':str(parent),'J1_SHA256':sha(parent),'R1b_manifest_sha256':sha(B/'PROVENANCE_SHA256.json'),'R1b_evidence_files_verified':len(old['evidence_files']),'R1b_source_files_verified':len(old['source_files']),'R1b_mismatches':bad,'worker_recovery':'initial postprocessing directory error repaired from saved successful prediction; no repeat folding','preflight_fixes':'synthetic builder minimum length20 and zero-valued undefined bootstrap components corrected before any critic optimizer step','protocol':'model1 pTM,3 recycles,seed151502,no templates,128:256 caps,CPU preprocessing+GPU folding,network denied','network_enforcement':'each AF worker replaces socket.connect with an exception and disables parameter download; R1c contains no network client'}
 write('RESOURCE_AND_PROVENANCE.json',resources)
 histtable='| Control | Split/role | N | Native recovery | Scalar | Residual (pred−actual) | MSA rows |\n|---|---|---:|---:|---:|---:|---:|\n'
 for r in h['rows']:histtable+=f"| {r['id']} | {r['role']} | {r['N']} | {r['measured_recovery']:.3f} | {r['predicted_recovery']:.3f} | {r['signed_residual']:+.3f} | {r['MSA_metrics']['raw_depth']} |\n"
 write('REWARD_CALIBRATION_FAILURE_ANALYSIS.md',f'''# Historical confidence calibration failure

The R1b result is unchanged: TRAIN holdout MAE {h['historical_TRAIN_heldout_MAE']:.6f}, independent VALIDATION MAE {h['historical_VALIDATION_MAE']:.6f} against the frozen 0.20 limit. Both VALIDATION cases fail individually: 3gzf is underpredicted by 0.244402 and 7aoi by 0.472103. The latter contributes 65.89% of total absolute error; removing it for sensitivity leaves MAE 0.244402, still above the historical limit. No control is removed or rescored with a newly fitted formula.

{histtable}

Historical formula: clip(1.0856201660148699 × mean pLDDT/100 × pTM × long-range PAE-good fraction, 0, 1). Separation is min(24,max(4,N//4)); PAE-good means <10 Å. At 7aoi the PAE factor is 0.180572, yielding 0.064073 despite observed mapped recovery 0.536176. At 3gzf the factors yield 0.650959 despite recovery 0.895361. This is confidence-product underprediction, not an execution failure. Related global confidence factors may double-count uncertainty; the data do not isolate that causal mechanism. The two high-recovery TRAIN holdouts (0.975/0.983) do not test broad calibration: n=2, narrow target range, no context-disagreement challenge.

TRAIN amyloid6gk3 is the opposite problem: 95.31 pLDDT, 0.77 pTM, PAE-good fraction0.9885 and scalar0.7876 coexist with mapped recovery0.1422. It remains in all analyses. Comparison to an assembled fibril is not a direct label of monomer sequence quality or thermodynamic instability. A reward intended to identify isolated globular structures needs an explicitly scoped, independently validated reliability domain.

`HISTORICAL_PER_CONTROL_DIAGNOSIS.json` contains every control's exact canonical sequence, PDB/entity/author-label mapping and context, observed positions, coordinate-source reference, exact predicted PDB path/hash, full scores, MSA metrics and residual. Original native mappings and independent alignment evidence are retained by reference to R1b. No original mapping error was found there. No new claim of a corrected correspondence is made here.

The four prospectively frozen signals are examined only on TRAIN: historical product, raw pTM, equal additive local/global confidence, and their minimum. Their identity-level bootstrap intervals and the original fit/holdout ownership are in `CONFIDENCE_VS_NATIVE_RECOVERY.json`. No new coefficient was fitted and no revised candidate was selected on VALIDATION. Better TRAIN MAE does not certify an RL reward, and all four still overvalue the context-disagreement case relative to fibril recovery. RL needs useful ranking as well as calibration; neither tiny holdout MAE nor a confidence correlation alone establishes that outside these controls.
''')
 write('BIOLOGICAL_CONTEXT_AND_AMYLOID_CASE.md','''# High-confidence amyloid fragment

The public PDB6gk3 control is the deposited64-residue beta-2-microglobulin amyloid fragment, author residues22–85, not the full natural protein. The experimental octamer comprises two protofilaments. R1b verified its canonical construct sequence and residue correspondence. The local AF2 calculation predicts an isolated chain without partner chains/templates; it is not an assembled-fibril calculation.

The full alignment predicts a compact, high-confidence chain but does not reproduce the deposited extended fibril conformation. This is a structural-state/context disagreement, not proof that the natural sequence is poor, that AF's monomer is experimentally stable, or that the experimental coordinates are wrong. Neither this case nor VALIDATION's assembly-dependent ribosomal fragment is silently excluded.

With the fixed R1c protocol, adding evolutionary information raises confidence without recovering this experimental conformation. The depth table and exact predictions are retained. Deep alignment, good coverage, high pTM, high regional pLDDT and favorable PAE do not identify this disagreement. A gate using the unknown native fold would be unavailable to a generated candidate and is prohibited.

A known public annotation may identify assembly-dependent control scope prospectively; generated sequences usually lack such annotations. Composition/complexity checks do not establish an equivalent inference-time detector. Future reliability certification therefore needs a predefined mix of monomeric, assembly-dependent, fragment and conformational-change contexts and must report each stratum rather than discard inconvenient outcomes. No experimentally proven nonfolding label is assigned to this or to shuffled controls.
''')
 table='| TRAIN control | MSA rows | Input Neff80* | Recovery | pLDDT | pTM | PAE-good fraction |\n|---|---:|---:|---:|---:|---:|---:|\n'
 for r in rows:table+=f"| {r['control']} | {r['nominal_rows']} | {r['Neff80']:.1f} | {r['native_recovery']:.3f} | {r['features']['mean_pLDDT']:.1f} | {r['features']['ptm']:.2f} | {r['features']['long_range_PAE_good_fraction']:.3f} |\n"
 write('MSA_DEPTH_SUMMARY.md',f'''# Frozen depth stress test

{table}

*Neff80 is the sum of reciprocal80%-identity neighbor counts, non-gap-union identity, calculated on at most512 deterministic eligible unique input rows. It is not a full-MSA extrapolation or a measurement of the exact retained internal AF cluster features. All source/variant rows, query coverage/gap metrics, seeds, hashes and resolved inference settings are in `configs/DEPTH_STUDY_FROZEN.json` and `MSA_DEPTH_ABLATION.json`.

Three TRAIN identities × four levels were frozen before new inference. Rows4 and32 are nested distinct homolog sets with at least50% coverage, shuffled with seed2026101903 independent of recovery. Query-only and full MSAs are also new calls, not assumed identical caches. Same model1pTM, seed151502, three recycles, no templates,128:256 caps and corrected CPU-preprocessing/GPU-folding wrapper; no remote queries. All12 calls succeeded in {seconds:.2f} charged worker seconds, sampled peak {peak:.0f} MiB. The first prediction's output-directory postprocessing error was repaired from its saved structure, with original error evidence retained and zero repeat calls.

The128-residue domain and447-residue historical hard case improve strongly and their confidence signals broadly track recovery. Four rows already improve both to about0.70,32 rows to0.90/0.95; this does not establish a universal minimum alignment depth. The amyloid fragment's full-versus-query recovery changes by only−0.0013 while pTM increases0.29 and pLDDT30.55. At32 rows it has high confidence but mapped recovery0.177. Simple depth/confidence eligibility therefore retains known disagreement. Full repeats reproduce all three R1b mapped scores exactly; all settings and original failures remain versioned separately.

There is one homolog subsample/seed per depth, three independent identities and no VALIDATION folds. Depth rows are repeated measures, not12 independent proteins. No broad population confidence interval or reliable length/context interaction can be inferred. Natural-sequence thinning leaves the query's learned/family priors intact and does not simulate a novel sequence distribution.
''')
 write('NOVEL_SEQUENCE_RELIABILITY_ASSESSMENT.md','''# Supported and unsupported transfer

| Candidate situation | Evidence | Interpretation |
|---|---|---|
| Natural protein with deep MSA | R1b eight TRAIN/two VALIDATION plus R1c three repeats | Excellent recovery often, but biological-context and confidence-calibration failures remain |
| Natural protein with shallow MSA | Three TRAIN identities at1/4/32/full | Removal can substantially impair recovery; confidence warns in two examples but is not universally reliable |
| Novel generated sequence with a few homologs | No generated sequence tested or submitted | No reliability certificate; homolog identity, coverage and actual evolutionary support must be verified |
| Generated sequence without useful homologs | No generated-sequence study; natural query-only analogs only | Reward NO-GO; natural sequence thinning is an incomplete proxy |

Precomputed AFDB predictions belong to their own exact sequences/providers/protocols. A similar database entry is neither a fresh prediction of a generated sequence nor evidence that its oracle is calibrated. AFDB and local confidence are not pooled here. Existing AFDB/native results remain descriptive external context.

No newly proposed reward is fitted on or selected using the previously inspected VALIDATION examples. No new external calls, generated-sequence submission or J1 update occurred. Better native confidence/recovery does not establish folding uniqueness, solubility, thermodynamic stability or function. Novelty alone is not quality.
''')
 write('PROPOSED_SEQUENCE_REWARD_AND_ABSTENTION_POLICY.md','''# Proposed reliability and abstention policy — not production enabled

Keep completed-sequence assessment primary. Candidate signals, reported separately, are pTM, full-chain mean/q10 pLDDT, fraction confident, long-range PAE and confidence concentration, input effective MSA depth/coverage, and sequence complexity/composition. Do not multiply sequence reward by J1–AF agreement. J1 geometry receives a separate bounded auxiliary assessment; invalid construction remains an engineering failure.

R1c froze four interpretable diagnostic scores before new folds. No new regression was fitted. The exploratory eligibility hypothesis was Neff80≥8, mean input coverage≥0.7, pTM≥0.6 and pLDDTq10≥50. It is explicitly rejected as a sufficient production domain: it admits deep/32-row amyloid predictions with low experimental-state recovery. No threshold is changed to remove that case. Confidence-factor multiplication can penalize uncertain but substantially recovered structures; simply replacing it with pTM or an additive signal does not resolve context ambiguity.

Production eligibility is currently NONE. A valid technical result can be assessed descriptively without being a calibrated reward. Missing pTM/PAE, incomplete mapping, failed/nonfinite predictions, unsupported length/depth/context or suspected extrapolation result in UNSCORED/UNCERTAIN, not biological reward0. PENDING/RUNNING remain pending; timeout and service failure are technical states. Do not optimize an actor against this incomplete domain.

For a future certified domain, assess reliability entirely from available sequence, predicted structure/confidence and alignment information; never from unknown native recovery. Use a fixed attempted-episode budget and immutable queue identities, retain all abstentions/failures, compare eligibility rates with a frozen reference, and require a separate eligibility-rate safeguard. Do not rank only scored survivors or renormalize rewards to favor oracle avoidance. An unsupported candidate cannot receive a bonus by evading evaluation; if eligibility changes materially, stop optimization and audit the distribution. Eligible reward and retention/KL objectives would be separately bounded after certification, not assigned now.

Next prospective panel is in `configs/NEXT_CONTROL_PANEL_PROPOSED.json`:12 unrun TRAIN controls (8 fit/4 heldout by canonical SHA) and4 disjoint previously unrun local VALIDATION controls, selected deterministically from the original32-control panel before new local results. All have existing AFDB/mapping history, which remains disclosed; they are new local-protocol controls, not completely unseen biological information. Prospectively audit assembly/context strata. Request at most16 serial public-native MSA queries and24 local folds (16 full plus query/4-row pairs on four TRAIN controls), hard45-minute aggregate worker cap, at most250MiB incremental artifacts. None is executed now. Prior VALIDATION cases stay diagnostic, never become fitting data. A larger independent panel and a separately authorized generated-sequence study would still be needed for novel-sequence claims. Inspect confidence/accuracy ranking, calibration error and abstention coverage by depth/context; retain historical thresholds rather than lowering them to pass.
''')
 # Freeze proposed new controls, not predictions or network jobs.
 panel=json.load(open(B/'configs/NATIVE_PANEL_32.json'))['controls'];used={r['id'] for r in h['rows']};new={s:sorted([c for c in panel if c['split']==s and c['id'] not in used],key=lambda c:c['sequence_id'])[:n] for s,n in [('TRAIN',12),('VALIDATION',4)]}
 write('configs/NEXT_CONTROL_PANEL_PROPOSED.json',{'status':'PROPOSAL_ONLY_NOT_EXECUTED','controls':[{k:c[k] for k in ['id','split','sequence_id','entry_id','N']} for s in ['TRAIN','VALIDATION'] for c in new[s]],'TRAIN_fit':[c['id'] for c in new['TRAIN'][:8]],'TRAIN_holdout':[c['id'] for c in new['TRAIN'][8:]],'VALIDATION_new_local_audit':[c['id'] for c in new['VALIDATION']],'selection':'canonical SHA order among original32 panel excluding10 prior local controls; existing AFDB outcomes disclosed','additional_budget_requires_approval':{'MSA_queries':16,'local_calls':24,'aggregate_worker_seconds':2700,'artifact_bytes_max':250*2**20}})
 ct='| Critic | IID Q1/Q2 RMSE | OOD Q1/Q2 RMSE | Seconds | Both twins below0.100? |\n|---|---|---|---:|---|\n'
 for c in critic['rows']:ct+=f"| {c['arm']} | {' / '.join(f'{v:.4f}' for v in c['metrics']['IID']['RMSE_expected_return'])} | {' / '.join(f'{v:.4f}' for v in c['metrics']['OOD']['RMSE_expected_return'])} | {c['seconds']:.2f} | {c['passes_ID_and_OOD_both_twins']} |\n"
 write('SAC_VS_POLICY_GRADIENT_DECISION.md',f'''# Critic decision

{ct}

One matched800-update, batch32, AdamW0.002 configuration per arm, independent twins, no search or J1 model load. Complete synthetic episodes have TRAIN lengths20/24/32/48 and held-out OOD lengths64/96 with broader behavioral geometry actions. TRAIN64, IID16 and OOD16 episodes are disjoint; eight transitions per episode include early/bootstrap and terminal decisions. Both arms receive the same causal candidate-action and prefix-summary features; Transformer additionally encodes full current/candidate prefixes. This is a generous sufficient-statistic baseline, not a claim that protein returns have such summaries.

Returns are in[0,1], gamma1, terminal-only. Persistent dependence on early AA/geometry choices and length-normalized local scores creates temporal credit assignment. Critic targets are analytic conditional expected terminal returns checked against128 actual Monte Carlo continuations per state-action; IID/OOD errors against the MC mean and individual sampled returns, noise floor, calibration bins, action/progress strata and twin disagreement are saved separately. OOD prefixes are off-policy; all continuation values use the same declared reference policy. No future suffix/realized terminal reward is an input. Natural terminal episode returns and immutable action/probability traces are retained.

Correct categorical probabilities and transformed continuous densities/Jacobians/masks are independently checked. Replay coordinates are bitwise equal. Actual GPU interruption/restart reproduces each critic's model, AdamW state, target EMA and RNG exactly. Initial CPU checks caught an invalid proposed length12 and nonzero undefined bootstrap components; the fixtures were corrected to the unchanged existing contracts before any critic optimizer steps. No production actor/geometry convention changed.

The Transformer twins pass this declared full-episode regression gate; the MLP twins fail the OOD gate. These results do not rewrite the earlier fixed-prefix scores. The Transformer now has28,937 parameters per Q and the MLP22,913; both see additional causal sufficient-statistic summaries, so this is not an unchanged-input rerun of R1b. MC-mean versus analytic-target RMSE is0.002247; action log-density inverse error is1.07e-14 and replay coordinate error is0. This experiment is Monte Carlo value regression, not a full bootstrapped SAC learning success. The frozen RMSE limit remains0.100; even a pass would not certify critics on real AF returns, generated protein states or distribution drift. Constant-return and exact analytic baselines are reported to expose easy target variance. Recommended first algorithm, after independent reward certification and separate authorization: episode-level sequence-policy gradient with a return/length baseline, strong frozen-reference categorical KL, PDB behavioral cloning and constrained geometry. Use actual completed eligible returns, not critic-predicted quality. Do not backpropagate through integer AA samples; use score-function log probabilities. Report separate geometry density/KL and constrain shared drift.

Do not run SAC or policy-gradient optimization now. With delayed expensive rewards and no certified sequence reward, a simpler Monte Carlo baseline removes one unvalidated dependency. Any later SAC proposal must show real held-out return calibration and stability under matched query budgets in addition to these synthetic checks.
''')
 write('NEXT_RL_GO_NO_GO.md','''# E016 R1c readiness

| Gate | Decision |
|---|---|
| Local AlphaFold execution | GO for bounded inference under the sealed protocol |
| Native structural recovery | Strong on the two non-amyloid depth controls; earlier aggregate native gate remains passed |
| Confidence/reward calibration | NO-GO; historical independent MAE failure unchanged, new simple signals uncertified |
| Shallow-MSA reliability | Scope limited; confidence tracks two examples, does not resolve context disagreement |
| New J1 sequence transfer | Unverified; no generated sequence was submitted or tested |
| SAC critics | Transformer twins PASS the new synthetic IID/OOD regression gate; MLP twins FAIL OOD; real protein-return calibration remains unestablished |
| Production RL | NO-GO |

Next action is a separately approved prospective reliability-control study, not J1 updates. If reward reliability is established, start with a bounded KL-regularized episode-level policy-gradient comparison against equal-budget generation-and-selection. No RL optimizer or oracle campaign is authorized by this report. Workers stop after evidence publication.
''')
 write('EXPERIMENT_DECISIONS.md',f'''# R1c decisions

Historical R1b source `{old['source_commit']}` and all prior failures remain immutable. Current source `{source}`. No J1, supervised or production RL updates; no TEST scoring, new remote queries or deletion.

1. **Calibration diagnosis:** retain the exact TRAIN-fitted product and its failed VALIDATION MAE. Examine individual residuals before proposing signals. Both audit cases underpredict recovery; amyloid overpredicts agreement to an assembled state. Interpretation distinguishes sequence quality from isolated-chain/native-state recovery. Evidence: per-control diagnosis and unchanged R1b hashes.
2. **Depth intervention:** predeclare three TRAIN identities, nested1/4/32/full MSAs, seed151502, fixed128:256 caps/model1/three recycles/no templates. Reuse A3Ms, never remote searches. Twelve new calls avoid unproven cache equivalence. Full scores reproduce R1b; depth response strong in two cases and misleading confidence in the amyloid. Source config hash `{sha(R/'configs/DEPTH_STUDY_FROZEN.json')}`.
3. **Reliability hypotheses:** four fixed interpretable scores and an exploratory input-available eligibility rule sealed before depth folds. No new fitting on VALIDATION. Rule fails to exclude context-disagreement; retain NO-GO, propose an independent next panel rather than tune thresholds against it.
4. **Critics:** one full-episode comparison with matched data, updates and optimizer, exact targets verified against actual MC branches. Never load or update J1. CPU preflight corrected fixture-contract mistakes; GPU exact restart verified. Compare against frozen0.100, report all arms/OOD failures, no repeated architecture search. Recommendation remains a simpler episode-return policy gradient after reward certification.
5. **Recovery:** first AF call succeeded but comparison output directory was absent. Saved original error and repaired scoring from existing prediction; no folding retry. All12 calls count, final status12 succeeded. Publication uses atomic files and preserves10 GiB reserve.
6. **Stop:** no automatic further fold, external query, critic sweep or actor optimization. Proposed future panel/network budget is documentation only.
''')
 write('CHANGELOG.md',f'''# R1c changes

- `{source}`: local-only frozen MSA-depth pipeline, historical per-control residual analysis, complete synthetic episode/value-regression benchmark with deterministic GPU restart.
- Earlier source2542602 added bounded scripts;995c0ba fixed synthetic builder minimum length and zero-for-undefined components before critic training. Production model/distribution/geometry code unchanged.
- Missing alignment-directory postprocessing repaired from saved first prediction; original failure preserved.
- Reports, proposed abstention policy and compact provenance published after all authorized tests. No model selection or historical gate rewriting.
''')
 write('FINAL_E016_R1C_REWARD_RELIABILITY_REPORT.md',f'''# R1c: native folding operational; AlphaFold reward remains NO-GO

Completed12 serial local depth-ablation folds, historical per-control calibration/context analysis and one matched full-episode synthetic critic comparison. No J1 updates, production RL, remote queries, TEST evaluation, downloads or deletion.

{table}

Two non-amyloid controls improve strongly with depth; query→full gains are0.579 and0.720. Confidence broadly tracks those changes. Amyloid recovery remains0.14–0.18 despite high deep-MSA confidence; its experimental fibril and isolated-chain prediction are different biological contexts. A depth/confidence gate cannot safely erase this disagreement using information available for a novel generated candidate. Natural-sequence thinning does not certify novel-sequence behavior.

The historical VALIDATION MAE0.358252 remains a failure. Both cases contribute (absolute errors0.244402/0.472103); the larger accounts for65.89%, but excluding it for sensitivity still fails0.20. The confidence product underpredicts these partially/strongly recovered structures. In contrast, it overpredicts amyloid-state recovery. New fixed signals are reported on TRAIN only, without fitting or selection on VALIDATION. No replacement reward is certified.

{ct}

Transformer twins pass; MLP twins fail OOD. Critic gates apply to both twins and OOD, with the0.100 limit unchanged. Complete episodes, MC calibration, action densities, gamma1 terminal handling and deterministic GPU restart are retained. This value-regression experiment is not a SAC policy-learning certificate; no critic has been validated on actual protein-return trajectories. Prefer a KL-regularized episode-level sequence-policy-gradient baseline only after reward certification and separate approval.

Resources:12/12 successful folds, {seconds:.2f}s charged worker time /1500s cap, sampled AF GPU peak {peak:.0f}MiB. Critics {cseconds:.2f}s, final D: free {free/2**30:.2f}GiB,10GiB reserve maintained. R1b manifest verifies {len(old['evidence_files'])} evidence files and {len(old['source_files'])} source files unchanged. Frozen J1 SHA remains `{sha(parent)}`. Source `{source}`.

Readiness: **GO for bounded native AF inference; NO-GO for sequence-reward certification, novel-sequence transfer and production RL.** A specific16-control prospective panel with a separately approved16-query/24-fold/45-minute budget is proposed, not executed. No AlphaFold confidence result demonstrates experimental stability or function. Workers are stopped after publication.

Detailed evidence: [calibration](REWARD_CALIBRATION_FAILURE_ANALYSIS.md), [depth](MSA_DEPTH_SUMMARY.md), [context](BIOLOGICAL_CONTEXT_AND_AMYLOID_CASE.md), [novelty](NOVEL_SEQUENCE_RELIABILITY_ASSESSMENT.md), [policy](PROPOSED_SEQUENCE_REWARD_AND_ABSTENTION_POLICY.md), [critics](SAC_VS_POLICY_GRADIENT_DECISION.md), [gate](NEXT_RL_GO_NO_GO.md), [resources](RESOURCE_AND_PROVENANCE.json).
''')
 # Scientific plot from numeric results; no raster image editing.
 try:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  fig,axes=plt.subplots(1,3,figsize=(11,3.5))
  for ax,ident in zip(axes,['train_64','train_128','train_historical_2qmi_A']):
   vs=[v for v in rows if v['control']==ident];ax.plot(range(4),[v['native_recovery'] for v in vs],'o-',label='native recovery');ax.plot(range(4),[v['features']['ptm'] for v in vs],'s--',label='pTM');ax.plot(range(4),[v['features']['mean_pLDDT']/100 for v in vs],'^:',label='pLDDT/100');ax.set_xticks(range(4),['1','4','32','full']);ax.set_ylim(0,1.05);ax.set_title(ident.replace('train_',''));ax.set_xlabel('Input MSA depth');ax.grid(alpha=.2)
  axes[0].set_ylabel('Score');axes[-1].legend(fontsize=8);fig.tight_layout();fig.savefig(R/'MSA_DEPTH_CONFIDENCE.png',dpi=150);plt.close(fig)
 except ImportError:pass
 print(json.dumps({'report':str(R/'FINAL_E016_R1C_REWARD_RELIABILITY_REPORT.md'),'AF_seconds':seconds,'free_GiB':free/2**30,'source':source,'critics':[{k:v[k] for k in ['arm','passes_ID_and_OOD_both_twins','seconds']} for v in critic['rows']]}))
if __name__=='__main__':main()
