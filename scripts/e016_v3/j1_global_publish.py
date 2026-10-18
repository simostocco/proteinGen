"""Publish the measured J1 research assessment; no model execution."""
import json,math,time,re
from pathlib import Path
import j1_global as j
def run():
 R=j.R;read=lambda p:json.load(open(R/p))
 b=read('SEQUENCE_BASELINES.json');s=read('FINAL_STATISTICAL_COMPARISON.json');d=read('runs/J1/COMPLETED.json');f=read('FINAL_SELECTION.json');v=read('runs/J1/evaluation_30000.json');g=read('NATIVE_VS_GENERATED_GLOBAL_GEOMETRY.json');res=read('RESOURCE_REPORT.json');grad=read('BASE_LOSS_GRADIENT_COMPONENTS.json')
 table='| Predictor | TRAIN PPL | VALIDATION PPL |\n|---|---:|---:|\n'
 for label,a,z in [('Uniform',20.,20.),('TRAIN identity-first unigram',b['J1']['TRAIN']['probability_baselines']['matched_joint/unigram_identity_first']['perplexity'],b['J1']['VALIDATION']['probability_baselines']['matched_joint/unigram_identity_first']['perplexity']),('TRAIN identity-first Markov',b['J1']['TRAIN']['probability_baselines']['matched_joint/markov_identity_first']['perplexity'],b['J1']['VALIDATION']['probability_baselines']['matched_joint/markov_identity_first']['perplexity']),('Original J1-FULL',b['J1']['TRAIN']['perplexity'],b['J1']['VALIDATION']['perplexity']),('Trained J1 +30k',v['TRAIN']['perplexity'],v['VALIDATION']['perplexity'])]:table+=f'| {label} | {a:.6f} | {z:.6f} |\n'
 gt='| Length | Original Rg A | Trained Rg A | Original clashes/100res | Trained clashes/100res |\n|---:|---:|---:|---:|---:|\n'
 for n in ['64','128','256','384','500']:
  a=g['baseline'][n];z=g['latest_actual_policy'][n];gt+=f"| {n} | {a['Rg_A']['mean']:.3f} | {z['Rg_A']['mean']:.3f} | {a['clashes_per100_residues']['mean']:.4f} | {z['clashes_per100_residues']['mean']:.4f} |\n"
 ci=s['latest_vs_parent_paired_identity_AA_NLL']['bootstrap_95pct']
 text=f"""# Final J1 global pretraining report

**VERIFIED: all30,000 additional real-PDB optimizer updates executed. The new checkpoint improved held-out AA/local likelihood modestly, but failed the frozen final generation safeguard and was NOT promoted. The original J1-FULL remains the selected fallback.**

Training implementation commit2695487; branch e016-j1-global-pretraining-20261017-v1. Source and all scientific inputs/outputs are sealed separately by PROVENANCE_SHA256.json. Historical datasets, priors and checkpoints remain unchanged.

## What ran

One existing J1, with 12,351,746 trainable parameters: width352,8blocks,8heads(head dimension44),FFN4, dropout0, dense full-prefix geometric attention with sequence RoPE. Chemistry stays disabled; the approved14-feature registry and action system are unchanged. FP32 weights, BF16 neural matrix operations, FP32 geometry-head projection, FP64 coordinate invariants/density/builder operations.

AdamW moments restored from the30000-total-update parent. New250-update warmup, peak LR5e-5, cosine decay to5e-6, weight decay0.01, clipping1. Effective batch64, live 16, 4 accumulation passes. Ordinary decisions: 56 complete joint plus8partial AA-only, identity-first then observation/position sampling. Partial geometric inputs retain authoritative masks and positions; available partial geometry is intentionally not a training target in this AA-only branch. Missing coordinates were not fabricated or reclassified.

Base loss remains AA CE plus per-decision mean of valid conditional distance/angle/signed-torsion NLL. The periodic auxiliary objective is exactly
L(u)=L_base+I[u mod16=0]*ramp(u)*(0.13868601683008452*L_Rg+0.04997356779049633*L_clash).
Ramp reaches1at1000updates. Effective average post-ramp coefficients per ordinary update are0.0086678761and0.0031233480. No compensating16x importance factor was used.

Rg is Huber(delta0.5) of the log generated/native Rg ratio at identical prefix endpoints/count. Repulsion is squared penetration below2A for sequence separation>=3 and at least one generated endpoint, summed per generated residue. TRAIN512-identity calibration had9033260eligible nonlocal pairs, none below2A. Native future coordinates were loss targets only. Rollouts introduced true AAs one at a time after each AA prediction, sampled actual transformed-Gaussian geometry through reparameterization, and fed attached generated coordinates back into subsequent encoders. They are not fully free-running joint samples.

Exactly1920000ordinary decisions: 1,680,000 complete, 240,000 partial AA-only, including15968targets at missing-CA positions. There were1875global updates and59367differentiable generated residues (horizons17–32); no full500-step differentiable rollout was trained. Two separate, discarded engineering recovery proofs executed34optimizer updates. Main lineage now totals60000updates.

## Sequence evidence

Both frozen evaluation panels have2947distinct identities and23576fixed decisions, 8 per identity using the same first-sorted-observation/rounded-linspace protocol and matched length strata. This includes BOS once per identity; it is not whole-chain residue-weighted perplexity. TRAIN and VALIDATION identities do not overlap. TEST was not evaluated. All fits use TRAIN only; smoothing is0.5per AA/transition cell, with BOS represented only as a Markov context. Exact probabilities, both identity-first/residue-weighted variants and full-PDB statistical controls are in SEQUENCE_BASELINES.json.

{table}

Latest VALIDATION AA NLL2.755765591 versus2.764163531 originally: delta−0.008397941nats, paired identity-bootstrap95%interval{ci}. This is about0.84%lower perplexity. Intervals are conditional on validation-based checkpoint selection and do not remove family dependence or establish universal homology independence. The accepted practical split still has its documented residual similarities/context overlap.

TRAIN–VALIDATION NLL gap widened0.374808→0.536213. Most new fitting occurred on TRAIN; this is not evidence that the generalization gap was solved. VALIDATION top1accuracy15.838%→16.182%; top3accuracy33.810%→34.467%. Bits/residue are3.975729on VALIDATION and3.202137on TRAIN. These are native-prefix-geometry-conditioned AA metrics, not masked-language-model perplexities or unconditional sequence-only scores.

## Local and global geometry

VALIDATION geometry NLL per decision improved−0.213709→−0.253267. Distance/angle/torsion NLL changed from[-1.917618,-0.173647,1.358548]to[-1.951040,-0.198334,1.281031], with20629valid targets per component. Density units remain A/radian; negative continuous NLL is legitimate.

The actual trained checkpoint produced160/160finite complete outputs,32independent seeds at each length, all32sequences unique within every length. Baseline samples were reused only for the identical original checkpoint/seed/protocol. The actual last-trained policy is reported even though it was rejected.

{gt}

All five mean Rg point estimates decreased slightly, but every paired95%interval for native-normalized Rg-error change includes0. There is no demonstrated broad compactness improvement; long chains remain substantially more extended than length-matched native references. There is no evidence of gross chain collapse.

Clash means increased at4of5lengths and decreased at128; all paired clash-change intervals also include0. At64residues,2.783203clashes/100res exceeds the frozen point-estimate tolerance1.25*1.953125+0.25=2.691406. The paired clash-change interval[-0.048828,1.806641]is uncertain, but no post hoc tolerance relaxation or reseeding was used. The candidate failed this declared safeguard; other final checks passed. More compactness alone was not accepted as success.

Long-contact fractions rose at64/128/256/384and fell slightly at500. At500,the trained fraction0.003107remains below the native TRAIN/VALIDATION reference distributions (about0.0062). CA spacing, virtual-angle/circular-torsion summaries, degeneracy, composition and normalized shape diversity remain in NATIVE_VS_GENERATED_GLOBAL_GEOMETRY.json. These are CA proxies, not all-atom chemistry, experimental folding, free energy or function.

## Verification and interpretation

35distinct focused CPU tests passed:29mask/geometry/controller tests plus3assessment and3recovery-launcher tests. One initial pytest descriptor-capture failure occurred before collection on D:; the preserved retry with capture disabled passed. Tests cover native reconstruction, rotation/translation invariance, masked/future independence, zero-auxiliary base parity, finite-difference builder gradients and eager/recomputed parity.

Actual full-model GPU checks passed through rollout horizon128. Global gradients reached geometry heads and Transformer weights. BF16/FP32relative gradient error was0.0052503at horizon4, max coordinate difference0.030157A; eager/recomputed gradients matched exactly. Real interrupted/resumed global-update comparison had model error0, loss error0, optimizer maximum difference7.28e-12, with identical sampler/scheduler/RNG/rollout ordering.

A separate10-prefix TRAIN diagnostic found local geometry/distance gradient L2about239–246versus AA9–11before clipping; this small, explicitly frozen panel is diagnostic, not a population estimate or a reason to silently change the base objective. Sparse short rollouts and native-prefix/true-AA conditioning remain major limits on full free-generation credit assignment. Raw first/last100global TRAIN losses are not matched panels and do not demonstrate a monotonic Rg-loss reduction.

## Resources

Main elapsed15744.21seconds(4.373hours), including milestone evaluation/checkpoints; timed updates13478.50seconds, including4804.13global-loss seconds. Throughput was about142.45ordinary decisions/s over timed updates and121.95/s over the complete main run. Peak allocated/reserved CUDA memory3.473/3.607GiB; host RSS high-water6.821GiB; sampled device-wide GPU peak5345MiB. No swap, OOM, nonfinite optimizer step or disk-reserve breach was observed. Source/runtime/environment hashes and measured storage are in RESOURCE_REPORT.json and PROVENANCE_SHA256.json.

## Direct answers to the15scientific questions

1. Uniform perplexity: exactly20.
2. TRAIN-derived identity-first unigram: TRAIN18.670699, VALIDATION18.905275; exact probabilities/smoothing and residue-weighted alternatives are preserved.
3. First-order Markov: TRAIN17.560356, VALIDATION17.135997.
4. Starting J1: TRAIN10.906462, VALIDATION15.865763.
5. Gap: substantial0.374808nats initially; it widened to0.536213.
6. Best and final trained checkpoint by AA NLL is +30,000: TRAIN9.203208, VALIDATION15.733081. The quality-selected fallback remains the original10.906462/15.865763model.
7. Latest AA gains beyond unigram/Markov are0.183675/0.085416nats on VALIDATION; incremental gain beyond the original J1 is0.008398nats.
8. YES: nonzero verified pathwise gradients reach generated CA, geometry heads and upstream Transformer parameters.
9. NOT ESTABLISHED: Rg point estimates improved modestly without collapse, but length-specific uncertainty includes no change and chains remain too extended.
10. NO consistent reduction: clashes rose at4of5lengths; long contacts improved at4of5but500remains deficient.
11. YES for native-prefix likelihood: all three component NLL point estimates improved; free-running global quality did not become validated.
12. NO defensible simultaneous AA/global improvement was promoted: likelihood improved, final generation safeguard failed.
13.30,000 additional main updates, 1,920,000 ordinary decisions, 59,367 rollout residues; 34 separate engineering updates.
14. Retained trained checkpoint: {d['latest_checkpoint']}, SHA256 {d['latest_sha256']}. Selected fallback: {f['selected_checkpoint']}, SHA256 {f['selected_sha256']}.
15. Remaining limits: growing generalization gap, short/true-AA-conditioned rollout mismatch, unresolved long-range organization/clashes, sparse auxiliary strength, CA-only proxies, practical split limitations, and no calibrated oracle or experimental validation. No RL readiness or folding success is established.

Data sampling and loss changed together; the observed likelihood gain cannot be attributed exclusively to Rg. The full authorized experiment is complete; scientific improvement across both objectives was not achieved. No TEST, J2/S1/S2training, SAC, AlphaFold or E015work occurred.
"""
 # Keep prose readable without changing paths, numeric values or SHA identities.
 hashes={}
 def protect(m):
  key='HASH_PLACEHOLDER_'+chr(65+len(hashes));hashes[key]=m.group(0);return key
 text=re.sub(r'\b[0-9a-f]{40,64}\b',protect,text)
 text=re.sub(r'([a-z]{2,})([0-9])',r'\1 \2',text)
 text=re.sub(r'(TRAIN|VALIDATION|LR|FFN)(?=[0-9])',r'\1 ',text)
 text=re.sub(r'([0-9])([A-Za-z]{2,})',r'\1 \2',text)
 text=re.sub(r'(?m)^([0-9]{1,2})\.(?=\S)',r'\1. ',text)
 text=text.replace('width 352,8 blocks,8 heads(head dimension 44),FFN4','width 352, 8 blocks, 8 heads (head dimension 44), FFN expansion 4')
 for key,value in hashes.items():text=text.replace(key,value)
 text=text.replace('2A','2 Å')
 assert d['latest_sha256'] in text and f['selected_sha256'] in text, 'report must preserve exact hashes'
 (R/'FINAL_J1_GLOBAL_PRETRAINING_REPORT.md').write_text(text)
 spec=(R/'GLOBAL_GEOMETRY_LOSS_SPEC.md').read_text().split('\nExact periodic weighting:')[0];spec+='\nExact periodic weighting: L(u)=L_base+I[u%16==0]*ramp(u)*(lambda_Rg*L_Rg+lambda_clash*L_clash). No16x rescaling; average post-ramp strengths are0.0086678761and0.0031233480perordinary update. Input and output units/actions are unchanged.\n';(R/'GLOBAL_GEOMETRY_LOSS_SPEC.md').write_text(spec)
 rec=(R/'CHECKPOINT_AND_RECOVERY.md').read_text().split('\nVERIFIED final recovery proof:')[0];rec+='\nVERIFIED final recovery proof: model max error0, optimizer7.28e-12, AA/Rg/clash loss differences0, identical sampler/scheduler/RNG. Main checkpoint counters were independently reloaded:30000additional,60000optimizer total,30000scheduler epoch,1920000ordinary decisions. Use checked j1_global_resume.py; it refuses active/completed jobs and mismatched critical dependencies. The actual completed run is not to be reopened.\n';(R/'CHECKPOINT_AND_RECOVERY.md').write_text(rec)
 (R/'NEXT_RESEARCH_RECOMMENDATIONS.md').write_text("""# Next research decisions

The latest trained checkpoint is a modest AA/local-likelihood improvement, not a promoted global generator. Keep the original J1-FULL fallback selected and preserve the +30000candidate.

Prioritize a separately approved, fixed-data loss-balance/rollout study: review the measured AA-versus-distance gradient scales; assess longer generated horizons using measured128-step feasibility; predeclare steric/compactness balance and fixed larger generation panels. Keep data mixture fixed for causal interpretation. Do not retune the finished lineage or repeat seeds until a safeguard passes.

Continue using VALIDATION for development; the previous TEST has already been inspected and was not reused here. Any future independent evaluation protocol must disclose that history and practical split limitations. Additional training, new acceptance criteria or an independent replication require a new versioned protocol. No AlphaFold, SAC or E015 should be launched automatically from this report.

AA likelihood and CA compactness/clash proxies do not establish foldability, stability, expression, function or unique native state.
""")
 print('SCIENTIFIC_REPORT_UPDATED',flush=True)
if __name__=='__main__':run()
