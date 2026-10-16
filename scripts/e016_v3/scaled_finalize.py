"""One final TEST event only after completed training and frozen selections; publish evidence."""
import json,math,random,subprocess,shutil,time
from pathlib import Path
import numpy as np
import torch
import scaled_bc as c
from scaled_train import Engine
from protein_joint_sac.data.scaled_supervised import Data,internal,canonical,P,PILOT
from protein_joint_sac.data.accepted_supervised import CoordinateReader
from protein_joint_sac.data.certified_unified import file_sha
R=c.R

def panel(data,matched):
 ids=data.jointids['TEST'] if matched else data.ids['TEST'];rng=random.Random(4026);chosen=[]
 for lo,hi in [(20,64),(65,128),(129,256),(257,384),(385,500)]:
  pool=[h for h in ids if lo<=len(data.seqs[h])<=hi];rng.shuffle(pool);groups=set();picked=[]
  for h in pool:
   g=data.proposal.sequences[h]['protected_group_id']
   if g not in groups:groups.add(g);picked.append(h)
   if len(picked)==64:break
  if len(picked)!=64:raise ValueError('TEST coverage')
  chosen.extend(picked)
 return sorted([(h,data.owners[h][0] if h in data.owners else None,t) for h in chosen for t in sorted(rng.sample(range(len(data.seqs[h])),4))],key=lambda e:e[2])
@torch.no_grad()
def assess(eng,targets):
 sums=np.zeros(5);counts=np.zeros(3);bands={str(i):np.zeros(3) for i in range(5)};eng.model.eval()
 for k in range(0,len(targets),eng.c['live_batch']):
  ex=targets[k:k+eng.c['live_batch']]
  if any(eng.data.proposal.sequences[h]['split']!='TEST' for h,o,t in ex):raise ValueError('TEST-only assessment')
  b,y,g=c.make(eng.data,ex,eng.joint);_,aa,geo,parts=c.loss_call(eng.model,b,y,g,eng.c);sums+=np.array([float(aa.sum()),float(geo.sum()),*parts.sum(0).cpu().tolist()]);counts+=np.array([sum(t>=d for h,o,t in ex) for d in [1,2,3]]) if eng.joint else 0
  for (h,o,t),a,g in zip(ex,aa.cpu().tolist(),geo.cpu().tolist()):
   n=len(eng.data.seqs[h]);band=next(i for i,(lo,hi) in enumerate([(20,64),(65,128),(129,256),(257,384),(385,500)]) if lo<=n<=hi);bands[str(band)]+=np.array([a,g,1])
 n=len(targets);return {'AA_NLL':sums[0]/n,'perplexity':math.exp(sums[0]/n),'geometry_component_mean_NLL':sums[1]/n,'component_NLL':(sums[2:]/np.maximum(counts,1)).tolist(),'identities':len(set(h for h,o,t in targets)),'prefixes':n,'bands':{k:(v/np.array([max(1,v[2]),max(1,v[2]),1])).tolist() for k,v in bands.items()},'panel_sha256':c.digest(targets)}

def main():
 c.setup();names=['S1','J1','J2','S2'];completed={n:json.load(open(R/'runs'/n/'COMPLETED.json')) for n in names};configs=json.load(open(R/'configs/FROZEN.json'))
 if any(x['updates']!=10000 for x in completed.values()):raise ValueError('no TEST before all four10000-update completions')
 selections={n:{'path':x['selected_checkpoint'],'sha256':x['selected_sha256']} for n,x in completed.items()}
 schedules={}
 for n in ['S1','J1','J2']:
  values={}
  for log in sorted((R/'runs'/n).glob('updates_from_*.jsonl')):
   for line in log.read_text().splitlines():
    row=json.loads(line);u=row['update'];h=row['sample_order_sha256']
    if u in values and values[u]!=h:raise ValueError('conflicting exposure history')
    values[u]=h
  if sorted(values)!=list(range(1,10001)):raise ValueError('incomplete matched exposure history')
  schedules[n]=values
 if not (schedules['S1']==schedules['J1']==schedules['J2']):raise ValueError('matched exposure schedule violated')
 c.write(R/'MATCHED_EXPOSURE_VERIFICATION.json',{'status':'PASS','updates_compared':10000,'prefix_decisions_per_arm':640000,'models':['S1','J1','J2'],'same_identity_observation_prefix_schedule':True})
 for s in selections.values():assert file_sha(s['path'])==s['sha256']
 seal=R/'CHECKPOINT_SELECTIONS_FROZEN.json'
 if not seal.exists():c.write(seal,{'selections':selections,'protocol_sha256':file_sha(R/'configs/FINAL_TEST_PROTOCOL.json'),'TEST_evaluated':False})
 else:assert json.load(open(seal))['selections']==selections
 event=R/'FINAL_TEST_EVENT.json'
 if not event.exists():c.write(event,{'event_id':'SINGLE_FINAL_TEST_V1','selection_seal_sha256':file_sha(seal),'started':time.time(),'no_subsequent_tuning':True})
 data=Data(False);rows=list(data.proposal.iter_joint('TEST'));reader=CoordinateReader(data.proposal.external)
 for r in rows:data.rows[r['observation_id']]=r;data.owners[r['sequence_hash']].append(r['observation_id'])
 for h in data.jointids['TEST']:data.owners[h].sort()
 for r,x in reader.stream(rows):o=r['observation_id'];data.g[o]=internal(x);data.ca[o]=canonical(x)
 matched=panel(data,True);full=panel(data,False);c.write(R/'configs/FINAL_TEST_PANELS.json',{'matched':matched,'full':full});tests={}
 for n in names:
  path=R/'runs'/n/'FINAL_TEST.json'
  if path.exists():tests[n]=json.load(open(path));continue
  eng=Engine(n,data,configs[n]);eng.resume(Path(selections[n]['path']))
  generated=[]
  for length in [20,64,128,256,384,500]:
   for replicate in range(4):generated.append(eng.generate(length,913000+length*16+replicate))
  c.write(R/'runs'/n/'SELECTED_GENERATION_REPLICATES.json',generated)
  diversity={}
  TRAIN_strings={data.seqs[h] for h in data.ids['TRAIN']}
  for length in [20,64,128,256,384,500]:
   subset=[x for x in generated if x['N']==length];good=[x for x in subset if x['status']=='COMPLETE'];strings=[x['sequence'] for x in good];distances=[]
   if n.startswith('J'):
    for i in range(len(good)):
     for j in range(i):
      a=np.asarray(good[i]['generated_CA_A']);b=np.asarray(good[j]['generated_CA_A']);a=a-a.mean(0);b=b-b.mean(0);u,_,vt=np.linalg.svd(a.T@b);sign=np.eye(3);sign[-1,-1]=np.linalg.det(u@vt);rotation=u@sign@vt;distances.append(float(np.sqrt(np.square(a@rotation-b).sum()/length)))
   composition={letter:sum(s.count(letter) for s in strings)/max(1,sum(map(len,strings))) for letter in c.AA}
   diversity[str(length)]={'requested':4,'complete':len(good),'unique_sequences':len(set(strings)),'exact_TRAIN_sequence_overlap':len(set(strings)&TRAIN_strings),'composition':composition,'pairwise_proper_Kabsch_CA_RMSD_A':distances,'interpretation':'four-sample descriptive diversity only; neither distinct-fold certification nor stability evidence'}
  c.write(R/'runs'/n/'SELECTED_GENERATION_DIVERSITY.json',diversity)
  result={'primary':assess(eng,full if n=='S2' else matched)}
  if n=='S2':result['matched']=assess(eng,matched)
  c.write(path,result);tests[n]=result;del eng;torch.cuda.empty_cache()
 c.write(R/'FINAL_TEST_RESULTS.json',{'event_count':1,'models':tests,'no_tuning_after_TEST':True})
 publish(completed,selections,tests,data)

def publish(completed,selections,tests,data):
 counts=json.load(open(R/'MODEL_PARAMETER_COUNTS.json'));curves={};valid={};generation={}
 for n in completed:
  d=R/'runs'/n;curves[n]=[json.loads(line) for p in sorted(d.glob('updates_from_*.jsonl')) for line in p.read_text().splitlines()];valid[n]=[json.load(open(p)) for p in sorted(d.glob('validation_*.json'))];generation[n]=[{'update':int(p.stem.split('_')[-1]),'samples':json.load(open(p))} for p in sorted(d.glob('generation_*.json'))]
 for n in generation:
  generation[n]={'milestones':generation[n],'selected_replicates':json.load(open(R/'runs'/n/'SELECTED_GENERATION_REPLICATES.json')),'selected_diversity':json.load(open(R/'runs'/n/'SELECTED_GENERATION_DIVERSITY.json'))}
 c.write(R/'TRAINING_CURVES.json',curves);c.write(R/'VALIDATION_METRICS.json',{'models':valid,'final_TEST':tests});c.write(R/'FREE_GENERATION_METRICS.json',generation)
 telemetry=[json.loads(line) for line in (R/'RESOURCE_TELEMETRY.jsonl').read_text().splitlines()]
 runtime={}
 for n,x in completed.items():
  rows=[row for row in telemetry if row.get('name')==n and 'gpu' in row];parsed=[[float(v.strip()) for v in row['gpu'].split(',')] for row in rows];total=sum(row['seconds'] for row in curves[n]);last=valid[n][-1];previous=valid[n][-2];key='selection_loss' if n.startswith('J') else 'AA_NLL'
  runtime[n]={'elapsed_training_seconds':x['elapsed_seconds'],'summed_optimizer_update_seconds':total,'prefix_decisions_per_update_second':x['decisions']/total,'effective_batch':x['config']['effective_batch'],'live_batch':x['config']['live_batch'],'accumulation_steps':x['config']['effective_batch']//x['config']['live_batch'],'Torch_peaks_optimized_phase':x['resources'],'device_wide_memory_MiB_max_observed':max(z[0] for z in parsed) if parsed else None,'device_utilization_percent_mean_observed':sum(z[2] for z in parsed)/len(parsed) if parsed else None,'device_utilization_percent_max_observed':max(z[2] for z in parsed) if parsed else None,'last_interval_selection_change':last[key]-previous[key],'trend':'improved last interval' if last[key]<previous[key] else 'no final-interval improvement','checkpoint_bytes':Path(x['selected_checkpoint']).stat().st_size,'sampling_decisions':x['decisions'],'distinct_identities':x['distinct_TRAIN_identities']}
 matched_S2_final=json.load(open(R/'runs/S2/matched_validation_10000.json'))
 comparison={'scope':'ONE_SEED_DESCRIPTIVE_NO_SUPERIORITY_TEST','equal_10000_update_matched_VALIDATION_AA_NLL':{**{n:completed[n]['final_validation']['AA_NLL'] for n in ['S1','J1','J2']},'S2':matched_S2_final['AA_NLL']},'selected_checkpoint_matched_TEST_AA_NLL':{**{n:tests[n]['primary']['AA_NLL'] for n in ['S1','J1','J2']},'S2':tests['S2']['matched']['AA_NLL']},'dataset_difference':'S2 uses118219 TRAIN identities vs21475 for S1/J1/J2; all have640000 decisions','parameter_counts':counts,'runtime':runtime,'post_TEST_tuning':False}
 c.write(R/'MATCHED_COMPARISONS.json',comparison)

 refs=json.load(open(PILOT/'priors/SEQUENCE_IDENTITY_WEIGHTED_REFERENCES.json'));panels=json.load(open(R/'configs/VALIDATION_PANELS.json'));baseline={}
 for name,targets in [('MATCHED',panels['matched']),('FULL',panels['full'])]:
  freq=refs['statistics'][name]['frequency'];nll=sum(-math.log(freq[data.seqs[h][t]]) for h,o,t in targets)/len(targets);baseline[name]={'AA_NLL':nll,'perplexity':math.exp(nll),'fit_split':'TRAIN_ONLY'}
 c.write(R/'TRAIN_FREQUENCY_BASELINES.json',baseline)
 lines=['| Model | Trainable parameters | Updates | Prefix decisions | Distinct identities | Best VAL AA NLL | Final VAL AA NLL | TEST AA NLL |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for n,x in completed.items():lines.append(f"| {n} | {counts[n]['trainable_parameters']:,} | {x['updates']} | {x['decisions']:,} | {x['distinct_TRAIN_identities']:,} | {x['best_validation']['AA_NLL']:.5f} | {x['final_validation']['AA_NLL']:.5f} | {tests[n]['primary']['AA_NLL']:.5f} |")
 table='\n'.join(lines);commit=subprocess.check_output(['git','-C',str(R/'source'),'rev-parse','HEAD'],text=True).strip();status=subprocess.check_output(['git','-C',str(R/'source'),'status','--short'],text=True).strip()
 text=f"""# Completed scaled scratch PDB supervised comparison
VERIFIED: all four arms completed10,000 real-PDB updates, effective batch64. Source commit {commit}; worktree {status or 'clean'}.

{table}

Fixed stratified320-identity/1280-prefix development and final test panels; not an exhaustive evaluation of every held-out protein. S1/J1/J2 share matched identities, prefix schedules and target observations where applicable. S2 uses the larger TRAIN population and also reports matched held-out results. One final TEST event occurred only after checkpoint selections were sealed; no subsequent tuning.

Selection: minimum milestone VALIDATION AA NLL for S1/S2, minimum AA+normalized geometry NLL for J1/J2. Checkpoint/source/config/prior hashes, exposures, times, gradient norms, best/final metrics and source-anomaly exposure counts are in runs/*/COMPLETED.json, milestones and curves. Checkpoint selection and generation assessment are separate.

Dataset claim: exact identity separation and no detected violations of frozen50% identity/both80% coverage or70% identity/shorter80% coverage/minspan50 criteria. No universal remote-homology or experimental-context independence claim. Deposited constructs are not necessarily full natural-gene proteins. Native future coordinates/AA were never actor inputs.

Ordinary geometry initialization retained: empirical transformed-distance variance gives nearly2000 normalized NLL on known source extremes, above the frozen100 criterion. Six original coordinate pairs/five TRAIN identities retained and tracked, with no hidden filtering/repair or new geometry hard bounds.

A strict checkpoint-load failure at S1 update250 was repaired through a versioned metadata-only migration. Original bytes remain preserved; model, optimizer, scheduler, sampler and RNG were unchanged. New-model restart receipts document actual numerical equivalence. See recovery receipts and CHECKPOINT_AND_RESUME.md.

Single-seed point estimates are descriptive, not replicated proof of geometry/chemistry superiority. Free-running CA plausibility and failures must be assessed independently of teacher-forced NLL; neither proves folding, thermodynamic stability, unique native state, expression or function. No SAC, AF, E015 or editing was executed.

Selected full-state checkpoints:\n"""+'\n'.join(f"- {n}: {s['path']} SHA256 {s['sha256']}" for n,s in selections.items())+'\n'
 text+='\nMeasured runtime and final-interval behavior:\n'+'\n'.join(f"- {n}: {x['elapsed_training_seconds']/3600:.3f}h; live/effective batch {x['live_batch']}/{x['effective_batch']}; observed device utilization mean/max {x['device_utilization_percent_mean_observed']:.1f}/{x['device_utilization_percent_max_observed']:.1f}%; {x['trend']}." for n,x in runtime.items())+'\n\nSee MATCHED_COMPARISONS.json for same-panel AA contrasts at identical10000-update exposure and selected-checkpoint TEST metrics. Source/allocator recovery overhead is retained; optimized-phase Torch peaks and device-wide telemetry are separate from earlier oversubscribed attempts. Four samples per length give descriptive generation diagnostics, not a statistically complete diversity estimate.\n'
 c.write(R/'FINAL_SCALED_TRAINING_REPORT.md',text)
 docs={
 'ARCHITECTURE_10M_CONFIGS.md':'Width352 /8blocks /8heads /head dimension44 /FFN4 /RoPE /dropout0 /max500. Sequence pair module dormant/frozen and excluded from trainable counts. Joint distributions, bootstrap and builder unchanged. All scratch; paired initialization hashes verified. See configs/FROZEN.json and MODEL_PARAMETER_COUNTS.json.',
 'GPU_BATCH_OPTIMIZATION.md':'Measured64/128/256/384/500 and live1/2/4/8/16/32/64 in FP32/BF16. WDDM oversubscription beyond physical8GB severely reduces throughput; oversized batches not selected. Effective64; sequence live32 and joint live16. BF16 parity passed0.000356 loss error and0.00725 relative gradient L2. CPU reference logits/loss/gradient/optimizer/padding/checkpoint tests passed. Activation checkpointing saved memory but was slower at selected batch.',
 'CHECKPOINT_AND_RESUME.md':'Immutable atomic full-state checkpoints plus SHA sidecars. Preserve model/AdamW/scheduler/RNG/sampler/exposure, config/data/geometry/chemistry/prior/source hashes and selection metadata. Preflight12M restart and each arm update251 reconstruction verified. S1 original250 metadata failure preserved; separately SHA-verified normalization converted NumPy validation scalars to built-in numbers without altering tensors or stochastic state. Same-runtime numerical guarantees only.',
 'PRIOR_AND_SOURCE_QUALITY_REPORT.md':'TRAIN prior/source membership hashes rechecked. Source-extreme likelihood preflight rejected empirical narrow-distance head initialization; same ordinary initialization in J1/J2. Six known native extreme pairs across five TRAIN identities retained, not repaired or silently excluded. Anomaly exposure and gradient clipping recorded. Priors describe CA statistics, not rotamers/energy/folding stability.',
 'NEXT_RESEARCH_RECOMMENDATIONS.md':'Compare S1/J1 and J1/J2 at matched exposure and compute using held-out metrics and independent free-running geometry. S2 has different data and fewer exposures per identity. Repeat seeds before superiority claims. No tuning on final TEST. SAC, AlphaFold calibration, E015 and editing remain separate unexecuted research stages.'}
 for n,s in docs.items():c.write(R/n,'# '+n.removesuffix('.md').replace('_',' ')+'\n'+s+'\n')
 c.write(R/'RESOURCE_AND_RUNTIME_REPORT.json',{'models':completed,'measured_runtime_and_utilization':runtime,'allocator_runtime_change':json.load(open(R/'ALLOCATOR_RUNTIME_CHANGE.json')),'D_free_bytes':shutil.disk_usage(R).free,'root_logical_bytes':c.size(R),'GPU':c.gpu(),'source_commit':commit,'source_status':status,'TEST_events':1,'SAC_updates':0,'AF_calls':0,'protected_deletions':0})
 artifacts={str(p.relative_to(R)):file_sha(p) for p in sorted(R.rglob('*')) if p.is_file() and '.git' not in p.parts and p.name not in ['PROVENANCE_SHA256.json','FINALIZE.log']}
 c.write(R/'PROVENANCE_SHA256.json',{'source_commit':commit,'artifact_sha256':artifacts,'selected_checkpoints':selections,'dataset_manifest_sha256':file_sha(P/'PROPOSAL_MANIFEST.json'),'prior_seal_sha256':file_sha(PILOT/'PROVENANCE_SHA256.json'),'final_TEST_events':1})
 print('ALL_FOUR_SCALED_RUNS_COMPLETE_AND_SEALED',flush=True)

if __name__=='__main__':main()
