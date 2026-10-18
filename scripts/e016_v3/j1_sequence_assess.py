"""Post-run independent reporting. No optimization; TRAIN/VALIDATION only."""
import json,math,random,time,gc,collections
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import j1_sequence_first as s
from j1_sequence_first import R,PREVIOUS,j,write,PartialData,Engine,load_checked,file_sha,set_dropout
from j1_global_finalize import metric_summary,paired_ci
from partial_finalize import metrics
j.R=R;j.core.R=R
@torch.no_grad()
def partial_eval(model,data,panel,role):
 model.eval();by=collections.defaultdict(list);band=collections.defaultdict(list);missing=0;start=time.monotonic()
 for off in range(0,len(panel),16):
  ex=panel[off:off+16]
  if any(data.proposal.sequences[h]['split']!=role for h,o,t in ex):raise ValueError('partial evaluation role')
  b,y,g=data.batch(ex)
  with torch.autocast('cuda',dtype=torch.bfloat16):hstate=s.encode_batch(model,b);logits=model.aa_head(hstate).float()
  aa=torch.nn.functional.cross_entropy(logits,y,reduction='none');correct=logits.topk(3,dim=-1).indices.eq(y[:,None])
  for (h,o,t),loss,cor in zip(ex,aa.cpu().tolist(),correct.cpu().tolist()):
   by[h].append((loss,float(cor[0]),float(any(cor))));band[str(j.band(len(data.seqs[h])))].append(loss);missing+=int(not data.cache[o]['seen'][t])
 mean={h:np.mean(v,axis=0).tolist() for h,v in by.items()};a=float(np.mean([v[0] for v in mean.values()]))
 return {'role':role,'identities':len(mean),'decisions':len(panel),'AA_NLL':a,'perplexity':math.exp(a),'top1':float(np.mean([v[1] for v in mean.values()])),'top3':float(np.mean([v[2] for v in mean.values()])),'per_identity':mean,'length_bands':{k:{'decisions':len(v),'AA_NLL':float(np.mean(v)),'perplexity':float(np.exp(np.mean(v)))} for k,v in band.items()},'AA_at_missing_CA':missing,'seconds':time.monotonic()-start}
def train_partial_panel(data):
 val=json.load(open(R/'configs/PARTIAL_VALIDATION_PANEL.json'));counts=collections.Counter(j.band(len(data.seqs[h])) for h in {x[0] for x in val});rng=random.Random(180041);out=[]
 for k,n in counts.items():
  ids=[h for h in data.ids['TRAIN'] if j.band(len(data.seqs[h]))==k];rng.shuffle(ids)
  for h in sorted(ids[:n]):
   o=data.owners['TRAIN'][h][0];N=len(data.seqs[h]);r=data.cache[o];pos=set(np.rint(np.linspace(0,N-1,8)).astype(int).tolist());miss=np.flatnonzero(~r['seen'])
   if len(miss):pos.add(int(miss[len(miss)//2]))
   for c in range(3):
    av=np.flatnonzero(r['mask'][:,c])
    if len(av):pos.add(int(av[len(av)//2]))
   for t in np.rint(np.linspace(0,N-1,16)).astype(int):
    if len(pos)>=12:break
    pos.add(int(t))
   if len(pos)!=12:raise ValueError('partial panel positions')
   out.extend((h,o,int(t)) for t in sorted(pos))
 return out
@torch.no_grad()
def nonlocal_evaluate(model,data):
 model.eval();set_dropout(model,.15);panel=json.load(open(R/'configs/EVALUATION_PANELS.json'))['VALIDATION'];ex=[]
 for k in range(5):
  ex.extend([tuple(e) for e in panel if j.band(len(data.seqs[e[0]]))==k and e[2]>=24][:32])
 torch.manual_seed(180122);torch.cuda.manual_seed_all(180122);values=[];counts=[]
 for off in range(0,len(ex),4):
  b,y,g=data.batch(ex[off:off+4]);native=torch.tensor(np.asarray([data.base.ca[o][t] for h,o,t in ex[off:off+4]]),device='cuda')
  loss,n=s.direct_nonlocal(model,b,y,native,K=4);values.append(float(loss));counts.extend(n)
 return {'mean_batch_energy_score':float(np.mean(values)),'targets':len(ex),'anchors':int(sum(counts)),'panel':ex,'K':4,'dropout_augmentation':False,'AA_condition':'true current label after encoder','not_independent_labels':True}
def generation_for(model,cp,sha,n=32):
 path=R/'runs/J1'/('final_generation_'+sha[:12]+'.json');rows=json.load(open(path)) if path.exists() else []
 expected={(N,k) for N in [64,128,256,384,500] for k in range(n)};keys={(r['N'],r['sample_index']) for r in rows}
 if len(keys)!=len(rows) or not keys<=expected or any(r['checkpoint_sha256']!=sha for r in rows):raise ValueError('generation cache identity')
 engine=SimpleNamespace(model=model,c={'name':'J1','precision':'BF16','geometry_projection_fp32':True},joint=True)
 from scaled_train import Engine as G
 for N in [64,128,256,384,500]:
  for k in range(n):
   if (N,k) in keys:continue
   with torch.no_grad():r=G.generate(engine,N,2026101700+100*N+k)
   r.update(sample_index=k,checkpoint_sha256=sha)
   if r['status']=='COMPLETE':r['metrics']=metrics(r['generated_CA_A'])
   rows.append(r);write(path,rows);s.status('FINAL_GENERATION',N=N,sample=k,checkpoint_sha256=sha)
 return rows
def check_candidate(summary,baseline,native):
 failures=[]
 for N,b in summary.items():
  p=baseline[N]
  if b['complete_valid']!=32:failures.append(N+': incomplete')
  if b['complete_valid']<2:continue
  if b['unique_sequences']<30:failures.append(N+': sequence collapse')
  if b['clashes_per100_residues']['mean']>1.25*p['clashes_per100_residues']['mean']+.25:failures.append(N+': clash regression')
  ref=np.median([v['metrics']['Rg_over_N_cuberoot'] for v in native['TRAIN'][N]['rows']])
  old=abs(math.log(p['Rg_over_N_cuberoot']['mean']/ref));new=abs(math.log(b['Rg_over_N_cuberoot']['mean']/ref))
  if new>1.2*old+.05:failures.append(N+': Rg regression')
  if b['mean_pair_sequence_Hamming_fraction']<.9*p['mean_pair_sequence_Hamming_fraction']:failures.append(N+': sequence diversity regression')
  if b['mean_pair_Rg_normalized_distance_fingerprint_RMS']<.7*p['mean_pair_Rg_normalized_distance_fingerprint_RMS']:failures.append(N+': structure diversity regression')
 return failures
def assess():
 assert json.load(open(R/'configs/FINAL_SELECTION_FROZEN.json'))['before_first_new_VALIDATION_evaluation'] is True
 j.core.setup();done=json.load(open(R/'runs/J1/COMPLETED.json'));data=PartialData();panels=json.load(open(R/'configs/EVALUATION_PANELS.json'));pt=train_partial_panel(data);write(R/'configs/PARTIAL_TRAIN_PANEL.json',pt)
 write(R/'POPULATION_RECEIPT.json',{'TRAIN_complete_identities':len(data.base.jointids['TRAIN']),'TRAIN_partial_identities':len(data.ids['TRAIN']),'TRAIN_union_identities':len(set(data.base.jointids['TRAIN'])|set(data.ids['TRAIN'])),'TRAIN_partial_observations':len([r for r in data.rows.values() if r['split']=='TRAIN']),'VALIDATION_complete_identities':len(data.base.jointids['VALIDATION']),'VALIDATION_partial_identities':len(data.ids['VALIDATION']),'partial_TRAIN_panel_identities':len({h for h,o,t in pt}),'partial_TRAIN_panel_seed':180041,'TEST_targets':0})
 model,state=j.parent_model();del state;set_dropout(model,.15)
 trainpart=partial_eval(model,data,pt,'TRAIN');valpart=partial_eval(model,data,json.load(open(R/'configs/PARTIAL_VALIDATION_PANEL.json')),'VALIDATION');parent_nonlocal=nonlocal_evaluate(model,data)
 write(R/'PARENT_PARTIAL_EVALUATION.json',{'TRAIN':trainpart,'VALIDATION':valpart,'nonlocal':parent_nonlocal})
 parentgen=json.load(open(j.OLD/'runs/J1-FULL/SELECTED_GENERATION.json'));parentrows=[]
 for r in parentgen:
  v=dict(r);v['metrics']=v['independent_CA_metrics'];parentrows.append(v)
 baseline_summary=metric_summary(parentrows);native=json.load(open(j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json'))['native']
 candidates=[Path(done['latest_checkpoint'])]
 if done['best_update'] and done['best_update']!=done['additional_updates']:candidates.append(R/'runs/J1'/f"checkpoint_{done['best_update']:05}.pt")
 reviewed=[];cache={}
 for cp in candidates:
  sha=file_sha(cp);st=load_checked(cp);model.load_state_dict(st['model'],strict=True);u=st['additional_update'];sampler=st['sampler'];del st;gc.collect();set_dropout(model,.15)
  evaluation=json.load(open(R/'runs/J1'/f'evaluation_{u:05}.json'))
  partialtrain=partial_eval(model,data,pt,'TRAIN');nonlocalv=nonlocal_evaluate(model,data);rows=generation_for(model,cp,sha);summary=metric_summary(rows);fail=check_candidate(summary,baseline_summary,native)
  val=evaluation['VALIDATION'];tr=evaluation['TRAIN'];base=json.load(open(R/'SEQUENCE_BASELINES.json'))['J1'];gap=val['AA_NLL']-tr['AA_NLL'];oldgap=base['VALIDATION']['AA_NLL']-base['TRAIN']['AA_NLL']
  if val['AA_NLL']>=base['VALIDATION']['AA_NLL']:fail.append('no AA validation improvement')
  if val['geometry_NLL']>base['VALIDATION']['geometry_NLL']+.10:fail.append('local geometry safeguard')
  if gap>oldgap:fail.append('generalization gap not narrowed')
  paired=paired_ci([val['per_identity'][h]['AA_NLL']-base['VALIDATION']['per_identity'][h]['AA_NLL'] for h in sorted(val['per_identity'])])
  comparison={}
  for N in [64,128,256,384,500]:
   old=sorted([r for r in parentrows if r['N']==N],key=lambda v:v['sample_index']);new=sorted([r for r in rows if r['N']==N],key=lambda v:v['sample_index']);ref=np.median([v['metrics']['Rg_over_N_cuberoot'] for v in native['TRAIN'][str(N)]['rows']])
   comparison[str(N)]={'clash_delta':paired_ci([b['metrics']['clashes_per100_residues']-a['metrics']['clashes_per100_residues'] for a,b in zip(old,new) if 'metrics' in b]),'native_log_Rg_error_delta':paired_ci([abs(math.log(b['metrics']['Rg_over_N_cuberoot']/ref))-abs(math.log(a['metrics']['Rg_over_N_cuberoot']/ref)) for a,b in zip(old,new) if 'metrics' in b])}
  v={'update':u,'checkpoint':str(cp),'sha256':sha,'passed':not fail,'failures':fail,'TRAIN':tr,'VALIDATION':val,'partial_TRAIN':partialtrain,'partial_VALIDATION':evaluation['PARTIAL_VALIDATION'],'nonlocal':nonlocalv,'AA_validation_paired_change':paired,'TRAIN_VALIDATION_gap':gap,'free_generation':summary,'paired_free_generation_change':comparison}
  reviewed.append(v);write(R/'FINAL_CANDIDATE_REVIEWS.json',reviewed)
  if cp==Path(done['latest_checkpoint']):
   actual=sum(x.bit_count() for x in sampler['coverage']['visited'].values());expected=0
   for key,n in sampler['coverage']['cursor'].items():
    if key.startswith('AA:'):
     h=key[3:];N=len(data.seqs[h]);expected+=N*(-math.expm1(n*math.log1p(-1/N)))
   coverage={**done['coverage'],'same_identity_visit_counts_with_replacement_expected_unique_positions':expected,'actual_distinct_positions':actual,'excess_unique_positions_over_expected':actual-expected,'comparison_scope':'Analytic expectation with same per-identity counts under uniform replacement; not a separately trained arm'}
   write(R/'POSITION_COVERAGE_REPORT.json',coverage)
 passing=[v for v in reviewed if v['passed']];selected=min(passing,key=lambda v:v['VALIDATION']['AA_NLL']) if passing else None
 selection={'selected_checkpoint':selected['checkpoint'] if selected else str(j.PARENT),'selected_sha256':selected['sha256'] if selected else j.PARENT_SHA,'selected_additional_update':selected['update'] if selected else 0,'selection_reason':'lowest AA NLL satisfying frozen geometry/diversity/gap safeguards' if selected else 'immutable original fallback; no candidate satisfies all frozen safeguards','candidates':[{'update':v['update'],'passed':v['passed'],'failures':v['failures']} for v in reviewed],'TEST':'NOT_EVALUATED'}
 write(R/'FINAL_SELECTION.json',selection);write(R/'NATIVE_VS_GENERATED_GEOMETRY.json',{'native_reference_path':str(j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json'),'native_reference_sha256':file_sha(j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json'),'baseline':baseline_summary,'candidates':{str(v['update']):{'summary':v['free_generation'],'paired_change':v['paired_free_generation_change']} for v in reviewed},'historical_60k_path':str(PREVIOUS/'NATIVE_VS_GENERATED_GLOBAL_GEOMETRY.json'),'historical_sha256':file_sha(PREVIOUS/'NATIVE_VS_GENERATED_GLOBAL_GEOMETRY.json')})
 # TRAIN-only gradient panel, same implementation and no optimizer; reported recalibration is diagnostic only.
 import j1_sequence_preflight as pf
 original=pf.write
 def redirect(p,v):
  name=Path(p).name
  if name=='GRADIENT_BALANCE_REPORT.json':
   actual=json.load(open(R/'configs/FROZEN.json'));v['weights_applied']= {k:val for k,val in actual.items() if k.startswith('lambda_')}
   v['hypothetical_recalibration_NOT_APPLIED']=v.pop('weights')
   aa=v['loss_gradients']['AA']['shared_all']['L2'];v['actual_weighted_shared_ratios']={k:actual['lambda_'+k]*v['loss_gradients'][k]['shared_all']['L2']/aa for k in ['local','nonlocal','Rg','clash','pair']}
   p=R/'FINAL_GRADIENT_BALANCE_DIAGNOSTIC.json'
  elif name=='GRADIENT_TRAIN_PANEL.json':p=R/'configs/FINAL_GRADIENT_TRAIN_PANEL.json'
  original(p,v)
 pf.write=redirect
 latest=load_checked(done['latest_checkpoint']);model.load_state_dict(latest['model'],strict=True);del latest;pf.profile(model,data)
 write(R/'ASSESSMENT_COMPLETE.json',{'status':'DONE','source_commit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip(),'selection':selection,'time':time.time()})
 print('SEQUENCE_FIRST_ASSESSMENT_COMPLETE',flush=True)
if __name__=='__main__':assess()
