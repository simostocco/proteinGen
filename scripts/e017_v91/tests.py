"""Deterministic CPU observability, reward, gradient and leak tests."""
import ast,copy,inspect,itertools,json,runpy,sys,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v91_ordered_goal_recovery_20261023_v1');p=runpy.run_path(str(R/'source/scripts/e017_v91/policy.py'));g=p['g'];c=p['c'];read=c['read'];save=c['save'];AA=c['AA'];OLD=p['OLD']
from protein_joint_sac.crisp_v3.density import log_prob

def checktests():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.perf_counter();spec=read(R/'REWARD_SPECIFICATION.json');goal=read(R/'manifests/TRAIN_GOALS.json')['goals'][0];base=g['canonical'](goal);cases={};m=base['motif'];a,z=m['allowed_region']
 for name in ['order','identity','position','position_shift','spacing','exposure','burial','region','length']:
  v=copy.deepcopy(base)
  if name=='order':v['motif']['tokens']=v['motif']['tokens'][::-1]
  if name=='identity':v['motif']['tokens'][0]=next(x for x in AA if x not in m['tokens'])
  if name=='position':v['motif']['absolute_positions']=[a,a+1]
  if name=='position_shift':v['motif']['absolute_positions']=[a+1,a+2]
  if name=='spacing':v['motif']['gaps']=[[1,1]]
  if name=='exposure':v['regions']['exposed']['threshold_units']+=500 if v['regions']['exposed']['threshold_units']<9500 else -500
  if name=='burial':v['regions']['buried']['threshold_units']+=500 if v['regions']['buried']['threshold_units']<9500 else -500
  if name=='region':v['regions']['buried']['interval']=[max(0,v['regions']['buried']['interval'][0]-1),v['regions']['buried']['interval'][1]]
  if name=='length':v['N']=65
  cases[name]=g['canonical'](v)
 def reversekeys(x):
  if isinstance(x,dict):return {k:reversekeys(v) for k,v in reversed(list(x.items()))}
  if isinstance(x,list):return [reversekeys(v) for v in x]
  return x
 equivalent=reversekeys(copy.deepcopy(base));assert g['serialize'](base)==g['serialize'](equivalent) and g['semantic_sha'](goal)==g['semantic_sha'](base)
 source=read(OLD/'results/TRAIN_B_V3_000.json')['records'][0];world=c['PropertyWorld'](goal,spec,source['seed']);checks=[];failures=[]
 def check(name,value,detail=None):
  checks.append(dict(name=name,PASS=bool(value),detail=detail))
  if not value:failures.append(name)
 def features(state,gg):
  w=c['PropertyWorld'](gg,spec,1);w.state=copy.deepcopy(state);prefix=p['collate']([w.state],'cpu');ref=dict(h=torch.zeros(1,352),logits=torch.zeros(1,20),mu=torch.zeros(1,20,3),ls=torch.zeros(1,20,3));return g['public_batch']([w],prefix,ref)
 def fingerprint(batch):
  tokens=sorted(batch['goals'][0,batch['goal_valid'][0]].tolist());return json.dumps(dict(tokens=tokens,context=batch['context'].tolist()),sort_keys=True)
 representation_checks=0
 for t in range(65):
  original=features(world.state,base);key=fingerprint(original);check('equivalent_features_t'+str(t),key==fingerprint(features(world.state,equivalent)))
  for name,gg in cases.items():check('distinct_'+name+'_t'+str(t),key!=fingerprint(features(world.state,gg)));representation_checks+=1
  if t<64:world.append(source['sequence'][t],source['geometry'][t])
 # Incorrect scope/extra fields and sub-grid differences fail closed.
 invalid=[]
 v=copy.deepcopy(goal);v['absolute_positions']=[a,a+1];invalid.append(v)
 v=copy.deepcopy(base);v['motif']['absolute_positions']=[a+1,a];invalid.append(v)
 v=copy.deepcopy(goal);v['native_future_coordinates']=[[0,0,0]];invalid.append(v)
 v=copy.deepcopy(goal);v['exposure_min']+=1e-6;invalid.append(v)
 for i,v in enumerate(invalid):
  rejected=False
  try:g['canonical'](v)
  except (ValueError,TypeError):rejected=True
  check('invalid_or_unsupported_semantics_rejected_'+str(i),rejected)
 for name in ['position','position_shift','spacing']:
  rejected=False
  try:g['production_raw'](cases[name])
  except ValueError:rejected=True
  check('engineering_extensions_not_silently_scored_as_original_'+name,rejected)
 # Completed sequence constraints agree exactly with original V9 definitions.
 rng=np.random.default_rng(9191);witness=read(OLD/'manifests/GOAL_WITNESSES_PRIVATE.json');reward_checks=0
 for raw in read(R/'manifests/TRAIN_GOALS.json')['goals']:
  canon=g['canonical'](raw);w=witness[raw['goal_id']];x=np.array(w['coordinates']);expected=c['score'](w['sequence'],x,raw,spec);other=c['score'](w['sequence'],x,g['production_raw'](canon),spec);check('native_reward_unchanged_'+raw['goal_id'],expected==other and expected['success']);reward_checks+=1
  for j in range(32):
   seq=''.join(rng.permutation(list(w['sequence'])));new=g['sequence_progress'](seq,canon)['utility'];old=c['motif_utility'](seq,raw);check('ordered_sequence_evaluator_'+raw['goal_id']+'_'+str(j),new==old)
  token1,token2=raw['motif'];fill=next(t for t in AA if t not in raw['motif']);seq=fill*raw['N'];aa,zz=raw['exposed'];seq=seq[:aa]+raw['motif']+seq[aa+2:];back=copy.deepcopy(raw);back['motif']=raw['motif'][::-1];check('reward_order_not_bag_'+raw['goal_id'],c['motif_utility'](seq,raw)==1 and c['motif_utility'](seq,back)<1)
 check('reward_spec_exact_SHA',c['sha'](R/'REWARD_SPECIFICATION.json')==c['sha'](OLD/'REWARD_SPECIFICATION.json'))
 check('reward_source_AST_unchanged',ast.dump(ast.parse(inspect.getsource(c['score'])))==ast.dump(ast.parse(inspect.getsource(p['legacy']['c']['score']))))
 # Legitimate schema only; no native witness tensor/IDs in feature generation.
 empty=c['PropertyWorld'](goal,spec,1);before=fingerprint(features(empty.state,base));private=copy.deepcopy(witness);private[goal['goal_id']]['coordinates']=np.full((64,3),999.).tolist();private[goal['goal_id']]['sequence']='W'*64;check('private_future_perturbation_no_features',before==fingerprint(features(empty.state,base)))
 metadata=copy.deepcopy(goal);metadata['witness_identity']='different';metadata['protected_group']='different';metadata['goal_id']='different';check('audit_IDs_not_neural_features',fingerprint(features(empty.state,metadata))==before)
 # Real parent/J1 neural forward and gradients; CPU only, no training updates.
 assert c['sha'](c['J1'])==c['J1SHA'] and c['sha'](c['PARENT'])==c['PSHA'];prior=p['legacy']['FrozenPrior']('cpu');state=torch.load(c['PARENT'],map_location='cpu',weights_only=False);weights={k[len('models.actor.'):]:v for k,v in state['system'].items() if k.startswith('models.actor.')};actor=p['legacy']['LowActor'](p['legacy']['PROFILES']['research']['actor']).eval();actor.load_state_dict(weights,strict=True);del state,weights;initial=c['state_digest'](actor.state_dict());neural=[]
 def realbatch(world):
  prefix=p['collate']([world.state],'cpu')
  with torch.no_grad():ref=prior(prefix)
  return g['public_batch']([world],prefix,ref)
 for t in [0,3]:
  worlds=[]
  for gg in [base,cases['order']]:
   w=c['PropertyWorld'](gg,spec,1)
   for j in range(t):w.append(source['sequence'][j],source['geometry'][j])
   worlds.append(w)
  batches=[realbatch(w) for w in worlds];outs=[actor(b) for b in batches];neural.append(dict(t=t,encoded_goal_difference=float((batches[0]['goals']-batches[1]['goals']).abs().max()),AA_logits_difference=float((outs[0][0]-outs[1][0]).abs().max().detach()),geometry_mu_difference=float((outs[0][1]-outs[1][1]).abs().max().detach())))
  actor.zero_grad(set_to_none=True);out=actor(batches[0]);label=AA.index(base['motif']['tokens'][0]);loss=-out[0].log_softmax(-1)[0,label];loss.backward();aa_grad=float(actor.encoder.goal.weight.grad.norm());check('AA_goal_projection_grad_t'+str(t),mathfinitepositive(aa_grad),aa_grad)
  if t==3:
   actor.zero_grad(set_to_none=True);out=actor(batches[0]);aa=AA.index(source['sequence'][3]);physical=torch.tensor([source['geometry'][3]],dtype=torch.float64);mask=torch.ones(1,3,dtype=torch.bool);loss=-log_prob(out[1][:,aa].double(),out[2][:,aa].double(),physical,mask).mean();loss.backward();geom_grad=float(actor.encoder.goal.weight.grad.norm());check('geometry_goal_projection_grad_active_t3',mathfinitepositive(geom_grad),geom_grad);check('geometry_head_grad',actor.geometry[-1].weight.grad is not None and float(actor.geometry[-1].weight.grad.norm())>0)
 # A zero residual head can hide behavioral sensitivity without erasing features.
 zero=copy.deepcopy(actor).eval()
 with torch.no_grad():zero.aa.weight.zero_();zero.aa.bias.zero_();zero.geometry[-1].weight.zero_();zero.geometry[-1].bias.zero_();z=[zero(b) for b in batches]
 check('zero_head_identical_behavior_is_not_representation_failure',all(torch.equal(x,y) for x,y in zip(*z)) and fingerprint(batches[0])!=fingerprint(batches[1]));actor.zero_grad(set_to_none=True);prior.verify();check('parent_actor_unchanged_by_tests',c['state_digest'](actor.state_dict())==initial);check('J1_no_parameter_gradients',all(q.grad is None for q in prior.model.parameters()))
 save('GOAL_OBSERVABILITY_TESTS.json',dict(PASS=not failures,checks=checks,failures=failures,representation_pair_checks=representation_checks,stages=65,neural_prelearning_sensitivity=neural,production_reward_checks=reward_checks,reward_AST_and_hash_unchanged=True,actor_parameters=sum(v.numel() for v in actor.parameters()),parent_fingerprint=initial,J1_fingerprint=prior.identity,J1_file_SHA256=c['J1SHA'],parent_file_SHA256=c['PSHA'],no_optimizer_updates=True,GPU_used=False,seconds=time.perf_counter()-start,partial_progress_note='Canonical partial progress may expose one known correct motif residue sooner than the legacy utility; it is observation only. Completed sequence utility and production reward remain exactly unchanged.'))
 save('OBSERVABILITY_ADMISSION.json',dict(PASS=not failures,failures=failures,controllability_pending=True,RL_authorized=False));print(json.dumps(dict(PASS=not failures,checks=len(checks),failures=failures,neural=neural)))
def mathfinitepositive(x):return np.isfinite(x) and x>1e-12
if __name__=='__main__':checktests()
