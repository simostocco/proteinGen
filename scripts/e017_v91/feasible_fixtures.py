"""TRAIN-only engineering fixture eligibility from exact frozen residual bounds."""
import copy,itertools,json,runpy,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v91_ordered_goal_recovery_20261023_v1');p=runpy.run_path(str(R/'source/scripts/e017_v91/policy.py'));c=p['c'];read=c['read'];save=c['save'];AA=c['AA'];OLD=p['OLD']
def bounds(logits,label):
 z=logits.double();other=z.clone();other[label]=-float('inf');gap=float(other.max()-z[label]);best=z-3;best[label]+=6;return dict(top_minus_target=gap,maximum_target_probability=float(best.softmax(-1)[label]),greedy_reachable=gap<6)
def make():
 torch.set_num_threads(1);test=read(R/'GOAL_OBSERVABILITY_TESTS.json');assert test['PASS'];assert all(c['sha'](R/'source/scripts/e017_v91'/n)==h for n,h in test['source_SHA256'].items());prior=p['legacy']['FrozenPrior']('cpu');spec=read(R/'REWARD_SPECIFICATION.json');files=sorted((OLD/'results').glob('TRAIN_B_V3_[0-9][0-9][0-9].json'),key=lambda f:(read(f)['goal']['exposed'][0],read(f)['goal']['witness_identity']));groups=[];checked=[];selected_pairs=set()
 for source_path in files[:16]:
  source=read(source_path);raw=source['goal'];a=raw['exposed'][0];valid=[r for r in source['records'] if r['metrics']['hard_valid']]
  if not valid:continue
  row=valid[0];w=c['PropertyWorld'](raw,spec,1)
  for t in range(a):w.append(row['sequence'][t],row['geometry'][t])
  with torch.no_grad():ref=prior(p['collate']([w.state],'cpu'))
  logits=ref['logits'][0];first=[j for j in range(20) if bounds(logits,j)['maximum_target_probability']>=.75 and bounds(logits,j)['top_minus_target']<4.5];first=sorted(first,key=lambda j:(-float(logits[j]),j))[:12];follow=[]
  for j in first:
   z=copy.deepcopy(w);z.append(AA[j],row['geometry'][a]);follow.append(z)
  if not follow:checked.append(dict(source=str(source_path),reason='no eligible first actions'));continue
  with torch.no_grad():nextref=prior(p['collate']([z.state for z in follow],'cpu'))
  index={aa:i for i,aa in enumerate(first)};pairs=[]
  for j,k in itertools.combinations(first,2):
   pair=tuple(sorted((AA[j],AA[k])))
   if pair in selected_pairs:continue
   proof=[bounds(logits,j),bounds(nextref['logits'][index[j]],k),bounds(logits,k),bounds(nextref['logits'][index[k]],j)]
   if all(q['maximum_target_probability']>=.75 and q['top_minus_target']<4.5 for q in proof):pairs.append((min(q['maximum_target_probability'] for q in proof),pair,proof))
  checked.append(dict(source=str(source_path),first_eligible_AAs=[AA[j] for j in first],feasible_reversible_pairs=len(pairs)))
  for margin,pair,proof in sorted(pairs,key=lambda q:(-q[0],q[1])):
   if len(groups)==2:break
   if pair in selected_pairs:continue
   selected_pairs.add(pair);motif=''.join(pair);groups.append(dict(source=str(source_path),candidate_index=row['candidate_index'],goal=raw,motifs=[motif,motif[::-1]],position=a,decision_support_proofs=proof,minimum_target_probability_upper_bound=margin))
  if len(groups)==2:break
 prior.verify();save('FEASIBLE_CONTROLLABILITY_FIXTURES.json',dict(PASS=len(groups)==2,groups=groups,checked=checked,maximum_TRAIN_prefixes=16,selection='deterministic TRAIN-only max-min frozen action-support margin, no learner outcomes; exactly2 reversible pairs/4 goals/8 decisions',admission='every target top-minus-target logit<4.5 and oracle maximum probability>=.75 under unchanged +/-3 residual law',initial_ineligible_attempt_preserved='CONTROLLABILITY_TEST.json and CONTROLLABILITY_FAILURE_AUDIT.json',no_optimizer_updates=True,no_GPU=True));assert len(groups)==2,'bounded eligible fixture construction failed';print(json.dumps(dict(PASS=True,groups=[dict(motifs=z['motifs'],position=z['position'],minimum_upper_bound=z['minimum_target_probability_upper_bound']) for z in groups])))
if __name__=='__main__':make()
