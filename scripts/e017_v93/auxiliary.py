"""CPU checks of KL objective/RLOO and future TRAIN eligibility census."""
import itertools,math,runpy,torch,numpy as np
from pathlib import Path
R=Path('/mnt/d/Simone/proteinGen_data/e017_v93_regional_controllability_20261024_v1');d=runpy.run_path(str(R/'source/scripts/e017_v93/common.py'));t=runpy.run_path(str(R/'source/scripts/e017_v93/tests.py'));P=runpy.run_path(str(R/'source/scripts/e017_v93/prepare.py'));torch.set_default_dtype(torch.float64)
def main():
 theta=torch.tensor([[.3,-.1,.2],[-.2,.4,.1]],requires_grad=True);lp=t['evaluate'](theta);pr=lp.exp();ref=t['evaluate'](torch.zeros_like(theta));rewards=torch.tensor([z[2] for z in t['traces']]);beta=.07;llr=lp-ref;objective=(pr*(rewards-beta*llr)).sum();exact=torch.autograd.grad(objective,theta,retain_graph=True)[0];aug=(rewards-beta*llr).detach();joint=pr[:,None].detach()*pr[None,:].detach();ad=aug[:,None]-aug[None,:];loss=-.5*ad*(lp[:,None]-lp[None,:])+.5*beta*(llr[:,None]+llr[None,:]);estimate=-torch.autograd.grad((joint*loss).sum(),theta)[0];assert torch.allclose(exact,estimate,atol=1e-12)
 # Goal public projection explicitly removes metadata only; unknown semantic
 # overrides still raise in canonical(), preventing silent field loss.
 tasks=d['read'](R/'manifests/PROSPECTIVE_FIXED.json')['tasks'];n=0
 for task in tasks:
  original=task['goal'];w=d['world'](original);assert all(w.goal[k]==v for k,v in original.items() if k not in ['difficulty','sequence_synthetic_motif_reversal']);assert d['g']['semantic_sha'](w.goal)==d['g']['semantic_sha']({k:v for k,v in original.items() if k not in ['difficulty','sequence_synthetic_motif_reversal']});n+=1
 try:d['world'](dict(tasks[0]['goal'],unsupported_new_constraint='must-not-ignore'))
 except ValueError:pass
 else:raise AssertionError('unrecognized semantics silently accepted')
 d['save']('KL_GRADIENT_AND_SCHEMA_TESTS.json',dict(PASS=True,beta=beta,exact_gradient=exact.tolist(),group2_RLOO_exact_gradient=estimate.tolist(),max_error=float((exact-estimate).abs().max()),enumerated_independent_groups=len(t['traces'])**2,public_goal_checks=n,unknown_semantics_fail_closed=True))
 original=d['read'](R/'manifests/TRAIN_TASKS.json')['tasks'];dev_ids={q['source_reference_identity'] for q in original};dev_groups={q['goal']['protected_group'] for q in original};rows=d['read'](d['V9']/'manifests/TRAIN_REFERENCE.json')['records'];pool=[];rejected=[]
 for row in rows:
  if row['identity'] in dev_ids or row['group'] in dev_groups:continue
  for level in ['witness','relaxed']:
   item=P['make'](row,'TRAIN_FUTURE',level)
   if item:pool.append(item)
   elif level=='witness':rejected.append(row['identity'])
 assert len(pool)>0
 d['save']('manifests/FUTURE_FIXED_TRAIN_POOL.json',dict(tasks=pool,identities=len({x['source_reference_identity'] for x in pool}),protected_groups=len({x['goal']['protected_group'] for x in pool}),source_identities=512,development_TRAIN_identities_reserved=16,excluded_ineligible=len(rejected),source_SHA256=d['sha'](d['V9']/'manifests/TRAIN_REFERENCE.json'),native_witnessed_goals_only=True,coordinates_not_supplied_to_controller=True,not_evaluated_or_trained_inV93=True))
 print('AUX_PASS','future TRAIN',len(pool),len({x['source_reference_identity'] for x in pool}),flush=True)
if __name__=='__main__':main()
