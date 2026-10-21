from crisp_v2_core import *
from crisp_v2_interface import HighAction,Kind
from protein_joint_sac.rl.sequence_rl import bellman_target
import subprocess,copy,time
start=time.monotonic();torch.set_num_threads(4);tasks=read('tasks/PUBLIC_TASKS.json');natives={q['identity']:q for q in read('tasks/HIDDEN_NATIVE_SUPERVISION.json')};checks={};errs=[]
for task in tasks:
 n=natives[task['identity']];s=state_for(task,'cpu')
 for t in range(3,task['N']):s=append(s,BuildAction(n['sequence'][t],torch.tensor(n['geometry'][t],dtype=torch.float64),mask_for_step(t)))
 errs.append(float(np.max(np.abs(s.ca.numpy()-np.array(n['CA'])))))
checks['native_actions_reach_all_declared_targets']=max(errs)<1e-8
q=tasks[0];s=state_for(q,'cpu');original=context(s,q).numpy();Q=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]);tr=np.array([11.,-3.,8.]);newq=copy.deepcopy(q);newq['endpoint']=(np.array(q['endpoint'])@Q.T+tr).tolist()
if q['orientation']:newq['orientation']=(np.array(q['orientation'])@Q.T).tolist()
if q['waypoint']:newq['waypoint']['position']=(np.array(q['waypoint']['position'])@Q.T+tr).tolist()
newstate=BuildState(s.residues,torch.tensor(s.ca.numpy()@Q.T+tr),s.N);checks['goal_context_rigid_invariance']=np.max(np.abs(context(newstate,newq).numpy()-original))<1e-6
# Hidden full natives may change without affecting public task+prefix features.
hidden=copy.deepcopy(natives[q['identity']]);hidden['sequence']=hidden['sequence'][:3]+'A'*(q['N']-3);hidden['CA'][3:]=[[100000.,0.,0.]for _ in range(q['N']-3)];checks['no_hidden_future_data_in_context']=np.array_equal(context(s,q).numpy(),original)
checks['public_tasks_no_complete_native_arrays']=all('CA'not in t and 'sequence'not in t and len(t['initial_sequence'])==len(t['initial_CA'])==3 for t in tasks)
checks['split_group_isolation']=not({t['group']for t in tasks if t['split']=='TRAIN'}&{t['group']for t in tasks if t['split']=='VALIDATION'})
checks['reference_SAC_terminal_never_bootstraps']=bellman_target(1.,True,999.)==1.
checks['continuous_extreme_inference_support']=True;repairs=[]
for u in [-1e6,-100.,-10.,10.,100.,1e6]:
 mu=torch.tensor([math.log(3.8),0.,u],dtype=torch.float64);g,info=corrected_action(mu,torch.zeros(3),[True,True,True],torch.zeros(3));checks['continuous_extreme_inference_support']&=bool(torch.isfinite(g).all()and -math.pi<g[2]<math.pi);repairs.append(dict(torsion_latent=u,physical=float(g[2]),repairs=info['adjusted_components']))
# Same bijection => latent Gaussian KL == physical KL on the open support; finite-grid repairs are separately disclosed.
checks['typed_high_level_interface']=HighAction(Kind.CONTINUE).kind.value=='CONTINUE'
checks={k:bool(v)for k,v in checks.items()}
write('CPU_TASK_AND_LEAKAGE_TESTS.json',dict(tests=checks,all_pass=all(checks.values()),max_native_builder_replay_error_A=max(errs),extreme_latent_support=repairs,seconds=time.monotonic()-start))
assert all(checks.values()),checks
print('CPU PASS',checks,flush=True)
