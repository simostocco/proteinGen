from crisp_v2_core import *
from crisp_v2_interface import *
import time
start=time.monotonic();actor=load_actor();adapter=Steering().cuda();adapter.load_state_dict(torch.load(R/'runs/adapter_00080.pt',weights_only=False)['adapter']);task=read('tasks/PUBLIC_TASKS.json')[0];e=Hierarchy(actor,adapter,task,791,aa=False,geometry=True);w=ProteinWorld(e);tests={};obs=w.observation();old=copy.deepcopy(e.task);obs['goal']['endpoint'][0]+=999;obs['intervention_results'].append(dict(fake=True));tests['observation_cannot_mutate_goal_or_history']=e.task==old and len(e.interventions)==0
r=w.transition(HighAction(Kind.CONTINUE,arguments=dict(steps=3)));t=r['transition'];tests['semi_MDP_transition_exposes_rewards_duration']=t['duration']==3 and len(t['primitive_rewards'])==3 and abs(t['option_reward']-sum(.995**i*v for i,v in enumerate(t['primitive_rewards'])))<1e-12 and t['discount']==.995**3 and not t['terminated']
q=copy.deepcopy(e.export());clone=Hierarchy(actor,Steering().cuda(),task,1,aa=True,geometry=False);clone.restore(q);tests['full_recovery_restores_policy_gates_and_adapters']=clone.aa==False and clone.geometry==True and all(torch.equal(v,clone.adapter.state_dict()[k])for k,v in e.adapter.state_dict().items());e.act('CONTINUE',steps=2);clone.act('CONTINUE',steps=2);tests['gated_policy_restart_same_actions']=torch.equal(e.state.ca,clone.state.ca)and e.state.residues==clone.state.residues
r=w.transition(HighAction(Kind.STOP));tests['STOP_exposes_terminal_failure']=r['transition']['terminated']and not r['transition']['accepted']
for m in ['crisp_v2_common','crisp_v2_core','crisp_v2_interface','crisp_v2_tasks','crisp_v2_pilot','crisp_v2_evaluate','crisp_v2_cpu_checks','crisp_v2_extended','crisp_v2_publish']:
 p=R/'source/scripts/e016_v3'/f'{m}.py';compile(p.read_text(),str(p),'exec')
write('HIGH_LEVEL_INTERFACE_TESTS.json',dict(tests=tests,all_pass=all(tests.values()),seconds=time.monotonic()-start));assert all(tests.values()),tests
print('INTERFACE PASS',tests,flush=True)
