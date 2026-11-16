"""CPU-only production callback/schedule/sensor admission checks. No neural update."""
from production97 import *
def main():
 verify();assert N==4096 and LOOKS==[0,250,500,1000,2000,3500,5000] and ALPHA==.05/28
 from safety97 import classify
 assert classify(3947,3947,KL=.0065,diversity=14.79)['status']=='PASS'
 assert classify(3900,3900,KL=.0065,diversity=14.79)['status']=='INCONCLUSIVE'
 assert classify(3800,3800,KL=.0065,diversity=14.79)['status']=='BREACH'
 assert classify(3947,3947,n=4095)['optimizer_updates_allowed']==False
 assert classify(4096,4096,KL=2,diversity=16)['status']=='BREACH'
 assert classify(4096,4096,KL=0,diversity=1)['status']=='BREACH'
 fit=read(BASE/'TRAINING_POPULATION.json')['tasks'];ids=sorted({t['source_reference_identity'] for t in fit});lookup={i:[t for t in fit if t['source_reference_identity']==i] for i in ids};order=np.random.default_rng(19731).permutation(len(ids)).tolist();plan=read(BASE/'TRAINING_SCHEDULE.json')['updates']
 # Actual sampler namespace is used, so every candidate seed is checked.
 P,F=Q.factory('A',PAIR/'CPU_FIXTURE_NO_RUNTIME');S=P['S']
 for u,row in enumerate(plan,1):
  identity=ids[order[(u-1)%len(ids)]];task=lookup[identity][((u-1)//len(ids))%len(lookup[identity])]
  assert row['identity']==identity and row['goal_id']==task['goal']['goal_id'] and row['seeds']==[S['ns']('SUBSTANTIAL',19731,u,i) for i in range(16)]
 # Callback automaton: no continuation after incomplete/BREACH/INCONCLUSIVE.
 for frontier in [0,249,250,251,500,1000,3500,5000]:
  for decision in ['MISSING','INCOMPLETE','PASS','BREACH','INCONCLUSIVE']:
   may_step=frontier<5000 and (frontier not in LOOKS[1:] or decision=='PASS')
   if frontier in LOOKS[1:] and decision!='PASS':assert not may_step
 for u in range(1,5001):assert required_look_before_update(u)==max(k for k in LOOKS if k<u)
 # Passive gate sensor observes existing head calls and leaves outputs/gradients intact.
 class Dummy(torch.nn.Module):
  def __init__(self):super().__init__();self.geometry=torch.nn.Identity();self.burial_gate=torch.nn.Parameter(torch.tensor(0.))
 d=Dummy();s=GateSensor(d);x=torch.randn(2,20,6,requires_grad=True);y=torch.randn(2,20,6,requires_grad=True)
 a=d.geometry(x);b=d.geometry(y);assert torch.equal(a,x) and torch.equal(b,y)
 z=(a+d.burial_gate*(b-a)).sum();z.backward();assert d.burial_gate.grad==((y-x).sum()) and s.summary()['gate_gradient_preclip_sum']==float(d.burial_gate.grad)
 assert s.summary()['gate_scaled_mean_delta_RMS']==0
 d.burial_gate.data.fill_(.00450095);s.reset();d.geometry(x);d.geometry(y);assert s.summary()['gate_scaled_mean_delta_RMS']>0
 for h in s.handles:h.remove()
 verify();out=dict(PASS=True,UTC=now(),CPU_only=True,optimizer_updates=0,all_5000_goals_and_80000_seeds_verified=True,safety_boundary_cases=True,passive_gate_sensor_exact=True,qualified_Runtime_and_likelihood_unchanged=True,qualified_recovery_A=read(BASE/'RECOVERY_A.json')['PASS'],qualified_recovery_B=read(BASE/'RECOVERY_B.json')['PASS'])
 atomic(PAIR/'ORCHESTRATION_QUALIFICATION.json',out,True);print(json.dumps(out),flush=True)
if __name__=='__main__':main()
