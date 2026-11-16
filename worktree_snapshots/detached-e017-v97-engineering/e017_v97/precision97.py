"""Additional literal-byte qualification; no optimizer and no candidate generation."""
from core97 import *
import torch,copy,dataclasses,subprocess
verify();extra=read(ROOT/'LITERAL_BITS_TEST_BINDING.json');assert sha(__file__)==extra['source_SHA256'];assert sha(ROOT/'SOURCE_BINDING.json')==extra['base_binding_SHA256']
assert not subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
P,F=factory('B',ROOT/'qualification');e,_=P['K']['load_engine']('C');e.actor.eval();base=e.actor._v97_original_forward;tested=[]
for v in read(ROOT/'INITIAL_EQUIVALENCE.json')['fixtures']:
 i,N,t=v['TRAIN_goal'],v['N'],v['t'];obj=read(PAIR.parent/f'e017_v96_arm_a_original_20261029_v1/production/diagnostics/DEV_00000_filtered_{i:02d}.json');state=P['S']['reconstruct'](obj['records'][0],64,t);state=dataclasses.replace(state,N=N);goal=copy.deepcopy(obj['task']['goal']);goal['N']=N;goal['goal_id']+='_ENGINEERING_N'+str(N)
 if N==32:goal.update(exposed=[4,12],buried=[20,28],motif=obj['task']['fixed_sequence'][4:6])
 w=P['S']['world'](goal,state);prefix=P['p']['collate']([state],'cuda')
 with torch.no_grad():ref=e.prior(prefix);b=P['p']['public_batch']([w],prefix,ref);before=b['goal_valid'].clone();a=base(b);z=e.actor(b)
 assert torch.equal(before,b['goal_valid'])
 flags=[torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)) for x,y in zip(a,z)];assert all(flags),(i,N,t,flags)
 tested.append(dict(TRAIN_goal=i,N=N,t=t,AA_mu_logstd_literal_bytes_equal=flags,original_goal_masks_unchanged=True))
e.prior.verify();save('LITERAL_BITWISE_EQUIVALENCE.json',dict(PASS=True,fixtures=tested,new_generation_attempts=0,optimizer_calls=0,gate_source_SHA256=sha(WT/'e017_v97/gate97.py')))
print('LITERAL_BITWISE_EQUIVALENCE_PASS',len(tested),flush=True)
