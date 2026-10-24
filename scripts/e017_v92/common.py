"""Read-only V9.1 components and exact irreversible-prefix predicates."""
import collections,copy,hashlib,json,math,runpy,time,os
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v92_prefix_feasibility_20261024_v1');OLD=R.parent/'e017_v91_ordered_goal_recovery_20261023_v1';V9=R.parent/'e017_v9_property_steering_20261023_v1';p=runpy.run_path(str(OLD/'source/scripts/e017_v91/policy.py'));c=p['c'];g=p['g'];read=c['read'];sha=c['sha'];AA=c['AA'];SPEC=read(OLD/'REWARD_SPECIFICATION.json');LIMITS=SPEC['hard_backbone_limits'];assert c['LIMITS']==LIMITS

def save(name,obj):
 path=R/name;tmp=path.with_name(path.name+'.part');assert not tmp.is_symlink()
 with tmp.open('w') as f:json.dump(obj,f,separators=(',',':'),allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def ns(*items):return int(hashlib.sha256(('E017V92|9241|'+'|'.join(map(str,items))).encode()).hexdigest()[:15],16)
def append(state,aa,geom):return c['append_state'](state,c['BuildAction'](aa,torch.tensor(geom,dtype=torch.float64),c['mask_for_step'](state.t)))
def prefix_status(state):
 x=state.ca.numpy();n=state.t;N=state.N;finite=bool(np.isfinite(x).all());bad=[];pair_count=0;contact_edges=0;rg=None;current_contact=0.;spacing=True;frames=True
 if not finite:bad.append('nonfinite_coordinates')
 if finite and n>1:
  bonds=np.diff(x,axis=0);d=np.linalg.norm(bonds,axis=-1);spacing=bool(d.min()>=LIMITS['adjacent_low_A'] and d.max()<=LIMITS['adjacent_high_A'])
  if not spacing:bad.append('local_spacing')
  if n>2:frames=bool((np.linalg.norm(np.cross(bonds[1:],bonds[:-1]),axis=-1)>1e-8).all())
  if not frames:bad.append('degenerate_frame')
 if finite and n:
  D=np.linalg.norm(x[:,None]-x[None,:],axis=-1);ix=np.arange(n);pair_count=int(((D<2)&(ix[:,None]-ix[None,:]>=3)).sum());contact_edges=int(((D<8)&(ix[:,None]-ix[None,:]>=8)).sum());rg=float(np.sqrt(np.square(x-x.mean(0)).sum(-1).mean()));current_contact=2*contact_edges/n
 if pair_count*100/N>LIMITS['clashes_per100_max']:bad.append('nonlocal_clash_budget')
 # These are necessary final-N bounds; current Rg/contact averages are NOT
 # classified as absorbing prefix failures or used by the geometry filter.
 seq=''.join(state.residues);ss=c['sequence_stats'](seq);sl=SPEC['sequence_limits'][str(N)];seqbounds=[]
 if ss['max_run']>sl['max_run_max']:seqbounds.append('homopolymer_run')
 if n and max(collections.Counter(seq).values())/N>sl['max_fraction_max']:seqbounds.append('final_max_AA_fraction_lower_bound')
 if ss['unique']+(N-n)<sl['unique_min']:seqbounds.append('insufficient_remaining_distinct_AAs')
 tris=[seq[i:i+3] for i in range(max(0,n-2))];duplicates=len(tris)-len(set(tris))
 if duplicates/max(1,N-2)>sl['repeated_trigram_fraction_max']:seqbounds.append('final_repeated_trigram_lower_bound')
 return dict(prefix_length=n,final_N=N,valid=not bad,irreversible_failures=bad,finite=finite,local_spacing_valid=spacing,nondegenerate_frames=frames,clash_pairs=pair_count,allowed_clash_pairs=math.floor(LIMITS['clashes_per100_max']*N/100),nonlocal_clash_valid=pair_count*100/N<=LIMITS['clashes_per100_max'],Rg_A=rg,Rg_scaled_by_current_n=rg/n**(1/3) if n and rg is not None else None,Rg_scaled_by_final_N=rg/N**(1/3) if rg is not None else None,current_contact_degree=current_contact,contact_edges=contact_edges,final_contact_degree_lower_bound=2*contact_edges/N,packing_final_lower_bound_already_exceeded=2*contact_edges/N>SPEC['packing_limits'][str(N)][1],irreversible_sequence_bounds=seqbounds,global_metrics_used_by_prefix_filter=False)

def extend_checked(state,aa,geom):
 before=prefix_status(state);mask=c['mask_for_step'](state.t);arr=np.asarray(geom,float);support=arr.shape==(3,) and np.isfinite(arr).all();reasons=[]
 if not support:reasons.append('invalid_or_nonfinite_action')
 else:
  if (mask[0] and arr[0]<=0) or (mask[1] and not 0<arr[1]<math.pi) or (mask[2] and not abs(arr[2])<math.pi):reasons.append('disallowed_policy_action_support')
 if not before['valid']:reasons.append('existing_prefix_already_invalid')
 if reasons:return None,dict(accepted=False,reasons=reasons,before=before,support_valid=support and 'disallowed_policy_action_support' not in reasons,new_clash_pairs=None)
 try:afterstate=append(state,aa,geom)
 except (ValueError,RuntimeError,AssertionError) as error:return None,dict(accepted=False,reasons=['builder_error'],builder_error=repr(error),before=before,support_valid=True,new_clash_pairs=None)
 after=prefix_status(afterstate);reasons=list(after['irreversible_failures']);return afterstate,dict(accepted=after['valid'],reasons=reasons,before=before,after=after,support_valid=True,new_clash_pairs=after['clash_pairs']-before['clash_pairs'])

def motif_info(seq,goal):
 motif=goal['motif'];a,z=goal['exposed'];region=seq[a:min(len(seq),z)];wanted=collections.Counter(motif)
 def contains_identity(s):v=collections.Counter(s);return all(v[k]>=n for k,n in wanted.items())
 def ordered(s):j=0
 # Explicit subsequence test separates identity, order and adjacency.
 def subsequence(s):
  j=0
  for aa in s:
   if j<len(motif) and aa==motif[j]:j+=1
  return j==len(motif)
 return dict(identity_anywhere=contains_identity(seq),identity_in_region=contains_identity(region),ordered_subsequence_anywhere=subsequence(seq),ordered_subsequence_in_region=subsequence(region),contiguous_anywhere=motif in seq,original_contiguous_in_region=motif in region,region_generated=max(0,min(len(seq),z)-a),region_closed=len(seq)>=z,no_exact_motif_start_assigned=True)

def motif_allowed(seq,goal):
 """DFA viability mask: preserves all legal motif locations; no new position."""
 t=len(seq);a,z=goal['exposed'];motif=goal['motif'];region=seq[a:min(t,z)]
 if t<a or motif in region or t>=z:return np.ones(20,bool) if not (t>=z and motif not in region) else np.zeros(20,bool)
 remaining=z-t-1;allowed=[]
 for aa in AA:
  s=region+aa
  if motif in s:allowed.append(True);continue
  matched=max((k for k in range(len(motif)) if s.endswith(motif[:k])),default=0);allowed.append(len(motif)-matched<=remaining)
 return np.array(allowed,bool)

def known_goal_dead(state,goal):
 status=prefix_status(state);mi=motif_info(''.join(state.residues),goal);observe=c['observe'](''.join(state.residues),state.ca.numpy(),goal,SPEC);reasons=[]
 if mi['region_closed'] and not mi['original_contiguous_in_region']:reasons.append('motif_window_closed_without_motif')
 if observe['regions']['exposed']['final_exposure_upper_bound']+1e-10<goal['exposure_min']:reasons.append('exposed_region_upper_bound_below_target')
 if status['packing_final_lower_bound_already_exceeded']:reasons.append('final_packing_lower_bound_exceeded')
 reasons.extend(status['irreversible_sequence_bounds']);return reasons

def reconstruct(record,N=64,stop=None):
 state=c['BuildState'].empty(N)
 for i,(aa,geom) in enumerate(zip(record['sequence'],record['geometry'])):
  if stop is not None and i>=stop:break
  state=append(state,aa,geom)
 return state

def load_engine(policy):
 e=p['Engine']();record=None
 if policy in ['D100','D50']:
  u=100 if policy=='D100' else 50;record=read(OLD/'checkpoints/protein/RECOVERY_INDEX.json')['owned'][str(u)];assert sha(Path(record['path']))==record['sha256'];state=torch.load(record['path'],map_location='cpu',weights_only=False);assert c['state_digest'](state)==record['state_digest'];assert state['goal_schema_SHA256']==sha(OLD/'source/scripts/e017_v91/goals.py');e.actor.load_state_dict(state['actor'],strict=True);e.actor.eval().requires_grad_(False);del state
 e.audit_actor_fingerprint=c['state_digest'](e.actor.state_dict());return e,record

def policy_name(policy):return {'A':'A_J1','B':'B_PREV9','C':'C_FIXED','D100':'D_RL','D50':'D_RL'}[policy]
def verify_engine(e):
 e.verify(trained=True);assert c['state_digest'](e.actor.state_dict())==e.audit_actor_fingerprint;assert all(not v.requires_grad and v.grad is None for model in [e.actor,e.parent,e.prior.model] for v in model.parameters())
