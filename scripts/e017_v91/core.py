"""Versioned property task, prefix-only diagnostics and true suffix corrections."""
from pathlib import Path
import collections,copy,hashlib,json,math,os,random
import numpy as np
import torch
from protein_joint_sac.contracts import AA,BuildState,BuildAction
from protein_joint_sac.geometry.ca_builder import append as append_state,mask_for_step
from protein_joint_sac.crisp_v3.recovery import atomic_json,sha,state_digest,StoragePolicy
from protein_joint_sac.crisp_v3.environment import assess
R=Path('/mnt/d/Simone/proteinGen_data/e017_v91_ordered_goal_recovery_20261023_v1')
J1=R.parent/'e016_j1_bidirectional_multitask_20261019_v1/runs/J1/checkpoint_07500.pt';J1SHA='881457ff0396975c1b20dd0a8e2caac8521e65b7e1cec76a7bb531dbc437b412'
PARENT=R.parent/'e017_crisp_v3_throughput_20261021_v1/training_runs/run_d3e78a804534473884df42f0f34b7c41/checkpoints/checkpoint_00010000_7d8c5d05.pt';PSHA='ccd11e9788a30a4f3c398b57ee655d24f3e4b636fec875469f080b711ee3e8e9'
LIMITS=json.loads((R.parent/'e017_crisp_v3_20261021_v1/configs/NATIVE_CALIBRATION.json').read_text())
DESIGN=dict(schema='E017_V9_property_v1',seed=9917,N=64,TRAIN_reference=512,VALIDATION_reference=128,SASA_fit=16,SASA_TRAIN_hold=16,SASA_VALIDATION=16,SASA_sphere_points=960,SASA_probe_A=1.4,
 proxy_candidates=['smooth8_sep3','smooth10_sep3','contact10_sep3'],proxy_selection='highest mean per-protein Spearman on16TRAINfit atomreferences; ties preferexisting smooth8',proxy_gate='heldTRAIN andVALIDATION mean proteinSpearman>=.5, identity95%lower>.3; at least16proteins each; finite exact source atom mapping; >=.9observed complete heavy-atom residues; no native coordinates at inference',
 reference_windows='one hash-indexed contiguous64residue window per qualified identity, finite/continuous2.0to4.5A observational-quality filter; not selected for packing; preserve masks and exclusions',
 sequence_limits='TRAINq.01 Shannon entropy and uniqueAA; TRAINq.99 maximumAAfraction, homopolymer run, repeated3gramfraction',packing_limits='TRAINq.01/q.99 longrange contactdegree(d<8,absseqsep>=8), original V3 backbone limits unchanged',
 AA_compatibility='TRAINFIT256 exposure-bin20AA counts Laplace1; admit only heldTRAIN andVALIDATION per-identity loglikelihoodgain>.01nats/res and paired95%lower>0, plus motif-preserving shuffle sensitivity; otherwise omit optionalcompatibility and restrictclaim',
 motif_length=2,region_length=8,goal_selection='32TRAIN property targets from valid FITnative witness,8DEV and16CONFIRM from disjointVALIDATION witnesses excluding atomreference IDs; max regional exposure contrast over starts4..52stride4, separation>=16; target exposure rounded permissively to.05grid, contrastreq.1; motif from exposedregion existing heterodipeptide. No witness values supplied to policy except declared properties/motif.',
 reward='R=hard_backbone_and_sequence_and_packing_valid*(.5*binary_full_success+.5*mean(bounded property/motif utilities)); per-constraint utility exp(-positive_residual/TRAINscale), motif best fractional match. Exact success separate; invalid/trivial candidates0; no stability/function claim',
 reward_gate='all finite; rigid-motion invariance; valid native goal witnesses success; collapsed/straight-extended/invalidbackbone/degenerateAAcontrols maxreward0; locallyvalid extendedhelix nofullsuccess and reward<=.5; absentmotif never fullsuccess; randomizedAAs audited; prefix observations invariant to discarded future',
 baseline_TRAIN_goals=32,baseline_DEV_goals=8,baseline_CONFIRM_goals=16,K=[1,2,4,8],N32_control='only ifN64admissionfails,16TRAINgoal/seedcontrols perpolicy; useN32TRAINcalibration, keepdistinct; never replaceN64primary',
 low_admission='B_V3 TRAIN64completion>=.95; backbonevalid>=.20 and combinedsequence/packingvalid>=.10 across>=16goals; rewardSD>=.03; nonzeroreward>=.10; each exposure/burial and motifutility varies; no nonfinite/repair samples fortraining; AAmeanKL<=.5 and activegeommeanKL<=1.0. Fullsuccess canbezero ifgradedreturninformative; no promotion onrewardalone',
 low_pilot='critic-free onpolicy completeepisode RLOO groups8; fixedTRAINper-goalclass rewardscale; sum exact AA+transformed geom logdensities pertrajectory/N; J1no_grad; freshAdamWlr1e-6,clip5; exact analytic jointKL; AAonly/Gonly override disabledfactor with immutableparentoutputs; total<=500accepted updates:joint<=400,AAonly<=50,Gonly<=50; engineering then25/50/100/250/400gates; noSAC',
 low_stops='nonfinite, original validity decline>5pp, AAkl>.5 orgeomkl>1, conditionalstepKL>.02 => rollback/stop; at100continueonlyDEV K1>=parent andTRAINrollingrewardgain; at250continueonlyDEV K1strictlyimproves; no threshold revisions',
 low_promotion='CONFIRM K1gain>=5pp and pairedgoal98.333333%lower>0 for3arms; preservedhardvalidity,diversity,KL; largerKnotworse; no native-prefix evidence',
 high_options=['CONTINUE','FOCUS_EXPOSED','FOCUS_BURIED','FOCUS_MOTIF','REWIND4','REWIND8','REWIND16','REGENERATE','ACCEPT_OR_STOP'],gamma=1.,high_admission='selectedlowpolicy fullvalidK1 improves; correctionvsBestofmatchedwork gain>=5pp andpaired95%lower>0; validated future-outcome diagnostic advantage; frozenlow and independenthigh,<=50highupdates only; otherwisepropose/block',
 resource=dict(reserve_bytes=2*2**30,near_growth_bytes=128*2**20,max_GPU_bytes=6*2**30,one_GPU_worker=True),TEST=False,AlphaFold=False,J1_updates=0)
POLICY=StoragePolicy(min_headroom_bytes=2*2**30,near_term_growth_bytes=128*2**20,checkpoint_hint_bytes=256*2**20)
def save(n,x):atomic_json(R/n,x)
def read(p):return json.loads(Path(p).read_text())
def ns(*x):return int(hashlib.sha256(('E017V9|9917|'+ '|'.join(map(str,x))).encode()).hexdigest()[:15],16)
def exposure(x,name):
 x=np.asarray(x,float);n=len(x);d=np.linalg.norm(x[:,None]-x[None,:],axis=-1);ix=np.arange(n);eligible=abs(ix[:,None]-ix[None,:])>=3
 if name=='contact10_sep3':return 1/(1+((d<10)&eligible).sum(-1)/6.)
 radius=8 if name=='smooth8_sep3' else 10;return 1/(1+(np.maximum(0,1-d/radius)**2*eligible).sum(-1))
def sequence_stats(seq):
 if not seq:return dict(entropy_bits=0.,unique=0,max_fraction=1.,max_run=0,repeated_trigram_fraction=0.)
 c=collections.Counter(seq);p=np.array(list(c.values()),float)/len(seq)
 # Explicit run scan (regex capture returns one character, not its full run).
 longest=now=0;last=None
 for a in seq:now=now+1 if a==last else 1;longest=max(longest,now);last=a
 tri=[seq[i:i+3] for i in range(max(0,len(seq)-2))];rep=1-len(set(tri))/len(tri) if tri else 0.
 return dict(entropy_bits=float(-(p*np.log2(p)).sum()),unique=len(c),max_fraction=float(p.max()),max_run=longest,repeated_trigram_fraction=rep)
def geometry_stats(x):
 x=np.asarray(x,float);n=len(x)
 if n<3 or x.shape!=(n,3) or not np.isfinite(x).all():return dict(valid=False,finite=False,failures=['incomplete_or_nonfinite'],longrange_contact_degree=0.,Rg_normalized=None,clashes_per100=None)
 d=np.linalg.norm(np.diff(x,axis=0),axis=-1);D=np.linalg.norm(x[:,None]-x[None,:],axis=-1);ix=np.arange(n);clash=int(((D<2)&(ix[:,None]-ix[None,:]>=3)).sum());rg=float(np.sqrt(np.square(x-x.mean(0)).sum(-1).mean())/n**(1/3));cross=np.linalg.norm(np.cross(np.diff(x,axis=0)[1:],np.diff(x,axis=0)[:-1]),axis=-1);fail=[]
 if d.min()<LIMITS['adjacent_low_A'] or d.max()>LIMITS['adjacent_high_A']:fail.append('spacing')
 if clash*100/n>LIMITS['clashes_per100_max']:fail.append('nonlocal_clashes')
 if not LIMITS['Rg_normalized_low']<=rg<=LIMITS['Rg_normalized_high']:fail.append('Rg')
 if (cross<=1e-8).any():fail.append('degenerate_frames')
 contacts=((D<8)&(abs(ix[:,None]-ix[None,:])>=8)).sum(-1)
 return dict(valid=not fail,finite=True,failures=fail,adjacent_min_A=float(d.min()),adjacent_max_A=float(d.max()),clash_count=clash,clashes_per100=clash*100/n,Rg_normalized=rg,longrange_contact_degree=float(contacts.mean()))
def motif_utility(seq,goal):
 a,z=goal['exposed'];motif=goal['motif'];region=seq[a:min(z,len(seq))];m=len(motif)
 return max((sum(p==q for p,q in zip(region[i:i+m],motif))/m for i in range(max(0,len(region)-m+1))),default=0.)
def score(seq,x,goal,spec):
 g=geometry_stats(x);stats=sequence_stats(seq);N=goal['N'];complete=len(seq)==N and len(x)==N and not set(seq)-set(AA);sl=spec['sequence_limits'][str(N)];seqvalid=complete and stats['entropy_bits']>=sl['entropy_bits_min'] and stats['unique']>=sl['unique_min'] and stats['max_fraction']<=sl['max_fraction_max'] and stats['max_run']<=sl['max_run_max'] and stats['repeated_trigram_fraction']<=sl['repeated_trigram_fraction_max'];lo,hi=spec['packing_limits'][str(N)];packed=g['finite'] and lo<=g['longrange_contact_degree']<=hi;hard=complete and g['valid'] and seqvalid and packed
 if not complete or not g['finite']:return dict(complete=complete,success=False,reward=0.,backbone_valid=False,sequence_valid=seqvalid,packing_valid=False,hard_valid=False,geometry=g,sequence=stats)
 e=exposure(x,spec['proxy']);a,z=goal['exposed'];c,d=goal['buried'];ex=float(e[a:z].mean());bu=float(e[c:d].mean());contrast=ex-bu;motif=motif_utility(seq,goal);res=[max(0,goal['exposure_min']-ex),max(0,bu-goal['burial_max']),max(0,goal['contrast_min']-contrast)];scale=spec['exposure_residual_scale'];u=[math.exp(-q/scale) for q in res]+[motif];success=bool(hard and max(res)<=1e-10 and motif==1);compat=None
 if spec.get('compatibility_enabled'):
  model=np.array(spec['compatibility']['probabilities']);marg=np.array(spec['compatibility']['marginal']);bins=np.minimum(3,(e*4).astype(int));ids=np.array([AA.index(a) for a in seq]);compat=float(np.mean(np.log(model[bins,ids])-np.log(marg[ids])));u.append(float(np.clip((compat-spec['compatibility']['low'])/spec['compatibility']['scale'],0,1)))
 reward=float(hard*(.5*success+.5*np.mean(u)));assert 0<=reward<=1 and (hard or reward==0)
 return dict(complete=complete,success=success,reward=reward,backbone_valid=g['valid'],sequence_valid=bool(seqvalid),packing_valid=bool(packed),hard_valid=bool(hard),geometry=g,sequence=stats,exposed_mean=ex,buried_mean=bu,contrast=contrast,motif_utility=motif,utilities=u,compatibility_score=compat,exposure=e.tolist())
def observe(seq,x,goal,spec,history=(),mode=0):
 n=len(seq);N=goal['N'];ex=exposure(x,spec['proxy']) if n else np.array([]);regions={}
 for label,(a,z) in [('exposed',goal['exposed']),('buried',goal['buried'])]:
  known=max(0,min(n,z)-a);values=ex[a:min(n,z)] if known else np.array([]);regions[label]=dict(generated=known,total=z-a,current_known_mean=float(values.mean()) if known else None,final_exposure_upper_bound=float((values.sum()+(z-a-known))/(z-a)),final_exposure_lower_bound=0.)
 clashes=0
 if n:
  d=np.linalg.norm(np.asarray(x)[:,None]-np.asarray(x)[None,:],axis=-1);ii=np.arange(n);clashes=int(((d<2)&(ii[:,None]-ii[None,:]>=3)).sum())
 return dict(schema='V9_public_prefix_v1',N=N,t=n,remaining=N-n,sequence=seq,coordinates=np.asarray(x).tolist(),goal=copy.deepcopy(goal),regions=regions,motif_current_utility=motif_utility(seq,goal),clash_count=clashes,irreversible_clash_violation=clashes*100/N>LIMITS['clashes_per100_max'],history=copy.deepcopy(list(history)),steering_mode=mode,estimated_reachability=None,reachability_status='not learned or admitted')
class PropertyWorld:
 """Uses existing typed state/builder; property reward replaces coordinate goals."""
 def __init__(self,goal,spec,seed):
  self.goal=copy.deepcopy(goal);self.spec=spec;self.seed=seed;self.rng=np.random.default_rng(seed);self.state=BuildState.empty(goal['N']);self.events=[];self.interventions=[];self.mode=0;self.branch=0;self.branch_counter=0;self.pending=False;self.terminated=False;self.snapshots={};self.snapshot()
 def snapshot(self):self.snapshots[self.state.t]=copy.deepcopy((self.state,self.rng.bit_generator.state,self.mode,self.branch))
 def append(self,a,g):
  if self.pending or self.terminated or self.state.t>=self.goal['N']:raise ValueError('construction masked')
  self.state=append_state(self.state,BuildAction(a,torch.tensor(g,dtype=torch.float64),mask_for_step(self.state.t)));self.events.append(dict(kind='RESIDUE',aa=a,geometry=list(g),branch=self.branch));self.snapshot()
 def mask(self):
  t=self.state.t;return [t<self.goal['N'] and not self.pending and not self.terminated]*4+[t>=k and t-k in self.snapshots and not self.pending and not self.terminated for k in [4,8,16]]+[self.pending and not self.terminated,not self.terminated]
 def high(self,a):
  if not self.mask()[a]:raise ValueError('masked high action')
  if a<4:self.mode=a;self.events.append(dict(kind='CONTEXT',mode=a));return dict(continue_generation=True,mode=a)
  if a in [4,5,6]:
   old=self.state.t;k=[4,8,16][a-4];self.state,rng,self.mode,self.branch=copy.deepcopy(self.snapshots[old-k]);self.rng.bit_generator.state=rng;self.snapshots={t:v for t,v in self.snapshots.items() if t<=old-k};self.pending=True;self.interventions.append(dict(kind='REWIND',from_t=old,to_t=old-k));self.events.append(dict(kind='REWIND',from_t=old,to_t=old-k));return dict(rewound=k)
  if a==7:self.branch_counter+=1;self.branch=self.branch_counter;self.rng=np.random.default_rng(ns(self.seed,'branch',self.branch,len(self.interventions)));self.pending=False;self.interventions.append(dict(kind='REGENERATE',branch=self.branch));self.events.append(dict(kind='REGENERATE',branch=self.branch));return dict(regenerated=True)
  self.terminated=True;r=score(''.join(self.state.residues),self.state.ca.numpy(),self.goal,self.spec);self.events.append(dict(kind='STOP',complete=r['complete'],success=r['success']));return r
 def observation(self):return observe(''.join(self.state.residues),self.state.ca.numpy(),self.goal,self.spec,self.interventions,self.mode)
 def export(self):return copy.deepcopy(self.__dict__|{'rng':self.rng.bit_generator.state})
 def restore(self,p):
  p=copy.deepcopy(p);rng=p.pop('rng');self.__dict__.update(p);self.rng=np.random.default_rng();self.rng.bit_generator.state=rng
