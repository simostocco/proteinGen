"""Frozen-panel read-only diagnostics. Optimizer steps are forbidden."""
import json,math,time,gc,resource,collections,itertools,shutil
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import scaled_bc as core
from protein_joint_sac.models.batched_bc import encode_batch,losses
from protein_joint_sac.rl.sequence_first import geometry_parameters,direct_nonlocal
from protein_joint_sac.rl.group_clipping import groups
from protein_joint_sac.data.scaled_supervised import Data
from protein_joint_sac.data.certified_unified import file_sha
R=Path(__file__).resolve().parents[3];B=R.parent;O=B/'e016_j1_clip_precision_pilot_20261018_v1';Q=B/'e016_j1_sequence_first_20261018_v1';P=B/'e016_partial_supervision_20261016_v1'
CP={'original':P/'runs/J1-FULL/checkpoint_20000.pt','source5000':Q/'runs/J1/checkpoint_05000.pt',**{f'pilot{k}':O/f'runs/J1/model_{k:05d}.pt' for k in [500,1000]},**{f'pilot{k}':O/f'runs/J1/checkpoint_{k:05d}.pt' for k in [1500,2500]}}
def write(n,v):core.write(R/n,v)
def status(**v):write('DIAGNOSTIC_PROGRESS.json',{'time':time.time(),**v});print(json.dumps(v),flush=True)
def st(a):
 a=np.asarray(a,dtype=float);a=a[np.isfinite(a)]
 return {'n':len(a),'mean':float(a.mean()) if len(a) else None,'std':float(a.std()) if len(a) else None,'quantiles_0_001_01_05_25_50_75_95_99_999_100':np.quantile(a,[0,.001,.01,.05,.25,.5,.75,.95,.99,.999,1]).tolist() if len(a) else []}
def components(m,b,y,g):
 with torch.autocast('cuda',dtype=torch.bfloat16):h=encode_batch(m,b);aa=F.cross_entropy(m.aa_head(h).float(),y,reduction='none')
 p=geometry_parameters(m,h,y).float();mu=p[:,:3];ls=p[:,3:].clamp(-5,2);mask=(b['t'][:,None]>=torch.arange(1,4,device='cuda')[None])&b.get('geometry_target_valid',torch.ones(len(y),3,device='cuda',dtype=torch.bool));gg=g.double()
 u=torch.stack([torch.where(mask[:,0],gg[:,0],1.).log(),torch.logit(torch.where(mask[:,1],gg[:,1]/math.pi,.5)),torch.atanh(torch.where(mask[:,2],gg[:,2]/math.pi,0.))],-1)
 jac=torch.stack([u[:,0],math.log(math.pi)-F.softplus(-u[:,1])-F.softplus(u[:,1]),math.log(math.pi)+2*(math.log(2)-u[:,2]-F.softplus(-2*u[:,2]))],-1)
 nll=torch.where(mask,.5*((u-mu)/ls.exp()).square()+ls+.5*math.log(2*math.pi)+jac,0.)
 if not torch.isfinite(nll).all():raise FloatingPointError('nonfinite likelihood')
 return aa,nll,mask,p,u

def summarize(z,panel):
 nll=z['nll'];mask=z['mask'];uid=np.array([h for h,o,t in panel]);ret={'components':{},'decisions':len(panel),'identities':len(set(uid)),'historic_local_per_decision':float((nll.sum(1)/mask.sum(1).clip(1)).mean()),'AA_NLL':float(z['aa'].mean())}
 for c,name in enumerate(['distance','angle','torsion']):
  v=nll[mask[:,c],c];ids=uid[mask[:,c]];by=collections.defaultdict(list)
  for h,a in zip(ids,v):by[h].append(a)
  means=np.array([np.mean(by[h]) for h in sorted(by)]);rng=np.random.default_rng(318);ci=np.quantile([rng.choice(means,len(means),replace=True).mean() for _ in range(300)],[.025,.975]).tolist();worst={}
  for frac in [.001,.01,.05]:
   k=max(1,math.ceil(len(v)*frac));top=np.sort(v)[-k:];worst[str(frac)]={'targets':k,'top_mean':float(top.mean()),'contribution_to_mean':float(top.sum()/len(v)),'remaining_mean_analysis_only':float(np.sort(v)[:-k].mean())}
  ret['components'][name]={'target_weighted':st(v),'identity_weighted':st(means),'identity_bootstrap_mean95':ci,'worst_tail':worst}
 ok=mask[:,2];mu=z['p'][ok,2].astype(float);raw=z['p'][ok,5].astype(float);ls=np.clip(raw,-5,2);sig=np.exp(ls);u=z['u'][ok,2];tau=z['g'][ok,2];res=(u-mu)/sig
 nodes,weights=np.polynomial.hermite.hermgauss(64);pred=math.pi*np.tanh(mu[:,None]+math.sqrt(2)*sig[:,None]*nodes);w=weights/math.sqrt(math.pi);arith=pred@w;ss=np.sin(pred)@w;cc=np.cos(pred)@w;circ=np.arctan2(ss,cc);resultant=np.hypot(ss,cc);err=np.arctan2(np.sin(circ-tau),np.cos(circ-tau));pit=torch.special.ndtr(torch.tensor(res)).numpy();mass={}
 for deg in [1,5,10,20]:
  cut=np.arctanh((math.pi-math.radians(deg))/math.pi);mass[str(deg)]={'native_fraction':float((abs(tau)>math.pi-math.radians(deg)).mean()),'predicted_mean_mass':float((torch.special.ndtr(torch.tensor((-cut-mu)/sig))+1-torch.special.ndtr(torch.tensor((cut-mu)/sig))).mean())}
 nd,ww=np.polynomial.hermite.hermgauss(32);pr=math.pi*np.tanh(mu[:,None]+math.sqrt(2)*sig[:,None]*nd);c32=np.arctan2(np.sin(pr)@(ww/math.sqrt(math.pi)),np.cos(pr)@(ww/math.sqrt(math.pi)))
 ret['torsion_calibration']={'latent_mu':st(mu),'raw_log_std':st(raw),'clamped_log_std':st(ls),'logstd_clip_fraction':float(((raw<=-5)|(raw>=2)).mean()),'standardized_latent_residual':st(res),'PIT_histogram10':np.histogram(pit,bins=np.linspace(0,1,11))[0].tolist(),'PIT_uniform_expected_per_bin':len(pit)/10,'coverage50_80_95':{str(c):float((abs(res)<=q).mean()) for c,q in [(50,.674489750196),(80,1.281551565545),(95,1.95996398454)]},'circular_mean_prediction_abs_error_rad':st(abs(err)),'physical_mean_abs_circular_error_rad':st(abs(np.arctan2(np.sin(arith-tau),np.cos(arith-tau)))),'resultant':st(resultant),'quadrature64_vs32_circular_mean_abs_difference_rad':st(abs(np.arctan2(np.sin(circ-c32),np.cos(circ-c32)))),'near_seam_mass_degrees':mass,'FP32_tanh_mu_exact_saturation_fraction':float((abs(np.tanh(mu.astype(np.float32)))==1).mean()),'latent_abs_mu_above10_fraction':float((abs(mu)>=10).mean()),'interpretation':'coverage is equal-tailed latent/principal-angle monotone interval, not shortest circular arc; quadrature diagnostic approximation with convergence error reported'}
 N=z['N'];t=z['t'];defs={'length':np.digitize(N,[65,129,257,385]),'position_decile':np.minimum(9,(10*t/N).astype(int)),'absolute_prefix':np.digitize(t,[1,16,64,128,256,384]),'seam_native_deg10':(abs(z['g'][:,2])>math.pi-math.radians(10)).astype(int),'uncertainty_logstd':np.digitize(z['p'][:,5],[-2,-1,0,1]),'native_angle_band':np.digitize(z['g'][:,1],[math.pi/3,math.pi/2,2*math.pi/3]),'known_coordinate_anomaly':z['anomaly']}
 ret['strata']={key:{str(k):{'targets':int(((a==k)&ok).sum()),'torsion_NLL':float(nll[(a==k)&ok,2].mean()) if ((a==k)&ok).any() else None} for k in np.unique(a)} for key,a in defs.items()}
 ret['worst_torsion_targets']=[{'sequence_hash':panel[i][0],'observation_id':panel[i][1],'target_position0':int(panel[i][2]),'N':int(N[i]),'torsion_NLL':float(nll[i,2]),'native_torsion_rad':float(z['g'][i,2]),'latent_mu':float(z['p'][i,2]),'logstd':float(z['p'][i,5]),'known_anomaly':bool(z['anomaly'][i])} for i in np.argsort(nll[:,2])[-30:][::-1]]
 return ret
@torch.no_grad()
def evaluate(m,data,panels):
 ext=json.load(open(B/'e016_real_pdb_pilots_20261016_v1/priors/COORDINATE_EXTREMES.json'))['examples'];extobs={v['observation_id'] for v in ext}
 for tag,cp in CP.items():
  if all((R/f'summaries/{tag}_{role}.json').exists() for role in ['TRAIN','VALIDATION']):continue
  state=torch.load(cp,map_location='cpu',weights_only=True);m.load_state_dict(state['model'],strict=True);del state;gc.collect();m.eval()
  for role in ['TRAIN','VALIDATION']:
   panel=sorted(panels[role],key=lambda e:(e[2],e[0],e[1]));rows=collections.defaultdict(list)
   for off in range(0,len(panel),16):
    ex=panel[off:off+16]
    if any(data.proposal.sequences[h]['split']!=role for h,o,t in ex):raise ValueError('split contradiction')
    b,y,g=core.make(data,ex,True);aa,nll,mask,p,u=components(m,b,y,g)
    if off==0:
     with torch.autocast('cuda',dtype=torch.bfloat16):ra,rg,rp=losses(m,b,y,g,geometry_fp32=True)
     if not torch.equal(rp,nll) or not torch.equal(ra,aa):raise ValueError('original likelihood parity failed')
    for key,v in [('aa',aa),('nll',nll),('mask',mask),('p',p),('u',u),('g',g)]:rows[key].append(v.cpu().numpy())
    if off%4096==0:status(phase='FROZEN_PANEL',checkpoint=tag,role=role,completed=off)
   z={k:np.concatenate(v) for k,v in rows.items()};z.update(N=np.array([len(data.seqs[h]) for h,o,t in panel]),t=np.array([t for h,o,t in panel]),anomaly=np.array([o in extobs for h,o,t in panel]));(R/'arrays').mkdir(exist_ok=True);np.savez_compressed(R/'arrays'/f'{tag}_{role}.npz',**z);write(f'summaries/{tag}_{role}.json',summarize(z,panel));status(phase='PANEL_COMPLETE',checkpoint=tag,role=role)

def gradients(m,data):
 cfg=json.load(open(O/'configs/FROZEN.json'));limits=cfg['group_limits'];rng=np.random.default_rng(180721);ex=[]
 for lo,hi in [(20,64),(65,128),(129,256),(257,384),(385,500)]:
  ids=sorted(h for h in data.jointids['TRAIN'] if lo<=len(data.seqs[h])<=hi)
  for i,h in enumerate(rng.choice(ids,4,replace=False)):
   N=len(data.seqs[h]);ex.append((h,data.owners[h][0],max(3,min(N-1,int(N*[.15,.4,.7,.95][i])))))
 write('configs/COMPONENT_GRADIENT_TRAIN_PANEL.json',{'examples':ex,'seed':180721,'description':'20 TRAIN identities;4 perlength;early/mid/late;five minibatches4. Eval-mode native observed prefixes for component attribution. No optimizer calls.'})
 names=[n for n,p in m.named_parameters()];pars=list(m.parameters());group={n:k for k,v in groups(m)[0].items() for n,p in v};layers={n:('.'.join(n.split('.')[:3]) if n.startswith('encoder.blocks.') else n.split('.')[0]+('.pair' if n.startswith('encoder.pair.') else '')) for n in names};result={}
 def vec(loss,retain=False):return {n:v.detach().float().cpu() for n,v in zip(names,torch.autograd.grad(loss,pars,allow_unused=True,retain_graph=retain)) if v is not None}
 def norm(d,grp):return math.sqrt(sum(float(v.double().square().sum()) for n,v in d.items() if group[n]==grp))
 def cos(a,b):return sum(float((a[n].double()*b[n].double()).sum()) for n in a.keys()&b.keys() if group[n]=='S')/max(norm(a,'S')*norm(b,'S'),1e-30)
 for tag in ['original','source5000','pilot500','pilot1500','pilot2500']:
  state=torch.load(CP[tag],map_location='cpu',weights_only=True);m.load_state_dict(state['model']);m.eval();mom={}
  if 'optimizer' in state:
   opt=state['optimizer'];flat=[i for g in opt['param_groups'] for i in g['params']]
   if len(flat)!=len(names):raise ValueError('optimizer mapping not established')
   for n,i in zip(names,flat):
    v=opt['state'].get(i,{})
    if 'exp_avg' in v:mom[n]={'step':float(v['step']),'moment_L2':float(v['exp_avg'].double().norm()),'second_moment_mean':float(v['exp_avg_sq'].double().mean()),'second_moment_max':float(v['exp_avg_sq'].max())}
  del state;gc.collect();rr=[]
  for off in range(0,len(ex),4):
   b,y,g=core.make(data,ex[off:off+4],True);torch.manual_seed(721+off);aa,nll,mask,p,u=components(m,b,y,g);ls={'AA':aa.mean(),**{k:nll[:,c].mean() for c,k in enumerate(['distance','angle','torsion'])}};vv={k:vec(v,True) for k,v in ls.items()};del aa,nll,mask,p,u,ls
   targets=torch.tensor(np.asarray([data.ca[o][t] for h,o,t in ex[off:off+4]]),device='cuda');nl,counts=direct_nonlocal(m,b,y,targets,K=4,seed=180042);vv['nonlocal']=vec(nl);del nl,targets,b,y,g
   weighted={};weights={'distance':cfg['lambda_local']/3,'angle':cfg['lambda_local']/3,'torsion':cfg['lambda_local']/3,'nonlocal':cfg['lambda_nonlocal']}
   for key in vv:
    if key!='AA':
     for n,v in vv[key].items():weighted[n]=weighted.get(n,torch.zeros_like(v))+weights[key]*v
   aS=norm(vv['AA'],'S');raw=norm(weighted,'S');alpha=min(1,.5*aS/max(raw,1e-30));combined={n:vv['AA'].get(n,0)+alpha*weighted.get(n,0) for n in set(vv['AA'])|set(weighted)};before={k:norm(combined,k) for k in ['S','A','G']};sc={k:min(1,limits[k]/max(before[k],1e-30)) for k in before}
   stats={k:{'raw_L2':{gr:norm(v,gr) for gr in ['S','A','G']},'weighted_shared_L2':norm(v,'S')*(1 if k=='AA' else weights[k]),'post_cap_shared_L2':norm(v,'S')*(1 if k=='AA' else weights[k]*alpha),'per_layer_L2':{layer:math.sqrt(sum(float(x.double().square().sum()) for n,x in v.items() if layers[n]==layer)) for layer in set(layers.values())},'final_output_rows_L2':{n:[float(row.double().norm()) for row in x] for n,x in v.items() if n in ['geom.2.weight','geom.2.bias']}} for k,v in vv.items()}
   rr.append({'offset':off,'component_gradients':stats,'shared_pairwise_cosine':{a+'/'+bb:cos(vv[a],vv[bb]) for a,bb in itertools.combinations(vv,2)},'aggregate_aux_raw_shared':raw,'aux_shared_scale':alpha,'aggregate_aux_postcap_shared':raw*alpha,'AA_shared':aS,'combined_group_before':before,'group_clip_scale':sc,'combined_group_after':{k:before[k]*sc[k] for k in before},'AA_shared_retained_fraction':sc['S'],'nonlocal_anchor_counts':counts,'note':'nonlocal included to stress aggregate cap;actual pilot every2updates on4/32 geometry examples;smallbatch clipping rates not population rates'})
   del vv,weighted,combined;gc.collect();status(phase='COMPONENT_GRADIENT',checkpoint=tag,batches=len(rr))
  result[tag]={'batches':rr,'adam_moments_by_parameter':mom if mom else 'UNKNOWN:model-only checkpoint','optimizer_steps_executed':0};write('GRADIENT_DIAGNOSTICS.json',result)
def main():
 start=time.monotonic();core.setup();torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
 def forbidden(*a,**k):raise RuntimeError('optimizer.step forbidden')
 torch.optim.AdamW.step=forbidden
 if shutil.disk_usage(R).free<10*2**30+55*2**20:raise RuntimeError('reserve plus55MiB diagnostic output required')
 man=json.load(open(O/'CHECKPOINT_MANIFEST.json'));expected={v['path']:v['sha256'] for v in man['retained']};expected[str(CP['original'])]=man['original_fallback_sha256'];expected[str(CP['source5000'])]=man['source_sha256'];hashes={str(p):file_sha(p) for p in CP.values()}
 if any(hashes[p]!=expected[p] for p in hashes):raise ValueError('checkpoint hash mismatch')
 write('INPUT_VERIFICATION.json',{'checkpoint_sha256':hashes,'panel_sha256':file_sha(O/'configs/EVALUATION_PANELS.json'),'optimizer_steps_allowed':0});report={};data=Data(True,report);write('DATA_LOADING.json',report);panels=json.load(open(O/'configs/EVALUATION_PANELS.json'));m=core.model('J1');evaluate(m,data,panels);gradients(m,data)
 write('RESOURCE_REPORT.json',{'seconds':time.monotonic()-start,'RSS_highwater_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_allocated_peak':torch.cuda.max_memory_allocated(),'GPU_reserved_peak':torch.cuda.max_memory_reserved(),'D_free_bytes':shutil.disk_usage(R).free,'optimizer_steps':0,'TEST':'NOT_EVALUATED'});status(phase='DIAGNOSTICS_COMPLETE')
if __name__=='__main__':main()
