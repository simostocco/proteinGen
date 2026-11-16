"""TRAIN-only property calibration and independent all-atom exposure checks."""
import collections,copy,gzip,importlib.util,json,math,os,runpy,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import gemmi
R=Path('/mnt/d/Simone/proteinGen_data/e017_v9_property_steering_20261023_v1');c=runpy.run_path(str(R/'source/scripts/e017_v9/core.py'));save=c['save'];read=c['read'];sha=c['sha'];ns=c['ns'];DESIGN=c['DESIGN'];AA=c['AA']
HEAVY={a:set(s.split()) for a,s in dict(A='N CA C O CB',C='N CA C O CB SG',D='N CA C O CB CG OD1 OD2',E='N CA C O CB CG CD OE1 OE2',F='N CA C O CB CG CD1 CD2 CE1 CE2 CZ',G='N CA C O',H='N CA C O CB CG ND1 CD2 CE1 NE2',I='N CA C O CB CG1 CG2 CD1',K='N CA C O CB CG CD CE NZ',L='N CA C O CB CG CD1 CD2',M='N CA C O CB CG SD CE',N='N CA C O CB CG OD1 ND2',P='N CA C O CB CG CD',Q='N CA C O CB CG CD OE1 NE2',R='N CA C O CB CG CD NE CZ NH1 NH2',S='N CA C O CB OG',T='N CA C O CB OG1 CG2',V='N CA C O CB CG1 CG2',W='N CA C O CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2',Y='N CA C O CB CG CD1 CD2 CE1 CE2 CZ OH').items()}
RADII=dict(C=1.7,N=1.55,O=1.52,S=1.8,SE=1.9,P=1.8)
def sphere(n=960):
 k=np.arange(n);z=1-2*(k+.5)/n;r=np.sqrt(1-z*z);phi=np.pi*(3-np.sqrt(5))*k;return np.c_[r*np.cos(phi),r*np.sin(phi),z]
SPHERE=sphere()
def atom_sasa(x,r):
 x=np.asarray(x);r=np.asarray(r)+1.4;tree=cKDTree(x);areas=[]
 for i,(p,ri) in enumerate(zip(x,r)):
  neighbors=[j for j in tree.query_ball_point(p,ri+r.max()) if j!=i];points=p+ri*SPHERE;free=np.ones(len(points),bool)
  for j in neighbors:
   if not free.any():break
   ix=np.flatnonzero(free);free[ix[np.square(points[ix]-x[j]).sum(-1)<r[j]**2]]=False
  areas.append(4*np.pi*ri**2*free.mean())
 return np.array(areas)
def boot(a,groups=None):
 a=np.asarray(a,float);n=len(a);draw=np.random.default_rng(19917).integers(0,n,(10000,n));v=a[draw].mean(1);return dict(estimate=float(a.mean()),CI95=np.quantile(v,[.025,.975]).tolist(),identities=n,replicates=10000,seed=19917)
def prepare():
 from protein_joint_sac.data.scaled_supervised import Data,canonical,internal
 from protein_joint_sac.data.accepted_supervised import CoordinateReader
 from protein_joint_sac.crisp_v3.execution import set_loader
 torch.set_num_threads(1);set_loader('arrow_scalar_lru4');data=Data(False);reader=CoordinateReader(data.proposal.external);allrows=[];rejected=[];start=time.time()
 for split,count in [('TRAIN',512),('VALIDATION',128)]:
  groups=set();chosen=[]
  for h in sorted(data.jointids[split],key=lambda h:ns('reference',split,h)):
   if len(chosen)==count:break
   g=data.proposal.sequences[h]['protected_group_id'];seq=data.seqs[h]
   if g in groups or len(seq)<64:continue
   row=data.rows[min(data.owners[h])]
   try:
    raw=reader.read(row);assert raw.shape==(len(seq),3);offset=ns('window',h)%(len(seq)-63);x=canonical(raw[offset:offset+64]);d=np.linalg.norm(np.diff(x,axis=0),axis=-1);assert np.isfinite(x).all() and d.min()>=2 and d.max()<=4.5;internal(x)
    if row['coordinate_kind']!='legacy_cache':
     mapping=reader.small(row['coordinate_path'])['observations'][int(row['coordinate_row'])];assert all(mapping['continuity_mask'][offset:offset+63]);assert not mapping['alternate_selection_ambiguous']
    rec=dict(identity=h,group=g,split=split,sequence=seq[offset:offset+64],offset=offset,coordinates=x.tolist(),source=row,geometry=c['geometry_stats'](x),sequence_stats=c['sequence_stats'](seq[offset:offset+64]),exposure={name:c['exposure'](x,name).tolist() for name in DESIGN['proxy_candidates']});chosen.append(rec);groups.add(g)
   except (ValueError,AssertionError,KeyError,FloatingPointError) as ex:rejected.append(dict(identity=h,split=split,reason=repr(ex)))
  assert len(chosen)==count,(split,len(chosen));allrows.extend(chosen);save('manifests/'+split+'_REFERENCE.json',dict(records=chosen,observational_quality_only=True));print('REFERENCE',split,len(chosen),flush=True)
 assert not {r['group'] for r in allrows if r['split']=='TRAIN'}&{r['group'] for r in allrows if r['split']=='VALIDATION'};save('manifests/DATA_EXCLUSIONS.json',rejected)
 paths=[R/'source/scripts/e017_v9/core.py',R/'source/scripts/e017_v9/calibrate.py',c['J1'],c['PARENT'],R.parent/'pdb_supervised_split_50id80cov_20261016_v1/PROPOSAL_MANIFEST.json']+list((R/'manifests').glob('*.json'));assert sha(c['J1'])==c['J1SHA'] and sha(c['PARENT'])==c['PSHA'];save('REFERENCE_CENSUS.json',dict(TRAIN=512,VALIDATION=128,TRAIN_fit=256,TRAIN_hold=256,window_length=64,one_window_per_identity=True,masked_exclusions=len(rejected),seconds=time.time()-start,coordinate_source_SHA256={r['source']['coordinate_path']:data.proposal.external[r['source']['coordinate_path']] for r in allrows},TEST_evaluated=False));save('INITIAL_INPUT_AUDIT.json',dict(input_SHA256={str(p):sha(p) for p in paths},J1_SHA=c['J1SHA'],selected_low_actor_SHA=c['PSHA'],scientific_selection='V3selected10000sequence/geometry actor, no property qualification; V5D250 is a specialized H2control',D_free_bytes=__import__('shutil').disk_usage(R).free))
def reference_atoms(rec):
 row=rec['source']
 if row['coordinate_kind']=='legacy_cache':
  import pyarrow.parquet as pq
  pf=pq.ParquetFile(row['coordinate_path']);index=int(row['coordinate_row']);base=0;meta=None
  for group in range(pf.num_row_groups):
   size=pf.metadata.row_group(group).num_rows
   if index<base+size:
    cols=['label_asym_id','model_id','chain_mapping_status','selected_calpha_coordinates','residue_names','atom_label_sequence_ids']
    meta=pf.read_row_group(group,columns=cols,use_threads=False).slice(index-base,1).to_pylist()[0];break
   base+=size
  assert meta is not None and meta['chain_mapping_status']=='unambiguous'
  names=json.loads(meta['residue_names']);coords=json.loads(meta['selected_calpha_coordinates']);ids=json.loads(meta['atom_label_sequence_ids']);assert len(names)==len(coords)==len(ids)
  offset=rec['offset'];mapping=[dict(label_seq_id=ids[i],native_monomer=names[i],ca=coords[i]) for i in range(offset,offset+64)]
  assert len({str(m['label_seq_id']) for m in mapping})==64
  raw=Path('/mnt/d/Simone/proteinGen/data/full/raw/mmcif')/(row['entry_id'].lower()+'.cif.gz');assert raw.stat().st_size<32*2**20 and sha(raw)==row['source_sha']
  with gzip.open(raw,'rb') as f:assert len(f.read(128*2**20+1))<=128*2**20
  obs=dict(label_asym_id=meta['label_asym_id'],model_num=meta['model_id'])
 else:
  parent=json.load(gzip.open(row['coordinate_path'],'rt'));obs=parent['observations'][int(row['coordinate_row'])];raw=Path(parent['source_path']);assert raw.stat().st_size<32*2**20 and parent['uncompressed_bytes']<128*2**20;assert sha(raw)==parent['source_sha256']==row['source_sha'];offset=rec['offset'];mapping=obs['residue_mapping'][offset:offset+64]
 labels={str(r['label_seq_id']):i for i,r in enumerate(mapping)};block=gemmi.cif.read(str(raw)).sole_block();table=block.find('_atom_site.',['label_asym_id','pdbx_PDB_model_num','label_seq_id','label_atom_id','type_symbol','Cartn_x','Cartn_y','Cartn_z','occupancy','label_alt_id']);by=collections.defaultdict(dict);ambiguous=set()
 for a in table:
  if a[0]!=obs['label_asym_id'] or a[1]!=str(obs['model_num']) or a[2] not in labels:continue
  element=str(a[4]).upper()
  if element in ['H','D']:continue
  if element not in RADII:raise ValueError('unsupported atom element')
  i=labels[a[2]];name=str(a[3]);xyz=np.array([float(a[j]) for j in [5,6,7]]);occupancy=float(a[8]);value=(occupancy,xyz,element);old=by[i].get(name)
  if old is not None and abs(old[0]-occupancy)<1e-6 and np.linalg.norm(old[1]-xyz)>.01:ambiguous.add(i)
  if old is None or occupancy>old[0]:by[i][name]=value
 standard={'ALA','CYS','ASP','GLU','PHE','GLY','HIS','ILE','LYS','LEU','MET','ASN','PRO','GLN','ARG','SER','THR','VAL','TRP','TYR'}
 valid=np.array([i not in ambiguous and mapping[i]['native_monomer'] in standard and HEAVY[rec['sequence'][i]]<=set(by[i]) for i in range(64)]);assert valid.mean()>=.9
 for i,m in enumerate(mapping):assert 'CA' in by[i] and np.linalg.norm(by[i]['CA'][1]-m['ca'])<1e-5
 # Gauge atom coordinates with the same first-three CA anchors as the proxy.
 ca=np.array([by[i]['CA'][1] for i in range(64)]);u=ca[1]-ca[0];u/=np.linalg.norm(u);w=ca[2]-ca[1];w-=w.dot(u)*u;w/=np.linalg.norm(w);basis=np.stack([u,w,np.cross(u,w)]);assert np.max(np.abs((ca-ca[0])@basis.T-np.array(rec['coordinates'])))<1e-5;x=[];radii=[];owner=[]
 for i in range(64):
  for name,(occ,xyz,element) in by[i].items():
   if occ<=0:continue
   x.append((xyz-ca[0])@basis.T);radii.append(RADII[element]);owner.append(i)
 x=np.array(x);radii=np.array(radii);owner=np.array(owner);areas=atom_sasa(x,radii);relative=np.zeros(64);isolated=[]
 for i in range(64):
  ii=owner==i;den=atom_sasa(x[ii],radii[ii]).sum();isolated.append(float(den));relative[i]=areas[ii].sum()/den
 mask=valid.copy();mask[:3]=False;mask[-3:]=False
 return dict(raw_path=str(raw),raw_SHA256=sha(raw),atoms=len(x),relative_accessibility=relative.tolist(),mask=mask.tolist(),reference='Shrake-Rupley960points,1.4Aprobe,fixedheavyatomradii; residueASA/context dividedbyidenticalresidueatomsalone; isolated64residuefragment,notDSSPRSA orfullassemblySASA',complete_heavy_fraction=float(valid.mean()))
def sasa():
 # Analytic surface checks precede all data references.
 one=atom_sasa([[0,0,0]],[1.7])[0];assert abs(one-4*np.pi*3.1**2)<1e-9;two=atom_sasa([[0,0,0],[3.1,0,0]],[1.7,1.7]);expected=3*np.pi*3.1**2;assert np.max(abs(two-expected))/expected<.025
 tr=read(R/'manifests/TRAIN_REFERENCE.json')['records'];val=read(R/'manifests/VALIDATION_REFERENCE.json')['records'];parts={'FIT':tr[:256],'HOLD':tr[256:],'VALIDATION':val};out=[];reject=[];start=time.time()
 for split,pop in parts.items():
  chosen=0
  for r in sorted(pop,key=lambda x:ns('SASA',x['identity'])):
   if chosen==16:break
   try:
    a=reference_atoms(r);mask=np.array(a['mask']);y=np.array(a['relative_accessibility']);scores={name:float(spearmanr(np.array(r['exposure'][name])[mask],y[mask]).statistic) for name in DESIGN['proxy_candidates']};undefined=[k for k,z in scores.items() if not np.isfinite(z)];scores={k:z if np.isfinite(z) else 0. for k,z in scores.items()};out.append(dict(identity=r['identity'],group=r['group'],partition=split,scores=scores,undefined_rank_contributes_zero_conservatively=undefined,**a));chosen+=1;save('results/SASA_REFERENCES.json',dict(records=out,rejected=reject));print('SASA',split,chosen,16,flush=True)
   except (AssertionError,ValueError,KeyError,RuntimeError,OSError) as ex:reject.append(dict(identity=r['identity'],partition=split,reason=repr(ex)))
  print('SASA_PARTITION',split,chosen,flush=True)
 save('SASA_COMPLETED.json',dict(records=out,rejected=reject,seconds=time.time()-start,analytic_sphere_tests=True))
def specification():
 tr=read(R/'manifests/TRAIN_REFERENCE.json')['records'];val=read(R/'manifests/VALIDATION_REFERENCE.json')['records'];ref=read(R/'SASA_COMPLETED.json')['records'];means={name:np.mean([r['scores'][name] for r in ref if r['partition']=='FIT']) if any(r['partition']=='FIT' for r in ref) else -1 for name in DESIGN['proxy_candidates']};proxy=max(means,key=means.get);gates={};proxy_ok=True
 for split in ['HOLD','VALIDATION']:
  scores=[r['scores'][proxy] for r in ref if r['partition']==split];v=boot(scores) if scores else None;ok=len(scores)>=16 and v['estimate']>=.5 and v['CI95'][0]>.3;gates[split]=dict(PASS=ok,correlation=v);proxy_ok &=ok
 seq={};packing={}
 for N in [32,64]:
  ss=[c['sequence_stats'](r['sequence'][:N]) for r in tr];gs=[c['geometry_stats'](np.array(r['coordinates'])[:N]) for r in tr];seq[str(N)]=dict(entropy_bits_min=float(np.quantile([s['entropy_bits'] for s in ss],.01)),unique_min=int(np.floor(np.quantile([s['unique'] for s in ss],.01))),max_fraction_max=float(np.quantile([s['max_fraction'] for s in ss],.99)),max_run_max=int(np.ceil(np.quantile([s['max_run'] for s in ss],.99))),repeated_trigram_fraction_max=float(np.quantile([s['repeated_trigram_fraction'] for s in ss],.99)));packing[str(N)]=np.quantile([g['longrange_contact_degree'] for g in gs],[.01,.99]).tolist()
 # Identity-equal exposure-bin AA conditional model; assessed on other identities.
 counts=np.ones((4,20));marg=np.ones(20)
 for r in tr[:256]:
  e=np.array(r['exposure'][proxy]);bins=np.minimum(3,(e*4).astype(int));ids=np.array([AA.index(a) for a in r['sequence']]);np.add.at(counts,(bins,ids),1/64);np.add.at(marg,ids,1/64)
 probabilities=counts/counts.sum(1,keepdims=True);marg=marg/marg.sum();comp={};allgain=[]
 for split,pop in [('HOLD',tr[256:]),('VALIDATION',val)]:
  gains=[]
  for r in pop:
   e=np.array(r['exposure'][proxy]);ids=np.array([AA.index(a) for a in r['sequence']]);bins=np.minimum(3,(e*4).astype(int));gains.append(float(np.mean(np.log(probabilities[bins,ids])-np.log(marg[ids]))))
  comp[split]=boot(gains);allgain.extend(gains)
 compat=all(v['estimate']>.01 and v['CI95'][0]>0 for v in comp.values());nativegain=[]
 for r in tr[:256]:
  e=np.array(r['exposure'][proxy]);ids=np.array([AA.index(a) for a in r['sequence']]);bins=np.minimum(3,(e*4).astype(int));nativegain.append(float(np.mean(np.log(probabilities[bins,ids])-np.log(marg[ids]))))
 scale=max(.05,float(np.quantile(nativegain,.75)-np.quantile(nativegain,.25)));spec=dict(schema='V9_reward_v1',N=64,proxy=proxy,proxy_PASS=bool(proxy_ok),proxy_selection_TRAIN_means={k:float(v) for k,v in means.items()},proxy_gates=gates,sequence_limits=seq,packing_limits=packing,exposure_residual_scale=max(.05,float(np.std([np.mean(r['exposure'][proxy]) for r in tr]))),compatibility_enabled=compat,compatibility=dict(probabilities=probabilities.tolist(),marginal=marg.tolist(),assessments=comp,low=float(np.quantile(nativegain,.01)),scale=scale,meaning='bounded AA/environment association score; not foldability'),hard_backbone_limits=c['LIMITS'],threshold_source='TRAIN512only; proxyselectionTRAINfit16; hold/VALIDATION used only admission,not tuning')
 save('REWARD_SPECIFICATION.json',spec)
 goals={};witness={};used_sasa={r['identity'] for r in ref};rejections=[];seen_signatures=set()
 for panel,pop,number in [('TRAIN',tr[:256],32),('DEV',[r for r in val if r['identity'] not in used_sasa][:64],8),('CONFIRM',[r for r in val if r['identity'] not in used_sasa][64:],16)]:
  out=[]
  for r in sorted(pop,key=lambda r:ns('goals',panel,r['identity'])):
   if len(out)==number:break
   x=np.array(r['coordinates']);E=np.array(r['exposure'][proxy]);starts=list(range(4,53,4));pairs=[(float(E[a:a+8].mean()-E[b:b+8].mean()),a,b) for a in starts for b in starts if abs(a-b)>=16];gap,a,z=max(pairs)
   if gap<.15:rejections.append(dict(identity=r['identity'],panel=panel,reason='native regional contrast<.15'));continue
   motifs=[r['sequence'][i:i+2] for i in range(a,a+7) if r['sequence'][i]!=r['sequence'][i+1]]
   if not motifs:continue
   goal=dict(goal_id='V9_'+panel+'_'+r['identity'][:16],witness_identity=r['identity'],protected_group=r['group'],N=64,exposed=[a,a+8],buried=[z,z+8],exposure_min=float(np.floor(E[a:a+8].mean()*20)/20),burial_max=float(np.ceil(E[z:z+8].mean()*20)/20),contrast_min=.1,motif=motifs[ns('motif',r['identity'])%len(motifs)],motif_scope='anywhere within exposedregion',native_coordinates_provided=False)
   signature=__import__('hashlib').sha256(json.dumps({k:goal[k] for k in ['N','exposed','buried','exposure_min','burial_max','contrast_min','motif']},sort_keys=True).encode()).hexdigest()
   if signature in seen_signatures:continue
   goal['property_signature']=signature;rr=c['score'](r['sequence'],x,goal,spec)
   if not rr['success']:rejections.append(dict(identity=r['identity'],panel=panel,reason='native witnessfails hard task/sequence/packing limits'));continue
   out.append(goal);seen_signatures.add(signature);witness[goal['goal_id']]=dict(sequence=r['sequence'],coordinates=r['coordinates'],identity=r['identity'],group=r['group'])
  goals[panel]=out;save('manifests/'+panel+'_GOALS.json',dict(goals=out,requested=number,coordinate_free=True));print('GOALS',panel,len(out),number,flush=True)
 save('manifests/GOAL_WITNESSES_PRIVATE.json',witness);save('manifests/GOAL_EXCLUSIONS.json',rejections);save('CALIBRATION_ADMISSION.json',dict(PASS=bool(proxy_ok and all(len(goals[p])==n for p,n in [('TRAIN',32),('DEV',8),('CONFIRM',16)])),proxy_PASS=bool(proxy_ok),goal_counts={p:len(g) for p,g in goals.items()},compatibility_enabled=compat,stage='calibration only, no policy sampling or training'))
if __name__=='__main__':globals()[sys.argv[1]]()
