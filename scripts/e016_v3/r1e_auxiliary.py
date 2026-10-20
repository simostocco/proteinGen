from r1e_common import *
import itertools,subprocess,importlib.util,os,time
panel=read('FROZEN_J1_GENERATION_PANEL.json');aux=read('AUXILIARY_STRUCTURE_EVALUATION.json');div={};paired={}
for g in ['J1_7500','J1_500']:
 div[g]={}
 for N in [64,128,256,384,500]:
  rr=[r for r in panel['rows'] if r['group']==g and r['N']==N];cache=json.loads(Path(rr[0]['source']).read_text());xx=[np.array(cache[r['source_index']]['generated_CA_A']) for r in rr if r['status']=='COMPLETE'];rmsd=[]
  for a,b in itertools.combinations(xx,2):
   a=a-a.mean(0);b=b-b.mean(0);u,s,v=np.linalg.svd(a.T@b);rot=u@np.diag([1,1,np.linalg.det(u@v)])@v;rmsd.append(float(np.sqrt(((a@rot-b)**2).sum(1).mean())))
  div[g][str(N)]=dict(pairwise_Kabsch_CA_RMSD_A=desc(rmsd),note='fixed sequence-position correspondence and proper rotation; pair values dependent; descriptive spread, mean bootstrap is not family-level evidence')
for N in [64,128,256,384,500]:
 a=[r for r in panel['rows'] if r['group']=='J1_7500' and r['N']==N];b=[r for r in panel['rows'] if r['group']=='J1_500' and r['N']==N];paired[str(N)]={k:desc([x['geometry_metrics'][k]-y['geometry_metrics'][k] for x,y in zip(a,b)]) for k in ['Rg_A','clashes_per100_residues','long_contact_fraction']}
native=D/'e016_partial_supervision_20261016_v1/NATIVE_VS_GENERATED_GEOMETRY.json';z=json.loads(native.read_text())['native'];ref={}
for split,by in z.items():
 ref[split]={}
 for N,entry in by.items():
  rr=entry['rows'];ref[split][N]=dict(n=len(rr),length_window=entry['length_window'],metrics={k:desc([r['metrics'][k] for r in rr]) for k in ['Rg_A','Rg_over_N_cuberoot','clashes_per100_residues','long_contact_fraction']})
aux.update(structural_diversity=div,paired_J1_7500_minus_500=paired,native_reference=ref,native_reference_path=str(native),native_reference_sha256=sha(native),native_note='previous frozen +/-10%length windows; reported separately from exact-length sequence controls; TRAIN reference only used for normalization')
for g in aux['metrics']:
 for N,v in aux['metrics'][g].items():v['native_TRAIN_normalized_Rg_ratio']=v['Rg_over_N_cuberoot']['mean']/ref['TRAIN'][N]['metrics']['Rg_over_N_cuberoot']['median']
# Check preserved AF generated-sequence records; no new folds and no homolog substitution.
P=D/'e016_sequence_first_rl_readiness_20261019_v1';af=[]
for p in (P/'oracle').glob('v3_*generated*/RESULT.json'):
 x=json.loads(p.read_text());af.append(dict(path=str(p),sha256=sha(p),record=x))
aux['historical_generated_AF_cache_candidates']=af;aux['panel_AF_exact_matches']=[]
seqs={r['sequence']:r['id'] for r in panel['rows']}
for q in af:
 for k,v in q['record'].items():
  if isinstance(v,str) and v in seqs:aux['panel_AF_exact_matches'].append(dict(id=seqs[v],path=q['path'],status='sequence_exact; model/protocol require separate receipt validation'))
aux['AF_scope']='Historical generated64 receipts retained as context only; no new folding. No exact panel-sequence/model/protocol-certified AF match => no panel external structural qualification.'
write('AUXILIARY_STRUCTURE_EVALUATION.json',aux)
tools=Path('/mnt/d/Simone/proteinGen/e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1/external_designability_loop_v1/tools')
audit=dict(mmseqs=dict(path='/usr/bin/mmseqs',version=subprocess.check_output(['/usr/bin/mmseqs','version'],text=True).strip(),license='GPLv3',seconds=sum(v['seconds'] for v in read('SEQUENCE_NOVELTY_AND_HOMOLOGY.json')['runs']),ram_limit_bytes=2*2**30),ESM2=dict(implementation=read('data/ESM_IMPLEMENTATION_PROVENANCE.json'),benchmark=read('ESM_BENCHMARK.json'),weights_sha256=sha(R/'weights/esm2_t12_35M_UR50D.pt'),weights_bytes=(R/'weights/esm2_t12_35M_UR50D.pt').stat().st_size,license='MIT',legacy_150M_weights='/mnt/d/Simone/proteinGen/external_models/e007_phase4b_v1/esm2_150m/model.safetensors',legacy_150M_status='present but no compatible Transformers package in chosen environment;35M chosen as expressly authorized bounded evaluator; not a new trained model'),IUPred3=dict(status='not found in inspected project/tool locations or PATH; omitted; no install',license='not assessed absent'),ESMFold=dict(status='no installed folding weights in inspected tools; omitted; no download',license='not assessed absent'),ProteinMPNN=dict(path=str(tools/'ProteinMPNN'),status='source and historical CA-only weights present; no verified E016 CA-only scoring protocol adopted; standard full-backbone scores unavailable for J1',license='MIT; verify notices in preserved source',J1_scores=False),USalign=dict(path=str(tools/'USalign/USalign'),status='installed historical alignment audit; not needed for sequence scoring',license='consult existing tool license; not redistributed'),AlphaFold=dict(path=str(tools/'colabfold_env'),version='ColabFold1.6.3,AFcolabfold2.3.20,JAX0.6.2 per immutable R1b/R1c install audit',weights=str(tools/'alphafold_weights/params/params_model_1_ptm.npz'),weights_sha256='5e564f79af5bcd54ccef6e2a6bb0ff01015d01650ebc41d4575e35f0de9ecc84',weights_license='CC BY-NC-SA4.0; source notices govern use',new_calls=0,measured_prior_seconds='R1d12calls476.633s aggregate; lengths44–69; not new R1e timing',VRAM_prior_peak_MiB=5081),storage=dict(free_bytes=shutil.disk_usage(R).free,reserve_bytes=10*2**30),no_training=True)
write('EVALUATOR_AVAILABILITY.json',audit)
print('auxiliary ready, AF receipts',len(af),'panel sequence matches',len(aux['panel_AF_exact_matches']),flush=True)
