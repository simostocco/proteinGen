"""Freeze family-separated variants and experimental coordinate references before folding."""
from r1d_common import *
import numpy as np
from protein_joint_sac.data.scaled_supervised import Data
from protein_joint_sac.data.accepted_supervised import CoordinateReader

def main():
 guard();assert not (R/'configs/VARIANT_INFERENCE_FROZEN.json').exists();inv=json.load(open(R/'analysis/ELIGIBLE_PARENT_INVENTORY.json'));candidates=sorted(inv['candidate_contrast_parents'],key=lambda c:c['selection_key']);data=Data(False);reader=CoordinateReader(data.proposal.external);available=[];previous=json.load(open(B/'configs/NATIVE_PANEL_32.json'))['controls'];previousseq={c['sequence_id'] for c in previous};usedfamilies=set()
 # Scan only indexed, qualified TRAIN structural records; no TEST coordinates or labels.
 for c in candidates:
  if c['family_id'] in usedfamilies:continue
  seq=c['sequence'];found=None
  for h in data.jointids['TRAIN']:
   full=data.seqs[h];start=full.find(seq)
   if start<0 or full.find(seq,start+1)>=0:continue
   # Prefer a source accession corresponding to the assay parent's PDB code.
   owners=sorted(data.owners[h],key=lambda o:data.rows[o]['entry_id'].lower()!=c['PDB_accession'])
   for oid in owners:
    row=data.rows[oid]
    if row['entry_id'].lower()!=c['PDB_accession']:continue
    x=reader.read(row);native=x[start:start+len(seq)];assert native.shape==(len(seq),3) and np.isfinite(native).all()
    found={'sequence_id':h,'full_sequence_sha256':h,'start0':start,'length':len(seq),'coordinate_record':row,'native_ca':native.tolist(),'qualified_complete_TRAIN':True,'mapping':'exact unique substring in qualified native full canonical polymer; observed CA slice preserves consecutive original positions'};break
   if found:break
  if found and hashlib.sha256(seq.encode()).hexdigest() not in previousseq:available.append(dict(c,native_reference=found));usedfamilies.add(c['family_id'])
  if len(available)>=12:break
 # Selection gives two overlap-range and two shorter domains if independent candidates exist.
 selected=[];families=set()
 for pred,quota in [(lambda c:64<=c['N']<=72,2),(lambda c:40<=c['N']<64,2)]:
  for c in available:
   if len([x for x in selected if pred(x)])>=quota:break
   if pred(c) and c['family_id'] not in families:selected.append(c);families.add(c['family_id'])
 for c in available:
  if len(selected)>=4:break
  if c['family_id'] not in families:selected.append(c);families.add(c['family_id'])
 if len(selected)<4:
  write('INFERENCE_BLOCKER.json',{'reason':'fewer than four independent operational family controls with exact qualified native mapping under frozen thresholds','available':len(selected),'no_new_AF_calls':True});return
 selected=sorted(selected,key=lambda c:hashlib.sha256(('R1dRole20261020'+c['family_id']).encode()).hexdigest());dev=[];audit=[];jobs=[];natives={};summary=[]
 for idx,c in enumerate(selected):
  role='DEVELOPMENT' if idx<2 else 'AUDIT';parentkey='parent'+str(idx+1);natives[parentkey]=dict(c['native_reference'],sequence=c['sequence'],parent_name=c['parent']);labs=[]
  for j,v in enumerate(c['variants']):
   ident=f'{parentkey}_variant{j}';a3m=R/'data'/f'{ident}.a3m';a3m.write_text('>'+ident+'\n'+v['sequence']+'\n');label=dict(v,id=ident,parent_key=parentkey,parent=c['parent'],family_id=c['family_id'],role=role);labs.append(label);jobs.append({'id':ident,'parent_key':parentkey,'parent':c['parent'],'family_id':c['family_id'],'role':role,'sequence':v['sequence'],'sequence_sha256':hashlib.sha256(v['sequence'].encode()).hexdigest(),'N':len(v['sequence']),'a3m_path':str(a3m),'a3m_sha256':sha(a3m),'WT':j==0})
  (dev if role=='DEVELOPMENT' else audit).extend(labs);summary.append({'parent_key':parentkey,'parent':c['parent'],'family_id':c['family_id'],'N':c['N'],'role':role,'variants':[{'id':l['id'],'sequence':l['sequence'],'source_rows':[r['source_row'] for r in l['source_rows']]} for l in labs],'PDB_accession':c['PDB_accession'],'native_reference_sequence_id':c['native_reference']['sequence_id']})
 write('analysis/DEVELOPMENT_LABELS.json',dev);write('analysis/AUDIT_LABELS_SEALED.json',audit);write('configs/NATIVE_REFERENCES.json',natives);write('STABILITY_VARIANT_BENCHMARK.json',{'selected_families':summary,'available_qualified_native_families':len(available),'inventory':'analysis/ELIGIBLE_PARENT_INVENTORY.json','labels_development':'analysis/DEVELOPMENT_LABELS.json','labels_audit':'analysis/AUDIT_LABELS_SEALED.json','audit_labels_used_for_development_fit':False,'WT_note':'author median WT reference, conservative envelope of qualified per-replicate95% intervals; individual WT ddG deviations retained in source rows'})
 saved=json.load(open(B/'COLABFOLD_MSA_QUERY_MANIFEST.json'))['rows'];matching=[]
 for c in selected:
  for m in saved:
   if m['status']=='SUCCEEDED' and m['sequence_sha256']==hashlib.sha256(c['sequence'].encode()).hexdigest():matching.append({'parent':c['parent'],'saved_MSA_id':m['id']})
 cfg={'version':'R1d_query_only_variants_v1','freeze_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'jobs':jobs,'models':1,'model_order':[1],'model_type':'alphafold2_ptm','seed':151502,'recycles':3,'recycle_tolerance':0,'templates':False,'relaxation':False,'msa_mode':'single_sequence','max_MSA':'1:1','max_calls':12,'max_worker_seconds':1500,'timeout_seconds':115,'precision':'existing internal BF16 / output FP32, GPU default and CPU preprocessing; no fast kernels; same weight hash as R1b/R1c','inference_setting':'genuine query-only input for every WT and mutant; no independently queried MSA or transferred homolog background','saved_MSA_queries_inspected':len(saved),'exact_selected_parent_MSA_matches':matching,'selection_config_sha256':sha(R/'configs/SELECTION_AND_ANALYSIS_PREDECLARED_v2.json'),'development_labels_sha256':sha(R/'analysis/DEVELOPMENT_LABELS.json'),'audit_labels_sha256':sha(R/'analysis/AUDIT_LABELS_SEALED.json'),'ranking_formula_policy':'pTM primary; prespecified secondary and fixed additive composite, no fitting or metric-selection on audit','baseline_adequacy_rule':'after two DEVELOPMENT WT calls, stop if both native mapped scores<.5 and both pTM<.5; otherwise run remaining sealed jobs, no quality-based replacement','scope':'natural-domain limited-information mutation ranking feasibility; not deep-MSA or novel-sequence certification','J1_updates':0,'remote_MSA_queries':0}
 # Fold development WT controls first for the predeclared adequacy/resource check.
 cfg['jobs']=sorted(jobs,key=lambda j:(not(j['role']=='DEVELOPMENT' and j['WT']),j['role']=='AUDIT',j['parent_key'],j['id']))
 write('configs/VARIANT_INFERENCE_FROZEN.json',cfg)
 write('FAMILY_SPLIT_AND_CONTAMINATION_AUDIT.json',{'selected':summary,'family_graph':'analysis/FAMILY_GRAPH.json','graph_v1_rejected':'analysis/rejected_family_proxy_v1','operational_family_policy':'author clusters + strong50%/80% global links + accepted E016 protected group/source PDB; no claimed complete remote homology exclusion','E016_TEST_context':'identity/split-ownership audit only; no TEST structures, experimental labels, model metrics or tuning; selected operational families with any exact construct link to TEST excluded','E016_TRAIN_overlap':'all four parent constructs map exactly to qualified TRAIN observations; not unseen structures or sequences to J1; audit is parent/family-disjoint from this experimental development subset','AlphaFold_training_overlap':'natural PDB constructs and public sequences can overlap AF parameter training; exact AF training membership unknown; labels2023 not an AF supervised stability target','experimental_family_counts':{'development':2,'audit':2},'all_variant_sequences_unique':len({j['sequence_sha256'] for j in jobs})==len(jobs),'no_parent_or_family_split_between_roles':len({c['family_id'] for c in selected})==4,'audit_protocol_locked_before_predictions':True})
 print(json.dumps({'selected':[{k:c[k] for k in ['parent','N','family_id']} for c in selected],'exact_saved_MSA_matches':matching,'frozen_jobs':len(jobs)}),flush=True)
if __name__=='__main__':main()
