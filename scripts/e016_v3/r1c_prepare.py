"""Freeze native selection and nested MSA variants before any R1c folding."""
from r1c_common import *
import re,numpy as np
IDS=['train_64','train_128','train_historical_2qmi_A']
def parse(text):
 rows=[];h=None;s=''
 for l in text.splitlines():
  if l.startswith('#') or not l.strip():continue
  if l.startswith('>'):
   if h is not None:rows.append((h,s))
   h=l;s=''
  else:s+=l.strip()
 if h is not None:rows.append((h,s))
 return rows
def aligned(s):return re.sub('[a-z.]','',s).upper()
def metrics(rows,q):
 a=np.array([list(aligned(v[1])) for v in rows]);assert a.shape[1]==len(q) and ''.join(a[0])==q
 unique=list(dict.fromkeys(''.join(v) for v in a));u=np.array([list(v) for v in unique]);cov=(u!='-').mean(1);eligible=np.flatnonzero(cov>=.5);ix=eligible[np.linspace(0,len(eligible)-1,min(512,len(eligible)),dtype=int)];z=u[ix];neff=0
 for row in z:
  sim=((z==row)&(z!='-')&(row!='-')).sum(1)/np.maximum(((z!='-')|(row!='-')).sum(1),1);neff+=1/max(1,(sim>=.8).sum())
 return {'raw_depth':len(a),'unique_depth':len(u),'coverage_mean':float((a!='-').mean()),'coverage_q10':float(np.quantile((a!='-').mean(1),.1)),'column_gap_q90':float(np.quantile((a=='-').mean(0),.9)),'Neff80_max512':float(neff),'Neff_rows':len(z),'Neff_definition':'sum reciprocal >=80% identity neighbours; shared non-gap matches / non-gap union; deterministic evenly spaced max512 unique rows with coverage>=.5; not extrapolated full Neff'}
def main():
 guard();assert not (R/'configs/DEPTH_STUDY_FROZEN.json').exists()
 panel=json.load(open(B/'configs/NATIVE_PANEL_32.json'));jobs=[]
 for ident in IDS:
  c=next(v for v in panel['controls'] if v['id']==ident);assert c['split']=='TRAIN'
  src=B/'msa'/ident/'query.a3m';receipt=json.load(open(B/'msa'/ident/'RECEIPT.json'));assert sha(src)==receipt['a3m_sha256'];rows=parse(src.read_text());assert aligned(rows[0][1])==c['sequence']
  # Canonical distinct, >=50% query-covered homologs; deterministic shuffle independent of recovery.
  seen={aligned(rows[0][1])};eligible=[]
  for k,row in enumerate(rows[1:],1):
   s=aligned(row[1]);assert len(s)==c['N']
   if s not in seen and s.count('-')<=len(s)/2:eligible.append(k);seen.add(s)
  rng=np.random.default_rng(2026101903);rng.shuffle(eligible)
  for depth in [1,4,32,'full']:
   indexes=[0]+eligible[:depth-1] if isinstance(depth,int) else list(range(len(rows)))
   selected=[rows[k] for k in indexes];text=''.join(h+'\n'+s+'\n' for h,s in selected);name=f'{ident}_depth{depth}';path=R/'msa'/f'{name}.a3m';path.write_text(text)
   jobs.append({'id':name,'control':ident,'depth_level':depth,'a3m_path':str(path),'a3m_sha256':sha(path),'source_a3m':str(src),'source_a3m_sha256':sha(src),'source_row_indices':indexes if depth!='full' else 'all original rows, unchanged order','metrics':metrics(selected,c['sequence'])})
 write('configs/NATIVE_CONTROLS.json',{'controls':[v for v in panel['controls'] if v['id'] in IDS],'source_panel':str(B/'configs/NATIVE_PANEL_32.json'),'source_sha256':sha(B/'configs/NATIVE_PANEL_32.json')})
 write('configs/DEPTH_STUDY_FROZEN.json',{'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'selection_rationale':'TRAIN amyloid fragment64, domain128, historical hard447; span length/context and preserve historical TRAIN heldout447; no VALIDATION folds','jobs':jobs,'max_calls':12,'max_worker_seconds':1500,'seed':151502,'max_MSA':'128:256','recycles':3,'model':1,'templates':False,'new_remote_requests':0,'nesting':'query retained, same deterministic shuffled >=50% covered unique homolog order for4/32; full is complete archived A3M','cache_policy':'all12 new; same caps even query-only; no single-sequence R0 cache assumed identical','metrics':'fixed-correspondence fullN TM, observedN TM, Kabsch RMSD, independent USalign, long-range contacts, pLDDT/pTM/PAE; all depth variants clustered by identity','historical_training_holdout':'train_historical_2qmi_A and train_r1b_03 retained; no identity-level independence across depths'})
 write('configs/REWARD_PROPOSALS_FROZEN.json',{'scope':'TRAIN-only descriptive comparisons, not reward certification; no selection using historical VALIDATION','candidates':['historical frozen beta*plddt*pTM*PAEgood','raw pTM, no fitted parameters','0.5*pTM+0.5*mean_pLDDT/100, no fitted parameters','min(pTM,mean_pLDDT/100), no fitted parameters'],'depth_reliability_hypothesis':'Neff80>=8, row query coverage mean>=.7, pTM>=.6, pLDDTq10>=50; exploratory eligible domain, not production certificate','abstention':'unknown quality, never reward0 or penalty; balanced attempted-candidate accounting; production blocked pending independent prospective controls','historical_fit_holdout_file':str(B/'folds/TRAIN_CALIBRATION_BEFORE_VALIDATION.json'),'VALIDATION_refit':False})
 print(json.dumps([{'id':j['id'],'depth':j['metrics']['raw_depth'],'Neff':j['metrics']['Neff80_max512']} for j in jobs]))
if __name__=='__main__':main()
