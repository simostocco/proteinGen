"""Read-only historical residual analysis and TRAIN-only depth confidence analysis."""
from r1c_common import *
import numpy as np
from scipy.stats import spearmanr

def interval(v):
 rng=np.random.default_rng(190299);a=np.asarray(v);return np.quantile([np.mean(rng.choice(a,len(a),replace=True)) for _ in range(2000)],[.025,.975]).tolist()
def scores(f,beta):
 return {'historical_product':float(np.clip(beta*f['sequence_confidence_product'],0,1)),'pTM':f['ptm'],'additive_equal':.5*f['ptm']+.5*f['mean_pLDDT']/100,'minimum':min(f['ptm'],f['mean_pLDDT']/100)}
def main():
 h=json.load(open(B/'REWARD_CALIBRATION_RESIDUALS.json'));hist=json.load(open(B/'MSA_ASSISTED_NATIVE_RECOVERY.json'))['rows'];panel=json.load(open(B/'configs/NATIVE_PANEL_32.json'))['controls'];mapping={v['id']:v for v in json.load(open(B/'native/MAPPING_REAUDIT.json'))['rows']};roles={r['id']:r['role'] for r in h['rows']};cal=[];rows=[]
 contexts={'train_64':'6gk3 beta-2-microglobulin amyloid fragment; deposited64-residue construct, author22-85, octamer/two protofilaments; not full natural chain','train_128':'9wzy BRD4 BD1 domain, ligand/molecular-glue complex','train_256':'29tx rhinovirus capsid chain; capsid assembly context','train_384':'5it7 ribosomal assembly/IRES/eEF2 context','train_historical_3fm1_A':'3fm1 fungal peroxidase, heme/cofactor context,319/331 observed','train_historical_2qmi_A':'2qmi archaeal penicillin-binding homologue, octameric assembly','train_r1b_03':'8vcp galactose-binding lectin rMcL-1, dimeric raffinose complex;152-residue chainB, X-ray1.58A','train_r1b_05':'2a8x Mycobacterium tuberculosis lipoamide dehydrogenase, FAD-containing homodimer;464-residue chainA, X-ray2.4A; native MSE mapped canonically to MET','validation_96':'3gzf feline-coronavirus nsp4 C-terminal domain, dimer context','validation_256':'7aoi mitochondrial large ribosomal subunit assembly intermediate'}
 total=sum(v['absolute_error'] for v in h['rows'] if v['role']=='VALIDATION_AUDIT')
 # Role labels vary across archived schemas; identify VALIDATION from authoritative result.
 valids={r['id'] for r in hist if r['split']=='VALIDATION'};total=sum(v['absolute_error'] for v in h['rows'] if v['id'] in valids)
 for z in h['rows']:
  r=next(v for v in hist if v['id']==z['id']);c=next(v for v in panel if v['id']==z['id']);folder=B/'folds'/z['id'];pp=list((folder/'prediction').glob('*unrelaxed_rank_001*.pdb'));rr=dict(z,signed_residual=z['predicted_recovery']-z['measured_recovery'],split=r['split'],N=c['N'],sequence_sha256=c['sequence_id'],entry_id=c['entry_id'],observed_positions=c['observed_positions'],canonical_sequence=c['sequence'],biological_context=contexts.get(z['id']),native_mapping_evidence=mapping.get(z['id']),native_source=c.get('source_path',c.get('observation_metadata')),prediction_path=str(pp[0]),prediction_sha256=sha(pp[0]),prediction_result=str(folder/'RESULT.json'),prediction_result_sha256=sha(folder/'RESULT.json'),features=r['features'],MSA_metrics=r['MSA_depth'],recovery=r['recovery'],historical_protocol=str(folder/'PROTOCOL.json'))
  if z['id'] in valids:rr['fraction_of_VALIDATION_sum_absolute_error']=z['absolute_error']/total;rr['failure_type']='confidence-product underprediction of measured recovery; experimental assembly context may affect coordinate comparison, not proven causal'
  elif z['id']=='train_64':rr['failure_type']='high-confidence prediction differs from experimental assembled amyloid conformation; native-state disagreement not proof of an invalid sequence'
  else:rr['failure_type']='retained control; prediction/uncertainty/context examined separately, no inferred biological quality label'
  rows.append(rr)
  if r['split']=='TRAIN':cal.append({'id':r['id'],'role':roles[r['id']],'measured':z['measured_recovery'],'signals':scores(r['features'],h['beta'])})
 proposal=[]
 for name in cal[0]['signals']:
  rr={'signal':name,'fitted_new_parameters':0,'TRAIN_only':True,'VALIDATION_scored_for_selection':False}
  for role in ['TRAIN_FIT','TRAIN_HELDOUT']:
   vs=[abs(v['signals'][name]-v['measured']) for v in cal if v['role']==role];rr[role]={'n':len(vs),'MAE':float(np.mean(vs)),'identity_bootstrap95_MAE':interval(vs)}
  rr['TRAIN_highconfidence_lowrecovery']=[v['id'] for v in cal if v['signals'][name]>=.7 and v['measured']<.5];proposal.append(rr)
 write('HISTORICAL_PER_CONTROL_DIAGNOSIS.json',{'rows':rows,'historical_beta':h['beta'],'historical_TRAIN_heldout_MAE':float(np.mean([z['absolute_error'] for z in h['rows'] if z['role']=='TRAIN_HELDOUT'])),'historical_VALIDATION_MAE':total/len(valids),'historical_failure_preserved':True,'no_VALIDATION_fit_or_new_formula_selection':True})
 depth=json.load(open(R/'MSA_DEPTH_ABLATION.json'));dr=[]
 for r in depth['rows']:
  if r['status']!='SUCCEEDED':continue
  f=r['features'];c=next(v for v in panel if v['id']==r['control']);y=r['recovery']['robust_fixed_map_TM_similarity_fullN'];dr.append({'id':r['id'],'control':r['control'],'historical_role':roles[r['control']],'N':c['N'],'depth_level':r['depth_level'],'nominal_rows':r['metrics']['raw_depth'],'Neff80':r['metrics']['Neff80_max512'],'coverage':r['metrics']['coverage_mean'],'native_recovery':y,'features':f,'signals':scores(f,h['beta']),'proposed_exploratory_eligible':r['metrics']['Neff80_max512']>=8 and r['metrics']['coverage_mean']>=.7 and f['ptm']>=.6 and f['pLDDT_q10']>=50,'prediction_sha256':sha(R/'folds'/r['id']/'RESULT.json'),'protocol_sha256':r['protocol_sha256']})
 trends=[]
 for ident in sorted(set(v['control'] for v in dr)):
  vv=[v for v in dr if v['control']==ident];first=next((v for v in vv if v['depth_level']==1),None);last=next((v for v in vv if v['depth_level']=='full'),None)
  if first and last:trends.append({'control':ident,'full_minus_query_recovery':last['native_recovery']-first['native_recovery'],'full_minus_query_pTM':last['features']['ptm']-first['features']['ptm'],'full_minus_query_pLDDT':last['features']['mean_pLDDT']-first['features']['mean_pLDDT'],'full_minus_query_PAEgood':last['features']['long_range_PAE_good_fraction']-first['features']['long_range_PAE_good_fraction'],'full_repeat_minus_R1b_native_recovery':last['native_recovery']-next(v for v in hist if v['id']==ident)['recovery']['robust_fixed_map_TM_similarity_fullN']})
 write('CONFIDENCE_VS_NATIVE_RECOVERY.json',{'historical_TRAIN_proposals':proposal,'depth_rows':dr,'identity_paired_depth_trends':trends,'highconfidence_lowrecovery_cases':[v['id'] for v in dr if v['features']['ptm']>=.7 and v['features']['mean_pLDDT']>=80 and v['native_recovery']<.5],'eligible_but_recovery_below_.5':[v['id'] for v in dr if v['proposed_exploratory_eligible'] and v['native_recovery']<.5],'independent_unit':'protein identity; three identities, not12 independent controls','fit_on_VALIDATION':False,'production_reward_GO':False})
 print(json.dumps({'depth_successes':len(dr),'trends':trends,'historical_VAL_MAE':total/len(valids),'TRAIN_proposals':proposal},indent=2))
if __name__=='__main__':main()
