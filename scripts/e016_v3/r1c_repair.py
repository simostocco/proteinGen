from r1c_common import *
from r1b_prepare import comparisons
(R/'alignment').mkdir(exist_ok=True)
p=R/'MSA_DEPTH_ABLATION.json';data=json.load(open(p));row=data['rows'][0];assert row['status']=='FAILED' and 'alignment/' in row['error'];write('folds/INITIAL_POSTPROCESSING_FAILURE.json',data)
c=next(v for v in json.load(open(R/'configs/NATIVE_CONTROLS.json'))['controls'] if v['id']==row['control']);res=json.load(open(R/'folds'/row['id']/'RESULT.json'))
from protein_joint_sac.rl.sequence_rewards import sequence_features
row.update(status='SUCCEEDED',fold_status='SUCCEEDED',postprocessing_initial_failure=row.pop('error'),recovery=comparisons(c,res['predicted_ca'],row['id']),features=sequence_features(c['sequence'],res['plddt'],res['ptm'],res['pae'],res['predicted_ca']),resolved_config=json.load(open(R/'folds'/row['id']/'prediction/config.json')))
write('MSA_DEPTH_ABLATION.json',data);write('folds/'+row['id']+'/RECEIPT.json',row);write('folds/POSTPROCESSING_REPAIR.json',{'rerun_folding_calls':0,'repair':'create missing alignment output directory and score existing validated prediction','original_failure':'folds/INITIAL_POSTPROCESSING_FAILURE.json'})
print(row['recovery']['robust_fixed_map_TM_similarity_fullN'])
