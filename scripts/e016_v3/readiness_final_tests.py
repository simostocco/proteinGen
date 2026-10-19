"""Defensive final CPU checks; synthetic fixtures cannot certify an AF reward."""
import json,copy,hashlib,torch
from pathlib import Path
from protein_joint_sac.rl.sequence_rewards import calibrated_sequence_reward
from protein_joint_sac.rl.sequence_sac import SequenceFirstSAC
from protein_joint_sac.rl.sequence_replay import SequenceReplay
from protein_joint_sac.contracts import BuildState,GeneratedProtein,RewardCard
R=Path(__file__).resolve().parents[3]
def main():
 for case in [{'GO':False},{'GO':True,'weights':None},{'GO':True,'weights':{'x':1},'scale':{'x':0},'center':{'x':0}}]:
  try:calibrated_sequence_reward({'global_confidence_product':1,'x':1},case)
  except PermissionError:pass
  else:raise AssertionError('unsealed reward accepted')
 obj=object.__new__(SequenceFirstSAC);obj.config={k:True for k in ['production_authorized','AF_reward_certified','SAC_real_episode_certified','gradient_composer_certified']};obj.config['calibration_sha256']='1'*64
 for reward,receipt,err in [(None,None,ValueError),(1.,None,PermissionError),(1.,{'status':'TIMEOUT'},PermissionError)]:
  try:obj.update_resolved(None,None,reward,None,False,PDB_AA=None,PDB_geometry=None,receipt=receipt)
  except err:pass
  else:raise AssertionError('invalid episode admitted')
 try:obj.update_resolved(None,None,1.,None,False,PDB_AA=None,PDB_geometry=None,receipt={'status':'SUCCEEDED','calibrated':True,'calibration_sha256':'1'*64,'evidence_type':'EXTERNAL_PREDICTION'})
 except ValueError as e:assert 'separately capped retention' in str(e)
 else:raise AssertionError('geometry KL routed through primary AA objective')
 from protein_joint_sac.rl.sequence_rewards import fit_reward_gate
 rows=json.loads((R/'ALPHAFOLD_NATIVE_CONTROL_RESULTS.json').read_text())['rows'];native=copy.deepcopy([v for v in rows if v['status']=='SUCCEEDED' and v['kind']=='native' and v['split']=='TRAIN']);native[0]['features']['ptm']=None;native[0]['features']['global_confidence_product']=None;native[0]['features']['sequence_confidence_product']=None;assert not fit_reward_gate(native,[])['GO']
 # A genuinely reconstructed generated episode with mock reward remains ineligible.
 d=json.loads((R/'generated/J1_initial64.json').read_text());state=BuildState(tuple(d['sequence']),torch.tensor(d['ca'],dtype=torch.float64),len(d['sequence']));protein=GeneratedProtein(state);rp=SequenceReplay(R/'tests/final_mock_replay.sqlite');rp.add_trace('synthetic_receipt_test',len(d['sequence']),d['trace'],protein,{'scope':'CPU mock eligibility only'})
 card=RewardCard('SYNTHETIC_ONLY',protein.sequence_sha256,protein.geometry_sha256,'1'*64,'MOCK_UNIT_TEST',{},False);rp.attach('synthetic_receipt_test',card);assert not rp.eligible('synthetic_receipt_test');rp.close()
 files=list((R/'source/src/protein_joint_sac/rl').glob('sequence_*.py'))+list((R/'source/scripts/e016_v3').glob('readiness_*.py'))
 for p in files:compile(p.read_text(),str(p),'exec')
 result={'status':'PASS','unsealed_reward_refused':True,'unknown_failed_or_uncalibrated_receipt_refused_before_any_step':True,'mock_reward_replay_ineligible':True,'missing_global_metrics_calibration_NO_GO':True,'geometry_KL_primary_objective_bypass_refused':True,'compiled_source_files':len(files),'J1_optimizer_updates':0,'AF_calls':0,'scope':'defensive CPU tests, not AF certification'};(R/'FINAL_DEFENSIVE_TESTS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
