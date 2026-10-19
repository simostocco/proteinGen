"""CPU queue/reward correctness tests; synthetic evidence never AF calibration."""
import json,os,sqlite3,subprocess,sys,hashlib
from pathlib import Path
import numpy as np
import torch
from protein_joint_sac.contracts import *
from protein_joint_sac.geometry.ca_builder import replay,mask_for_step
from protein_joint_sac.oracle.queue import EvaluationQueue
from protein_joint_sac.oracle.mock_oracle import MockOracle
from protein_joint_sac.rl.sequence_replay import SequenceReplay,OracleWallBudget
from protein_joint_sac.rl.sequence_rewards import sequence_features,calibrated_sequence_reward,combine_rewards,mapped_structure_scores
R=Path(__file__).resolve().parents[3]
def write(n,v):(R/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 actions=tuple(BuildAction('A',torch.tensor([3.8 if t>=1 else 0,1.9 if t>=2 else 0,.7 if t>=3 else 0],dtype=torch.float64),mask_for_step(t)) for t in range(20));state=replay(20,actions);protein=GeneratedProtein(state);rot=state.ca@torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],dtype=torch.float64)+2;other=GeneratedProtein(BuildState(state.residues,rot,20));proto=OracleProtocol('MOCK_UNIT_TEST','test_v1',(),'none','none',(7,),0,'SYNTHETIC_ONLY');path=R/'tests/queue.sqlite'
 q=EvaluationQueue(path,max_requests=8,max_calls=3);q.enqueue('first',protein,proto,7);q.enqueue('same_sequence_other_backbone',other,proto,7);assert q.claim('first');result=MockOracle(proto).evaluate(protein,7);q.finish('first',result);assert not q.claim('same_sequence_other_backbone') and q.calls_used==1;assert q.row('same_sequence_other_backbone')['geometry_sha']!=q.row('first')['geometry_sha'];q.enqueue('interrupted',protein,OracleProtocol('MOCK_UNIT_TEST','test_v2',(),'none','none',(7,),0,'SYNTHETIC_ONLY'),7);q.claim('interrupted');q.close();q=EvaluationQueue(path,max_requests=8,max_calls=3);assert q.recover_running()==1 and q.row('interrupted')['status']=='TIMEOUT' and q.calls_used==2
 budget=OracleWallBudget(R/'tests/wall.sqlite',16,2700);budget.reserve('crash_attempt',360);budget.db.close();budget=OracleWallBudget(R/'tests/wall.sqlite',16,2700);assert budget.report()['charged_seconds']==360
 db=R/'tests/uncommitted.sqlite';con=sqlite3.connect(db);con.execute('CREATE TABLE IF NOT EXISTS t(x)');con.commit();con.close();subprocess.run([sys.executable,'-c',"import sqlite3,os; c=sqlite3.connect("+repr(str(db))+");c.execute('BEGIN IMMEDIATE');c.execute('INSERT INTO t VALUES(1)');os._exit(0)"],check=True);con=sqlite3.connect(db);assert con.execute('SELECT count(*) FROM t').fetchone()[0]==0;con.close()
 seq='ACDEFGHIKLMNPQRSTVWY';x=state.ca.numpy();pae=np.full((20,20),5.);f=sequence_features(seq,np.full(20,85.),.8,pae,x);bad=sequence_features(seq,np.full(20,95.),.2,np.full((20,20),25.),x);assert f['global_confidence_product']>bad['global_confidence_product'];missing=sequence_features(seq,np.full(20,95.),None,None,x);assert missing['global_confidence_product'] is None
 rejected=False
 try:calibrated_sequence_reward(f,{'GO':False})
 except PermissionError:rejected=True
 assert rejected and combine_rewards(1.,-1.,.05)['total']>combine_rewards(.1,1.,.05)['total']
 repeat=sequence_features('A'*20,np.full(20,99.),.9,pae,x);assert repeat['composition_entropy_nats']==0 and repeat['longest_identical_run']==20
 truth=mapped_structure_scores(x,x);rigid=mapped_structure_scores(x,x@np.array([[0.,-1,0.],[1.,0,0.],[0,0,1.]])+5);collapse=mapped_structure_scores(x,np.zeros_like(x));assert truth['robust_fixed_map_TM_similarity_fullN']>1-1e-10 and rigid['robust_fixed_map_TM_similarity_fullN']>1-1e-10 and collapse['robust_fixed_map_TM_similarity_fullN']<.5
 write('REPLAY_QUEUE_CORRECTNESS.json',{'status':'PASS','same_sequence_different_geometry_cache_shared':True,'prediction_calls_for_two_proposals':1,'geometry_association_separate':True,'interrupted_RUNNING_to_TIMEOUT':True,'budget_preserved_after_restart':True,'unclosed_wall_reservation_seconds':360,'uncommitted_transaction_rollback_after_process_exit':True,'scope':'SQLite crash/transaction tests; not a simulated whole-PC power failure','mock_evidence_not_real_AF':True})
 write('REWARD_EXPLOITATION_TESTS.json',{'status':'PASS','synthetic_only_test_fixtures':True,'good_sequence_not_erased_by_poor_proposal_geometry':True,'no_sequence_reward_times_backbone_agreement':True,'high_local_low_global_confidence_not_sufficient':True,'missing_pTM_PAE_not_fabricated':True,'uncalibrated_reward_refused':True,'homopolymer_complexity_alarm':True,'rigid_transform_invariance':True,'exact_native_reconstruction_score':truth,'collapsed_trace_detected':True,'variant_nonfolding_status':'UNKNOWN, not experimental negatives'})
 print('CPU replay, cache, failure, wall-budget and separate-reward tests PASS',flush=True)
if __name__=='__main__':main()
