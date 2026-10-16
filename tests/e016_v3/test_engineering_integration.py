import copy,json,math,random
from dataclasses import replace
import numpy as np
import pytest,torch
from protein_joint_sac.rl.bc_training import BCConfig,Population,BCTrainer
from protein_joint_sac.data.engineering_fixtures import records
from protein_joint_sac.data.certified_unified import canonical_teacher_prefix
from protein_joint_sac.geometry.ca_builder import append,replay,mask_for_step
from protein_joint_sac.contracts import BuildAction,BuildEpisode,GeneratedProtein,RewardCard,OracleProtocol
from protein_joint_sac.models.joint_actor import JointActor
from protein_joint_sac.models.build_critic import TwinCritics
from protein_joint_sac.models.dense_geometry_attention import ModelConfig
from protein_joint_sac.rl.hybrid_sac_build import HybridSAC
from protein_joint_sac.rl.delayed_episode_replay import EpisodeReplay
from protein_joint_sac.rl.replay_controller import ReplayController
from protein_joint_sac.evaluation.bc_metrics import generate
from governance.e016_v1.allocation import Group,Evidence,build_groups,allocate

def small(mode='joint',chemistry=True,device='cpu'):
    return BCConfig(mode=mode,width=16,heads=2,layers=1,chemistry=chemistry,batch_size=2,scheduler_updates=20,device=device)
def population():return Population(records(),'ENGINEERING_FIXTURE')
def sac(implementation='streamed_exact',device='cpu'):
    torch.manual_seed(4);c=ModelConfig(width=16,heads=2,layers=1)
    return HybridSAC(JointActor(c).to(device),TwinCritics(c).to(device),actor_backward=implementation)
def actions(N=20):
    return tuple(BuildAction('A',torch.tensor([3.8 if i else 0.,1.9 if i>=2 else 0.,.4 if i>=3 else 0.],dtype=torch.float64),mask_for_step(i)) for i in range(N))

@pytest.mark.parametrize('mode,chemistry',[('sequence',True),('joint',True),('joint',False)])
def test_training_validation_generation_and_exact_restart(d_tmp,mode,chemistry):
    pop=population();t=BCTrainer(small(mode,chemistry),pop)
    for _ in range(3):t.step()
    t.validate(2);path=d_tmp/'checkpoint.pt';t.checkpoint(path)
    expected=[t.step() for _ in range(3)]
    state={k:v.clone() for k,v in t.model.state_dict().items()};opt=copy.deepcopy(t.optimizer.state_dict());next_random=(random.random(),float(np.random.random()),torch.rand(3))
    resumed=BCTrainer(small(mode,chemistry),pop);resumed.resume(path);actual=[resumed.step() for _ in range(3)]
    assert actual==expected
    for k,v in state.items():torch.testing.assert_close(resumed.model.state_dict()[k],v,atol=0,rtol=0)
    for k,vs in opt['state'].items():
        for name,v in vs.items():
            if isinstance(v,torch.Tensor):torch.testing.assert_close(resumed.optimizer.state_dict()['state'][k][name],v,atol=0,rtol=0)
    assert random.random()==next_random[0] and float(np.random.random())==next_random[1]
    torch.testing.assert_close(torch.rand(3),next_random[2],atol=0,rtol=0)
    assert math.isfinite(resumed.validate(3)['aa_perplexity'])
    g=generate(resumed.model,20,mode);assert len(g['sequence'])==20 and (g['geometry'] is None)==(mode=='sequence')

def test_resume_rejects_changed_config_and_population(d_tmp):
    p=population();t=BCTrainer(small(),p);t.step();f=d_tmp/'a.pt';t.checkpoint(f)
    with pytest.raises(ValueError,match='mismatch'):BCTrainer(replace(small(),chemistry=False),p).resume(f)
    rows=records();rows[0]['source_sha']='f'*64
    with pytest.raises(ValueError,match='mismatch'):BCTrainer(small(),Population(rows,'ENGINEERING_FIXTURE')).resume(f)
    with pytest.raises(ValueError,match='exists'):t.checkpoint(f)

def test_sequence_mode_reads_no_geometry():
    rows=records();rows[0].pop('coordinates');p=Population(rows,'ENGINEERING_FIXTURE');t=BCTrainer(small('sequence'),p);t.step()
    s,label,g=p.target(0,12,'sequence','cpu');assert g is None and not s.valid.any()

@pytest.mark.parametrize('field,value',[('fragment',True),('label_positions',list(range(2,22))),('role','NATIVE_AF_CALIBRATION')])
def test_full_polymer_and_role_admission(field,value):
    rows=records();rows[0][field]=value
    with pytest.raises(ValueError):Population(rows,'ENGINEERING_FIXTURE')

def test_literal_prefix_rotation_and_next_action():
    row=records()[0];x=torch.tensor(row['coordinates'],dtype=torch.float64);rot=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))[0]
    if torch.linalg.det(rot)<0:rot[:,0]*=-1
    for t in [0,1,2,3,12,19]:
        a=canonical_teacher_prefix(row['sequence'],x,t,row['source_sha']);b=canonical_teacher_prefix(row['sequence'],x@rot+20,t,row['source_sha'])
        torch.testing.assert_close(a.state.ca,b.state.ca,atol=1e-9,rtol=1e-9);torch.testing.assert_close(a.next_action.geometry,b.next_action.geometry,atol=1e-9,rtol=1e-9)
        assert append(a.state,a.next_action).residues[-1]==row['sequence'][t]

def test_sac_update_eager_streamed_full_parameter_parity():
    eager=sac('eager_exact');streamed=sac();a=actions();s=replay(20,a[:4]);nxt=append(s,a[4])
    torch.manual_seed(202);em=eager.update_one(s,a[4],.3,nxt,False)
    torch.manual_seed(202);sm=streamed.update_one(s,a[4],.3,nxt,False)
    for m1,m2 in [(eager.actor,streamed.actor),(eager.critics,streamed.critics),(eager.temperatures,streamed.temperatures)]:
        for p,q in zip(m1.parameters(),m2.parameters()):
            torch.testing.assert_close(p,q,atol=3e-6,rtol=3e-5)
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=3e-6,rtol=3e-5)
    assert sm['actor_backward']=='streamed_exact'
    for key in ['actor_loss','critic_loss','temperature_loss']:assert sm[key]==pytest.approx(em[key],abs=3e-6)

def test_replay_pending_terminal_credit_and_restart(d_tmp):
    a=actions();ep=BuildEpisode('episode',20,a);db=EpisodeReplay(d_tmp/'a.sqlite');protein=GeneratedProtein(replay(20,a));db.store(ep,protein);c=ReplayController(sac(),db)
    with pytest.raises(ValueError,match='pending'):c.transition('episode',19,'ORACLE_RESOLVED',[0.]*20)
    db.attach('episode',RewardCard('toy',protein.sequence_sha256,protein.geometry_sha256,'a'*64,'MOCK_UNIT_TEST',{'value':2.}))
    assert c.transition('episode',19,'ORACLE_RESOLVED',[0.]*20)[2]==2. and c.transition('episode',0,'ORACLE_RESOLVED',[0.]*20)[2]==0.
    c.sample_update('ORACLE_RESOLVED',{'episode':[0.]*20});path=d_tmp/'sac.pt';c.snapshot(path)
    expected=c.sample_update('ORACLE_RESOLVED',{'episode':[0.]*20});params=[p.clone() for p in c.algorithm.actor.parameters()]
    db2=EpisodeReplay(d_tmp/'b.sqlite');restored=ReplayController(sac(),db2);restored.restore(path);actual=restored.sample_update('ORACLE_RESOLVED',{'episode':[0.]*20});assert actual==expected
    for p,q in zip(params,restored.algorithm.actor.parameters()):torch.testing.assert_close(p,q,atol=0,rtol=0)
    db.close();db2.close()

def test_long_horizon_credit_is_not_near_terminal_transition():
    from protein_joint_sac.rl.soft_targets import monte_carlo_returns,n_step_target
    rewards=[0.]*499+[1.];r=monte_carlo_returns(rewards,.99)
    assert r[0]==pytest.approx(.99**499) and r[-1]==1. and r[0]<.007
    assert n_step_target(rewards,1.,0.,True)==1.

def test_typed_evidence_rejects_forest_as_complete():
    e=Evidence('a','b','similarity','a'*64,'RETAINED_FOREST_ONLY')
    with pytest.raises(ValueError,match='incomplete'):build_groups(['a','b'],[e],{'similarity':'HARD_CLOSURE'},scope='SYNTHETIC_ONLY')
    with pytest.raises(ValueError,match='approval'):build_groups(['a','b'],[],{},scope='PROVISIONAL_REAL_DIAGNOSTIC')

def test_allocator_joint_views_calibration_and_no_singleton_repair():
    groups=[Group(str(i),10,5,5) for i in range(12)];accept={'groups':2,'joint_groups':2,'sequences':10,'joint_observations':5,'joint_identities':5}
    r=allocate(groups,[('0','CALIBRATION','REWARD_DEVELOPMENT')],accept)
    assert r['status']=='DIAGNOSTIC_PASS_NOT_CERTIFICATION' and not r['certified'];assert r==allocate(groups,[('0','CALIBRATION','REWARD_DEVELOPMENT')],accept);assert r['assignments']['0']=='CALIBRATION'
    assert allocate([Group('a',20,0,0)],[],accept)['status'].startswith('INFEASIBLE')
    assert allocate(groups,[('0','TRAIN','ORDINARY'),('0','TEST','ORDINARY')],accept)['status']=='INFEASIBLE_RESERVATION_CONFLICT'

def test_saved_AF_output_mapping_cache_and_comparison(d_tmp):
    import gemmi
    from protein_joint_sac.oracle.adapter import import_saved_result,process_queued_saved
    from protein_joint_sac.oracle.queue import EvaluationQueue
    protein=GeneratedProtein(replay(20,actions()));seq=protein.sequence;pdb=d_tmp/'prediction.pdb';structure=gemmi.Structure();model=gemmi.Model('1');chain=gemmi.Chain('A')
    for i in range(20):
        res=gemmi.Residue();res.name='ALA';res.seqid=gemmi.SeqId(i+1,' ');atom=gemmi.Atom();atom.name='CA';atom.element=gemmi.Element('C');atom.pos=gemmi.Position(i*3.8,0,0);res.add_atom(atom);chain.add_residue(res)
    model.add_chain(chain);structure.add_model(model);structure.write_pdb(str(pdb));scores=d_tmp/'scores.json';scores.write_text(json.dumps({'plddt':[80.]*20,'ptm':.6,'pae':[[2.]*20]*20}))
    protocol=OracleProtocol('LOCAL_AF2','test',('a'*64,),'single_sequence','none',(7,),3,'PLANNED_UNCALIBRATED');result=import_saved_result(seq,pdb,scores,protocol.protocol_sha256,7)
    q=EvaluationQueue(d_tmp/'q.sqlite',max_calls=1);q.enqueue('ep',protein,protocol,7);assert q.claim('ep');compared=process_queued_saved(q,'ep',protein,result);assert compared['optimized_TM_score'] is None
    q.enqueue('ep2',protein,protocol,7);assert q.row('ep2')['status']=='SUCCEEDED' and q.calls_used==1
    scores.write_text(json.dumps({'pae':[[float('nan')]*20]*20}))
    with pytest.raises(ValueError,match='PAE'):import_saved_result(seq,pdb,scores,protocol.protocol_sha256,7)
    with pytest.raises(ValueError,match='mismatch'):import_saved_result('C'*20,pdb,scores,protocol.protocol_sha256,7)
    q.close()

@pytest.mark.skipif(not torch.cuda.is_available(),reason='bounded CUDA changed-path check')
def test_changed_SAC_and_BC_CPU_CUDA_small_equivalence(d_tmp):
    import time,os
    from pathlib import Path
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    cpu=BCTrainer(small(),population());gpu=BCTrainer(small(device='cuda'),population());gpu.model.load_state_dict(cpu.model.state_dict());a=cpu.loss(0,4)[0];b=gpu.loss(0,4)[0]
    torch.testing.assert_close(a,b.cpu(),atol=1e-4,rtol=1e-4);a.backward();b.backward()
    for p,q in zip(cpu.model.parameters(),gpu.model.parameters()):
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad.cpu(),atol=1e-4,rtol=1e-3)
    alg=sac(device='cuda');acts=actions();s=replay(20,acts[:4],device='cuda');action=BuildAction(acts[4].residue,acts[4].geometry.cuda(),acts[4].mask);result=alg.update_one(s,action,.1,append(s,action),False)
    assert math.isfinite(result['actor_loss'])
    torch.cuda.synchronize()
    receipt={'scope':'SMALL_CHANGED_PATH_ENGINEERING_ONLY','seconds':time.monotonic()-start,'CUDA_allocated_peak_bytes':torch.cuda.max_memory_allocated(),'CUDA_reserved_peak_bytes':torch.cuda.max_memory_reserved(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'GPU':torch.cuda.get_device_name(),'BC_CPU_CUDA_loss_absolute_error':float((a-b.cpu()).abs().detach()),'SAC_actor_backward':result['actor_backward'],'production_training':False}
    (Path(os.environ['E016_ARTIFACT_ROOT'])/'tests/CUDA_CHANGED_PATH.json').write_text(json.dumps(receipt,indent=2))


def test_actual_candidate_release_remains_rejected():
    from protein_joint_sac.contracts import DatasetRef
    ref=json.load(open('/mnt/d/Simone/proteinGen_data/e016_gpu_unified_pdb_20261015_v1/DATASET_REF.json'));ref.pop('version')
    with pytest.raises(ValueError,match='certified'):Population.certified(DatasetRef(**ref),'joint')


def test_uniform_identity_sampling_not_observation_multiplicity():
    from protein_joint_sac.rl.bc_training import PrefixSampler
    rows=records();alias=copy.deepcopy(rows[0]);alias['observation_id']='toy-alias';rows.append(alias)
    p=Population(rows,'ENGINEERING_FIXTURE');sampler=PrefixSampler(p,53);counts={h:0 for h in p.groups['TRAIN']}
    for i,t in sampler.draw(4000):counts[p.rows[i]['sequence_hash']]+=1
    assert max(counts.values())/min(counts.values())<1.25


def test_allocator_reports_strata_and_explicit_context_conflicts():
    groups=[Group(str(i),10,5,5,(('20-64',10),),(('X-RAY',10),)) for i in range(10)]
    accept={'groups':2,'joint_groups':2,'sequences':10,'joint_observations':5,'joint_identities':5};r=allocate(groups,[],accept)
    assert sum(x['length_counts'].get('20-64',0) for x in r['strata'].values())==100
    a=next(g for g,sp in r['assignments'].items() if sp=='TRAIN');b=next(g for g,sp in r['assignments'].items() if sp=='TEST')
    challenged=allocate(groups,[],accept,context_restrictions=[(a,b,'synthetic-direct-source-embargo')])
    assert challenged['status']=='NO_VALID_ALLOCATION_FOUND' and challenged['context_conflicts']


def test_real_backend_dry_plan_identity_and_no_unapproved_launch(d_tmp):
    from protein_joint_sac.oracle.adapter import AF2Backend,RealAlphaFoldAdapter
    backend=AF2Backend(str(d_tmp/'python'),str(d_tmp/'source'),'b'*40,'c'*64,str(d_tmp/'weights'),'a'*64,str(d_tmp/'seal.json'),'d'*64)
    protocol=OracleProtocol('LOCAL_COLABFOLD_AF2_PTM','b'*40,('a'*64,),'single_sequence','none',(7,),3,'PLANNED_UNCALIBRATED')
    adapter=RealAlphaFoldAdapter(protocol,backend);protein=GeneratedProtein(replay(20,actions()))
    plan=adapter.prepare(protein.sequence,7,d_tmp/'output')
    assert plan['protocol_sha256']==adapter.bound_protocol.protocol_sha256 and not plan['execution_authorized']
    assert 'single_sequence' in plan['argv'] and '--templates' not in plan['argv']
    changed=RealAlphaFoldAdapter(protocol,replace(backend,precision_contract='DIFFERENT_EXPERIMENT'))
    assert changed.prepare(protein.sequence,7,d_tmp/'other')['protocol_sha256']!=plan['protocol_sha256']
    with pytest.raises(PermissionError):adapter.evaluate(protein,7,d_tmp/'output')
    assert not (d_tmp/'output').exists()


def test_native_control_preparation_preserves_five_existing_identities():
    from protein_joint_sac.oracle.native_calibration import prepare_controls
    panel=json.load(open('/mnt/d/Simone/proteinGen_data/e016_v3_blocker_resolution_20261014_v1/NATIVE_CONTROL_PANEL.json'))
    result=prepare_controls(panel);assert len(result['controls'])==5 and result['real_AF_calls']==0 and not result['calibration_pass']


def test_completed_comparison_supports_FP32_CA():
    from protein_joint_sac.oracle.fold_agreement import compare_completed
    from protein_joint_sac.oracle.mock_oracle import MockOracle
    from protein_joint_sac.contracts import BuildState
    a=replay(20,actions());protein=GeneratedProtein(BuildState(a.residues,a.ca.float(),20));proto=OracleProtocol('MOCK_UNIT_TEST','toy',(),'none','none',(7,),0,'SYNTHETIC_ONLY')
    result=compare_completed(protein,MockOracle(proto).evaluate(protein,7));assert math.isfinite(result['fixed_mapping_Kabsch_RMSD_A'])


def test_real_population_constructor_cannot_bypass_admission():
    with pytest.raises(ValueError,match='factory'):Population(records(),'CERTIFIED_PILOT')


def test_bounded_candidate_evidence_recovery_is_not_forest_subtraction():
    from governance.e016_v1.recovery import missing_candidate_body,recover_bounded
    original=b'1 50 0\n2 20 0\n3 10 0\n';aligned=b'2 20 0\n'
    assert missing_candidate_body(original,aligned)==b'1 50 0\n3 10 0\n'
    with pytest.raises(ValueError):missing_candidate_body(original,b'9 10 0\n')
    class Writer:
        def __init__(self):self.rows=[]
        def add(self,k,v):self.rows.append((k,v))
    w=Writer();r=recover_bounded([(0,original),(1,original)],lambda key:aligned,w,max_queries=1,max_output_bytes=100)
    assert r['query_keys']==[0] and r['next_unprocessed_query']==1 and r['missing_candidate_pairs_extracted']==2 and not r['alignment_executed']


def test_many_observations_in_one_group_cannot_satisfy_joint_independence():
    groups=[Group('one-joint-family',100,1000,100)]+[Group(str(i),10,0,0) for i in range(10)]
    accept={'groups':2,'joint_groups':2,'sequences':10,'joint_observations':5,'joint_identities':5}
    r=allocate(groups,[],accept)
    assert r['status']=='INFEASIBLE_NECESSARY_EVALUATION_COUNTS' and r['available']['joint_groups']==1
