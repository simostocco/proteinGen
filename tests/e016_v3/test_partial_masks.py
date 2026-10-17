import copy
import numpy as np
import pytest
import torch
from protein_joint_sac.data.engineering_fixtures import records
from protein_joint_sac.data.partial_geometry import derive,target,batch_targets
from protein_joint_sac.data.scaled_supervised import internal
from protein_joint_sac.data.certified_unified import canonical_teacher_prefix
from protein_joint_sac.models.batched_bc import collate,features_batch,encode_batch,losses
from protein_joint_sac.models.joint_actor import JointActor
from protein_joint_sac.models.dense_geometry_attention import ModelConfig

def observation():
 r=records()[0];x=np.array(r['coordinates']);return {'sequence':r['sequence'],'x':x,'seen':np.ones(len(x),bool),'links':np.ones(len(x)-1,bool)}
def model(chem=True):
 torch.manual_seed(7);return JointActor(ModelConfig(width=16,heads=2,layers=1,chemistry=chem))
def make(r,t=14):return batch_targets({'o':r},[('o',t)],'cpu')

def test_full_geometry_support_and_legacy_loss_parity():
 r=observation();g,m,seg=derive(r['x'],r['seen'],r['links']);np.testing.assert_allclose(g,internal(r['x']),atol=1e-12)
 for t in [0,1,2,3,12]:
  q=canonical_teacher_prefix(r['sequence'],r['x'],t,'a'*64);b,y,geo=make(r,t);a=collate([q.state],'cpu');net=model()
  torch.testing.assert_close(encode_batch(net,a),encode_batch(net,b),atol=2e-6,rtol=2e-5)
  old=losses(net,a,y,q.next_action.geometry[None]);new=losses(net,b,y,geo)
  for x,z in zip(old,new):torch.testing.assert_close(x,z,atol=2e-6,rtol=2e-5)

def test_single_gap_invalidates_all_dependent_windows():
 r=observation();r['seen'][7]=False;g,m,s=derive(r['x'],r['seen'],r['links'])
 for c,span in enumerate([2,3,4]):
  for i in range(20):assert m[i,c]==(i>=span-1 and not(i-span+1<=7<=i))
 assert s[6]!=s[8]

def test_extended_gap_does_not_reset_positions_or_create_adjacency():
 r=observation();r['seen'][5:10]=False;g,m,seg=derive(r['x'],r['seen'],r['links'])
 assert not m[10].any();assert m[11].tolist()==[True,False,False];assert m[12].tolist()==[True,True,False];assert m[13].all()
 b,y,g=make(r,12);assert b['t'].item()==12 and b['N'].item()==20;assert b['valid'].sum()==7
 assert not b['bond_valid'][0,10];assert b['bond_valid'][0,11]

def test_known_aa_with_missing_target_ca_remains_supervised():
 r=observation();r['seen'][14]=False;b,y,g=make(r);a,geo,parts=losses(model(),b,y,g)
 assert a.item()>0 and geo.item()==0 and not parts.any()

def test_fully_missing_context_has_zero_geometric_pairs():
 r=observation();r['seen'][:]=False;b,y,g=make(r);f,(pair,pv),present=features_batch(b)
 assert not pv.any();assert not pair[...,:8].any();net=model();a,geo,_=losses(net,b,y,g);assert torch.isfinite(a).all() and not geo.any()

def test_masked_placeholder_values_cannot_change_prediction_or_loss():
 r=observation();r['seen'][3:8]=False;b,y,g=make(r);c=copy.deepcopy(b);c['x'][~c['valid']]=float('nan');net=model()
 for a,z in zip(losses(net,b,y,g),losses(net,c,y,g)):torch.testing.assert_close(a,z,atol=0,rtol=0)

def test_native_suffix_swaps_cannot_change_prefix_encoding():
 r=observation();s=copy.deepcopy(r);s['sequence']=s['sequence'][:14]+'G'*6;s['x'][14:]=float('nan')
 # Construct actor prefix independently; target label/coordinate is not actor information.
 state=target(r['sequence'],r['x'],r['seen'],r['links'],13)[0]
 other=target(s['sequence'],s['x'],s['seen'],s['links'],13)[0]
 net=model();torch.testing.assert_close(encode_batch(net,collate([state],'cpu')),encode_batch(net,collate([other],'cpu')),atol=0,rtol=0)

def test_rigid_transform_invariance_for_partial_prefix():
 r=observation();r['seen'][3:5]=False;s=copy.deepcopy(r);q,_=np.linalg.qr(np.random.default_rng(4).normal(size=(3,3)));q[:,0]*=np.linalg.det(q);s['x']=s['x']@q+np.array([19.,-8.,3.]);b,y,g=make(r);c,z,h=make(s);net=model()
 torch.testing.assert_close(encode_batch(net,b),encode_batch(net,c),atol=2e-6,rtol=2e-5);torch.testing.assert_close(g,h,atol=1e-11,rtol=1e-11)

def test_observed_chain_break_is_not_missing_and_has_no_local_link():
 r=observation();r['links'][8]=False;g,m,s=derive(r['x'],r['seen'],r['links']);assert r['seen'].all();assert not m[9].any();assert m[10,0] and not m[10,1];assert s[8]!=s[9]

def test_unknown_aa_is_rejected_and_gly_pro_cys_preserved():
 r=observation();r['sequence']='X'+r['sequence'][1:]
 with pytest.raises(ValueError):make(r)
 r=observation();assert all(a in r['sequence'] for a in 'GPC')

def test_degenerate_observed_geometry_is_not_silently_masked():
 r=observation();r['x'][8]=r['x'][7]
 with pytest.raises(ValueError,match='OBSERVED_GEOMETRY'):derive(r['x'],r['seen'],r['links'])

def test_mixed_batch_backward_and_state_dict_compatibility():
 a=observation();b=copy.deepcopy(a);b['seen'][5:10]=False;net=model();net.load_state_dict(copy.deepcopy(net.state_dict()),strict=True)
 x,y,g=batch_targets({'a':a,'b':b},[('a',14),('b',8),('b',14),('a',0)],'cpu');aa,ge,_=losses(net,x,y,g);(aa+ge).mean().backward()
 assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_cached_mapping_string_labels_and_model_conformer_guards(monkeypatch):
 import prepare_partial_view as prep
 r=observation();seen=r['seen'].copy();seen[-2:]=False;N=len(seen)
 value={'chain_mapping_status':'unambiguous','model_id':'1','ambiguous_altloc_position_count':0,'deterministic_altloc_tie_count':0,'label_sequence_ids':[str(i+1) for i in range(N)],'residue_tokens':list(r['sequence']),'selected_altlocs':[None]*N,'missing_calpha_mask':(~seen).tolist(),'selected_calpha_coordinates':[v.tolist() if ok else None for v,ok in zip(r['x'],seen)],'auth_sequence_ids':[str(i+1) for i in range(N)],'insertion_codes':['A' if i==3 else '' for i in range(N)]}
 monkeypatch.setattr(prep,'raw',lambda row:value)
 x,mask,author,alt=prep.legacy({'model':'1'},{'canonical_full_sequence':r['sequence']});assert np.array_equal(mask,seen);assert author[('4A','A')]==3
 with pytest.raises(ValueError,match='MODEL'):prep.legacy({'model':'2'},{'canonical_full_sequence':r['sequence']})
 value['ambiguous_altloc_position_count']=1
 with pytest.raises(ValueError,match='CONFORMER'):prep.legacy({'model':'1'},{'canonical_full_sequence':r['sequence']})


def test_explicit_fp32_geometry_projection_preserves_fp32_density():
 r=observation();r['seen'][5:8]=False;b,y,g=make(r);net=model()
 for a,z in zip(losses(net,b,y,g),losses(net,b,y,g,geometry_fp32=True)):torch.testing.assert_close(a,z,atol=0,rtol=0)

def test_partial_signed_torsion_reflects_and_distance_angle_do_not():
 r=observation();r['seen'][4:6]=False;s=copy.deepcopy(r);s['x'][:,2]*=-1
 a,m,_=derive(r['x'],r['seen'],r['links']);b,n,_=derive(s['x'],s['seen'],s['links'])
 assert np.array_equal(m,n);np.testing.assert_allclose(a[:,:2],b[:,:2],atol=1e-12);np.testing.assert_allclose(a[:,2],-b[:,2],atol=1e-12)


def test_invalid_target_mask_shape_and_anchor_dependencies_fail_closed():
 r=observation();b,y,g=make(r);net=model();b['geometry_target_valid']=torch.ones(3,dtype=torch.bool)
 with pytest.raises(ValueError,match='B-by-3'):losses(net,b,y,g)
 b['geometry_target_valid']=torch.tensor([[False,True,False]])
 with pytest.raises(ValueError,match='anchors'):losses(net,b,y,g)

def test_new_lineage_retention_preserves_best_latest_and_rejects_links(tmp_path,monkeypatch):
 import partial_train as training
 from types import SimpleNamespace
 import hashlib,os
 monkeypatch.setattr(training,'R',tmp_path);d=tmp_path/'runs'/'J1-FULL';d.mkdir(parents=True)
 for u in [2500,5000,10000]:
  p=d/f'checkpoint_{u:05}.pt';p.write_bytes(str(u).encode());p.with_name(p.name+'.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest())
 engine=SimpleNamespace(update=10000,validation=[{'update':2500,'objective':1.},{'update':5000,'objective':2.},{'update':10000,'objective':3.}])
 training.retain_best_latest(d,engine);assert (d/'checkpoint_02500.pt').exists();assert (d/'checkpoint_10000.pt').exists();assert not (d/'checkpoint_05000.pt').exists();assert (d/'CHECKPOINT_RETENTION.jsonl').exists()
 p=d/'checkpoint_07500.pt';p.write_bytes(b'owned');p.with_name(p.name+'.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest());os.link(p,d/'unrelated_alias')
 with pytest.raises(RuntimeError,match='unsafe'):training.retain_best_latest(d,engine)
 assert p.exists()
