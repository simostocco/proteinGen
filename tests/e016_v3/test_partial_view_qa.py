"""Engineering-only independent view QA fixtures, never real split assignments."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import verify_partial_view as qa
from protein_joint_sac.data.partial_geometry import derive
def fixture_view(tmp_path,monkeypatch):
 root=tmp_path;u=root/'U';p=root/'P';(root/'view/continuity_recovery_journal').mkdir(parents=True);(u/'release/intake_v1').mkdir(parents=True);p.mkdir();(p/'PROPOSAL_MANIFEST.json').write_text('{}')
 monkeypatch.setattr(qa,'R',root);monkeypatch.setattr(qa,'U',u);monkeypatch.setattr(qa,'P',p)
 parents={};rows=[];intake=[];journal=[]
 for j,s in enumerate(['TRAIN','VALIDATION']):
  h='synthetic'+str(j);oid='synthetic-obs'+str(j);N=20;t=np.arange(N);x=np.column_stack([np.cos(t),np.sin(t),1.5*t]);seen=np.ones(N,bool);seen[5:7]=False;links=seen[:-1]&seen[1:];g,m,seg=derive(x,seen,links);parents[h]={'split':s,'length':N,'joint_eligible':False}
  row={'observation_id':oid,'sequence_hash':h,'split':s,'polymer_length':N,'aa_known':[True]*N,'ca_observed':seen.tolist(),'continuity':links.tolist(),'coordinate_segment_id':seg.tolist(),'distance_targets':int(m[:,0].sum()),'angle_targets':int(m[:,1].sum()),'torsion_targets':int(m[:,2].sum()),'AA_targets_at_missing_CA':2,'AA_only_targets':int((~m.any(1)).sum()),'method':'X-RAY DIFFRACTION','coordinate_kind':'legacy_cache','source_sha':'synthetic-source','coordinate_path':str(root/'synthetic-unused-source'),'coordinate_row':j,'model':'1'}
  rows.append(row);intake.append({k:row[k] for k in ['observation_id','sequence_hash','source_sha','coordinate_path','coordinate_row','model']});intake[-1].update(complete_CA=False,mapping_status='VERIFIED_LABEL_CORRESPONDENCE');journal.append([oid,np.flatnonzero(links).tolist(),{'path':'synthetic-sidecar','row':j}])
 pq.write_table(pa.Table.from_pylist(rows),root/'view/PARTIAL_OBSERVATIONS.parquet');pq.write_table(pa.Table.from_pylist(intake),u/'release/intake_v1/chain_observations.parquet');qa.write(root/'view/continuity_recovery_journal/synthetic.json',{'version':'continuity-recovery-v1','links':journal});monkeypatch.setattr(qa,'PDBSplitProposal',lambda _:type('Fixture',(),{'sequences':parents})())
 return root,rows,journal
def test_independent_view_qa_and_journal_recovery(tmp_path,monkeypatch):
 r,rows,journal=fixture_view(tmp_path,monkeypatch);qa.main()
 import json
 receipt=json.loads((r/'view/PARTIAL_VIEW_QA.json').read_text());assert receipt['status']=='PASS';assert receipt['splits']['TRAIN']['observations_and_targets']['AA_at_missing_CA']==2
 journal[0][1].pop();qa.write(r/'view/continuity_recovery_journal/synthetic.json',{'version':'continuity-recovery-v1','links':journal})
 with pytest.raises(ValueError,match='durable continuity replay'):qa.main()
def test_independent_qa_rejects_false_target_count(tmp_path,monkeypatch):
 r,rows,_=fixture_view(tmp_path,monkeypatch);rows[0]['distance_targets']-=1;pq.write_table(pa.Table.from_pylist(rows),r/'view/PARTIAL_OBSERVATIONS.parquet')
 with pytest.raises(ValueError,match='stencil counts'):qa.main()
