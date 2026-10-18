import torch
from j1_sequence_archive import archive_model
from protein_joint_sac.rl.checkpointing import load_checked,save_atomic
def test_model_archive_keeps_exact_weights_and_source_provenance(tmp_path):
 source=tmp_path/'source.pt';dest=tmp_path/'model.pt'
 sha=save_atomic(source,{'model':{'weight':torch.arange(6.)},'lineage':{'synthetic':True},'additional_update':2500,'optimizer':{'synthetic':True}})
 archive_model(source,dest);v=load_checked(dest)
 torch.testing.assert_close(v['model']['weight'],torch.arange(6.),atol=0,rtol=0)
 assert v['source_full_checkpoint_sha256']==sha and v['additional_update']==2500 and 'optimizer' not in v
