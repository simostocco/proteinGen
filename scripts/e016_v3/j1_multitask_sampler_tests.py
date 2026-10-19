"""CPU-only epoch-boundary and MASK-span tests; no model, optimizer or CUDA context."""
import copy,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from protein_joint_sac.models.bidirectional_multitask import SpanSampler
from protein_joint_sac.data.aa_only_supervised import ExpandedSampler
from protein_joint_sac.rl.sequence_first import Coverage
from protein_joint_sac.rl.checkpointing import save_atomic,load_checked
R=Path(__file__).resolve().parents[3]

def main():
 ids=[f'{i:064x}' for i in range(1,34)];co=ids[:11];pa=ids[11:22];un=ids[22:];seq={h:'A'*[20,64,128,256,384,500][i%6] for i,h in enumerate(ids)};owners={h:[h+'_obs'] for h in ids};pops={'TRAIN':{'complete':co,'partial_only':pa,'unused':un}};fake=SimpleNamespace(seqs=seq,populations=pops,base=SimpleNamespace(ids={'TRAIN':ids},jointids={'TRAIN':co},owners=owners),owners={'TRAIN':owners},cache={o:{'seen':np.array([(j%5)!=0 for j in range(len(seq[h]))],bool)} for h,obs in owners.items() for o in obs})
 ml=SpanSampler(fake,42);original=ExpandedSampler(fake,{'coverage':Coverage(42).state_dict(),'cursor':0});seenplans=[]
 for _ in range(3):seenplans.extend(ml.draw(16));original.draw()
 # Both samplers have wrapped multiple tiny fixture identity epochs.
 before={'masked':ml.state_dict(),'nextAA':original.state_dict()};p=R/'preflight/sampler_epoch_fixture.pt';save_atomic(p,before);loaded=load_checked(p);expected=[(ml.draw(16),original.draw()) for _ in range(5)];ml.load_state_dict(loaded['masked']);original.load_state_dict(loaded['nextAA']);actual=[(ml.draw(16),original.draw()) for _ in range(5)];assert actual==expected
 for masked,next_ in actual:
  for h,positions in masked:
   assert len(positions)==max(3,round(.15*len(seq[h]))) and len(set(positions))==len(positions);runs=[];run=1
   for a,b in zip(positions,positions[1:]):
    if b==a+1:run+=1
    else:runs.append(run);run=1
   runs.append(run);assert all(3<=v<=8 for v in runs)
 assert ml.epoch>=3 and not torch.cuda.is_initialized();out={'status':'PASS','controlled_fixture_only':True,'masked_sampler_epoch':ml.epoch,'AA_sampler_epochs':dict(original.epoch),'state_serialization_resume_exact_across_multiple_epochs':True,'MASK_effective_contiguous_runs_3_to8':True,'MASK_count_exact_rounded15pct':True,'CUDA_context_initialized':False,'optimizer_calls':0,'fixture_checkpoint':str(p)};(R/'SAMPLER_EPOCH_BOUNDARY_TESTS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
