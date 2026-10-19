"""Compact final report clarification; no optimizer construction or updates."""
import json,sys,hashlib,platform
from pathlib import Path
import torch,numpy as np
import j1_aa_head_only as a
R=a.R;read=a.read;write=a.write

def main():
 p=R/'FINAL_J1_AA_HEAD_ONLY_REPORT.md';text=p.read_text()
 fixes={'Dropout0.15':'Dropout 0.15','augmentation0.075':'augmentation 0.075','LR5e-5,50-update':'LR 5e-5, 50-update','to5e-6 at1000':'to 5e-6 at 1,000','used32':'used 32','fixed10identities,2072decisions':'fixed 10 identities, 2,072 decisions','GiB;10GiB':'GiB; 10-GiB','full-model2500':'full-model +2,500','PPL17.303353':'PPL 17.303353','gap0.212570':'gap 0.212570','localVAL−0.108511':'local VAL −0.108511'}
 for x,y in fixes.items():text=text.replace(x,y)
 before,tail=text.split('## Final scientific assessment\n',1);end=tail[tail.index('Split limitation:'):]
 answers='''## Final scientific assessment

1. **7,060 parameters trained**: the independent AA linear layer only, with fresh AdamW moments. No shared output adapter or tied embedding was updated.
2. **All 12,344,686 other parameters and every buffer remained bitwise unchanged**, including every saved recovery state. Frozen gradients were absent; dropout remained active for fitting and disabled for scoring.
3. **Teacher-forced geometry was preserved exactly**, including distance, angle, torsion NLL, valid-target counts and per-target distribution parameters. VALIDATION local NLL remains −0.127963.
4. **The head learned from 32,000 real AA decisions on 32,000 distinct TRAIN identities**. Allocation was exactly 12,800 complete / 9,600 partial-only / 9,600 unused. All 118,219 identities were available; only 14.20% of the unused pool was visited. No geometry training loss was evaluated.
5. **Geometry-unavailable VALIDATION PPL improved 17.925712 → 17.597277**. NLL improvement was 0.018492 nats; paired identity-bootstrap 95% interval for the change was [−0.019515, −0.017508]. TRAIN PPL improved 16.537191 → 16.173069.
6. **Native-geometry VALIDATION PPL changed 15.597933 → 15.593204**, a very small point improvement. Native TRAIN PPL changed 11.384579 → 11.350801. Subgroup, length, position, accuracy and paired uncertainty results remain in the metric JSONs.
7. **Both gaps widened slightly**: native NLL gap 0.314879 → 0.317547; geometry-free gap 0.080624 → 0.084397. The latter increase, 0.003772 nats, is below the frozen 0.05 allowance.
8. **Generated-prefix AA gain was negligible** on ten fixed VALIDATION identities: PPL 19.631964 → 19.626155. NLL change 95% interval [−0.004549, +0.005161] includes zero. Matched native/unavailable/generated scores are reported separately.
9. **All 160 free generations completed, but the N=384 clash safeguard failed**: 3.653971 clashes/100 residues versus the frozen cap 3.545898. Parent mean was 2.832031; paired increase 95% interval [0.097453, 1.562500]. At that length Rg decreased 50.577 → 43.995 Å, while contacts increased; compactness without steric protection is insufficient. Sequence/structure diversity and other length-specific safeguards passed; this does not establish fold quality.
10. **No extension**: completed 500 unique head updates, plus two reexecuted restart-test calls. The recent +250→+500 held-out gain was only 0.000713 nats, below the frozen 0.001 requirement, and generation failed the clash safeguard.
11. **Retain the accepted parent**. The new +500 checkpoint is useful sequence research evidence, but does not qualify as a better joint generator.
12. **Selected full-state checkpoint**: `/mnt/d/Simone/proteinGen_data/e016_j1_torsion_stabilization_20261018_v1/replay/checkpoint_00500_full.pt`; SHA256 `9208263300fb9e88519ff0d40e6d9d29a9744b709d9a6181819cd0894b593000`. New research state: `/mnt/d/Simone/proteinGen_data/e016_j1_aa_head_only_20261019_v1/runs/J1/checkpoint_00500.pt`; SHA256 `4115eff6f1207046c9769cbb04e6036bc61a0a789b6491ccb8a9943702cf0275`. Both are fully recoverable; the parent remains immutable.
13. **Recommend RL preparation from the accepted parent**, without another large supervised continuation. Do not execute RL yet. Required preparation includes length-stratified native positive controls and deliberately invalid controls; calibrated reward components; an immutable reference policy with verified action densities/KL; and tests for reward exploitation, extreme extension, collapse, missingness and generation failures. Choose reward coefficients only after those measurements. AlphaFold confidence and Cα heuristics cannot certify biological stability.

'''
 p.write_text(before+answers+end)
 resources=read(R/'RESOURCE_AND_STORAGE_REPORT.json');resources['environment']={'Python':sys.version,'Torch':torch.__version__,'CUDA_build':torch.version.cuda,'NumPy':np.__version__,'platform':platform.platform()};write(R/'RESOURCE_AND_STORAGE_REPORT.json',resources)
 cp=a.s.load_checked(R/'runs/J1/checkpoint_00500.pt');groups=[]
 for pg in cp['optimizer']['param_groups']:groups.append({k:v for k,v in pg.items() if k!='params'})
 write(R/'ACTUAL_HEAD_OPTIMIZER_STATE_METADATA.json',{'fresh_head_only_lineage':True,'optimizer_head_steps':sorted({int(v['step']) for v in cp['optimizer']['state'].values()}),'parameter_groups':groups,'frozen_before_baseline_config_sha256':a.sha(R/'HEAD_ONLY_OPTIMIZER_CONFIG.json'),'source_frozen_receipt':read(R/'DECISIONS_FROZEN_BEFORE_BASELINE.json')})
 docs=R/'source/docs';docs.mkdir(exist_ok=True);(docs/'E016_J1_AA_HEAD_ONLY_RESULT.md').write_text('''# E016 J1 AA-head-only result

Training source commit: 462bb59. Evidence: `/mnt/d/Simone/proteinGen_data/e016_j1_aa_head_only_20261019_v1/FINAL_J1_AA_HEAD_ONLY_REPORT.md`.

Completed 500 unique head-only updates with 7,060 trainable parameters. Frozen parameters, buffers and teacher-forced geometry remained bitwise identical. Geometry-free VALIDATION PPL improved17.925712→17.597277, but N=384 free-generation clashes failed the unchanged safeguard. No extension or promotion; retain accepted parent and recommend RL preparation only. No TEST or RL executed. Full checkpoint paths/hashes, paired intervals, coverage, resource cost and exact restart evidence are in the evidence root.
''')
 print('Final report and actual optimizer/environment metadata published; no optimizer updates.')
if __name__=='__main__':main()
