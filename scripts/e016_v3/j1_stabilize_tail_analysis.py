"""Paired loss-tail decomposition on frozen saved arrays; no model or optimizer."""
import json,math
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[3];hist=json.load(open(R/'TRAIN_VALIDATION_PERPLEXITY.json'));out={}
for role in ['TRAIN','VALIDATION']:
 base=np.load(R/f'calibration/{role}_00000.npz');out[role]={}
 for h in hist[1:]:
  u=h['new_update'];z=np.load(R/f'calibration/{role}_{u:05d}.npz')
  if any(not np.array_equal(base[k],z[k]) for k in ['g','mask','N','t']):raise ValueError('frozen native target arrays changed')
  ok=base['mask'][:,2];a=base['nll'][ok,2];b=z['nll'][ok,2];delta=b-a;n=len(a);k=math.ceil(.01*n);src=np.argsort(a)[-k:];end=np.argsort(b)[-k:];union=np.union1d(src,end);seam=np.abs(base['g'][ok,2])>math.pi-math.radians(10)
  def part(ix):return {'targets':len(ix),'contribution_to_mean_delta_nats':float(delta[ix].sum()/n),'remaining_contribution_nats':float((delta.sum()-delta[ix].sum())/n)}
  out[role][str(u)]={'valid_torsion_targets':n,'paired_mean_delta_nats':float(delta.mean()),'source_defined_worst1pct':part(src),'candidate_defined_worst1pct':part(end),'union_worst1pct':part(union),'native_seam_within10deg':part(np.flatnonzero(seam)),'analysis_only_no_target_excluded':True}
  z.close()
 base.close()
(R/'TORSION_PAIRED_TAIL_ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n');print('PAIRED_TAIL_DECOMPOSITION_COMPLETE')
