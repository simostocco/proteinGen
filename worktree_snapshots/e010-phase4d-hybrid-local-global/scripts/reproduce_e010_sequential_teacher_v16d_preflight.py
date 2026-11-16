"""Independently verify saved one-step preflight states; no optimization."""
import argparse,json
import numpy as np
import torch
from protein_distance_diffusion.training.e010_phase4d_diagnostic import file_hash
from scripts.run_e010_sequential_teacher_v16d import OUT,ROOT,teacher,synthetic,write


def reproduce(n):
    torch.set_num_threads(1)
    row=json.loads((OUT/'preflight'/f'length_{n}.json').read_text())
    path=ROOT/row['state_file'];assert file_hash(path)==row['state_file_sha256']
    with np.load(path) as raw:
        a=raw['delta'].copy();p=raw['prediction'].copy();pg=raw['pg'].copy();target=raw['target'].copy();mask=raw['mask'].copy()
    b=synthetic(n);c=teacher.Context(b);s=teacher.OneStep(c,b['pg'])
    assert np.array_equal(pg,b['pg'].numpy()) and np.array_equal(target,b['target'].numpy()) and np.array_equal(mask,b['mask'].numpy())
    assert np.array_equal(p,pg+a) and (a[~s.eligible.numpy()]==0).all()
    assert np.array_equal(teacher.project(a)[0],a)
    assert teacher.digest(a)==row['action_sha256']
    safety=c.safety(torch.from_numpy(p));assert json.loads(json.dumps(safety))==row['safety'] and safety['passes']
    value=float(c.values(torch.from_numpy(p))['local']/c.baseline['local'])
    assert value==row['solver']['accepted_normalized_local']
    u=np.einsum('bnji,bnj->bni',s.frame.numpy(),a)
    assert np.allclose(np.einsum('bnij,bnj->bni',s.frame.numpy(),u),a,atol=1e-14,rtol=1e-12)
    write(OUT/'preflight_reproduction'/f'length_{n}.json',dict(length=n,exact=True,optimizer_invocations=0,
              state_sha256=file_hash(path),local_action_sha256=teacher.digest(u),all_safety_pass=True))
    print('preflight independently reproduced',n,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('lengths',nargs='+',type=int)
    for n in p.parse_args().lengths:reproduce(n)
