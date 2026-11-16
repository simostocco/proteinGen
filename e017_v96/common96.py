import copy, datetime, gzip, hashlib, json, os, runpy, sys
from pathlib import Path
ROOT = Path('/mnt/d/Simone/proteinGen_data/e017_v96_burial_conditioning_20261029_v1')
V95 = ROOT.parent / 'e017_v95_expanded_validation_20261028_v1'
R3 = ROOT.parent / 'e017_v94r3_substantial_geometry_rl_20261026_v1'
sys.path.insert(0, str(V95 / 'source'))
import common as C
import numpy as np


def read(p):
    p = Path(p)
    with (gzip.open(p, 'rt') if p.suffix == '.gz' else p.open()) as f:
        return json.load(f)


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(name, obj):
    path = ROOT / name
    assert path.resolve().is_relative_to(ROOT.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name+'.part')
    payload = json.dumps(obj, indent=2, allow_nan=False).encode()
    if path.suffix == '.gz':
        payload = gzip.compress(payload, mtime=0)
    if path.exists():
        assert path.read_bytes()==payload, ('IMMUTABLE_PUBLICATION_CHANGED',str(path))
        return
    with partial.open('xb') as f:
        f.write(payload); f.flush(); os.fsync(f.fileno())
    os.replace(partial, path)


def load():
    return C.load()


def ns(*items):
    return int(hashlib.sha256(('E017_V96|'+ '|'.join(map(str,items))).encode()).hexdigest()[:15],16)


def alternatives(task, reference, core, spec):
    """Same sequence, thresholds, motif: only the requested interval changes."""
    seq = task['fixed_sequence']; goal = task['goal']; x = np.asarray(reference['coordinates'])
    found = {}
    for tag in ['burial', 'exposure']:
        region = 'buried' if tag == 'burial' else 'exposed'
        other = 'exposed' if tag == 'burial' else 'buried'
        for start in range(4, 53, 4):
            if abs(start-goal[region][0]) < 8 or abs(start-goal[other][0]) < 16:
                continue
            alt = copy.deepcopy(task); alt['goal'][region] = [start,start+8]
            alt['goal']['goal_id'] = goal['goal_id']+'_V96_'+tag
            if core['score'](seq,x,alt['goal'],spec)['success']:
                found[tag] = alt
                break
    return found


def diagnostic_triple(task,reference,core,spec):
    """Explicit matched diagnostic base; a common motif in both exposure regions.

    This changes neither primary tasks nor their thresholds. The diagnostic
    base is separately generated, so a changed motif is never paired against
    an original-primary trajectory carrying a different motif.
    """
    a,z=task['goal']['exposed'];seq=task['fixed_sequence']
    motifs=[task['goal']['motif']]+sorted({seq[j:j+2] for j in range(a,z-1) if seq[j]!=seq[j+1]})
    for motif in dict.fromkeys(motifs):
        base=copy.deepcopy(task);base['goal']['motif']=motif
        base['goal']['goal_id']=task['goal']['goal_id']+'_V96_DIAG_BASE'
        alt=alternatives(base,reference,core,spec)
        if len(alt)==2:
            assert core['score'](seq,np.asarray(reference['coordinates']),base['goal'],spec)['success']
            return dict(base_task=base,alternatives=alt,source_primary_goal_id=task['goal']['goal_id'],base_difficulty=task['goal']['difficulty'],motif_is_identical_across_diagnostic_triple=True)
    # If original primary targets admit no matched triple, use explicitly
    # separate native-supported diagnostic targets. Both members of each
    # comparison retain exactly the same thresholds/motif; primaries untouched.
    exposure=np.asarray(reference['exposure'][spec['proxy']]);b0=task['goal']['buried'][0]
    b_options=[j for j in range(4,53,4) if abs(j-b0)>=8 and abs(j-a)>=16 and exposure[a:z].mean()-exposure[j:j+8].mean()>=.15]
    if not b_options:return None
    b1=min(b_options,key=lambda j:float(exposure[j:j+8].mean()))
    slack=.05 if task['goal']['difficulty']=='relaxed' else 0.
    for a1 in sorted(range(4,53,4),key=lambda j:-float(exposure[j:j+8].mean())):
        if abs(a1-a)<8 or abs(a1-b0)<16 or exposure[a1:a1+8].mean()-exposure[b0:b0+8].mean()<.15:continue
        shared=[m for m in dict.fromkeys(motifs) if m in seq[a1:a1+8]]
        if not shared:continue
        base=copy.deepcopy(task);g=base['goal'];g['goal_id']=task['goal']['goal_id']+'_V96_DIAG_BASE';g['motif']=shared[0]
        g['exposure_min']=max(0.,round(np.floor(min(exposure[a:z].mean(),exposure[a1:a1+8].mean())*20)/20-slack,8))
        g['burial_max']=min(1.,round(np.ceil(max(exposure[b0:b0+8].mean(),exposure[b1:b1+8].mean())*20)/20+slack,8))
        burial=copy.deepcopy(base);burial['goal']['buried']=[b1,b1+8];burial['goal']['goal_id']+='_burial'
        exposed=copy.deepcopy(base);exposed['goal']['exposed']=[a1,a1+8];exposed['goal']['goal_id']+='_exposure'
        if all(core['score'](seq,np.asarray(reference['coordinates']),t['goal'],spec)['success'] for t in [base,burial,exposed]):
            return dict(base_task=base,alternatives=dict(burial=burial,exposure=exposed),source_primary_goal_id=task['goal']['goal_id'],base_difficulty=task['goal']['difficulty'],motif_is_identical_across_diagnostic_triple=True,threshold_policy='shared_native_supported_original_grid_and_slack; explicit separate diagnostics, not primary thresholds',source_primary_thresholds=dict(exposure_min=task['goal']['exposure_min'],burial_max=task['goal']['burial_max']))
    return None
