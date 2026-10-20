from r1c_common import *
from r1c_critic import dataset
import ast
for p in (R/'source/scripts/e016_v3').glob('r1c_*.py'):ast.parse(p.read_text())
s,c,x,y,mc,m=dataset()
assert len(s)==768 and x.shape==(768,48)
assert len({v['episode'] for v in m if v['split']=='TRAIN'})==64
assert not ({v['episode'] for v in m if v['split']=='TRAIN'}&{v['episode'] for v in m if v['split']!='TRAIN'})
write('critic/CPU_PREFLIGHT.json',{'transitions':len(s),'features':x.shape[1],'finite':bool(__import__('numpy').isfinite(x).all()),'episode_split_disjoint':True,'full_episode_file_sha256':sha(R/'critic/FULL_EPISODES.json'),'target_file_sha256':sha(R/'critic/STATE_ACTION_TARGETS.json'),'no_GPU_workload':True,'no_J1_updates':True})
print('CPU episode, logprob, replay, causal-feature and target checks passed')
