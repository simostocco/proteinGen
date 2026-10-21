"""Explicit execution-only migration; no scientific schedule/optimizer reset."""
import copy,json,sqlite3
from pathlib import Path
from .prior import file_sha,SHA,tensor_sha
from .storage import restore,digest,rng_state
from .recovery import state_digest

def scientific_config(c):
 c=copy.deepcopy(c)
 for key in ['data_manifest','data_input_hashes','native_limits_sha256','validation_panel_sha256','matched_TRAIN_panel_sha256','source_commit','execution_parent','execution_baseline_validation','execution_protocol']:c.pop(key,None)
 c['microbatch']=1
 return c

def migrate_execution(config,system,learner,samplers,effective,replay,run):
 import torch
 p=config['execution_parent'];path=p['path']
 if file_sha(path)!=p['sha256']:raise ValueError('execution parent SHA')
 parent=torch.load(path,map_location='cpu',weights_only=False)
 if file_sha(config['execution_baseline_validation'])!=p['baseline_validation_sha256']:raise ValueError('paired baseline validation SHA')
 if parent['j1_sha256']!=SHA or parent['config']['source_commit']!=p['source_commit'] or parent['counters']['updates']!=p['update']:raise ValueError('execution parent identity/update')
 if parent['counters'].get('scientific_stop',{}).get('stop'):raise ValueError('scientific stop cannot be bypassed by execution migration')
 if scientific_config(config)!=scientific_config(parent['config']):raise ValueError('execution migration changed scientific configuration')
 if parent['config']['data_input_hashes']!=effective['data_input_hashes']:raise ValueError('execution data provenance')
 # Restore with EXACT parent effective config; then only the next checkpoint's
 # execution binding changes. No new random sampling, LR reset, moments or optimizer steps.
 state=restore(path,system,learner,samplers,parent['config'])
 assert state_digest(rng_state())==state_digest(state['rng'])
 assert state_digest({k:o.state_dict() for k,o in learner.optimizers.items()})==state_digest(state['optimizers'])
 assert state_digest({k:o.state_dict() for k,o in learner.schedulers.items()})==state_digest(state['schedulers'])
 snap=state['replay_snapshot'];assert file_sha(snap['path'])==snap['sha256'];replay.close();src=sqlite3.connect('file:'+snap['path']+'?mode=ro',uri=True);dst=sqlite3.connect(run/'replay.sqlite');src.backup(dst);src.close();dst.close()
 # Python/CPU/CUDA/NumPy RNG is deliberately not touched by this metadata-only record.
 from .recovery import atomic_json
 atomic_json(run/'EXECUTION_MIGRATION.json',dict(schema='crisp_execution_migration_v1',parent= p,parent_effective_config_sha256=parent['config_sha256'],new_effective_config_sha256=digest(effective),only_scientific_config_difference='microbatch execution ordering; effective batch and objective identical',model_state_exact=True,optimizer_scheduler_exact=True,RNG_exact=True,samplers_exact=True,next_global_update=state['counters']['updates']+1))
 return state['counters']

# Execution-only, bounded four-row-group cache. Convert only the requested scalar
# instead of materializing every coordinate JSON string in a large row group.
from functools import lru_cache
from protein_joint_sac.data.accepted_supervised import CoordinateReader
_EAGER_GROUP=CoordinateReader.group
class ArrowRows:
 def __init__(self,column):self.column=column
 def __getitem__(self,index):return self.column[index].as_py()
 def __len__(self):return len(self.column)
@lru_cache(maxsize=4)
def arrow_group(self,path,index):
 import pyarrow.parquet as pq
 return ArrowRows(pq.ParquetFile(path).read_row_group(index,columns=['selected_calpha_coordinates'],use_threads=False).column(0))
ACTIVE_LOADER='eager'
def set_loader(mode):
 global ACTIVE_LOADER
 ACTIVE_LOADER=mode
 old=CoordinateReader.group;old.cache_clear();arrow_group.cache_clear();_EAGER_GROUP.cache_clear()
 if mode=='eager':CoordinateReader.group=_EAGER_GROUP
 elif mode=='arrow_scalar_lru4':CoordinateReader.group=arrow_group
 elif mode in ['qualified_ram_512MiB','qualified_ram_1024MiB']:
  from .fast_execution import qualified_group,QUALIFIED
  QUALIFIED.limit=(1024 if mode=='qualified_ram_1024MiB' else 512)*2**20
  CoordinateReader.group=qualified_group
 elif mode=='arrow_ram_256MiB':
  from .fast_execution import ram_group
  CoordinateReader.group=ram_group
 else:raise ValueError('unknown execution loader')
def enable_loader(config):
 set_loader(config.get('execution_protocol',{}).get('loader','eager'))
 from .fast_execution import enable
 enable(config)
