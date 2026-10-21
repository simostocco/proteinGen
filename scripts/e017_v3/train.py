"""Validate/dry-run by default. Substantive training is NOT authorized in this handoff."""
import argparse,json
from pathlib import Path
from protein_joint_sac.crisp_v3.training import seal_config,execute
from protein_joint_sac.crisp_v3.recovery import StoragePause
from protein_joint_sac.crisp_v3.networks import System
import torch
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--execute',action='store_true');p.add_argument('--authorization-file');p.add_argument('--resume');p.add_argument('--output');a=p.parse_args();c=json.load(open(a.config));h=seal_config(c)
with torch.device('meta'):m=System(c['profile'],c.get('shapes'))
print(json.dumps(dict(config_sha256=h,counts=m.counts(),execution=a.execute,stage_updated_parameters=m.counts()['trainable']['actor'] if c.get('admission_protocol') else m.counts()['total_trainable'],updated_components=['actor'] if c.get('admission_protocol') else 'stage dependent'),indent=2),flush=True)
if a.execute:
 if not a.output:p.error('--output required')
 try:run=execute(c,a.output,a.authorization_file,a.resume)
 except StoragePause as error:p.exit(75,'Storage admission denied: '+str(error)+'\n')
 print(run)
 if json.load(open(Path(run)/'RUN_STATUS.json'))['status']=='PAUSED_STORAGE':p.exit(75,'Paused safely for storage; no automatic continuation.\n')
