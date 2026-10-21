"""Validate/dry-run by default. Substantive training is NOT authorized in this handoff."""
import argparse,json
from pathlib import Path
from protein_joint_sac.crisp_v3.training import seal_config,execute
from protein_joint_sac.crisp_v3.networks import System
import torch
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--execute',action='store_true');p.add_argument('--authorization-file');p.add_argument('--resume');p.add_argument('--output');a=p.parse_args();c=json.load(open(a.config));h=seal_config(c)
with torch.device('meta'):m=System(c['profile'],c.get('shapes'))
print(json.dumps(dict(config_sha256=h,counts=m.counts(),execution=a.execute),indent=2),flush=True)
if a.execute:
 if not a.output:p.error('--output required')
 print(execute(c,a.output,a.authorization_file,a.resume))
