"""Read-only R2 gate. Never constructs a GPU model or executes policy updates."""
import argparse,json,sys
from pathlib import Path

def inspect(root):
 root=Path(root);c=json.loads((root/'configs/R2_PILOT_PROPOSED.json').read_text());af=json.loads((root/'AF_REWARD_GO_NO_GO.json').read_text());critic=json.loads((root/'TRANSFORMER_CRITIC_DIAGNOSTIC.json').read_text())
 reasons=[]
 if not c.get('production_authorized'):reasons.append('R2 requires separate user authorization')
 if not af.get('GO'):reasons.append('AF native recovery and reward calibration NO-GO')
 if critic.get('status')!='PASS':reasons.append('Transformer critic held-out calibration NO-GO')
 if not c.get('coefficients_sealed'):reasons.append('Production KL, entropy and PDB coefficients remain unsealed')
 if not c.get('real_episode_critic_certified'):reasons.append('No critic certification against real complete-episode calibrated returns')
 return {'GO':not reasons,'reasons':reasons,'GPU_model_constructed':False,'production_optimizer_updates':0,'oracle_calls':0}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3]);a=p.parse_args();out=inspect(a.root);print(json.dumps(out,indent=2));sys.exit(0 if out['GO'] else 2)
