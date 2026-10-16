"""Single-controller consistent replay/optimizer snapshots and explicit delayed-reward streams."""
import json,random
from dataclasses import asdict
from ..contracts import GEOMETRY_VERSION
from ..chemistry.reference_adapter import ref as chemistry_ref
from ..contracts import GeneratedProtein,digest
from ..geometry.ca_builder import replay
from .checkpointing import save_atomic,load_checked
from .bc_training import rng_state,restore_rng,code_hash

class ReplayController:
    def __init__(self,algorithm,replay_store,seed=42):
        self.algorithm=algorithm;self.store=replay_store;self.rng=random.Random(seed);self.updates=0;self.cheap_rewards=None
    def transition(self,episode_id,t,stream,cheap_rewards):
        episode=self.store.episode(episode_id)
        if len(episode.actions)!=episode.N or len(cheap_rewards)!=episode.N or not 0<=t<episode.N:raise ValueError('complete episode/action index/reward trace')
        terminal=t==episode.N-1; reward=float(cheap_rewards[t])
        if stream=='ORACLE_RESOLVED':
            card=self.store.oracle_return(episode_id)
            if card['evidence_type']!='MOCK_UNIT_TEST' and not card['calibrated']:raise ValueError('external reward not calibrated')
            value=card['components'].get('value') if card['evidence_type']=='MOCK_UNIT_TEST' else card['components'].get('terminal_reward')
            if value is None:raise ValueError('explicit terminal reward required')
            if terminal:reward+=float(value)
        elif stream!='CHEAP_ONLY':raise ValueError('replay stream')
        device=next(self.algorithm.actor.parameters()).device
        s=replay(episode.N,episode.actions[:t],device=device);nxt=replay(episode.N,episode.actions[:t+1],device=device)
        from ..contracts import BuildAction
        a=episode.actions[t];action=BuildAction(a.residue,a.geometry.to(device),a.mask)
        return s,action,reward,nxt,terminal
    def sample_update(self,stream,cheap_rewards=None):
        if self.cheap_rewards is None:
            if cheap_rewards is None:raise ValueError("explicit cheap reward trace required")
            self.cheap_rewards=json.loads(json.dumps(cheap_rewards,allow_nan=False))
        elif cheap_rewards is not None and digest(cheap_rewards)!=digest(self.cheap_rewards):raise ValueError("cheap reward lineage changed")
        cheap_rewards=self.cheap_rewards
        ids=self.store.resolved_ids() if stream=='ORACLE_RESOLVED' else [r[0] for r in self.store.db.execute('SELECT id FROM episodes ORDER BY id')]
        if not ids:raise ValueError('no eligible replay; pending is not zero')
        identity=self.rng.choice(ids);ep=self.store.episode(identity);t=self.rng.randrange(ep.N)
        result=self.algorithm.update_one(*self.transition(identity,t,stream,cheap_rewards[identity]));self.updates+=1
        return {**result,'episode':identity,'step':t,'updates':self.updates,'stream':stream}
    def lineage(self):
        a=self.algorithm
        return {"actor":asdict(a.actor.encoder.config),"critic":asdict(a.critics.q1.current.config),"gamma":a.gamma,"target_aa":a.temperatures.target_aa,"target_geom":a.temperatures.target_geom,"backward":a.actor_backward,"geometry":GEOMETRY_VERSION,"chemistry":asdict(chemistry_ref()),"distribution":"exp_sigmoid_tanh_gaussian_reference_v1"}
    def snapshot(self,path):
        from pathlib import Path
        if Path(path).exists():raise ValueError('immutable checkpoint')
        a=self.algorithm;rows=list(self.store.db.execute('SELECT id,payload,reward FROM episodes ORDER BY id'))
        return save_atomic(path,{'version':'SAC_REPLAY_REFERENCE_V1','lineage':self.lineage(),'cheap_rewards':self.cheap_rewards,'code':code_hash(),'actor_backward':a.actor_backward,'gamma':a.gamma,'actor':a.actor.state_dict(),'critics':a.critics.state_dict(),'temperatures':a.temperatures.state_dict(),'actor_optimizer':a.actor_optimizer.state_dict(),'critic_optimizer':a.critic_optimizer.state_dict(),'alpha_optimizer':a.alpha_optimizer.state_dict(),'replay_rows':rows,'replay_sha':digest(rows),'rng':rng_state(),'sampler_rng':self.rng.getstate(),'updates':self.updates})
    def restore(self,path):
        s=load_checked(path);a=self.algorithm
        if s['lineage']!=self.lineage() or s['version']!='SAC_REPLAY_REFERENCE_V1' or s['code']!=code_hash() or s['gamma']!=a.gamma or s['actor_backward']!=a.actor_backward or digest(s['replay_rows'])!=s['replay_sha']:raise ValueError('SAC lineage/replay mismatch')
        current=list(self.store.db.execute('SELECT id,payload,reward FROM episodes ORDER BY id'))
        if current and digest(current)!=s['replay_sha']:raise ValueError('live replay differs from snapshot; restore into empty store')
        with self.store.db:
            for row in s['replay_rows']:self.store.db.execute('INSERT OR IGNORE INTO episodes VALUES(?,?,?)',row)
        for key,module in [('actor',a.actor),('critics',a.critics),('temperatures',a.temperatures)]:module.load_state_dict(s[key],strict=True)
        for key,opt in [('actor_optimizer',a.actor_optimizer),('critic_optimizer',a.critic_optimizer),('alpha_optimizer',a.alpha_optimizer)]:opt.load_state_dict(s[key])
        self.cheap_rewards=s['cheap_rewards'];self.rng.setstate(s['sampler_rng']);self.updates=s['updates'];restore_rng(s['rng'])
