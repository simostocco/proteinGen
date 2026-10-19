"""Compact trace metadata around existing crash-safe replay and oracle queue."""
import json,hashlib,sqlite3,time
import torch
from ..contracts import BuildEpisode,BuildAction,GeneratedProtein
from .delayed_episode_replay import EpisodeReplay
from ..geometry.ca_builder import replay

class SequenceReplay(EpisodeReplay):
    def __init__(self,path):
        super().__init__(path);self.db.execute('CREATE TABLE IF NOT EXISTS action_traces(id TEXT PRIMARY KEY,payload TEXT NOT NULL,sha TEXT NOT NULL)')
    def add_trace(self,identity,N,trace,protein,metadata):
        actions=tuple(BuildAction(v['residue'],torch.tensor(v['geometry'],dtype=torch.float64),tuple(v['mask'])) for v in trace);episode=BuildEpisode(identity,N,actions)
        p=json.dumps({'trace':trace,'metadata':metadata},sort_keys=True,allow_nan=False);h=hashlib.sha256(p.encode()).hexdigest()
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.store(episode,protein)
            old=self.db.execute('SELECT sha FROM action_traces WHERE id=?',(identity,)).fetchone()
            if old and old[0]!=h:raise ValueError('immutable action probability trace conflict')
            self.db.execute('INSERT OR IGNORE INTO action_traces VALUES(?,?,?)',(identity,p,h));self.db.execute('COMMIT')
        except Exception:self.db.execute('ROLLBACK');raise
        return h
    def trace(self,identity):
        p,h=self.db.execute('SELECT payload,sha FROM action_traces WHERE id=?',(identity,)).fetchone()
        if hashlib.sha256(p.encode()).hexdigest()!=h:raise ValueError('action trace corrupted')
        return json.loads(p)
    def reconstruct(self,identity):return replay(self.episode(identity).N,self.episode(identity).actions)
    def eligible(self,identity):
        card=self.oracle_return(identity) # Raises for pending: never zero reward.
        valid=(card.get('calibrated') is True and card.get('evidence_type')=='EXTERNAL_PREDICTION' and card.get('components',{}).get('oracle_status')=='SUCCEEDED')
        return valid and not any(v['support_repaired'] for v in self.trace(identity)['trace'])

class OracleWallBudget:
    """Conservative crash accounting: unclosed calls retain their full reservation."""
    def __init__(self,path,max_calls=16,max_seconds=2700):
        self.db=sqlite3.connect(path,isolation_level=None);self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL');self.db.execute('CREATE TABLE IF NOT EXISTS wall(id TEXT PRIMARY KEY,reserved REAL,actual REAL,start REAL)');self.max_calls=max_calls;self.max_seconds=max_seconds
        self.db.execute('CREATE TABLE IF NOT EXISTS config(id INTEGER PRIMARY KEY,calls INTEGER,seconds REAL)');self.db.execute('INSERT OR IGNORE INTO config VALUES(1,?,?)',(max_calls,max_seconds));assert self.db.execute('SELECT calls,seconds FROM config').fetchone()==(max_calls,max_seconds)
    def reserve(self,identity,seconds):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            used,calls=self.db.execute('SELECT COALESCE(sum(COALESCE(actual,reserved)),0),count(*) FROM wall').fetchone()
            if calls>=self.max_calls or used+seconds>self.max_seconds:raise RuntimeError('hard R1 oracle call/time budget exhausted')
            self.db.execute('INSERT INTO wall VALUES(?,?,NULL,?)',(identity,seconds,time.time()));self.db.execute('COMMIT')
        except Exception:self.db.execute('ROLLBACK');raise
    def finish(self,identity,seconds):self.db.execute('UPDATE wall SET actual=? WHERE id=?',(seconds,identity))
    def report(self):return {'attempts_reserved':self.db.execute('SELECT count(*) FROM wall').fetchone()[0],'charged_seconds':self.db.execute('SELECT COALESCE(sum(COALESCE(actual,reserved)),0) FROM wall').fetchone()[0],'max_attempts':self.max_calls,'max_seconds':self.max_seconds,'unfinished_calls_charged_full_reservation':True}
