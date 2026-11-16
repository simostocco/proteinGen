"""CPU-only real torch-write interruption and hash-corruption rejection."""
import builtins,json,os,runpy,subprocess,sys
from pathlib import Path
R=Path('/mnt/d/Simone/proteinGen_data/e017_v94_fixed_sequence_geometry_rl_20261024_v1')
P=runpy.run_path(str(R/'source/scripts/e017_v94/production.py')); Q=R/'recovery_qualification_20261025_v1'
read=P['read'];sha=P['sha'];compact=P['compact'];torch=P['torch']
if len(sys.argv)>1 and sys.argv[1]=='interrupt-real-write':
    meta=read(Q/'uninterrupted/RECOVERY_INDEX.json')['latest']; source=Q/'uninterrupted'/meta['relative']
    assert sha(source)==meta['SHA256']; state=torch.load(source,map_location='cpu',weights_only=False)
    target=Q/'real_writer_interruption/probe.pt'; original=builtins.open
    class Interrupt:
        def __init__(self,f):self.f=f;self.n=0
        def __getattr__(self,n):return getattr(self.f,n)
        def __enter__(self):return self
        def __exit__(self,*args):return self.f.__exit__(*args)
        def write(self,b):
            n=self.f.write(b);self.n+=n
            if self.n>=8*2**20:
                self.f.flush();os.fsync(self.f.fileno());os._exit(76)
            return n
    def intercepted(file,*a,**kw):
        f=original(file,*a,**kw)
        return Interrupt(f) if str(file)==str(target)+'.part' and a and a[0]=='xb' else f
    builtins.open=intercepted
    P['atomic_checkpoint'](target,state,P['StoragePolicy'](checkpoint_hint_bytes=P['HINT']))
    raise RuntimeError('real-writer fault was not injected')
else:
    release=read(Q/'RELEASE.json'); assert release['PASS'] and release['binding']==P['bindings']()
    d=Q/'real_writer_interruption';d.mkdir(exist_ok=False)
    oldindex=sha(Q/'uninterrupted/RECOVERY_INDEX.json')
    with (d/'fault.log').open('xb') as f:
        rc=subprocess.call([str(R/'run.sh'),__file__,'interrupt-real-write'],stdout=f,stderr=subprocess.STDOUT)
    assert rc==76,rc
    partial=d/'probe.pt.part'; assert partial.stat().st_size>=8*2**20 and not (d/'probe.pt').exists()
    assert sha(Q/'uninterrupted/RECOVERY_INDEX.json')==oldindex
    rejected=False
    try:torch.load(partial,map_location='cpu',weights_only=False)
    except (RuntimeError,EOFError):rejected=True
    assert rejected,'incomplete archive unexpectedly valid'
    # Exercise the actual production restore hash guard before deserialization.
    d2=Q/'hash_corruption_negative';(d2/'checkpoints').mkdir(parents=True,exist_ok=False)
    fixture=d2/'checkpoints/corrupt.pt';fixture.write_bytes(b'NEW_OWNED_INTENTIONALLY_CORRUPT_QUALIFICATION_FIXTURE')
    compact(d2/'RECOVERY_INDEX.json',dict(latest=dict(relative='checkpoints/corrupt.pt',SHA256=release['uninterrupted']['checkpoint']['SHA256'])))
    rt=P['Runtime'].__new__(P['Runtime']);rt.root=d2; rejected=False
    try:rt.restore()
    except AssertionError as e:assert str(e)=='checkpoint hash mismatch';rejected=True
    assert rejected
    # Restore the complete real checkpoint through production code on CPU-side
    # validation: semantic integrity, every optimizer tensor and binding intact.
    meta=read(Q/'uninterrupted/RECOVERY_INDEX.json')['latest'];cp=Q/'uninterrupted'/meta['relative']
    assert sha(cp)==meta['SHA256'];state=torch.load(cp,map_location='cpu',weights_only=False)
    assert P['state_digest'](state)==meta['state_digest'] and state['binding']==P['bindings']()
    result=dict(PASS=True,source_commit=P['bindings']()['V94_source_commit'],binding=P['bindings'](),
        optimizer_journal_release_SHA256=sha(Q/'RELEASE.json'),actual_atomic_writer_interrupted=True,partial_archive_rejected=True,
        partial_bytes=partial.stat().st_size,partial_SHA256=sha(partial),authoritative_checkpoint_and_index_unchanged=True,
        actual_production_restore_hash_guard_rejected_corruption=True,complete_checkpoint_readback=True,
        supplement_source_SHA256=sha(Path(__file__)),production_optimizer_updates=0)
    compact(Q/'PRODUCTION_RECOVERY_RELEASE.json',result);print(json.dumps(result),flush=True)
