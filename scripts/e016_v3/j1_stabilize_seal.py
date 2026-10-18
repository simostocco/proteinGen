"""Final source/output seal and immediate independent rehash, no scientific mutation."""
import hashlib,json,os,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[3];S=R/'source'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 if not (R/'PUBLICATION_READY.json').exists():raise RuntimeError('final reports not ready')
 if (R/'PROVENANCE_SHA256.json').exists():raise RuntimeError('preserve existing published seal')
 if subprocess.check_output(['git','status','--porcelain'],cwd=S,text=True).strip():raise RuntimeError('commit validated source before seal')
 ret=json.load(open(R/'APPROVED_RETIREMENT_MANIFEST.json'));external={}
 for p,expected in ret['protected_sha256_after'].items():
  actual=sha(p)
  if actual!=expected:raise ValueError('protected input changed '+p)
  external[p]={'sha256':actual,'bytes':Path(p).stat().st_size}
 B=R.parent
 for p in [B/'e016_j1_clip_precision_pilot_20261018_v1/PROVENANCE_SHA256.json',B/'e016_j1_torsion_diagnostics_storage_20261018_v1/PROVENANCE_SHA256.json',B/'e016_j1_sequence_first_20261018_v1/PROVENANCE_SHA256.json',B/'e016_partial_supervision_20261016_v1/PROVENANCE_SHA256.json',B/'e016_j1_clip_precision_pilot_20261018_v1/runs/J1/checkpoint_01500.pt',B/'e016_j1_clip_precision_pilot_20261018_v1/runs/J1/checkpoint_02500.pt',B/'e016_partial_supervision_20261016_v1/runs/J1-FULL/checkpoint_20000.pt',B/'e016_j1_clip_precision_pilot_20261018_v1/generation/final00500_ca_fp64_interior_v1.json',B/'e016_j1_clip_precision_pilot_20261018_v1/generation/fallback_ca_fp64_interior_v1.json',B/'e016_partial_supervision_20261016_v1/NATIVE_VS_GENERATED_GEOMETRY.json',Path(ret['candidate_manifest'])]:external[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
 tracked=subprocess.check_output(['git','ls-files','-z'],cwd=S).decode().split('\0');src={p:sha(S/p) for p in tracked if p and (S/p).is_file()};outputs={}
 for p in sorted(R.rglob('*')):
  rel=p.relative_to(R)
  if not p.is_file() or p.is_symlink() or rel.parts[0] in ['source','runtime'] or p.name in ['PROVENANCE_SHA256.json','PROVENANCE_VERIFICATION.json','SEAL.log']:continue
  outputs[str(rel)]={'sha256':sha(p),'bytes':p.stat().st_size}
 retired=[{'path':v['path'],'pre_retirement_sha256':v['sha256'],'currently_absent':not Path(v['path']).exists()} for v in ret['retired']]
 if len(retired)!=10 or not all(x['currently_absent'] for x in retired):raise ValueError('retirement mismatch')
 prov={'version':'E016_J1_TORSION_STABILIZATION_V1','branch':subprocess.check_output(['git','branch','--show-current'],cwd=S,text=True).strip(),'reporting_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip(),'training_commit':'049930ab86e058717aac577380d4033ca9d6c1db','historical_replay_entrypoint_revision':'2de6e21e81f1f34b912fca6bd1b2e2efe5b9fb6f','config_sha256':sha(R/'configs/FROZEN.json'),'panel_sha256':sha(R/'configs/EVALUATION_PANELS.json'),'tracked_source_sha256':src,'outputs':outputs,'protected_input_sha256':external,'approved_retired_intermediates':retired,'excluded':'runtime caches, Git metadata, source ignored files, provenance self/verification and in-flight seal stdout; no raw coordinate duplication','scientific_scope':'TRAIN-only optimization, frozen TRAIN/VALIDATION panels, no TEST evaluation; no universal remote homology or folding claim'};dump(R/'PROVENANCE_SHA256.json',prov)
 failed=[];count=0
 for rel,v in outputs.items():
  count+=1
  if sha(R/rel)!=v['sha256']:failed.append(rel)
 for rel,v in src.items():
  count+=1
  if sha(S/rel)!=v:failed.append('source/'+rel)
 for p,v in external.items():
  count+=1
  if sha(p)!=v['sha256']:failed.append(p)
 dump(R/'PROVENANCE_VERIFICATION.json',{'status':'PASS' if not failed else 'FAIL','checked_files':count,'failures':failed,'provenance_sha256':sha(R/'PROVENANCE_SHA256.json'),'retired_expected_absent':10,'historical_protected_verification':json.load(open(R/'PROTECTED_INTEGRITY_VERIFICATION.json'))['status'],'optimizer_updates':0})
 if failed:raise ValueError('seal verification failures')
 print('PROVENANCE_VERIFIED',count)
if __name__=='__main__':main()
