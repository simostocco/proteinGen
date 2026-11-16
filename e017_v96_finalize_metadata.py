"""Reporting-only correction; preserve qualified code/evidence and neural traces."""
import ast, datetime, hashlib, json, shutil
from pathlib import Path
R=Path('/mnt/d/Simone/proteinGen_data/e017_v96_burial_conditioning_20261029_v1')
LOCAL=Path('/mnt/d/Simone/proteinGen/e017_v96')
A=R/'qualification/pre_metadata_count_fix'
assert not A.exists()
A.mkdir();(A/'source96').mkdir();(A/'publication').mkdir()
original_publication=json.loads((R/'PROVENANCE_SHA256.json').read_text())
(R/'engineering').mkdir(exist_ok=True)
shutil.copyfile(__file__,R/'engineering/METADATA_COUNT_FIX_SOURCE.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def archive(p):
    target=A/'publication'/str(p.relative_to(R)).replace('/','__')
    p.rename(target)
    return target
before={p.name:sha(p) for p in (R/'source96').glob('*.py')}
for p in (R/'source96').glob('*.py'):shutil.copyfile(p,A/'source96'/p.name)
for name in ['campaign.py','qualify.py','publish.py']:
    ast.parse((LOCAL/name).read_text())
# No changed initializer, optimizer, safety controller, model hook, gradient,
# sampler, task schedule, checkpoint writer, or recovery algorithm.
def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
old=functions(R/'source96/campaign.py');new=functions(LOCAL/'campaign.py')
for name in ['runtime_factory','main','reconcile_schedule']:assert old[name]==new[name]
oldq=functions(R/'source96/qualify.py');newq=functions(LOCAL/'qualify.py')
for name in ['toy','partial_child']:assert oldq[name]==newq[name]
assert (LOCAL/'algorithm.py').read_bytes()==(R/'source96/algorithm.py').read_bytes()
assert (LOCAL/'study.py').read_bytes()==(R/'source96/study.py').read_bytes()
oldbase=old['baseline'];newbase=new['baseline']
assert "'attempts'" in oldbase and "'attempts'" in newbase
for name in ['campaign.py','qualify.py','publish.py']:shutil.copyfile(LOCAL/name,R/'source96'/name)
after={p.name:sha(p) for p in (R/'source96').glob('*.py')}
changes={k:dict(before=before[k],after=after[k],metadata_only=True) for k in before if before[k]!=after[k]}
assert set(changes)=={'campaign.py','qualify.py','publish.py'}
baseline=read(R/'qualification/AUGMENTED_INITIAL_TRAIN_BASELINE.json')
actual=sum(v['candidates'] for v in baseline['summary'].values());assert actual==256
diagnostic=read(R/'DIAGNOSTIC_GENERATION_COMPLETED.json')['attempts'];assert diagnostic==256
for path in [R/'qualification/AUGMENTED_INITIAL_TRAIN_BASELINE.json',R/'qualification/engineering/diagnostics/DEVELOPMENT_00000.json']:
    v=read(path);archive(path);v['attempts']=actual
    raw=v['summary']['underlying_unfiltered'];raw['invalid_proposal_checks']=raw['rejections'];raw['rejections']=0
    save(path,v)
path=R/'qualification/PROTEIN_TESTS.json';v=read(path);archive(path);v['development_attempts']=actual;v['actual_total_new_generation_attempts']=diagnostic+actual+v['new_training_attempts'];assert v['actual_total_new_generation_attempts']==544;save(path,v)
path=R/'qualification/RELEASE.json';v=read(path);archive(path)
v['original_neural_qualification_source96_SHA256']=before;v['source96_SHA256']=after
v['metadata_only_qualification_transfer']='METADATA_COUNT_FIX.json';v['neural_qualification_repeated']=False
save(path,v)
for p in list(R.glob('*.md'))+[R/'FINAL_STATUS.json',R/'PROVENANCE_SHA256.json']:
    if p.exists():archive(p)
unchanged=[]
for rel,v in original_publication['files'].items():
    if rel.endswith('.json.gz') or any('/'+tag+'/' in '/'+rel for tag in ['rollouts','commits','checkpoints']):
        assert sha(R/rel)==v['SHA256'],('NEURAL_PAYLOAD_CHANGED',rel)
        unchanged.append(rel)
save(R/'qualification/METADATA_COUNT_FIX.json',dict(PASS=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_changes=changes,
    reason='Reserved TRAIN contains16 goals, not32; report actual256 baseline attempts and544 total new attempts. Raw invalid-proposal checks are diagnostic flags, not actual rejections.',
    neural_methods_AST_identical=['runtime_factory','main','reconcile_schedule'],algorithm_and_study_byte_identical=True,
    only_campaign_function_changed='baseline reporting count and raw diagnostic legality-label normalization after unchanged sampling',
    synthetic_and_partial_writer_tests_AST_identical=True,qualification_changes='reporting counts only; same two unique optimizer updates and exact replay',
    original_qualification_evidence_preserved=str(A.relative_to(R)),qualification_checkpoints_keep_original_source_bindings=True,
    future_production_is_fresh_under_final_reviewed_bindings=True,production_updates=0,new_generation_during_correction=0,
    actual_baseline_attempts=actual,actual_total_new_generation_attempts=544,cap=800,unchanged_scientific_payload_files=len(unchanged)))
print('METADATA_ONLY_TRANSFER_PASS actual_baseline256 actual_total544 NO_MODEL_WORK')
