from common_review import *
import shutil,subprocess
assert not ROOT.exists();ROOT.mkdir();(ROOT/'source').mkdir();(ROOT/'runtime').mkdir();(ROOT/'runtime/tmp').mkdir();(ROOT/'runtime/cache').mkdir()
for p in Path('/mnt/d/Simone/proteinGen/e017_v96_review').glob('*.py'):shutil.copyfile(p,ROOT/'source'/p.name)
script=(PAIR/'run.sh').read_text().replace(str(PAIR),str(ROOT));(ROOT/'run.sh').write_text(script);(ROOT/'run.sh').chmod(0o755)
write('ADMISSION_AND_SCOPE.json',dict(UTC=utc(),scope='Historical-record analysis; bounded forward/backward inference on existing TRAIN traces; no optimizer construction or step; no new candidates',production_updates=0,new_generation_attempts=0,TEST=False,space_free_bytes=shutil.disk_usage(ROOT).free,GPU_state=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout))
print(ROOT)
