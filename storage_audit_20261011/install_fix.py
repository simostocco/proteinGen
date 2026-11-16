from pathlib import Path
import hashlib,json,subprocess
O=Path(__file__).resolve().parent
source=Path('/home/simostocco/proteinGen-causal-rope/scripts/prepare_e012_sequence_corpus_v2b.py')
fixed=O/'e012_fix/scripts/prepare_e012_sequence_corpus_v2b.py'
before=source.read_bytes();after=fixed.read_bytes()
old=b"with sample.open() as src,Path(target).open('w') as dst:"
replacement=b"# Native Windows newline translation breaks MMseqs zero-copy\n            # createdb/result2flat on CRLF FASTA. Keep the same rank/cluster\n            # protocol and serialize diagnostic inputs with portable LF bytes.\n            with sample.open() as src,Path(target).open('w', newline='\\n') as dst:"
assert before.count(old)==1
assert before.replace(old,replacement)==after,'Refuse unrelated source changes'
# No native E012 owner or Linux MMseqs process may be running.
r=subprocess.run(['powershell.exe','-NoProfile','-Command','Get-CimInstance Win32_Process | Where-Object { $_.Name -match "python|mmseqs|linclust" } | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json'],capture_output=True,text=True,check=True)
assert not r.stdout.strip(),'Native worker active; defer future-writer source fix'
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:parts=(p/'cmdline').read_bytes().split(b'\0')
 except OSError:continue
 if parts and Path(parts[0].decode(errors='replace')).name.startswith('mmseqs'):raise RuntimeError('MMseqs active')
record={'path':str(source),'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':hashlib.sha256(after).hexdigest(),'source_fix_commit':subprocess.check_output(['git','-C',str(O/'e012_fix'),'rev-parse','HEAD'],text=True).strip(),'change':'Only native diversity subset LF serialization; no scientific parameters, retention, reservoir, protocol files or active E015 source changed','deployed':False}
(O/'source_fix.json').write_text(json.dumps(record,indent=2))
source.write_bytes(after)
record['deployed']=True;(O/'source_fix.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
