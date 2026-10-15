"""Run the unchanged parser from a small process, avoiding inherited parent RSS peaks.
Linux fork/exec high-water accounting can otherwise charge a large SQLite parent's
pre-exec RSS to a tiny parser, even while its3GiB address-space limit is intact.
"""
import os,sys,subprocess,json,resource
from pathlib import Path
P=Path('/mnt/d/Simone/proteinGen_data/pdb_only_v3_20261013')
source,entry,out=sys.argv[1:]
assert str(Path(source).resolve()).startswith('/mnt/d/') and str(Path(out).resolve()).startswith('/mnt/d/')
current=next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:'))
r=subprocess.run([sys.executable,'-B',str(P/'parse_entry.py'),source,entry,out],capture_output=True,text=True)
print(json.dumps({'broker_current_RSS_kib':int(current),'broker_inherited_highwater_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'parser_child_highwater_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,'parser_source_unchanged':True}),flush=True)
sys.stdout.write(r.stdout);sys.stderr.write(r.stderr);sys.exit(r.returncode)
