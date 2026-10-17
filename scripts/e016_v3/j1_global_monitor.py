"""Read-only resource sampling for this task; never stops another process."""
import json,os,resource,subprocess,time
from pathlib import Path
R=Path("/mnt/d/Simone/proteinGen_data/e016_j1_global_pretraining_20261017_v1")
def main():
 (R/"MONITOR_PID.json").write_text(json.dumps({"pid":os.getpid(),"start":time.time()}))
 with (R/"logs/resources.jsonl").open("a",buffering=1) as f:
  while not (R/"MONITOR_STOP").exists():
   mem={s.split(":")[0]:int(s.split()[1])*1024 for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith(("MemAvailable:","MemTotal:","SwapFree:","SwapTotal:"))}
   jobs=[]
   for p in Path("/proc").iterdir():
    if not p.name.isdigit():continue
    try:
     args=(p/"cmdline").read_bytes().split(b"\0")
     if not args or b"python" not in args[0] or not any(b"j1_global_train.py" in a or b"j1_global_finalize.py" in a for a in args[1:]):continue
     status=(p/"status").read_text();rss=int(next(s.split()[1] for s in status.splitlines() if s.startswith("VmRSS:")))*1024
     jobs.append({"pid":int(p.name),"rss_bytes":rss,"args":[s.decode(errors="replace") for s in args if s]})
    except (OSError,StopIteration):continue
   try:gpu=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True,timeout=5).strip()
   except Exception as ex:gpu=repr(ex)
   row={"time":time.time(),"memory":mem,"gpu_used_free_MiB_utilization":gpu,"jobs":jobs,"D_free_bytes":__import__("shutil").disk_usage(R).free}
   f.write(json.dumps(row)+"\n");time.sleep(10)
if __name__=="__main__":main()
