import json,os,time,hashlib,subprocess,datetime,collections
from pathlib import Path
O=Path(__file__).resolve().parent
R=Path('/mnt/d/Simone/proteinGen');D=Path('/mnt/d/Simone/proteinGen_data')
E12=D/'sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization/scientific_corrected_protein_v1'
E15=R/'e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1'
E7=R/'.e014_worktree/e007_matched_fullscale_v1'
def read(p):return json.loads(Path(p).read_text())
def save(name,data): (O/name).write_text(json.dumps(data,indent=2))
def owner(p):
 s=str(p).lower()
 for e in ['e012','e015','e007','e014']:
  if e in s or (e=='e012' and 'sequence_foundation' in s):return e.upper()
 if str(p).startswith(str(R/'data')) or str(p).startswith(str(R/'external_models')) or str(p).startswith(str(D)):return 'shared dataset'
 if any(x in s for x in ['worktree','/.git','/src/','/scripts/','/configs/']):return 'source/worktree'
 return 'unknown'
if not (O/'census_snapshot.json').exists():save('census_snapshot.json',read(O/'growth_snapshot_before.json'))
c=read(O/'census_snapshot.json');dirs=c['directories']
for p,r in dirs.items():r['owner']=owner(p)
for r in c['large_files']+c['checkpoints']+c['special']:r['owner']=owner(r['path'])
save('census_snapshot.json',c)
a=read(O/'targeted_before.json');b=read(O/'targeted_after.json');elapsed=b['timestamp']-a['timestamp'];rates=[]
for p,v in b['directories'].items():
 gained=v['bytes']-a['directories'][p]['bytes'];persec=gained/elapsed
 rates.append({'path':p,'bytes_before':a['directories'][p]['bytes'],'bytes_after':v['bytes'],'bytes_gained':gained,'elapsed_seconds':elapsed,'MiB_per_min':persec*60/1024**2,'GiB_per_hour':persec*3600/1024**3,'GiB_per_day':persec*86400/1024**3})
a['measurement']='Targeted recursive regular-file apparent sizes; symlinks excluded. Full census separate.'
b['growth_rates']=rates;b['measurement']=a['measurement'];save('growth_snapshot_before.json',a);save('growth_snapshot_after.json',b)
save('growth_rates.json',rates)
for name,rows,limit in [('largest_directories.tsv',[(p,r) for p,r in dirs.items()],50),('largest_files.tsv',[(r['path'],r) for r in c['large_files']],100)]:
 with (O/name).open('w') as f:
  f.write('bytes\towner\tfiles\tpath\n')
  for p,r in sorted(rows,key=lambda x:x[1]['bytes'],reverse=True)[:limit]:f.write(f"{r['bytes']}\t{r['owner']}\t{r.get('files',1)}\t{p}\n")
categories={
 'files_ge_1_GiB':[r for r in c['large_files'] if r['bytes']>=1024**3],
 'files_ge_250_MiB':c['large_files'],
 'many_small_files':[dict(path=p,**r) for p,r in dirs.items() if r['direct_files']>=1000],
 'checkpoint_directories':[dict(path=p,**r) for p,r in dirs.items() if Path(p).name=='checkpoints'],
 'caches':[dict(path=p,**r) for p,r in dirs.items() if any(k in Path(p).name.lower() for k in ['cache','triton'])],
 'evaluation_and_trajectory_directories':[dict(path=p,**r) for p,r in dirs.items() if any(k in Path(p).name.lower() for k in ['evaluation','trajector','sampling','ensemble'])],
 'log_directories':[dict(path=p,**r) for p,r in dirs.items() if Path(p).name=='logs'],
 'sqlite_wal_shm':[r for r in c['special'] if any(k in r['path'].lower() for k in ['sqlite','-wal','-shm'])],
 'archives':[r for r in c['large_files'] if any(r['path'].endswith(k) for k in ['.zip','.tar','.tgz','.tar.gz','.7z'])],
 'large_logs':[r for r in c['special'] if r['path'].endswith('.log') and r['bytes']>=10*1024**2],
 'partial_files':[r for r in c['large_files']+c['special'] if '.partial' in r['path'] or r['path'].endswith('.tmp')],
}
save('storage_categories.json',categories)
pilot=E12/'clusters/identity_50/pilot_tmp/11841646291735746956'
exports=[pilot/'rep_seq.fasta',pilot/'all_seqs.fasta'];candidates=[]
for p in exports:
 s=p.stat()
 with p.open('rb') as f:
  first=f.read(4096);f.seek(max(0,s.st_size-4096));last=f.read(4096)
 row={'absolute_path':str(p),'byte_size':s.st_size,'classification':'UNKNOWN','proposed_action':'Delete only after owner approval and bounded failure evidence preservation; never reuse these exports as scientific output','justification':'Malformed CRLF/zero-copy MMseqs export from failed 10000-sequence pilot; no completed pilot certificate. Unique failure artifact and current resumable namespace require review. Not automatically disposable.','sha256':None,'sha256_note':'Full-file hash not computed: these are not exact-duplicate candidates; preserve bytes pending approval.','sampled_first_last_4096_sha256':hashlib.sha256(first+last).hexdigest(),'nul_bytes_in_sample':(first+last).count(b'\0'),'expected_reclaimed_bytes_if_approved':s.st_blocks*512,'exactly_reproducible':'Bug behavior involves zero-copy CRLF reads beyond record boundaries; not proven bitwise reproducible','process_evidence':'No E012/MMseqs worker in Linux or Windows snapshots; no Linux deleted-open files observed. Windows-wide NTFS handles not fully enumerated.','state_reference_evidence':'Production STATE records failed diversity/easy-linclust and pilot.log. No pilot.complete.json. MMseqs easycluster.sh still tests existence of these files: naive retry would reuse malformed/partial exports.','execution_requires_user_approval':True}
 candidates.append(row)
save('proposed_deletions.json',{'automatic_subset':[],'requires_approval':candidates,'protected_paths_untouched':True})
save('executed_deletions.json',{'deletions':[],'reclaimed_bytes':0,'reason':'No large candidate met every automatic-deletion criterion; retain unique failed-attempt evidence and resumability state.'})
mm=[]
for p,r in dirs.items():
 if owner(p)!='E012':continue
 if not any(k in p for k in ['/mmseqs','/clusters','/tmp','/screen_optimization','/screen_architecture','/data2d_batch200_diagnostic','/benchmarks','/replays','/replay_restart','/screen_production']):continue
 if r['bytes']<1024**2 and not p.startswith(str(E12)):continue
 label='F';classification='PROTECTED';reason='Historical screening/diagnostic evidence; retained.'
 if p.startswith(str(E12/'clusters')):label='B';classification='KEEP FOR NOW';reason='Failed 50% identity pilot; no completed pilot marker. Preserve clustering DB/index/TSV and failure log; corrupted exports separately UNKNOWN. No 30% identity directory exists.'
 elif p.startswith(str(E12/'mmseqs')) or p.startswith(str(E12/'tmp')):label='D';classification='KEEP FOR NOW';reason='Currently empty corrected MMseqs/sort directory; no bytes to reclaim.'
 elif '/protected_index' in p:label='A';reason='Protected target DB reused read-only by corrected verification; handoff/replay input.'
 mm.append(dict(path=p,**r,phase_category=label,classification=classification,reason=reason))
save('e012_mmseqs_inventory.json',{'active_worker':False,'state':read(E12/'production/STATE.json'),'directories':mm,'historical_reservoir':str(E12.parent/'scientific/filtered/reservoir.sqlite'),'required_categories':{'A':'input databases','B':'resumability state','C':'completed scientific outputs in manifests/corpus','D':'empty completed scratch; no positive safe reclaim established','E':'failed pilot exports UNKNOWN pending review','F':'historical evidence protected'},'identity_30_present':(E12/'clusters/identity_30').exists(),'pilot_complete':(E12/'clusters/identity_50/pilot.complete.json').exists()})
protected=[]
def protect(p,reason,exclude=0):
 if str(p) not in dirs:return
 protected.append({'path':str(p),'classification':'PROTECTED','apparent_path_bytes':dirs[str(p)]['bytes']-exclude,'reason':reason,'exclude_bytes':exclude})
protect(R/'data','Shared raw/processed source datasets and scientific population; no rematerialization/retention authorization.')
protect(R/'reports','Unique scientific evaluation/certification and exact replay evidence; conservative protection.')
protect(E15,'Current production recovery, boundary checkpoints, smoke/replay certificates and FP32 EMA/optimizer/RNG state.')
protect(R/'e014_artifacts','Shared frozen population/cache opened by active E015; E014 scientific evidence.')
protect(R/'e013_artifacts','Historical model/evaluation provenance consumed by control protocols.')
protect(E7,'LATEST and COMMITTED reference sequence_pause_3580.pt; full hash verified unchanged.')
protect(R/'external_models','Shared model artifacts; no disposable role established.')
protect(D/'sequence_foundation','All corpus, source, certificates, historical reservoirs/WAL/SHM, corrected-screen and historical evidence; excludes only two UNKNOWN failed exports.',sum(x.stat().st_size for x in exports))
dup=read(O/'duplicate_inventory.json');hardlink_overcount=sum(r['bytes_per_path']*(len(r['paths'])-1) for r in dup if r['kind']=='same physical file via hardlinks' and all(p.startswith(str(D/'sequence_foundation')) for p in r['paths']))
totalprotected=sum(x['apparent_path_bytes'] for x in protected)-hardlink_overcount
save('protected_artifacts.json',{'disjoint_groups':protected,'known_hardlink_overcount_bytes':hardlink_overcount,'protected_unique_apparent_bytes_lower_bound':totalprotected,'limitation':'Conservative disjoint protected groups, deduplicated only known large hardlinks. Also preserve unique evidence/latest checkpoints throughout UNKNOWN/KEEP roots. Not an exhaustive inode-level protected-byte proof.','e007_verified_sha256':'47d58d914400c08f2d90af8d19a7f2751f7575efd4e63a8f137818fb9357693c','explicit_never_remove':['all 512 final corpus shard certificates','all final shards, hashes, manifests and certification','4649 historical evidence files','all historical reservoirs, WAL, SHM','current metadata referenced checkpoint files','opened shared coordinates.f32','pilot DBs and indices until reviewed recovery']})
save('reclaim_candidates.json',{'PROTECTED':protected,'KEEP FOR NOW':[{'path':str(R/'outputs'),'bytes':dirs[str(R/'outputs')]['bytes'],'reason':'263 GB historical experiments/checkpoint schedules; latest durable state and unique comparisons not fully adjudicated; no deletion.'}]+[r for r in dup if r.get('classification')=='KEEP FOR NOW'],'SAFE RECLAIM':[],'UNKNOWN':candidates+[r for r in dup if r.get('classification')=='UNKNOWN'],'safe_reclaim_bytes':0,'requires_approval_bytes':sum(r['byte_size'] for r in candidates)})
v=os.statvfs('/mnt/d');free=v.f_bavail*v.f_frsize;capacity=v.f_blocks*v.f_frsize;used=(v.f_blocks-v.f_bfree)*v.f_frsize
fs={'initial':{'capacity_bytes':1000202039296,'used_bytes':989465092096,'free_bytes':10736947200,'df_h':'932G capacity; 922G used; 10G available; 99%','df_i':'DrvFS synthetic: 999 inodes, -999001 used, 1000000 free; unusable as real NTFS inode telemetry'},'final':{'timestamp':time.time(),'capacity_bytes':capacity,'used_bytes':used,'free_bytes':free},'project_apparent_path_bytes':dirs[str(R)]['bytes'],'project_data_apparent_path_bytes':dirs[str(D)]['bytes'],'gnu_du_unique_apparent_bytes':{str(R):484699603203,str(D):167743468013},'census_errors':c['errors'],'native_census_errors':read(O/'native_census.json')['errors'],'native_note':'Independent native scan completed in ~72s but omitted 19 long-path entries (~530 MB). Linux census authoritative. Native hashtable Select-Object console summary is null; JSON file has valid measurements.','scope_note':'Whole-drive used bytes include substantial nonproject contents; only two requested project roots fully enumerated. Project census excludes audit folder and symlinks, avoids other mount points.'}
save('filesystem_snapshot.json',fs)
cp=read(O/'checkpoint_inventory.json');production=[r for r in cp if str(E15/'production/checkpoints')+'/' in r['path']];cpbytes=sum(r['bytes'] for r in cp if r['path'].startswith(str(E15)))
checkpoint_size=max(r['bytes'] for r in production);retained_boundary=sum('epoch_' in Path(r['path']).name for r in production)
remaining_min=8-retained_boundary;remaining_max=16-retained_boundary
future_min=(remaining_min+1)*checkpoint_size;future_max=(remaining_max+1)*checkpoint_size
floor=read(E12/'stats/resource_plan.json')['starting_free_bytes']*3//10
rate=next(r for r in rates if r['path']==str(E15/'production'));e15net=rate['bytes_gained']/elapsed
runway=free/e15net if e15net>0 else None
current=read(E15/'production/HEARTBEAT.json');committed=read(E15/'production/COMMITTED.json')
def gib(n):return f'{n/1024**3:.2f} GiB'
with (O/'checkpoint_policy.json').open('w') as f:json.dump({'e015':{'source':str(R/'.e015_worktree/src/protein_distance_diffusion/training/e015_trainer.py'),'atomic_save_source':str(R/'.e015_worktree/src/protein_distance_diffusion/training/e014_runtime.py'),'recovery_seconds':900,'recovery_slots':2,'older_recovery_behavior':'Alternating same-directory atomic replacement; no unbounded slot accumulation','checkpoint_payload':['model','FP32 EMA','optimizer','scaler','global RNG','engine RNG','epoch schedule','microbatch ends','exposure bitmap','position','completed evaluations','summaries','contract'],'current_checkpoint_bytes':checkpoint_size,'temporary_peak_extra_bytes':checkpoint_size,'fractions':[.25,.5,.75,1.0],'epochs_min':2,'epochs_max':4,'remaining_boundary_count_min':remaining_min,'remaining_boundary_count_max':remaining_max,'remaining_checkpoint_net_bytes_min':future_min,'remaining_checkpoint_net_bytes_max':future_max,'rolling_gross_GiB_per_hour':checkpoint_size*4/1024**3,'checkpoint_total_bytes':cpbytes},'e007':{'checkpoint_size_bytes':92321475,'rolling_seconds':900,'boundary_count_remaining_upper':16,'remaining_checkpoint_net_bytes_upper':16*92321475,'temporary_peak_extra_bytes':92321475,'payload':['model','optimizer','RNG','schedule hash','exposure bitmap','position','curves','contract','contention and execution policy'],'latest_checkpoint':str(E7/'checkpoints/sequence_pause_3580.pt'),'sha256_verified':True,'status':'paused'}},f,indent=2)
fix=read(O/'source_fix.json')
summary=f'''# proteinGen storage audit — 2026-10-11

D: is 99% full. E015 is actively training and can reach its next durable checkpoint with current space. E012 is **already failed**, not actively growing: its native diversity owner exited on DATA2-E (disk reserve) before this audit. Do not restart the failed pilot namespace unchanged. No data was deleted, moved, compacted or truncated.

## Filesystem and totals

- Capacity: {capacity:,} bytes ({gib(capacity)}).
- Initial used/free: 989,465,092,096 / 10,736,947,200 bytes.
- Final used/free: {used:,} / {free:,} bytes ({gib(free)} free). Audit output and normal writers explain the small decrease; reclaimed bytes: **0**.
- proteinGen apparent path total: {dirs[str(R)]['bytes']:,} bytes ({gib(dirs[str(R)]['bytes'])}); GNU du unique apparent total: 484,699,603,203 bytes.
- proteinGen_data apparent path total: {dirs[str(D)]['bytes']:,} bytes ({gib(dirs[str(D)]['bytes'])}); GNU du unique apparent total: 167,743,468,013 bytes. Known hardlinks explain 21,095,193,808 bytes of double-counted path size.
- The two project roots do not account for all D: usage. Approximately 337 GB of drive usage is outside their unique apparent totals or filesystem allocation overhead; this audit does not classify unrelated contents.
- DrvFS df -i is synthetic and internally inconsistent; it does not indicate real NTFS inode exhaustion.

## Largest consumers and classification

| Consumer | Apparent size | Classification |
|---|---:|---|
| outputs/ | {gib(dirs[str(R/'outputs')]['bytes'])} | KEEP FOR NOW / UNKNOWN: legacy scientific checkpoint schedules and experiment outputs |
| data/ | {gib(dirs[str(R/'data')]['bytes'])} | PROTECTED shared source/processed datasets |
| E012 sequence_foundation | {gib(dirs[str(D/'sequence_foundation')]['bytes'])} path-counted | PROTECTED except separately reviewed corrupt pilot exports |
| E012 failed identity_50 pilot_tmp | {gib(dirs[str(pilot)]['bytes'])} | KEEP FOR NOW resumability/forensics; two malformed exports UNKNOWN |
| reports/ | {gib(dirs[str(R/'reports')]['bytes'])} | PROTECTED conservative scientific evidence |
| E015 entire run | {gib(dirs[str(E15)]['bytes'])} | PROTECTED |

Legacy outputs contain 44.75 GiB in E001 checkpoints and about 38.64–38.78 GiB each in E002/E003/E004 checkpoints. These dominate long-term project storage but are not an automatic reclaim authorization. Shared data/full contains 223,709 raw mmCIF files and 333,973 processed-recovery samples; the census includes hidden worktrees. largest_directories.tsv holds the top 50; largest_files.tsv holds the top 100. storage_categories.json supplies >=1 GiB, >=250 MiB, small-file counts, checkpoint, cache, SQLite/WAL/SHM, log, archive and evaluation/trajectory inventories.

Conservatively identified protected unique apparent storage lower bound: **{totalprotected:,} bytes ({gib(totalprotected)})**. Additional unique evaluations/latest checkpoints in KEEP/UNKNOWN roots remain protected; this is not an exhaustive inode-level accounting claim. Unequivocal SAFE RECLAIM: **0 bytes**. Actually reclaimed: **0 bytes**.

## Active writers and measured growth

Linux supervisor PID 28874 owns E015 trainer PID 30192. The trainer has D: cache/temp environment settings, and both map the shared D: coordinates.f32. No Linux MMseqs/E012/E007 worker was found. Windows CIM found no Python/MMseqs/linclust worker on repeated queries. E007 state is paused, PID null. Linux lsof +L1 returned no deleted-open entries; native Windows-wide deleted-open handles were not exhaustively enumerable, so their absence is not proven.

Measured window: {elapsed:.1f} seconds ({elapsed/60:.2f} minutes), using targeted regular-file sizes while E015 continued normally.

| Directory | Bytes gained | MiB/min | GiB/hour | GiB/day |
|---|---:|---:|---:|---:|
'''
summary+='\n'.join(f"| {r['path'].replace(str(R)+'/','').replace(str(E12), 'E012 corrected')} | {r['bytes_gained']:,} | {r['MiB_per_min']:.5f} | {r['GiB_per_hour']:.6f} | {r['GiB_per_day']:.6f} |" for r in rates)
summary+=f'''

Nested rates overlap: never sum production + its checkpoint/log/evaluation children. E015 production is the fastest observed project writer, at {rate['MiB_per_min']:.5f} MiB/min net retained growth; E012 rate is **0**. No checkpoint burst occurred inside this measurement window. A later recovery save was observed successfully at update 3103, exposures 93047. Gross recovery serialization is about {checkpoint_size*4/1024**3:.2f} GiB/hour when every 15-minute save occurs, mostly overwriting two retained slots rather than adding space. /proc write_bytes is zero on DrvFS despite real writes, so it is not a reliable byte-rate meter here.

Free / observed net E015 rate gives approximately {runway/86400 if runway else 0:.1f} days of **log-only observational runway**. This is not a usable full-training ETA: retained quarter-epoch checkpoints produce ~0.5 GiB jumps, atomic saves transiently require another ~0.5 GiB, and unrelated D: writers are not attributed. E012's prior exports are bursty runaway writes, not current growth. No meaningful active E012 storage ETA exists while it is stopped.

## Checkpoint recoverability and continuation

Every checkpoint-like artifact found in the E015 artifact root/worktree and E007 control root is listed in checkpoint_inventory.json, with position metadata decoded without loading tensor payloads or running checkpoint globals. Recorded SHA256 values are included where available. Current COMMITTED/LATEST references take precedence over snapshot ages; no checkpoint was removed. E015 current heartbeat: update {current['update']}, exposures {current['exposures']}; durable COMMITTED position: {committed['position']}. E015 checkpoint storage: {cpbytes:,} bytes ({gib(cpbytes)}), including smoke and CUDA replay evidence; production inventory currently totals {sum(r['bytes'] for r in production):,} bytes.

Actual source policy: rolling recovery every 900 seconds, alternates recovery_0.pt/recovery_1.pt, atomic .pt.partial on D:, full reread/model/optimizer/RNG verification, SHA256 then os.replace and LATEST publication. Immutable saves occur at 25/50/75/100% of each epoch, initialization, final, and exceptional stops. Slots are overwritten by policy; scientific boundaries are not pruned. Full payload contains model, FP32 EMA, optimizer, scaler, global/engine RNG, schedule, exposure bitmap, exact position, evaluation set, summaries and contracts. No evidence of recovery pruning failure, unintended trajectory/tensor dumping, or growing atomic .partial copies was established. Per-protein telemetry is aggregated in bounded summaries; production logs are currently small.

E015 current full checkpoint is {checkpoint_size:,} bytes. Remaining checkpoints through two epochs: approximately {future_min:,} bytes net ({gib(future_min)}); through maximum four epochs: approximately {future_max:,} bytes ({gib(future_max)}). Add one {gib(checkpoint_size)} transient atomic save and evaluation/log/cache growth, which is not a hard upper bound. Therefore the next save is comfortably feasible; the full maximum schedule fits on checkpoint sizes alone but leaves only about {gib(free-future_max-checkpoint_size)} before other writes. Long-term simultaneous continuation is not certified safe.

E007 sequence_pause_3580.pt is 92,321,475 bytes. Its SHA256 was freshly computed and matches **47d58d914400c08f2d90af8d19a7f2751f7575efd4e63a8f137818fb9357693c**. LATEST/COMMITTED and shutdown/queue recovery metadata are preserved. Source has the same 15-minute two-slot recovery and quarter-epoch immutable boundary policy, up to four epochs. With all 16 boundaries remaining, checkpoint-only future net storage is at most approximately 1,477,143,600 bytes ({gib(1477143600)}), plus one ~92 MB transient save. Evaluation/control outputs and runtime duration cannot be bounded from current measurements; reassess D: before later resume.

## E012/MMseqs cause and required recovery

Current production STATE says phase=failed, stage_at_failure=diversity, error=runtime disk reserve breached. Its named native PID 30368 is absent. Certification/final markers remain on D:. The corrected corpus contains 512 shards and 512 shard certificates, plus canonical final corpus, per-row manifests and hashes. Historical reservoir scientific/filtered/reservoir.sqlite and all other reservoir copies/WAL/SHM remain untouched, including 4,649 historical evidence files.

The failed 10,000-sequence input is only 2,853,460 bytes, but rep_seq.fasta is 38,785,217,720 bytes and all_seqs.fasta is 13,167,845,376 bytes: **51,953,063,096 bytes ({gib(51953063096)}), approximately {51953063096/2853460:.0f}x input-byte amplification**. rep_seq export took ~18m35s, about 1.94 GiB/min averaged across that substep; that is historical log-based telemetry, not current write rate. Its FASTA contains NULs and out-of-record content. MMseqs ran createdb --createdb-mode 1, zero-copy symlinks to a Windows CRLF subset. Bounded 20-sequence tests reproduce corruption for CRLF/mode1, while LF/mode1 and copied mode0 outputs are clean and agree byte-for-byte. This proves the serialization bug without hashing/rereading the 20M corpus.

The source fix forces LF for the existing fixed-rank diagnostic subset writer. It keeps identity/coverage, ordering/rank, thread count, sample-size rule, scientific selection, retention, historical reservoir and all protocol files unchanged. Installed only while E012/MMseqs was inactive. **It does not repair the existing failed pilot**: easycluster.sh uses file-existence tests for rep_seq/all_seqs, so a naive rerun could publish those malformed/partial files and reuse an invalid pilot resource measurement. Preserve the remaining DB/index/cluster TSV, failed log, input and state until a reviewed replay plan establishes which substeps must be invalidated.

No identity_30 work exists yet. Corrected mmseqs/ and tmp/ roots are empty; historical MMseqs diagnostics remain F/protected. No SAFE A/B/F deletion exists. e012_mmseqs_inventory.json classifies these directories.

E012's existing reserve floor is **{floor:,} bytes ({gib(floor)})**, from 30% of starting_free_bytes. It is E012 runtime behavior, not an E015 scientific launch gate. Minimum reclaim merely to clear that floor now: **{max(0,floor-free):,} bytes ({gib(max(0,floor-free))})**, plus pilot scratch and concurrent E015 headroom. A conservative full 20 GB allowed diversity scratch peak plus a next E015 checkpoint requires about **{max(0,floor+20_000_000_000+checkpoint_size-free):,} bytes** reclaim. The existing predeclared sample-scaling rule may select a smaller diagnostic after a clean LF pilot; actual future scratch/RAM/time cannot be known until that pilot is measured. Two identities run sequentially; scratch is cleaned only after statistics/certification. Final diversity/handoff report sizes are small JSON/Markdown by source structure but not guaranteed exact byte counts.

Reclaiming the two corrupt exports, if approved, would give about {gib(free+51953063096)} free, enough to clear the reserve with a constrained pilot/diagnostic budget; it does not guarantee the full 20 GB maximum scratch plus the whole E015 schedule. E012 currently cannot complete diversity/handoff under its unchanged guard and invalid existing pilot state. Do not lower its guard, delete a reservoir, or silently select a different scientific policy.

## Duplicates and cleanup decision

E015 smoke and production initialization.pt are byte-identical (SHA256 66085f396f6b90b5cb994bf09a324d53f3e25979196f90a8de2546472fa803fc), but serve separate preflight/production evidence roles: KEEP FOR NOW. Actual/expected shutdown fingerprints are also identical but are replay evidence: keep. Audit/preservation reservoir.sqlite and WAL pairs are the same physical files through hardlinks; deleting one path reclaims **zero**. Original reservoir versus copied audited reservoir only matches sampled blocks; full equality is unproven and all are protected. Remaining size/sampled-block checkpoint similarities are documented without treating them as proof of byte equality. No corpus-wide hash scan was performed.

No deletion qualified as unequivocally recomputable, unreferenced, unopened and nonhistorical evidence. proposed_deletions.json records only the two UNKNOWN/high-value malformed exports requiring explicit review; its automatic subset is empty. executed_deletions.json records zero removals. No broad cleanup command was run.

## Source changes and next action

Fix commit: **{fix['source_fix_commit']}**, stored in the D: sparse source checkout `e012_fix/` based on recovery commit af9ada224543c6240c7919b88a4fe93a04a53e12. It changes scripts/prepare_e012_sequence_corpus_v2b.py (LF output only) and adds tests/test_e012_diversity_newlines.py. Two tests pass: simulated Windows serialization preserves exactly 10,000 predeclared ranks; real MMseqs LF/copied-CRLF outputs match. No E015/E007/E014 source or scientific protocol/retention policy changed. The existing E012 source file at /home/simostocco/proteinGen-causal-rope/scripts/prepare_e012_sequence_corpus_v2b.py received exactly the tested change, under filesystem approval; its source repository was not committed or cleaned. All new commit objects, tests, data, reports and scratch are on D:. Before/after source hashes and deployment checks are in source_fix.json. The sparse D: checkout borrows existing base git objects read-only; its new commit lives on D:.

Recommended immediate action: let E015 continue and monitor actual free bytes at recovery/boundary saves. Approve a specific recovery/reclaim plan for the two malformed E012 exports after retaining bounded forensic evidence and establishing fresh pilot invalidation/replay. Then measure a clean LF pilot under the unchanged resource rule before resuming diversity/handoff. For durable reserve beyond those stages, review legacy E001–E004 checkpoint retention separately; do not assume their 160+ GiB is disposable.

**Explicit preservation confirmation:** no protected scientific artifact was removed or modified. No dataset, checkpoint, log, report, cache, temporary file, SQLite/WAL/SHM, MMseqs state, trajectory, evaluation or model artifact was redirected off D:. E015 trainer/supervisor were neither killed, paused nor modified; E007 checkpoint and all E012 reservoirs/evidence/corpus remain intact.
'''
(O/'SUMMARY.md').write_text(summary)
terminal=f'''D: free before: 10,736,947,200 bytes (10.00 GiB)
D: free after: {free:,} bytes ({gib(free)})
largest consumer: proteinGen/outputs — {gib(dirs[str(R/'outputs')]['bytes'])}
fastest-growing consumer: E015 production — {rate['MiB_per_min']:.5f} MiB/min net; checkpoint bursts excluded
E015 checkpoint storage: {gib(cpbytes)} (production {gib(sum(r['bytes'] for r in production))})
E012 diversity/MMseqs storage: {gib(dirs[str(E12/'clusters')]['bytes'])}; corrected mmseqs/tmp empty
safe bytes reclaimed: 0
remaining candidates requiring approval: two malformed pilot exports, 51,953,063,096 bytes (48.39 GiB); not yet SAFE
estimated runway: bursty; log-only {runway/86400 if runway else 0:.1f} days is unsuitable for training planning
active E015 status: training update {current['update']}, exposures {current['exposures']}; next save has headroom
active E012 status: failed/stopped DATA2-E; requires reviewed pilot recovery and reserve
'''
(O/'terminal_summary.txt').write_text(terminal);print(terminal)
