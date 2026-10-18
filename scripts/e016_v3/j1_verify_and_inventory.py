"""Protected historical integrity, cumulative ClassA cleanup and approval-only candidates."""
import json,os,math,shutil,collections,subprocess,heapq,time
from pathlib import Path
import torch
import j1_torsion_diagnostic as d
R=d.R

def verify():
 prov=json.load(open(d.O/'PROVENANCE_SHA256.json'));checks=[]
 def check(p,expected):
  actual=d.file_sha(p)
  if actual!=expected:raise ValueError('protected artifact changed '+str(p))
  checks.append({'path':str(p),'sha256':actual})
 for rel,v in prov['outputs'].items():check(d.O/rel,v['sha256'])
 for rel,v in prov['tracked_source_sha256'].items():check(d.O/'source'/rel,v)
 for p,v in prov['protected_checkpoint_sha256'].items():check(p,v)
 for p,v in prov['input_source_data_and_reference_manifest_sha256'].items():check(p,v['sha256'])
 check(d.O/'PROVENANCE_SHA256.json','80b6c6e193b71ee17a023cc7dfa69ad3ced84b6637114c8f82db01b3a095a362')
 # Prior sequence-first manifest establishes that those published artifacts/sources also remain intact.
 old=json.load(open(d.Q/'PROVENANCE_SHA256.json'));extra={}
 for key in ['outputs','tracked_source_sha256']:
  count=0
  for rel,v in old.get(key,{}).items():
   p=d.Q/('source' if key=='tracked_source_sha256' else '')/rel;check(p,v['sha256'] if isinstance(v,dict) else v);count+=1
  extra[key]=count
 d.write('PROTECTED_INTEGRITY_VERIFICATION.json',{'status':'PASS','checked_files':len(checks),'checks':checks,'sequence_first_counts':extra,'raw_source_note':'source/mapping seals and loader source integrity verified; no new full raw archive rehash or reconstruction','source_worktree_status':subprocess.check_output(['git','status','--porcelain'],cwd=d.O/'source',text=True),'optimizer_steps':0})

def cleanup():
 plan=json.load(open(R/'SAFE_CLEANUP_ATTEMPT_02_PLAN.json'));journ=[json.loads(x) for x in open(R/'CLEANUP_DELETION_JOURNAL.jsonl')];by={x['path']:x for x in journ};missing=[v for v in plan['candidate_files'] if not Path(v['path']).exists()]
 if {v['path'] for v in missing}!=set(by):raise ValueError('deletion journal does not reconcile exactly')
 if any(not Path(v['path']).exists() for v in plan['preserved_cache_files']):raise ValueError('protected cache entry lost')
 categories=collections.Counter()
 for x in missing:
  kind='Python bytecode' if '__pycache__' in x['path'] or x['path'].endswith('.pyc') else 'compiler cache' if any(t in x['path'] for t in ['/triton/','/cuda/','/inductor/']) else 'pytest/cache metadata';categories[kind]+=x['allocated_bytes']
 receipts=[json.load(open(R/n)) for n in ['SAFE_CLEANUP_ATTEMPT_03_RECEIPT.json','SAFE_CLEANUP_MANIFEST.json']]
 d.write('SAFE_CLEANUP_MANIFEST.json',{'status':'COMPLETE','ClassA_deleted_files':len(missing),'allocated_bytes_removed':sum(x['allocated_bytes'] for x in missing),'logical_bytes_removed':sum(x['logical_bytes'] for x in missing),'free_before_first_cleanup_plan_bytes':plan['free_before_bytes'],'free_after_all_cleanup_and_diagnostics_bytes':shutil.disk_usage(R).free,'measured_net_free_change_includes_new_diagnostic_outputs_bytes':shutil.disk_usage(R).free-plan['free_before_bytes'],'categories_allocated_bytes':dict(categories),'deleted':missing,'preserved_initial_cache_files':len(plan['preserved_cache_files']),'all_preserved_initial_cache_entries_exist':True,'ClassB_deleted':0,'ClassC_deleted':0,'initial_plan':'SAFE_CLEANUP_ATTEMPT_02_PLAN.json','journal':'CLEANUP_DELETION_JOURNAL.jsonl','journal_exact_set_reconciliation':True,'attempts_note':'first scan interrupted before deletion; second scan interrupted during ClassA deletion with every deletion journaled; third scan conservatively preserved remaining caches because it indexed its own cleanup hashes; fourth excluded this diagnostic receipt from protection and completed disposal; all attempt logs retained; no training process interrupted','pass_receipts':receipts,'scope_limit':'cache inventory covers D: worktrees/experiment runtime cache locations; not every possible cache inside arbitrary historical output trees; published protection paths/SHA and Git tracked paths checked; no scientific fixtures retired'})

def candidates():
 old=json.load(open(R/'HOMOLOGY_STORAGE_CANDIDATES.json'));groups={}
 for name,prefix in [('full_scratch','pref_0'),('independent_all_pool_v1','prefilter')]:
  files=[v for v in old[name]['largest_files'] if Path(v['path']).name in [prefix+'.0',prefix+'.1',prefix+'.2',prefix+'.3',prefix+'.index']]
  groups[name]={'class':'B_APPROVAL_REQUIRED','files':files,'allocated_reclaimable_bytes_if_all_unlinked':sum(v['allocated_bytes'] for v in files if v['nlink']==1),'identical_durable_copy':'UNKNOWN; not demonstrated; the two candidates differ in shard sizes and are not interchangeable copies','reproducibility':'derived MMseqs candidate/prefilter material from retained FASTA/DB and commands, but bitwise regeneration/runtime not demonstrated','scientific_importance':'historical similarity-screen audit/recovery evidence; not raw proteins or trained weights','consequence':'retire direct inspection/restart from these candidate shards; preserve FASTA/queryDB/final edges/logs/receipts and explicitly record retirement if approved','recommended_action':'preserve pending explicit artifact-retirement approval; do not delete whole directory'}
 duplicates=[]
 for k in [1500,2500]:
  model=d.O/f'runs/J1/model_{k:05d}.pt';full=d.O/f'runs/J1/checkpoint_{k:05d}.pt';a=torch.load(model,map_location='cpu',weights_only=True)['model'];b=torch.load(full,map_location='cpu',weights_only=True)['model'];same=a.keys()==b.keys() and all(torch.equal(a[n],b[n]) for n in a)
  if not same:raise ValueError('duplicate model equivalence failed')
  s=model.stat();duplicates.append({'path':str(model),'allocated_bytes':s.st_blocks*512,'class':'B_APPROVAL_REQUIRED','identical_model_tensors_in':str(full),'byte_identical_serialized_file':False,'importance':'published candidate snapshot with historical SHA256','consequence':'old manifest would show retired file; new explicit retirement receipt/approval needed','recommended_action':'low priority, preserve now; never delete corresponding protected full checkpoint'})
 d.write('APPROVAL_ONLY_STORAGE_CANDIDATES.json',{'prefilters':groups,'model_only_duplicate_snapshots':duplicates,'all_preserved':True,'no_delete_authority_assumed':True})

def main():
 verify();cleanup();candidates();print('PROTECTED_INTEGRITY_AND_CLEANUP_RECONCILIATION_PASS')
if __name__=='__main__':main()
