"""Clarify the frozen effect-size gate without changing results or decisions."""
from pathlib import Path
import hashlib,json,os,time
R=Path('/mnt/d/Simone/proteinGen_data/e017_v72_counterfactual_ranking_20261023_v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,value):
 p=R/name;q=p.with_name(p.name+'.part')
 with q.open('w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
p=json.loads((R/'PROVENANCE_SHA256.json').read_text());assert all(sha(R/n)==h for n,h in p['output_SHA256'].items())
paragraph='The stage-0 MLP return gain is positive under the paired conditional bootstrap; the NO-GO does not negate that signal. Its gain of 0.04050 is below the prospectively frozen 0.05 minimum effect versus uniform selection. Replicated ranking covers 18 action pairs but only 3 independent proteins, below the required 12 proteins. The heuristic-comparison gain passes its own 0.025 effect threshold. Critic 1 has lower return RMSE (0.1280 versus MLP 0.1626), yet its independent selection gain interval includes zero: calibration error and useful action control are different endpoints. The independent EST Monte Carlo selector gains 0.05072 over uniform selection, but is a sampling procedure rather than the prospectively selected learned model.'
for name in ['FINAL_E017_V72_SCIENTIFIC_REPORT.md','NEXT_CRISP_RESEARCH_DECISION.md']:
 path=R/name;value=path.read_text();assert paragraph not in value;path.write_text(value+'\n\n'+paragraph+'\n')
resource=json.loads((R/'RESOURCE_AND_RECOVERY_REPORT.json').read_text());resource['watchdog']['scope']='Collection and initial prediction attempt. Resumed read-only prediction retained per-boundary disk/GPU admission checks; final host audit verified resource release. Watcher restart can exit before the resumed worker publishes RUNNING.';save('RESOURCE_AND_RECOVERY_REPORT.json',resource)
p['output_SHA256']={n:sha(R/n) for n in p['output_SHA256']};p['input_SHA256'][str(Path(__file__).resolve())]=sha(__file__);p['handoff_clarification']=dict(UTC=time.time(),scope='explain positive sign versus frozen minimum effect and independent-protein coverage; no result, threshold or decision changes');save('PROVENANCE_SHA256.json',p)
v=json.loads((R/'FINAL_VERIFICATION.json').read_text());v['provenance_SHA256']=sha(R/'PROVENANCE_SHA256.json');v['verified_input_hashes']=len(p['input_SHA256']);v['handoff_clarification_UTC']=time.time();save('FINAL_VERIFICATION.json',v)
assert all(sha(R/n)==h for n,h in p['output_SHA256'].items());print('Final effect-size clarification and output hashes verified')
