"""Presentation-only final pass; scientific data and checkpoint/source bindings unchanged."""
from crisp_v2_common import *
import re
for p in R.glob('*.md'):
 s=p.read_text();s=s.replace('-53,530','- 53,530').replace('J1+','J1 +')
 prefixes='from|and|are|all|All|of|full|frozen|about|no|proposed|only|lineage:|error|RSS|free|gamma'
 s=re.sub(r'\b('+prefixes+r')(?=\d)',r'\1 ',s)
 suffixes='discarded|identity|identities|across|claim|parents|degrees|radians|updates|steps|head|family|checkpoint|learning|clipping|examples|quota|residues|seconds|cases|components'
 s=re.sub(r'(\d)('+suffixes+r')\b',r'\1 \2',s)
 s=re.sub(r'(\d)(GiB|MiB|ms|s)\b',r'\1 \2',s);s=s.replace('%motif','% motif').replace('P1/P2tasks','P1/P2 tasks').replace('objective.3*N','objective. 3*N')
 s=re.sub(r'(\d(?:\.\d+)?e[+-]\d+)A\b',r'\1 A',s)
 p.write_text(s)
print('Report presentation checked')
