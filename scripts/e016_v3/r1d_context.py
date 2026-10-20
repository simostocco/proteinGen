"""Public PDBe parent context only; GET metadata, no sequence/MSA submissions."""
from r1d_common import *
import urllib.request

def main():
 bench=json.load(open(R/'STABILITY_VARIANT_BENCHMARK.json'));rows=[]
 for c in bench['selected_families']:
  pid=c['PDB_accession'];r={'parent':c['parent'],'parent_key':c['parent_key'],'pdb_id':pid}
  for key in ['summary','molecules']:
   url=f'https://www.ebi.ac.uk/pdbe/api/pdb/entry/{key}/{pid}';req=urllib.request.Request(url,headers={'User-Agent':'proteinGen-E016-R1d-public-parent-context/1.0'})
   with urllib.request.urlopen(req,timeout=45) as res:
    data=res.read(2**20+1);assert len(data)<=2**20;r[key]=json.loads(data);r[key+'_request']={'URL':url,'bytes':len(data),'SHA256':hashlib.sha256(data).hexdigest(),'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
   (R/'network'/f'PDB_{key}_{pid}.json').write_bytes(data)
  rows.append(r);write('analysis/PARENT_BIOLOGICAL_CONTEXT.json',{'rows':rows,'MSA_submissions':0});print(json.dumps({'parent':c['parent'],'summary':r['summary'],'proteins':[{'names':v.get('molecule_name'),'sequence':v.get('sequence')} for v in r['molecules'].get(pid,[]) if v.get('molecule_type','').startswith('polypeptide')]}),flush=True)
if __name__=='__main__':main()
