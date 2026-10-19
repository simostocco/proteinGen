"""Persist official sources and inspect live API/SIFTS schemas, no model downloads."""
from r1b_common import *
def main():
 urls=['https://alphafold.ebi.ac.uk/','https://alphafold.ebi.ac.uk/api-doc','https://alphafold.com/api-docs','https://www.ebi.ac.uk/pdbe/news/alphafold-database-release-notes','https://www.ebi.ac.uk/pdbe/news/breaking-changes-afdb-predictions-api','https://github.com/sokrypton/ColabFold','https://github.com/sokrypton/ColabFold/wiki','https://www.ebi.ac.uk/pdbe/','https://www.ebi.ac.uk/pdbe/docs/sifts/quick.html','https://alphafold.ebi.ac.uk/faq']
 rows=[]
 for i,u in enumerate(urls):
  try:p,m=get(u,f'docs/official_{i:02d}.html',8*2**20);rows.append(m)
  except Exception as e:rows.append({'url':u,'error':str(e)})
 s,m=getjson('https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/3fm1','network/SIFTS_3fm1.json');print('SIFTS_SCHEMA',str(s)[:2500],flush=True)
 # Metadata schema only, no unrelated model or MSA download.
 a,m=getjson('https://alphafold.ebi.ac.uk/api/prediction/P69905','docs/API_SCHEMA_EXAMPLE_P69905.json');print('AFDB_SCHEMA',str(a)[:4500],flush=True)
 write('docs/ACCESS_MANIFEST.json',{'sources':rows,'access_date_UTC':time.strftime('%Y-%m-%d',time.gmtime()),'API_example_is_schema_probe_only_no_model_download':True,'schema':a})
if __name__=='__main__':main()
