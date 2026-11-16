import json, os, stat, time, hashlib, shutil
from pathlib import Path
from datetime import datetime, timezone

OUT=Path('/mnt/d/Simone/proteinGen_data/thesis_storage_inventory_v1')
REPO=Path('/mnt/d/Simone/proteinGen')
V1=Path('/mnt/d/Simone/proteinGen_data/thesis_pdb_sequence_audit_v1')
V2=V1.with_name('thesis_pdb_sequence_audit_v2')
E012=Path('/mnt/d/Simone/proteinGen_data/sequence_foundation/e012_large_corpus_warmstart_20m_20261013_v1')
def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def gib(n): return f'{n/2**30:.3f}'
inv=load(OUT/'INVENTORY.json'); dirs=inv['directories']; G=2**30
def size(p): return dirs.get(str(p),{}).get('bytes',0)
def table(paths):
    return '\n'.join(f'| `{p}` | {gib(size(p))} | {dirs.get(str(p),{}).get("files",0):,} |' for p in paths)
protected_paths=set()
for root in [V1,V2]:
    for name in ['OUTPUT_SHA256.json','ADDITIONAL_OUTPUT_SHA256_V2.json']:
        p=root/name
        if p.exists():
            d=load(p)
            if isinstance(d,dict): protected_paths.update(str(root/k) for k in d if isinstance(k,str))

# Existing duplicate evidence is read as an inventory, never rehashed.
duplicates=load(REPO/'storage_audit_20261011/duplicate_readout.json')
hardlinks={}
for row in duplicates:
    current=[]
    for p in row['paths']:
        try:
            s=os.stat(p,follow_symlinks=False)
            current.append({'path':p,'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'device':s.st_dev,'inode':s.st_ino,'nlink':s.st_nlink,'mtime_ns':s.st_mtime_ns})
            if s.st_nlink>1 and stat.S_ISREG(s.st_mode):
                group=hardlinks.setdefault((s.st_dev,s.st_ino),{})
                group[p]=s.st_size
        except OSError: current.append({'path':p,'status':'MISSING'})
    row['current_metadata']=current
    row['current_content_equality']='UNKNOWN: previous SHA256/sample evidence only; no content reread'
    valid=[r for r in current if 'inode' in r]
    row['current_same_physical_file']=len(valid)==len(current) and len(valid)>1 and len({(r['device'],r['inode']) for r in valid})==1
known_overcount=sum((len(paths)-1)*next(iter(paths.values())) for paths in hardlinks.values())
(OUT/'DUPLICATE_EVIDENCE.json').write_text(json.dumps({'source':str(REPO/'storage_audit_20261011/duplicate_readout.json'),'current_stat_only':True,'known_hardlink_overcount_bytes':known_overcount,'scope':'known duplicate candidates only; not exhaustive','groups':duplicates},indent=2))

jobs=['additional_train_vs_validation','additional_validation_vs_train','full_validation_vs_train']
mm_paths=[str(V2/'mmseqs_search'/j/'tmp') for j in jobs]
assert all(load(V2/'mmseqs_search'/j/'COMPLETE.json')['status']=='completed' for j in jobs)
assert not any(any(p==r or p.startswith(r+'/') for r in mm_paths) for p in protected_paths)
cache_paths=[p for p in dirs if p.startswith(str(REPO)+'/') and p.endswith('/__pycache__')]
pytest_paths=[p for p in dirs if p.startswith(str(REPO)+'/') and p.endswith('/.pytest_cache')]
categories=[
 {'id':'C1','name':'Temporanei delle tre ricerche MMseqs audit v2','paths':mm_paths,'reason':'Ricerche completate; tmp esclusi dai 299 output certificati; input, allineamenti TSV, COMMAND/COMPLETE, versioni e log restano protetti.','resume_risk':'Nessun input di training/resume identificato in questi tre tmp; perdita del resume intermedio delle ricerche e di ispezioni forensi sui database.','reproducibility_risk':'Basso per risultati pubblicati conservati; moderato per replay dei passi intermedi. Rigenerazione richiede nuova autorizzazione CPU, stessa versione e ambiente; uguaglianza bitwise futura non garantita.','preconditions':['Approvazione esplicita C1 e dei tre percorsi selezionati','Ricontrollare COMPLETE, integrità dichiarata dei TSV/input e assenza di riferimenti esterni ai tmp','Nessun processo/handle attivo; non seguire symlink; verificare hardlink e file checkpoint inattesi','Non toccare nessun altro file degli audit v1/v2']},
 {'id':'C2','name':'Bytecode Python ricostruibile nel repository e worktree','paths':cache_paths,'reason':'Solo file .pyc dentro __pycache__ con corrispondente sorgente .py presente; non sono pesi o osservazioni scientifiche. Il totale censito è un limite superiore, non tutto automaticamente ammissibile.','resume_risk':'Basso solo dopo verifica dei sorgenti; piccolo costo al riavvio/import. Non coinvolge checkpoint, stato RNG, coordinate o corpus.','reproducibility_risk':'Basso per cache ricostruibile; UNKNOWN per bytecode senza sorgente o incluso in certificati, che va preservato.','preconditions':['Approvazione esplicita C2','Congelare lista dei soli .pyc con sorgente .py verificato presente; escludere file firmati da manifest e snapshot protetti','Preservare cache di soli bytecode senza sorgente; nessuna rimozione di directory ambiente o altri file']},
 {'id':'C3','name':'Cache pytest nel repository e worktree','paths':pytest_paths,'reason':'Solo cache di nodeids/lastfailed/stepwise; risultati, fixture, log e cartelle dei test restano conservati.','resume_risk':'Nessun effetto sul resume dei modelli; perdi selezioni pytest --lf/--ff e --stepwise.','reproducibility_risk':'Basso solo se non parte di evidenza certificata; risultati scientifici e test di replay non vengono proposti.','preconditions':['Approvazione esplicita C3','Escludere qualsiasi file incluso in inventari scientifici/checksum e verificare contenuto di sola cache pytest','Non toccare runtime/pytest, checkpoint di fixture o risultati dei test']}
]
all_paths=[]
for c in categories:
    c['approval_required']=True; c['approval_status']='NOT_REQUESTED / NOT_GRANTED'; c['executed']=False
    c['apparent_bytes_upper_bound']=sum(size(p) for p in c['paths'])
    c['recoverable_allocated_bytes']='UNKNOWN until eligible file list, hardlinks/handles and NTFS allocation verified'
    c['guaranteed_recoverable_bytes_now']=0
    for p in c['paths']:
        assert not any(p==q or p.startswith(q+'/') or q.startswith(p+'/') for q in all_paths)
        all_paths.append(p)
potential=sum(c['apparent_bytes_upper_bound'] for c in categories)
(OUT/'CLEANUP_PROPOSALS.json').write_text(json.dumps({'status':'PLAN ONLY; no cleanup performed','disjoint_categories':categories,'nonoverlapping_apparent_bytes_upper_bound':potential,'guaranteed_recoverable_bytes_now':0,'excluded':'All checkpoints, E012 reservoirs/WAL/SHM, certified 20M corpus and protected scientific evidence'},indent=2))

states={}
for name,p in [('E007',REPO/'.e014_worktree/e007_matched_fullscale_v1/STATE.json'),('E015',REPO/'e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1/STATE.json'),('E012',E012/'status.json')]:
    d=load(p); states[name]={'source':str(p),'state':d.get('phase',d.get('state')),'pid_recorded':d.get('pid'),'mtime_ns':p.stat().st_mtime_ns}
# Host process inspection: retain only relevant identifiers, never unrelated command lines/tokens.
active=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try:
        cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        if 'thesis_storage_inventory' in cmd or 'codex' in cmd: continue
        matches=[token for token in ['run_e012','train_e007','train_e015','run_e015','e007_matched_fullscale','mmseqs','colabfold','ProteinMPNN','foldseek'] if token.lower() in cmd.lower()]
        if matches: active.append({'pid':int(proc.name),'matched_tokens':matches})
    except OSError: pass
(OUT/'SAFETY_STATUS.json').write_text(json.dumps({'recorded_states':states,'local_linux_relevant_processes':active,'windows_and_remote_process_status':'UNKNOWN: no global handle or remote inspection','training_evaluation_download_launched':False,'cleanup_executed':False,'wsl_metadata_attempt':'Interrupted with SIGINT only on own inventory process; no scientific worker signalled','native_census_seconds':inv['finished_unix']-inv['started_unix']},indent=2))

main_paths=[str(REPO),str(REPO/'data/full'),str(Path('/mnt/d/Users/Simone Stocco/proteinGen_audits'))]+[r['path'] for r in inv['roots'][2:]]
sub_paths=[str(REPO/x) for x in ['outputs','reports','data/full/raw/mmcif','data/full/processed','data/full/processed_recovery/samples','data/full/splits_recovered_all_structures','reports/experiments/E007_matrix_sequence_cogeneration','.e014_worktree/e007_matched_fullscale_v1','e015_artifacts','e014_artifacts']]
e_dirs=[p for p in dirs if (p.startswith(str(REPO/'e015_artifacts')) or p.startswith(str(REPO/'.e014_worktree/e007_matched_fullscale_v1'))) and p.endswith(('/checkpoints','/evaluation','/evaluations'))]
e_dirs=sorted(e_dirs,key=size,reverse=True)[:10]
largest=sorted(inv['largest_files'],key=lambda r:r['bytes'],reverse=True)
assert len(largest)>0 and largest[0]['bytes']>G, 'Largest-file enumeration must include multi-GiB reservoirs'
with (OUT/'LARGEST_FILES.tsv').open('w') as f:
    f.write('bytes\tmtime_utc\tpath\n')
    for r in largest: f.write(f"{r['bytes']}\t{r['mtime_utc']}\t{r['path']}\n")
with (OUT/'DIRECTORIES.tsv').open('w') as f:
    f.write('bytes\tfiles\treparse_points\tpath\n')
    for p,r in sorted(dirs.items(),key=lambda x:x[1]['bytes'],reverse=True): f.write(f"{r['bytes']}\t{r['files']}\t{r['symlinks']}\t{p}\n")
counts=load(V2/'DATASET_COUNTS_V2.json')
rows=sum(counts['population_counts'].values()); eligible=sum(v['lm_eligible_records'] for v in counts['counts'].values())
mapping=size(V2/'structure_observations'); ca_upper=eligible*500*3*4
root_total=sum(r['bytes'] for r in inv['roots'])
when=datetime.fromtimestamp(inv['finished_unix'],timezone.utc).isoformat()
report=f'''# Inventario disco proteinGen — v1

**READ-ONLY sui dati esistenti**, concluso {when}. Nessuna cancellazione, spostamento, compressione, download, training o valutazione. Scritti soltanto questo inventario e i suoi script su D:. **VERIFIED** = metadati correnti o evidenza storica esplicitamente datata; **INFERRED** = stima; **UNKNOWN** = verifica mancante.

D: ha **{gib(inv['disk_after']['free'])} GiB liberi** ({inv['disk_after']['free']:,} byte; capacità {gib(inv['disk_after']['total'])} GiB, occupazione {(1-inv['disk_after']['free']/inv['disk_after']['total'])*100:.2f}%). Enumerati **{inv['regular_files']:,} file** in sei radici disgiunte, {len(inv['errors'])} errori, in {inv['finished_unix']-inv['started_unix']:.1f} secondi tramite NTFS nativo. Un precedente tentativo WSL, più lento, è stato interrotto sul solo processo dell'inventario. Nessun file di dataset è stato aperto o ricalcolato tramite hash.

Le dimensioni sotto sono **logiche/apparenti**, non spazio fisico recuperabile certificato. Non sommare padri e figli. Hardlink già noti e riconfermati tramite stat sovracontano **{gib(known_overcount)} GiB** nelle sei radici; la ricerca hardlink non è esaustiva. Totale delle sei radici: {gib(root_total)} GiB prima di questa correzione; il resto dell'occupazione D: non è classificato. Reparse point/symlink esclusi, archivio non fotografato atomicamente, handle Windows e host remoti non verificati. `INVENTORY.json` contiene byte esatti, denominatori e finestra temporale.

## Occupazione

| Directory richiesta (data/full è inclusa nel repository) | GiB apparenti | File |
|---|---:|---:|
{table(main_paths)}

| Principali sottoinsiemi, non sommare ai rispettivi padri | GiB | File |
|---|---:|---:|
{table(sub_paths)}

| Checkpoint/evaluation E007/E015, sottoinsiemi sovrapposti | GiB | File |
|---|---:|---:|
{table(e_dirs)}

`DIRECTORIES.tsv` dettaglia tutte le directory: include training, preflight, continuation, evaluation e loop esterno E015. I maggiori insiemi sono output storici, mmCIF grezzi e NPZ di recovery; le loro dimensioni non autorizzano la pulizia.

## File più voluminosi e copie

| File corrente, primi 8 del censimento | GiB |
|---|---:|
'''+ '\n'.join(f'| `{r["path"]}` | {gib(r["bytes"])} |' for r in largest[:8])+f'''

`LARGEST_FILES.tsv` riporta i primi 200 tra i file >=100 MiB, contando anche percorsi hardlink. **VERIFIED fisico:** reservoir `v2c_optimization/audit` e `preservation` condividono inode per SQLite/WAL; un singolo unlink recupererebbe zero. Tutti restano protetti. **VERIFIED storico, uguaglianza attuale UNKNOWN:** audit del 2026-10-11 aveva SHA identici per inizializzazione E015 smoke/production e fingerprint actual/expected; sono ruoli scientifici distinti e si preservano. Per altri checkpoint uguaglianza di dimensione o di blocchi campionati non dimostra copia eliminabile. `DUPLICATE_EVIDENCE.json` distingue queste prove senza nuovo hashing.

Raw/processed/recovery, sidecar/NPZ, split rappresentativi/all-structures e corpus FASTA/shard/manifest sono rappresentazioni o popolazioni diverse. `scientific` e `scientific_corrected_protein_v1` E012 documentano protocolli diversi, incluso il problema storico dell'alfabeto: non sono duplicati eliminabili. SQLite v1 e gli altri audit sono evidenza protetta. I due export corrotti da 51.953 GB risultano **già rimossi** nel manifest approvato del 2026-10-11; non rientrano nel recuperabile attuale.

## Dipendenze e materiale immutabile

| Catena documentata | Da preservare |
|---|---|
| mmCIF → processed/recovery → manifest/split → pairing → E006 → E012 storico / E007 | Raw, NPZ, split originali, pairing, tutti i sidecar, protocolli e hash |
| E007 accettato → popolazione/cache E014 → E015 | Popolazione sigillata, `coordinates.f32`, normalizzazione, config e contratti; cache E014 è input di resume |
| Corpus corretto UniRef50 20M → warm-start E012 → milestone/recovery → evaluation | Corpus certificato completo, shard/manifest/certificati, reservoir storico con WAL/SHM, checkpoint e pannelli congelati |
| Audit raw-full/E006/pairing + v1 → mapping/screening v2 → futura proposta v3 | Tabelle cache autorevoli, inventari/hash, corrispondenze, output, test e tentativi falliti preservati |
| Checkpoint → COMMITTED/LATEST + optimizer/RNG/schedule → resume | Tutti i checkpoint, anche smoke, replay, boundary, recovery e copie con ruolo distinto |

Fonti: `thesis_pdb_sequence_audit_v1/DATASET_LINEAGE.md`, `PROVENANCE_SHA256.json`; v2 `PROVENANCE_V2.json`, `OUTPUT_SHA256.json`, `ADDITIONAL_OUTPUT_SHA256_V2.json`, `RESOURCE_PLAN.md`; configurazioni `configs/e007_coordinate_real_continuation_to_10000_v1.yaml`, `.e015_worktree/configs/e015_integrated_multiscale_parity_selfconditioned_v1.yaml`; warm-start `audit.json`, `protocol.json`, `pause_verification.json`; vecchio `storage_audit_20261011/{{protected_artifacts,duplicate_readout,executed_deletions}}.json`. Tutti i percorsi sono espansi nei JSON del report o negli inventari citati. Hash esistenti consultati come dichiarazioni, non ricertificati leggendo i dati.

Stati registrati: E007 `{states['E007']['state']}`, E015 `{states['E015']['state']}`, E012 `{states['E012']['state']}`. Processi Linux pertinenti rilevati: {len(active)}; limite Windows/remoto UNKNOWN. Nessun segnale o modifica ai run scientifici.

## Piano di pulizia: solo proposte, approvazione separata

| Categoria disgiunta | Limite superiore apparente | Motivazione e rischio |
|---|---:|---|
| C1: tre `v2/mmseqs_search/<job>/tmp` | {gib(categories[0]['apparent_bytes_upper_bound'])} GiB | Completamento verificato, tmp fuori dagli output certificati; perde restart/intermedi MMseqs, conserva evidenza pubblicata. Rischio replay moderato, resume modelli basso |
| C2: `__pycache__` nel repository/worktree | {gib(categories[1]['apparent_bytes_upper_bound'])} GiB | Solo .pyc con sorgente presente e fuori dai certificati. Rischio basso dopo verifica; bytecode senza sorgente resta protetto |
| C3: `.pytest_cache` nel repository/worktree | {categories[2]['apparent_bytes_upper_bound']/2**20:.3f} MiB | Perde selezione lastfailed/stepwise, preserva risultati e fixture. Rischio basso dopo verifica |

**Somma non sovrapposta dei soli limiti superiori: {gib(potential)} GiB; recupero garantito adesso: 0 byte.** Allocazione NTFS e lista realmente ammissibile richiedono controllo prima dell'azione. Nessuna proposta multi-GiB è sostenuta dalle evidenze attuali senza intaccare materiale protetto. Gli altri tmp E012, pilot interrotti, shutdown/replay, database di screening e log sono conservati: nome `tmp` non dimostra disponibilità alla pulizia. Nessuna proposta su checkpoint, reservoir, corpus certificato o evidenze protette.

Ordine eventuale: approvazione esplicita **C1**, verifica riferimenti/handle/hardlink e conservazione di input/TSV/certificati; poi approvazione distinta **C2**, lista selettiva dei .pyc ricostruibili; poi approvazione distinta **C3**, sola cache pytest. Ricensire metadati e libero prima/dopo ciascuna categoria autorizzata. Qualsiasi contraddizione blocca la categoria. `CLEANUP_PROPOSALS.json` contiene percorsi esatti, motivazioni, rischi e prerequisiti; nessun comando di rimozione è eseguito o incluso.

## Budget PDB-only v3 (INFERRED, da approvare scientificamente)

Base dichiarata v2: **{rows:,} record** storici E012 TRAIN/VALIDATION + TEST rappresentativo; **{counts['full_lm_exact_unique_eligible']:,} sequenze complete canoniche uniche 20–500 aa**, collegate a **{eligible:,} record eleggibili**. Non è l'intero PDB e TEST/split v3 non sono ancora certificati. Misure correnti: mapping di 224 partizioni **{gib(mapping)} GiB**, parquet LM unico **{gib((V2/'proposed_lm_exact_unique.parquet').stat().st_size)} GiB**.

| Scenario, mantenendo tutte le fonti | Nuovo persistente | Picco build aggiuntivo | Riserva separata |
|---|---:|---:|---:|
| Stesso perimetro v2, tabelle/manifest/mapping compressi con riferimenti alle coordinate esistenti | 1–3 GiB | 10–14 GiB | 10 GiB |
| Stesso perimetro, anche coordinate Cα compatte autonome per i soli record eleggibili 20–500 | 3–5 GiB | 14–18 GiB | 10 GiB |

Budget include output finale, una copia di staging/validazione e scratch sequenziale limitato a 8 GiB come v2; non è un benchmark né autorizzazione a MMseqs. Il limite teorico Cα float32 è {gib(ca_upper)} GiB = {eligible:,} × 500 × 3 × 4 byte, più mapping/mask/metadata. Non vale per catene >500, altri atomi o tutti i modelli NMR. L'espansione al PDB completo è **UNKNOWN** finché perimetro, osservazioni e policy non sono congelati. Copiare raw/NPZ o generare matrici dense L×L esce da questi scenari e richiede un nuovo budget.

Con {gib(inv['disk_after']['free'])} GiB liberi il progetto limitato ai metadati appare fattibile con budget e riserva; la pulizia proposta libererebbe meno di 1 GiB. Libero da ricontrollare prima di un futuro build: crescita di altri processi e espansioni di scope non sono garantite. Nessun dataset v3 è stato costruito.

## Handoff

Inventario completato; dati storici preservati; run scientifici mantenuti in pausa. Autorizzazioni di pulizia **non richieste né ricevute**. Attendere una tua scelta esplicita per C1/C2/C3 o per il perimetro/budget v3. `CHECKS.json` valida ordinamento, categorie non sovrapposte e esclusione dei tmp dagli hash certificati; `REPORT_SHA256.json` riguarda soltanto i nuovi file del report.
'''
(OUT/'REPORT.md').write_text(report)
(OUT/'CHECKS.json').write_text(json.dumps({'native_errors':inv['errors'],'categories_nonoverlapping':True,'mmseqs_complete_and_tmp_outside_certified_paths':True,'largest_file_sort_correct':True,'checkpoint_or_reservoir_cleanup_proposed':False,'no_training_or_cleanup_launched':True,'note':'Metadata consistency checks only; no scientific reanalysis or test training'},indent=2))
for source in ['thesis_storage_inventory_native.ps1','thesis_storage_inventory_tool.py','thesis_storage_inventory_report.py']:
    shutil.copyfile(REPO/source,OUT/source)
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='REPORT_SHA256.json'}
(OUT/'REPORT_SHA256.json').write_text(json.dumps(hashes,indent=2))
print(json.dumps({'report':str(OUT/'REPORT.md'),'free_gib':inv['disk_after']['free']/G,'proposed_upper_bound_gib':potential/G,'known_hardlink_overcount_gib':known_overcount/G,'largest_file':largest[0],'active_relevant_linux_processes':active,'outputs':len(hashes)+1},indent=2))
