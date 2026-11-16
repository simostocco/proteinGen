"""Versioned v2 sequence contracts. No historical parser or scientific inputs changed."""
import hashlib,json
AA='ACDEFGHIKLMNPQRSTVWY'
AA3=dict(zip(['ALA','CYS','ASP','GLU','PHE','GLY','HIS','ILE','LYS','LEU','MET','ASN','PRO','GLN','ARG','SER','THR','VAL','TRP','TYR'],AA))
PARENTS={'MSE':'MET'}
def hash_sequence(s):return hashlib.sha256(s.encode()).hexdigest()
def cached_canonical(value):
    """Decode an identified legacy serialization artifact, not biological trimming.
    The legacy collector joined raw CIF lexical tokens, retaining enclosing semicolons.
    Require agreement with full scheme monomers before accepting this as sequence.
    """
    if value is None:raise ValueError('canonical entity sequence unavailable')
    s=''.join(str(value).split()).upper()
    if s.startswith(';') and s.endswith(';'):s=s[1:-1]
    if s.startswith("'") and s.endswith("'"):s=s[1:-1]
    if not s or any(c not in AA for c in s):raise ValueError('unresolved/noncanonical canonical entity sequence')
    return s
def array(value):
    if value is None:raise ValueError('missing array evidence')
    return json.loads(value) if isinstance(value,str) else value
def reconstruct(chain,selected_sequence,mapped_indices=None):
    full=cached_canonical(chain['entity_poly_sequence'])
    labels=[int(x) for x in array(chain['label_sequence_ids'])]
    names=array(chain['residue_names']);tokens=array(chain['residue_tokens'])
    auth=array(chain['auth_sequence_ids']);ins=array(chain['insertion_codes'])
    if len(full)>10000:raise ValueError('full-polymer length exceeds frozen reconstruction bound')
    if labels!=list(range(1,len(full)+1)):raise ValueError('full deposited label coverage unavailable/contradictory')
    if not all(len(x)==len(full) for x in [names,tokens,auth,ins]):raise ValueError('full-polymer mapping lengths contradict')
    canonical=''.join(AA3.get(PARENTS.get(n,n),'?') for n in names)
    if canonical!=full or ''.join(tokens)!=full:raise ValueError('entity canonical/scheme monomer identity contradiction')
    if chain.get('chain_mapping_status')!='unambiguous' or not chain.get('entity_id'):raise ValueError('ambiguous entity-chain mapping')
    missing=array(chain['missing_residue_mask']);ca_missing=array(chain['missing_calpha_mask'])
    if len(missing)!=len(full) or len(ca_missing)!=len(full):raise ValueError('missing-residue mask lengths contradict')
    observed=''.join(a for a,m in zip(full,ca_missing) if not m)
    if observed!=(chain.get('atom_sequence') or ''):raise ValueError('observed CA sequence contradiction')
    selected_labels=None;bos='UNKNOWN';retained_missing=None
    if mapped_indices is not None:
        idx=[int(x) for x in mapped_indices]
        if any(i<0 or i>=len(full) for i in idx) or any(a>=b for a,b in zip(idx,idx[1:])):raise ValueError('retained N-to-C indices invalid')
        if ''.join(full[i] for i in idx)!=selected_sequence:raise ValueError('historical retained identity contradiction')
        if any(ca_missing[i] for i in idx):raise ValueError('historical selected position has no cached CA')
        selected_labels=[labels[i] for i in idx];bos='polymer_N_terminus' if idx and idx[0]==0 else 'fragment_start'
        kept=set(idx);retained_missing=[i not in kept for i in range(len(full))]
    obs_positions=[i for i,m in enumerate(missing) if not m]
    first=obs_positions[0] if obs_positions else None;last=obs_positions[-1] if obs_positions else None
    ca_positions=[i for i,m in enumerate(ca_missing) if not m]
    start=ca_positions[0] if ca_positions else None;end=ca_positions[-1] if ca_positions else None
    coord_index={pos:i for i,pos in enumerate(ca_positions)}
    mapping=[{'label_seq_id':label,'auth_seq_id':None if str(a) in ['?','.','None',''] else str(a),'insertion_code':str(code or ''),'deposited_monomer':name,'canonical_residue':aa,'residue_observed':not miss,'ca_observed':not ca_miss,'coordinate_index':coord_index.get(i)} for i,(label,a,code,name,aa,miss,ca_miss) in enumerate(zip(labels,auth,ins,names,full,missing,ca_missing))]
    return {'canonical_complete_sequence':full,'canonical_complete_sha256':hash_sequence(full),'deposited_polymer_monomers':names,
      'deposited_one_letter_lexeme':None,'deposited_one_letter_status':'UNKNOWN: uncanonicalized one-letter field not retained in legacy cache',
      'observed_coordinate_sequence':observed,'observed_coordinate_sha256':hash_sequence(observed),'residue_mapping':mapping,
      'missing_residue_mask':missing,'missing_ca_mask':ca_missing,'n_terminal_missing_residues':first,'c_terminal_missing_residues':None if last is None else len(full)-last-1,
      'internal_missing_positions':[] if first is None else [i+1 for i in range(first,last+1) if missing[i]],
      'ca_n_terminal_missing':start,'ca_c_terminal_missing':None if end is None else len(full)-end-1,
      'retained_label_seq_ids':selected_labels,'historical_retained_missing_mask':retained_missing,'historical_bos_semantics':bos,
      'true_terminal_status':'deposited_construct termini established; native gene/proteolysis termini UNKNOWN',
      'modified_residues':[{'label_seq_id':i+1,'monomer':n,'parent':PARENTS[n]} for i,n in enumerate(names) if n in PARENTS],
      'canonicalization_version':'pdb_full_scheme_standard_plus_MSE_v2','full_sequence_lm_bos_semantics':'deposited_polymer_N_terminus'}
def length_stratum(n):
    for a,b in [(20,64),(65,128),(129,256),(257,384),(385,500)]:
        if a<=n<=b:return f'{a}-{b}'
    return 'outside_20_500'
def future_bos_annotation(full_hash,retained_hash,historical_bos):
    if full_hash is None:return {'future_view_bos':'UNKNOWN','starts_at_deposited_N_terminus':None,'complete_deposited_sequence':None}
    complete=full_hash==retained_hash
    return {'future_view_bos':'complete_deposited_polymer_start' if complete else 'fragment_start',
            'starts_at_deposited_N_terminus':True if historical_bos=='polymer_N_terminus' else False if historical_bos=='fragment_start' else None,
            'complete_deposited_sequence':complete}
def choose_evidence(items):
    """Do not pick one of conflicting strong correspondences."""
    matches={(r['physical_candidate_id'],tuple(array(r['mapped_indices']))) for r in items if r['strong_evidential_match'] and r['identity_mapping_available'] and not r['identity_mapping_ambiguous'] and r.get('author_chain_match')}
    if len(matches)>1:raise ValueError('contradictory strong residue correspondences')
    return next(iter(matches)) if matches else None
