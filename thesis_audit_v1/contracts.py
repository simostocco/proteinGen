"""Isolated proposed-v1 validation; never used by historical training."""
import hashlib
def validate_mapping(full_sequence, positions, observed_sequence, ca_mask):
    if not (len(positions)==len(observed_sequence)==len(ca_mask)):
        raise ValueError('mapping lengths differ')
    if any(not isinstance(p,int) or p<1 or p>len(full_sequence) for p in positions):
        raise ValueError('invalid canonical position')
    if any(a>=b for a,b in zip(positions,positions[1:])):
        raise ValueError('positions must increase N-to-C')
    if any(full_sequence[p-1]!=aa for p,aa in zip(positions,observed_sequence)):
        raise ValueError('sequence correspondence contradiction')
    return {'missing_residue_mask':[i not in positions for i in range(1,len(full_sequence)+1)],
            'n_terminal_trim':positions[0]-1 if positions else None,
            'c_terminal_trim':len(full_sequence)-positions[-1] if positions else None,
            'complete_ca':all(ca_mask),
            'peptide_adjacency':[b==a+1 and va and vb for a,b,va,vb in zip(positions,positions[1:],ca_mask,ca_mask[1:])]}
def assert_disjoint(rows):
    seen={}
    for r in rows:
        for key in ['sample_id','canonical_sequence_sha256','pdb_id','homology_cluster_id']:
            value=r.get(key)
            if value is None:raise ValueError('missing split safety identity')
            old=seen.setdefault((key,value),r['split'])
            if old!=r['split']:raise ValueError('cross-split leakage: '+key)
