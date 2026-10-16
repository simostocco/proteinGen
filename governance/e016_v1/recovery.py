"""Bounded recovery of existing candidate evidence. No sequence aligner is invoked."""
import hashlib


def _lines(body):
    rows={}
    for line in body.splitlines(keepends=True):
        fields=line.split()
        if len(fields)!=3:raise ValueError('native prefilter triple required')
        key=int(fields[0])
        if key in rows:raise ValueError('duplicate candidate target')
        rows[key]=line
    return rows


def missing_candidate_body(original,previously_aligned):
    """Pure candidate-ID difference; not a biological sequence predicate."""
    all_rows=_lines(original);known=_lines(previously_aligned)
    if not known.keys()<=all_rows.keys():raise ValueError('aligned evidence outside original candidate population')
    return b''.join(row for key,row in all_rows.items() if key not in known)


def recover_bounded(original_entries,aligned_body_for_query,writer,*,max_queries,max_output_bytes):
    """Adapt existing native Reader.entries / Writer.add. Caller owns D-only paths and source seals.

    Stops before the next record that exceeds the frozen budget. The returned cursor
    identifies exactly what was extracted; it never declares the relationship set complete.
    Native alignment of extracted candidates is a separate, unexecuted approval gate.
    """
    if not 0<max_queries<=32 or not 0<max_output_bytes<=128*2**20:raise ValueError('bounded evidence pilot envelope')
    keys=[];pairs=total=0;hasher=hashlib.sha256();deferred=None
    for key,body in original_entries:
        if len(keys)>=max_queries:deferred=key;break
        missing=missing_candidate_body(body,aligned_body_for_query(key))
        if total+len(missing)>max_output_bytes:deferred=key;break
        writer.add(key,missing);keys.append(key);pairs+=len(missing.splitlines());total+=len(missing);hasher.update(str(key).encode()+b'\n'+missing)
    return {'query_keys':keys,'missing_candidate_pairs_extracted':pairs,'output_body_bytes':total,'body_sha256':hasher.hexdigest(),'next_unprocessed_query':deferred,'scope':'EXISTING_CANDIDATE_IDS_ONLY','alignment_executed':False,'relationship_completeness':'NOT_CERTIFIED_REQUIRES_REVIEWED_FULL_RECOVERY'}
