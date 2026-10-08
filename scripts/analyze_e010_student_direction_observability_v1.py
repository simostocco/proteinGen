"""Label-blind nearest neighbors and geometry-matched observable-context tests."""
import concurrent.futures
import json
import multiprocessing
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
import e010_student_direction_observability_v1 as e
import faiss
CONTEXTS=e.PROTOCOL['contexts']+['initial_coarse_correction','rollout_history','denoiser_geometry']


def open_array(name,mode='r'):return np.load(e.ROOT/'panel'/f'{name}.npy',mmap_mode=mode)


def create_array(name,shape,dtype):return np.lib.format.open_memmap(e.ROOT/'panel'/f'{name}.npy',mode='w+',dtype=dtype,shape=shape)


def panel():
    assert (e.ROOT/'EXTRACTION_COMPLETED.json').exists()
    marker=e.ROOT/'PANEL_COMPLETED.json'
    if marker.exists():return
    rows=e.rows();splits=json.loads((e.old.ROOT/'splits.json').read_text())['assignments']
    group_map={s:i for i,s in enumerate(sorted({r['split_group_id'] for r in rows}))}
    cluster_map={s:i for i,s in enumerate(sorted({r['cluster_id'] for r in rows}))}
    n=2000*3*16*8
    arrays={name:create_array(name,(n,dim),dtype) for name,dim,dtype in
        [('geometry',533,'float32'),('latent',64,'float32'),('coarse_scalar_latent',416,'float32'),
         ('coarse_output_context',47,'float32'),('teacher',3,'float64'),('margins',5,'float64'),('sequence',7,'uint8'),('meta',11,'int32')]}
    pos=0;start=time.perf_counter()
    for row in rows:
        with np.load(e.ROOT/'representations'/f'identity_{row["index"]:04d}.npz',allow_pickle=False) as z:
            seq=z['sequence_tokens'];length=len(seq)
            seqpad=np.pad(seq,(3,3),constant_values=21)
            for ci,condition in enumerate((50,250,450)):
                prefix=str(condition);features=z[f'{prefix}_features'];lat=z[f'{prefix}_latent'];eligible=z[f'{prefix}_eligible']
                teacher=z[f'{prefix}_teacher_local'];margins=z[f'{prefix}_margins'];coarse=z[f'{prefix}_coarse_scalar'];context=z[f'{prefix}_coarse_output_context']
                mapping=z[f'{prefix}_logical_state_index']
                for step,state in enumerate(mapping):
                    choices=np.flatnonzero(eligible[state])
                    chosen=sorted(choices,key=lambda residue:e.rank(f'{row["sample_id"]}|{condition}|{step}|{residue}'))[:8]
                    assert len(chosen)==8
                    padded=np.pad(features[state],((6,6),(0,0)))
                    for slot,residue in enumerate(chosen):
                        arrays['geometry'][pos]=padded[residue:residue+13].reshape(-1)
                        arrays['latent'][pos]=lat[state,residue];arrays['coarse_scalar_latent'][pos]=coarse[residue]
                        arrays['coarse_output_context'][pos]=context[state,residue]
                        arrays['teacher'][pos]=teacher[state,residue];arrays['margins'][pos]=margins[state,residue]
                        arrays['sequence'][pos]=seqpad[residue:residue+7]
                        arrays['meta'][pos]=[row['index'],ci,step,residue,length,e.old.STRATA.index(row['stratum']),
                            {'train':0,'validation':1,'test':2}[splits[row['sample_id']]],slot,
                            group_map[row['split_group_id']],cluster_map[row['cluster_id']],row['index']*48+ci*16+step]
                        pos+=1
        if (row['index']+1)%100==0:print(json.dumps(dict(panel_identities=row['index']+1,points=pos,wall_seconds=time.perf_counter()-start)),flush=True)
    assert pos==n
    for array in arrays.values():array.flush()
    meta=arrays['meta'];bank=np.flatnonzero((meta[:,6]==0)&(meta[:,7]<4))
    np.save(e.ROOT/'panel/reference_rows.npy',bank.astype(np.int32),allow_pickle=False)
    transforms={}
    for name in ('geometry','latent','coarse_scalar_latent','coarse_output_context'):
        # Equal number of reference-panel points per identity/state; no action labels enter these moments.
        values=np.asarray(arrays[name][bank],dtype=np.float64)
        mean=values.mean(0);scale=np.maximum(values.std(0),1e-3)
        transforms[name]=dict(mean=mean.tolist(),scale=scale.tolist())
    e.write(e.ROOT/'panel/transforms.json',transforms)
    e.write(marker,dict(points=n,reference_points=len(bank),identities=2000,logical_states=96000,
        reference_split='train',reference_residues_per_state=4,query_residues_per_state=8,protocol_sha256=e.old.sha(e.ROOT/'protocol.json')))


def standardized(name,indices,transforms):
    values=np.asarray(open_array(name)[indices],dtype=np.float32)
    transform=transforms[name]
    return np.ascontiguousarray((values-np.asarray(transform['mean'],np.float32))/np.asarray(transform['scale'],np.float32))


def valid_candidates(q,ids,dist,meta,bank):
    indices=bank[ids[ids>=0]];ds=dist[ids>=0]
    valid=(meta[indices,0]!=meta[q,0])&(meta[indices,8]!=meta[q,8])&(meta[indices,9]!=meta[q,9])
    indices=indices[valid];ds=ds[valid]
    order=np.lexsort((indices,ds));return indices[order],ds[order]


def unique_indices(indices,meta,k=8):
    found=[];seen=set()
    for j in indices:
        identity=int(meta[j,0])
        if identity not in seen:
            seen.add(identity);found.append(int(j))
            if len(found)==k:break
    return found


def heldout_search(index,query,query_rows,meta,bank):
    """Retrieve beyond excluded groups, then keep the closest 256 allowed candidates."""
    count=e.PROTOCOL['candidate_k'];retrieval=min(512,len(bank))
    d,i=index.search(query,retrieval)
    result_i=np.full((len(query),count),-1,dtype=np.int32);result_d=np.full((len(query),count),np.inf,dtype=np.float32)
    for n,q in enumerate(query_rows):
        current_i=i[n];current_d=d[n]
        while True:
            global_i=bank[current_i[current_i>=0]];valid=(meta[global_i,0]!=meta[q,0])&(meta[global_i,8]!=meta[q,8])&(meta[global_i,9]!=meta[q,9])
            allowed_i=current_i[current_i>=0][valid];allowed_d=current_d[current_i>=0][valid]
            order=np.lexsort((bank[allowed_i],allowed_d));allowed_i=allowed_i[order];allowed_d=allowed_d[order]
            if len(allowed_i)>=count and len(unique_indices(bank[allowed_i[:count]],meta))==8:break
            retrieval_next=min(max(len(current_i)*2,2048),len(bank))
            assert retrieval_next>len(current_i), 'Insufficient held-out identities in complete reference bank'
            nd,ni=index.search(query[n:n+1],retrieval_next);current_i=ni[0];current_d=nd[0]
        result_i[n]=allowed_i[:count];result_d[n]=allowed_d[:count]
    return result_d,result_i


def search(name):
    marker=e.ROOT/'neighbors'/f'{name}_SEARCH_COMPLETED.json'
    if marker.exists():return
    transforms=json.loads((e.ROOT/'panel/transforms.json').read_text());bank=open_array('reference_rows');meta=open_array('meta')
    index_path=e.ROOT/'neighbors'/f'{name}.faiss'
    faiss.omp_set_num_threads(1)
    if index_path.exists():index=faiss.read_index(str(index_path))
    else:
        data=standardized(name,bank,transforms)
        index=faiss.IndexHNSWFlat(data.shape[1],e.PROTOCOL['hnsw_M']);index.hnsw.efConstruction=e.PROTOCOL['efConstruction']
        if hasattr(index.hnsw,'rng'):index.hnsw.rng=faiss.RandomGenerator(e.PROTOCOL['seed'])
        start=time.perf_counter()
        for lo in range(0,len(bank),10000):
            index.add(data[lo:lo+10000])
            if lo%50000==0:print(json.dumps(dict(index=name,indexed=min(lo+10000,len(bank)),total=len(bank),wall_seconds=time.perf_counter()-start)),flush=True)
        faiss.write_index(index,str(index_path))
    # Exact label-blind recall audit, including exclusion and unique-identity aggregation.
    audit_path=e.ROOT/'neighbors'/f'{name}_audit.json'
    if audit_path.exists():audit=json.loads(audit_path.read_text());index.hnsw.efSearch=audit['efSearch']
    else:
        sample=np.array(sorted(range(len(meta)),key=lambda i:e.rank(f'audit|{i}'))[:512])
        query=standardized(name,sample,transforms)
        exact=faiss.IndexFlatL2(index.d);exact.add(standardized(name,bank,transforms));faiss.omp_set_num_threads(8)
        ed,ei=heldout_search(exact,query,sample,meta,bank)
        recalls=[]
        for ef in (512,1024,2048):
            index.hnsw.efSearch=ef;ad,ai=heldout_search(index,query,sample,meta,bank);recall=[]
            for n,q in enumerate(sample):
                ex,_=valid_candidates(q,ei[n],ed[n],meta,bank);ap,_=valid_candidates(q,ai[n],ad[n],meta,bank)
                truth=set(unique_indices(ex,meta));approx=set(unique_indices(ap,meta))
                recall.append(len(truth&approx)/len(truth) if truth else 1.)
            recalls.append(dict(efSearch=ef,recall_at_8=float(np.mean(recall))))
            if np.mean(recall)>=.90:break
        assert recalls[-1]['recall_at_8']>=.90, 'ANN quality gate failed; do not analyze unreliable neighbor directions'
        audit=dict(name=name,queries=len(sample),efSearch=ef,ladder=recalls,teacher_labels_used=False,
            exact_reference_points=len(bank),reference_identity_exclusion=True)
        e.write(audit_path,audit)
    faiss.omp_set_num_threads(8)
    n=len(meta);k=e.PROTOCOL['candidate_k']
    ids=np.lib.format.open_memmap(e.ROOT/'neighbors'/f'{name}_ids.npy',mode='w+',dtype='int32',shape=(n,k))
    distances=np.lib.format.open_memmap(e.ROOT/'neighbors'/f'{name}_distances.npy',mode='w+',dtype='float32',shape=(n,k))
    start=time.perf_counter()
    for lo in range(0,n,1024):
        query=standardized(name,slice(lo,lo+1024),transforms);d,i=heldout_search(index,query,np.arange(lo,lo+len(query)),meta,bank)
        ids[lo:lo+len(i)]=i;distances[lo:lo+len(d)]=d
        if lo%51200==0:print(json.dumps(dict(search=name,queries=min(lo+1024,n),total=n,wall_seconds=time.perf_counter()-start)),flush=True)
    ids.flush();distances.flush()
    e.write(marker,dict(name=name,queries=n,audit=audit,wall_seconds=time.perf_counter()-start,protocol_sha256=e.old.sha(e.ROOT/'protocol.json')))


def init_analysis():
    global ARRAYS,TRANSFORMS,BANK,NEIGHBORS
    e.storage()
    ARRAYS={name:open_array(name) for name in ('geometry','latent','coarse_scalar_latent','coarse_output_context','teacher','margins','sequence','meta')}
    BANK=open_array('reference_rows');TRANSFORMS=json.loads((e.ROOT/'panel/transforms.json').read_text())
    NEIGHBORS={name:(np.load(e.ROOT/'neighbors'/f'{name}_ids.npy',mmap_mode='r'),np.load(e.ROOT/'neighbors'/f'{name}_distances.npy',mmap_mode='r')) for name in ('geometry','latent')}


def context_distance(name,q,indices):
    meta=ARRAYS['meta'];seq=ARRAYS['sequence']
    if name=='condition':return (meta[indices,1]!=meta[q,1]).astype(float)
    if name=='step':return np.abs(meta[indices,2]-meta[q,2])/15.
    if name=='residue_identity':return (seq[indices,3]!=seq[q,3]).astype(float)
    if name in ('sequence_local','sequence_condition'):
        value=(seq[indices]!=seq[q]).mean(1)
        return value+2*(meta[indices,1]!=meta[q,1]) if name=='sequence_condition' else value
    if name in ('initial_coarse_correction','rollout_history','denoiser_geometry'):
        sl={'initial_coarse_correction':slice(0,3),'rollout_history':slice(3,6),'denoiser_geometry':slice(6,47)}[name]
        delta=(ARRAYS['coarse_output_context'][indices,sl]-ARRAYS['coarse_output_context'][q,sl])/np.asarray(TRANSFORMS['coarse_output_context']['scale'],np.float32)[sl]
        return np.einsum('ij,ij->i',delta,delta)/delta.shape[1]
    delta=(ARRAYS[name][indices]-ARRAYS[name][q])/np.array(TRANSFORMS[name]['scale'],dtype=np.float32)
    return np.einsum('ij,ij->i',delta,delta)/delta.shape[1]


def choose_context(q,indices,ds,name):
    context=context_distance(name,q,indices);order=np.lexsort((indices,ds,context))
    chosen=unique_indices(indices[order],ARRAYS['meta'])
    position={int(j):i for i,j in enumerate(indices)}
    # Joint label-blind assignment prevents a greedy last-row dead end and gives
    # distinct control identities with minimum total geometric-distance mismatch.
    groups,inverse=np.unique(ARRAYS['meta'][indices,0],return_inverse=True)
    chosen_distance=np.array([ds[position[j]] for j in chosen])
    delta=np.abs(ds[None,:]-chosen_distance[:,None])/np.maximum(chosen_distance[:,None],1e-12)
    cost=np.full((8,len(groups)),np.inf)
    np.minimum.at(cost,(np.arange(8)[:,None],inverse[None,:]),delta)
    cost[np.arange(8),np.searchsorted(groups,ARRAYS['meta'][chosen,0])]=np.inf
    rr,cc=linear_sum_assignment(cost)
    assert np.array_equal(rr,np.arange(8))
    controls=[int(indices[np.argmin(np.where(inverse==group,delta[i],np.inf))]) for i,group in enumerate(cc)]
    return chosen,controls,position


def evaluate_pairs(q,chosen,controls,indices,ds,position):
    out=np.full((8,12),np.nan,dtype=np.float64)
    teacher=ARRAYS['teacher'];margins=ARRAYS['margins'];qn=float(np.linalg.norm(teacher[q]));qu=teacher[q]/max(qn,1e-300)
    for slot,(j,c) in enumerate(zip(chosen,controls)):
        if c<0:continue
        jn=float(np.linalg.norm(teacher[j]));cn=float(np.linalg.norm(teacher[c]))
        dj=float(ds[position[j]]);dc=float(ds[position[c]]);difference=abs(dj-dc)/max(dj,1e-12)
        out[slot,2:6]=[dj,dc,difference,jn]
        out[slot,6]=cn;out[slot,7]=qn
        if qn>1e-8 and jn>1e-8:out[slot,0]=np.clip(np.dot(qu,teacher[j]/jn),-1,1)
        if qn>1e-8 and cn>1e-8:out[slot,1]=np.clip(np.dot(qu,teacher[c]/cn),-1,1)
        # Margin variation is scored offline and never influences matching.
        valid=(margins[q]<1e20)&(margins[j]<1e20)
        out[slot,8]=float(np.abs(margins[q,:2]-margins[j,:2]).mean())
        out[slot,9]=float(np.abs(margins[q,2:4][valid[2:4]]-margins[j,2:4][valid[2:4]]).mean()) if valid[2:4].any() else np.nan
        out[slot,10]=abs(float(margins[q,4]-margins[j,4]));out[slot,11]=float(ARRAYS['meta'][j,0])
    good=(out[:,4]<=.10)&np.isfinite(out[:,0])&np.isfinite(out[:,1])
    vector=np.full(4,np.nan)
    if good.sum()>=2:
        a=np.array(chosen)[good];b=np.array(controls)[good]
        ua=teacher[a]/np.linalg.norm(teacher[a],axis=1)[:,None];ub=teacher[b]/np.linalg.norm(teacher[b],axis=1)[:,None]
        am=ua.mean(0);bm=ub.mean(0)
        vector=[np.square(qu-am).sum(),np.square(qu-bm).sum(),1-np.square(am).sum(),1-np.square(bm).sum()]
    return out,np.asarray(vector)


def analyze_identity(row):
    destination=e.ROOT/'neighbors'/f'identity_{row["index"]:04d}.npz';marker=destination.with_suffix('.json')
    if marker.exists():assert json.loads(marker.read_text())['archive_sha256']==e.old.sha(destination);return str(marker)
    meta=ARRAYS['meta'];query=np.flatnonzero(meta[:,0]==row['index']);assert len(query)==384
    arrays={'query_rows':query};start=time.perf_counter();minimum_candidates=100000
    for representation,(ids,distances) in NEIGHBORS.items():
        metrics={name:[] for name in ['baseline']+CONTEXTS};vectors={name:[] for name in ['baseline']+CONTEXTS}
        reference_ids={name:[] for name in ['baseline']+CONTEXTS}
        for q in query:
            indices,ds=valid_candidates(q,ids[q],distances[q],meta,BANK)
            minimum_candidates=min(minimum_candidates,len(set(meta[indices,0])))
            assert len(unique_indices(indices,meta))==8, 'Insufficient distinct held-out neighbors'
            baseline=unique_indices(indices,meta);position={int(j):i for i,j in enumerate(indices)}
            a,v=evaluate_pairs(q,baseline,baseline,indices,ds,position)
            metrics['baseline'].append(a);vectors['baseline'].append(v);reference_ids['baseline'].append(np.array(baseline))
            for context in CONTEXTS:
                chosen,controls,position=choose_context(q,indices,ds,context)
                a,v=evaluate_pairs(q,chosen,controls,indices,ds,position)
                metrics[context].append(a);vectors[context].append(v);reference_ids[context].append(np.stack([chosen,controls]))
        for name,values in metrics.items():arrays[f'{representation}_{name}_pairs']=np.stack(values)
        for name,values in vectors.items():arrays[f'{representation}_{name}_vector']=np.stack(values)
        for name,values in reference_ids.items():arrays[f'{representation}_{name}_reference_rows']=np.stack(values)
    e.old.frozen.old.atomic_npz(destination,**arrays)
    e.write(marker,dict(identity=row['sample_id'],index=row['index'],length_bin=row['stratum'],queries=len(query),
        minimum_distinct_reference_identities=minimum_candidates,archive_sha256=e.old.sha(destination),
        protocol_sha256=e.old.sha(e.ROOT/'protocol.json'),wall_seconds=time.perf_counter()-start))
    return str(marker)


def run_analysis():
    start=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn'),initializer=init_analysis) as pool:
        for i,p in enumerate(pool.map(analyze_identity,e.rows()),1):
            if i%20==0:print(json.dumps(dict(analyzed=i,total=2000,wall_seconds=time.perf_counter()-start,last=p)),flush=True)
    e.write(e.ROOT/'ANALYSIS_COMPLETED.json',dict(identities=2000,protocol_sha256=e.old.sha(e.ROOT/'protocol.json'),wall_seconds=time.perf_counter()-start))


def main():
    panel()
    for name in ('geometry','latent'):search(name)
    run_analysis()


if __name__=='__main__':main()
