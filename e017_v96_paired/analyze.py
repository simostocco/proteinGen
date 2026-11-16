"""Predeclared paired-model group-bootstrap analysis; no model inference."""
from paired import *
_paired_evaluator=runpy.run_path(str(PAIR/'source/evaluate.py'))
METHODS,specs,validate=(_paired_evaluator[name] for name in ['METHODS','specs','validate'])

def ci(values,alpha=.05):
    x=np.asarray(values,float);assert x.ndim==1 and len(x)>0;rng=np.random.default_rng(19329);means=[]
    for i in range(0,10000,500):means.extend(x[rng.integers(len(x),size=(500,len(x)))].mean(1))
    return dict(effect=float(x.mean()),CI=np.quantile(means,[alpha/2,1-alpha/2]).tolist(),confidence=1-alpha,clusters=len(x),bootstrap_seed=19329,bootstrap_draws=10000)
def endpoint(group,k):
    if k=='K1':return float(group['records'][0]['metrics']['success'])
    if k=='K8':return float(any(r['metrics']['success'] for r in group['records']))
    return float(np.mean([r['metrics'][k] for r in group['records']]))
def vector(groups,k):
    out=collections.defaultdict(list)
    for g in groups:out[g['task']['goal']['protected_group']].append(endpoint(g,k))
    return {i:float(np.mean(v)) for i,v in out.items()}
def difference(a,b):
    assert a.keys()==b.keys();return {i:b[i]-a[i] for i in sorted(a)}
def effect(a,b,alpha=.05):return ci(list(difference(a,b).values()),alpha)
def primary_supported(p):
    return p['Bfinal_minus_Afinal']['effect']>=.01 and p['Bfinal_minus_Afinal']['CI'][0]>0 and p['difference_in_changes']['effect']>=.01 and p['difference_in_changes']['CI'][0]>0 and p['B_change']['CI'][0]>0 and p['Bfinal_response']['effect']>=.01 and p['Bfinal_response']['CI'][0]>0
def specificity(panel,groups):
    lookup={g['task']['goal']['goal_id']:g for g in groups};out={}
    for tag,region in [('burial','buried'),('exposure','exposed')]:
        vals={};valid={};valid_only={}
        for d in panel['diagnostics']:
            task=d['base_task'];alt=d['alternatives'][tag];group=task['goal']['protected_group'];base=lookup[task['goal']['goal_id']]['records'];other=lookup[alt['goal']['goal_id']]['records']
            score=goal_score(base,other,task['goal'],alt['goal'],region);vals[group]=score['S'];valid[group]=score['valid_pairs']
            valid_only[group]=score['S']*8/score['valid_pairs'] if score['valid_pairs'] else None
        out[tag]=dict(values=vals,mean=ci(list(vals.values())),paired_valid=sum(valid.values()),paired_candidates=8*len(vals),groups_with_valid_pairs=sum(x>0 for x in valid.values()),valid_pairs_by_group=valid,valid_only_descriptive=valid_only)
    return out
def load():
    panel=read(PAIR/'VALIDATION_PANEL_MANIFEST.json');done=read(PAIR/'evaluation/EVALUATION_COMPLETED.json');assert done['attempts']==89088
    E=runpy.run_path(str(R3/'source/scripts/e017_v94r/evaluation.py'));summaries={};panels={};checksums={};all_counts=collections.Counter()
    for method,policy,model in METHODS:
        sums=collections.Counter();div=[];errors=[];penalty=[];KL=[];slim=[];counts=collections.Counter()
        for index,task,seeds,scope in specs(panel,method):
            cp=PAIR/'evaluation/commits'/f'{method}_{index:05d}.json';seal=read(cp);path=PAIR/'evaluation'/seal['path'];assert sha(path)==seal['SHA256']
            obj=read(path);validate(obj,task,seeds,method,done['models'][method]['actor_fingerprint'],policy,index);checksums[str(cp.relative_to(PAIR))]=sha(cp)
            counts['attempts']+=8;counts['complete']+=sum(r['metrics']['complete'] for r in obj['records']);counts['aborts']+=sum(r['termination'] is not None for r in obj['records'])
            if scope=='primary':
                s=E['summarize']([obj]);sums['goals']+=1;sums['attempts']+=8;sums['seconds']+=obj['seconds']
                for k in ['K1','K8']:sums[k]+=s[k]
                for k in ['hard_valid','backbone_valid','complete','exposure_success','burial_success','full_success','reward']:sums[k]+=8*s[k]
                sums['aborts']+=s['aborts'];sums['overpacking']+=s['global_contact_violations'];sums['proposals']+=s['proposals']
                if sum(r['metrics']['complete'] for r in obj['records'])>=2:div.append(s['structural_pair_RMSD'])
                for r in obj['records']:
                    m=r['metrics'];g=task['goal'];KL.extend(r['geometry_KL']);sums['packing_valid']+=m['packing_valid']
                    sums['clash_failures']+=m['complete'] and 'nonlocal_clashes' in m['geometry']['failures']
                    sums['actual_rejections']+=sum(not p[6] for p in r['proposals']) if policy=='C' else 0
                    sums['simultaneous_exposure_burial']+=m.get('exposed_mean',-1)>=g['exposure_min'] and m.get('buried_mean',2)<=g['burial_max']
                    err=max(0,g['exposure_min']-m['exposed_mean'])+max(0,m['buried_mean']-g['burial_max'])+max(0,g['contrast_min']-m['exposed_mean']+m['buried_mean']) if 'exposed_mean' in m else None
                    if err is not None:errors.append(err)
                    penalty.append(err if err is not None else 3.)
            slim.append(dict(task=task,records=[dict(metrics=r['metrics'],seed=r['seed'],termination=r['termination']) for r in obj['records']]))
            if index%200==0:print('ANALYSIS',method,index,flush=True)
        all_counts.update(counts);n=sums['attempts'];row=dict(sums)
        for k in ['hard_valid','backbone_valid','complete','exposure_success','burial_success','full_success']:row[k]=round(row[k]);row[k+'_rate']=row[k]/n
        row['K1_rate']=row['K1']/sums['goals'];row['K8_rate']=row['K8']/sums['goals'];row['reward']/=n
        row.update(structural_pair_RMSD=float(np.mean(div)),target_error_complete=float(np.mean(errors)),target_error_abort_penalty3=float(np.mean(penalty)),geometry_KL_per_active=float(np.mean(KL)),all_scope_counts=dict(counts))
        summaries[method]=row;panels[method]=slim
    assert all_counts['attempts']==89088 and all_counts['complete']+all_counts['aborts']==89088
    return panel,done,summaries,panels,checksums,dict(all_counts)
def scientific(panel,summaries,panels,outcomes):
    n=len(panel['tasks']);prim={k:v[:n] for k,v in panels.items()};specific={k:specificity(panel,v) for k,v in panels.items() if k!='J1'}
    # Sole primary scientific family: raw burial, an intersection requirement.
    v={m:specific[m+'_raw']['burial']['values'] for m in ['A0','B0','Af','Bf']}
    deltaA=difference(v['A0'],v['Af']);deltaB=difference(v['B0'],v['Bf']);dd=difference(deltaA,deltaB)
    primary=dict(Bfinal_minus_Afinal=effect(v['Af'],v['Bf']),A_change=ci(list(deltaA.values())),B_change=ci(list(deltaB.values())),difference_in_changes=ci(list(dd.values())),Bfinal_response=ci(list(v['Bf'].values())))
    raw_support=primary_supported(primary)
    # 8 endpoint families x4 planned contrasts, simultaneous Bonferroni intervals.
    matched={};legacy={};identity=[]
    for mode in ['raw','filtered']:
        vectors={m:{k:vector(prim[m+'_'+mode],k) for k in ['K1','K8']} for m in ['A0','B0','Af','Bf']}
        for k in ['K1','K8','burial','exposure']:
            x={m:(vectors[m][k] if k.startswith('K') else specific[m+'_'+mode][k]['values']) for m in ['A0','B0','Af','Bf']}
            da=difference(x['A0'],x['Af']);db=difference(x['B0'],x['Bf'])
            matched[mode+'_'+k]=dict(A_change=ci(list(da.values()),.05/32),B_change=ci(list(db.values()),.05/32),Bfinal_minus_Afinal=effect(x['Af'],x['Bf'],.05/32),difference_in_changes=ci(list(difference(da,db).values()),.05/32))
        legacy[mode]={}
        for model in ['Af','Bf']:
            legacy[mode][model]={k:effect(vector(prim['A0_'+mode],k),vector(prim[model+'_'+mode],k),.05/6 if k=='K8' else .05) for k in ['K1','K8','success','hard_valid','backbone_valid']}
            legacy[mode][model]['discordance_K1']=dict(collections.Counter(f"{int(a['records'][0]['metrics']['success'])}->{int(b['records'][0]['metrics']['success'])}" for a,b in zip(prim['A0_'+mode],prim[model+'_'+mode])))
        for tuples in zip(*(prim[m+'_'+mode] for m in ['A0','B0','Af','Bf'])):
            identity.append(dict(mode=mode,identity=tuples[0]['task']['source_reference_identity'],group=tuples[0]['task']['goal']['protected_group'],goal=tuples[0]['task']['goal']['goal_id'],values={m:{k:endpoint(g,k) for k in ['K1','K8','success','hard_valid','backbone_valid']} for m,g in zip(['A0','B0','Af','Bf'],tuples)}))
    original_specificity={mode:{tag:dict(change=effect(specific['B0_'+mode][tag]['values'],specific['Bf_'+mode][tag]['values'],.05/6),initial_response=ci(list(specific['B0_'+mode][tag]['values'].values()),.05/6),trained_response=ci(list(specific['Bf_'+mode][tag]['values'].values()),.05/6)) for tag in ['burial','exposure']} for mode in ['raw','filtered']}
    equal=all(outcomes[a]['updates']==5000 and outcomes[a]['all_generated_training_attempts']==80000 and outcomes[a]['status']=='COMPLETED' for a in ['A','B'])
    guards={};promotion={}
    for model in ['Af','Bf']:
        a=summaries['A0_filtered'];b=summaries[model+'_filtered'];s=legacy['filtered'][model]
        guard=b['hard_valid_rate']>=a['hard_valid_rate']-.05 and b['backbone_valid_rate']>=a['backbone_valid_rate']-.05 and b['backbone_valid_rate']>=.95 and b['geometry_KL_per_active']<=1 and b['structural_pair_RMSD']>=.8*a['structural_pair_RMSD'];guards[model]=guard
        promotion[model]=s['K1']['effect']>=.05 and s['K1']['CI'][0]>0 and s['K8']['effect']>=0 and guard
    exposure=specific['Bf_raw']['exposure']['mean']['effect']>=.01 and specific['Bf_raw']['exposure']['mean']['CI'][0]>0 and specific['Bf_raw']['exposure']['mean']['effect']>=specific['A0_raw']['exposure']['mean']['effect']
    decisions=dict(BURIAL_SPECIFICITY_IMPROVEMENT='SUPPORTED' if raw_support and equal else 'NOT_ESTABLISHED' if equal else 'PARTIAL_BUDGET_ONLY',TOKEN_VS_ADDITIONAL_TRAINING='SUPPORTED_FOR_THIS_SEED' if raw_support and equal else 'NOT_ESTABLISHED',EXPOSURE_PRESERVATION='POINT_AND_RESPONSE_GATES_PASS' if exposure else 'NOT_ESTABLISHED',K1_PRACTICAL_PROMOTION=promotion,K8_IMPROVEMENT={m:legacy['filtered'][m]['K8']['CI'][0]>0 for m in ['Af','Bf']},RAW_ACTOR_LEARNING={m:legacy['raw'][m]['K8']['CI'][0]>0 for m in ['Af','Bf']},VALIDITY_TRADEOFF=guards,TRAINING_STABILITY={a:outcomes[a]['status'] for a in ['A','B']},GOAL_CONDITIONING='BURIAL_LEARNING_SUPPORTED' if raw_support and equal else 'INCOMPLETE_OR_NOT_ESTABLISHED',TEST_SEAL='UNKNOWN_HISTORICAL_ACCESS',equal_full_budget=equal,one_seed_per_arm=True)
    return dict(primary=primary,matched=matched,legacy=legacy,legacy_specificity=original_specificity,specificity=specific,decisions=decisions),identity
def reports(results,outcomes,allcounts):
    summaries=results['summaries'];analysis=results['analysis'];dec=analysis['decisions']
    table=['| Policy | K1 /1024 | K8 /1024 | Success /8192 | Hard /8192 | Backbone /8192 | Aborts | RMSD | Proposal KL |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in summaries.items():table.append(f"| {name} | {s['K1']:.0f} | {s['K8']:.0f} | {s['full_success']} | {s['hard_valid']} | {s['backbone_valid']} | {s['aborts']} | {s['structural_pair_RMSD']:.6f} | {s['geometry_KL_per_active']:.8f} |")
    main='\n'.join(table)
    def doc(name,text):(PAIR/name).write_text(text+'\n')
    for a,root in ARMS.items():
        text=f"# Arm {a}: training and recovery\n\n"+json.dumps(outcomes[a],indent=2)+"\n\nAdmission and exact checkpoint/journal/RNG/partial-write qualification are preserved in qualification/ADMISSION.json. No stopped arm was restarted with changed hyperparameters."
        (root/'FINAL_TRAINING_AND_RECOVERY_REPORT.md').write_text(text+'\n');doc(f'ARM_{a}_TRAINING_AND_RECOVERY_REPORT.md',text)
    doc('K1_K8_VALIDATION_RESULTS.md','# K1/K8 development VALIDATION\n\n'+main+'\n\n'+json.dumps(analysis['legacy'],indent=2)+'\n\nK1 is candidate0, K8 any success in the same eight. All aborts remain failures. Original >=5pp K1 and lower95>0 contract retained against A0; matched between-arm effects are separate. Diagnostics are excluded from these denominators.')
    doc('BURIAL_SPECIFICITY_COMPARISON.md','# Burial specificity\n\n'+json.dumps(analysis['primary'],indent=2)+'\n\n'+json.dumps({k:v for k,v in analysis['matched'].items() if k.endswith('burial')},indent=2)+'\n\nThe primary raw-burial claim requires B-A>=.01 with positive95% lower bound, difference-in-changes>=.01 with positive lower bound, positive B-own change, and meaningful positive B response>=.01. Invalid matched pairs contribute0. Valid-only estimates are descriptive. Native-compatible diagnostic goals can be looser than primary goals.')
    doc('EXPOSURE_PRESERVATION_ANALYSIS.md','# Exposure preservation\n\n'+json.dumps({k:v for k,v in analysis['matched'].items() if k.endswith('exposure')},indent=2)+'\n\nExposure diagnostic triples hold sequence, motif and thresholds fixed within the triple. No isolated-exposure claim is made from the old V9.5 coupled motif/exposure diagnostic. Preservation point gates and intervals are reported separately, not as an unregistered noninferiority test.')
    doc('RAW_VS_FILTERED_ACTOR_ASSESSMENT.md','# Raw versus filtered\n\n'+main+'\n\n'+json.dumps(analysis['decisions'],indent=2)+'\n\nDeterministic filtering is unchanged and is not learned behavior. Read raw goal responsiveness and raw success separately from filtered yield. Gaussian reference KL is not the KL of the effective rejection-conditioned controller.')
    doc('VALIDITY_AND_DIVERSITY_TRADEOFF.md','# Validity and diversity\n\n'+json.dumps(summaries,indent=2)+'\n\nAll attempt denominators include aborts. Completed geometric validity does not establish experimental folding stability or function. Passing a tolerance gate does not mean no validity loss occurred.')
    costs=dict(training=outcomes,evaluation=read(PAIR/'evaluation/EVALUATION_COMPLETED.json'),all_evaluation_candidate_counts=allcounts)
    atomic(PAIR/'TRAINING_RESOURCE_ACCOUNTING.json',costs,True)
    doc('TRAINING_RESOURCE_ACCOUNTING.md','# Resource accounting\n\n'+json.dumps(costs,indent=2)+'\n\nEngineering admission, fixed TRAIN monitoring, fitting attempts and final VALIDATION generation are separate scopes. Pending unsafe/uncommitted training rollouts remain in all-generated-attempt accounting.')
    lines=['# Final controlled V9.6 scientific recommendation','',json.dumps(dec,indent=2),'',main,'', 'Training exposure: '+json.dumps({a:dict(updates=x['updates'],attempts=x['all_generated_training_attempts'],status=x['status']) for a,x in outcomes.items()}),'', 'Equal-budget causal interpretation is permitted only if both arms completed5000 updates/80000 fitting attempts. Otherwise this is a partial/asymmetric study and cannot establish a controlled full-budget token advantage. One training seed per arm does not establish algorithmic robustness; confidence intervals resample validation identities conditional on these fitted models.','', 'Continue low-level conditioning/credit research unless meaningful K1 promotion, raw burial specificity, exposure preservation and validity/diversity safeguards all support escalation. Do not automatically launch AA training, hierarchical RL, another experiment, or TEST. Historical V9.5 and V9.6 preparation artifacts remain unchanged.']
    doc('FINAL_SCIENTIFIC_RECOMMENDATION.md','\n'.join(lines))
    doc('CHECKPOINT_AND_SOURCE_PROVENANCE.md','# Provenance\n\nFrozen source/input hashes are in FROZEN_SOURCE_AND_INPUTS.json. Arm-specific checkpoint hashes and model fingerprints are in TRAINING_OUTCOME.json and evaluation/EVALUATION_COMPLETED.json. TEST was not evaluated; complete historical access cannot be certified.')
def main():
    verify();panel,done,summaries,panels,checksums,allcounts=load();outcomes={a:read(root/'TRAINING_OUTCOME.json') for a,root in ARMS.items()}
    analysis,identity=scientific(panel,summaries,panels,outcomes)
    results=dict(UTC=now(),summaries=summaries,analysis=analysis,training_outcomes=outcomes,all_evaluation_counts=allcounts,input_commit_SHA256=checksums,validation_manifest_SHA256=sha(PAIR/'VALIDATION_PANEL_MANIFEST.json'))
    atomic(PAIR/'PAIRED_SCIENTIFIC_RESULTS.json',results,True);atomic(PAIR/'IDENTITY_PAIRED_RESULTS.json',dict(records=identity),True)
    reports(results,outcomes,allcounts)
    for arm,root in ARMS.items():
        curves=[read(p) for p in sorted((root/'production/metrics').glob('u*.json'))]
        atomic(PAIR/f'ARM_{arm}_LEARNING_CURVES.json',dict(arm=arm,records=curves),True)
    (PAIR/'MATCHED_DEVELOPMENT_LEARNING_CURVES.md').write_text('# Matched development learning curves\n\nEach arm uses the same frozen TRAIN identities/goals/seeds at the registered update checkpoints. JSON per-update records are in ARM_A/B_LEARNING_CURVES.json. Fixed-development reports remain under each arm production/diagnostics/. Stratify changing TRAIN task difficulty before interpreting reward trends.\n')
    # Workers have exited before controller launches CPU publication. Active
    # controller/monitor logs remain excluded to avoid stale open-log hashes.
    files={str(p):dict(SHA256=sha(p),bytes=p.stat().st_size) for root in [PAIR,*ARMS.values()] for p in sorted(root.rglob('*')) if p.is_file() and 'runtime' not in p.relative_to(root).parts and p.name!='PROVENANCE_SHA256.json'}
    atomic(PAIR/'PROVENANCE_SHA256.json',dict(files=files,self_excluded=True,runtime_excluded=True,TEST=False),True)
    assert all(sha(path)==v['SHA256'] for path,v in files.items())
    atomic(PAIR/'runtime/PUBLICATION_VERIFICATION.json',dict(UTC=now(),PASS=True,files_verified=len(files),attempts=89088,decisions=analysis['decisions']),True)
    print('PAIRED_PUBLICATION_VERIFIED',len(files),json.dumps(analysis['decisions']),flush=True)
if __name__=='__main__':main()
