from pathlib import Path
import collections,json,math
r=Path('/mnt/d/Simone/proteinGen_data/e017_v93_regional_controllability_20261024_v1');read=lambda p:json.loads(p.read_text());x=read(r/'FIXED_SEQUENCE_NUMERICAL_RESULTS.json');delayed=read(r/'DELAYED_EFFECT_NUMERICAL_RESULTS.json');taxonomy=[]
for p in sorted((r/'results').glob('PROSPECTIVE_C_[0-9][0-9][0-9].json')):
 g=read(p);rows=g['records'];n=sum(z['metrics']['success'] for z in rows);hard=sum(z['metrics']['hard_valid'] for z in rows);taxonomy.append(dict(goal_id=g['task']['goal']['goal_id'],identity=g['task']['goal']['witness_identity'],difficulty=g['task']['goal']['difficulty'],full=n,total=8,hard=hard,status='NO_OBSERVED_C8_SUPPORT' if n==0 else 'RARE_SUPPORTED_PROPOSAL' if n==1 else 'REPEATED_SUPPORTED_PROPOSALS',native_computational_witness=True))
counts=dict(collections.Counter(v['status'] for v in taxonomy));(r/'GOAL_SUPPORT_TAXONOMY.json').write_text(json.dumps(dict(records=taxonomy,counts=counts,zero_samples_not_impossibility=True,selection='fixed first8 candidates; no failed-sample replacement'),indent=2)+'\n')
summary='''

## Reviewed scientific interpretation

Frozen filtered sampling achieves21/256 full regional successes and11/32 goals atK8 on the new panel; K1 remains0/32. The same-model filtered baseline—not unfiltered B—is the primary reference for any future trained-policy comparison. Goal-clustered C-minus-B K8 gain is21.875pp (95% interval9.375–34.375pp), and candidate full-success gain is5.859pp (2.344–9.766pp). These measure inference filtering, not RL learning. Current K1 zero-event bootstrap degeneracy is explicitly guarded by the conservative identity-event interval; it does not prove zero proposal success probability.

Goal-support decomposition: '''+str(counts)+'''. All these tasks have checked computational witnesses; failure in a pool of8 does not establish impossibility. Exact terminal selection solves a useful subset, so RL is not uniquely required for those individual goals. Its first substantial hypothesis is amortizing regional-property success into higher K1 under the same validity controller.

In delayed interventions, independently confirmed selected-action improvements were not established at stages8,32 or48: confidence intervals include zero. None of the12 selected prefixes already violated the exposed-region upper bound. Thus this study does not demonstrate a reliable action-ranking method, a particular early-stage bottleneck, or a need for hierarchy. These limitations must not be replaced by claims from shaped reward alone. An on-policy score-gradient campaign does not require a learned value/ranking critic; its performance and useful temporal credit assignment remain hypotheses for the declared substantial experiment.

Four same-sequence reversed regional patterns had observed hard-valid generated witnesses. Frozen goal-context effects on signed regional contrast were small (approximately0.002–0.030) and TRAIN-witness-selected; intentional property controllability is not confirmed by support existence alone. The bounded rewind search solved0/4 tasks, versus1/4 for whole-chain Bestof16, at roughlyhalf the simulated-proposal count; it does not demonstrate superiority over independent sampling.

The exact next step remains one separately authorized5,000-update fixed-sequence geometry campaign, conditional on storage expansion and full production optimizer/journal replay qualification. Primary promotion must require at least5pp K1 gain versus the immutable actor with the **same finite filtering**, positive paired95% lower interval, unchanged original validity/KL/diversity safeguards and K8 not worse. Compare terminal Bestof8 and non-RL search as explicit compute-accounted alternatives. Do not count motif enforcement, the250-update CPU toy, or the untrained V93 baselines as protein RL success. No additional small critic or protein-RL diagnostic is proposed here.
'''
for name in ['FINAL_E017_V93_SCIENTIFIC_REPORT.md','NEXT_CRISP_RESEARCH_DECISION.md','REGIONAL_PROPERTY_CONTROLLABILITY.md']:
 p=r/name;p.write_text(p.read_text()+summary)
p=r/'SUBSTANTIAL_RL_TRAINING_PLAN.md';p.write_text(p.read_text()+'\n\nPrimary trained-policy comparator: immutable parent actor with the same finite8 validity controller. Require K1 gain≥5pp and paired95% lower>0; unfiltered-policy improvements alone cannot qualify. Original validity, conditional proposal KL and diversity safeguards remain mandatory. No hierarchical training is admitted by these results.\n')
# Plan acceptance criteria are frozen before any protein optimization. Existing
# V93 outcomes and thresholds are never rewritten or relabeled as trained.
p=r/'SUBSTANTIAL_CAMPAIGN_SPEC.json';v=read(p);v['primary_performance_comparator']='immutable parent with identical finite8 validity controller';v['final_performance_acceptance']='K1gain>=.05 and paired95%lower>0; original hard/backbone/KL/diversity safeguards andK8notworse';p.write_text(json.dumps(v,indent=2)+'\n')
print('INTERPRETATION_REVIEWED',counts)
