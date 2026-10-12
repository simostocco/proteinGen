"""Compact numerical reports and cards for the frozen external E015 panel."""
from pathlib import Path
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from e015_external_designability import OUT, ROOT, WORK, read, write, sha, resource, progress


def table(name):
    p=OUT/name
    return pq.read_table(p,use_threads=False).to_pylist() if p.exists() else []


def ca(p):
    return np.array([[float(l[a:b]) for a,b in ((30,38),(38,46),(46,54))] for l in Path(p).read_text().splitlines() if l.startswith('ATOM') and l[12:16].strip()=='CA'])


def correlations(rows,x,y):
    pairs=[(r[x],r[y]) for r in rows if r.get(x) is not None and r.get(y) is not None]
    if len(pairs)<3:return dict(n=len(pairs),pearson=None,spearman=None)
    from scipy.stats import spearmanr
    a,b=np.array(pairs).T
    if np.std(a)==0 or np.std(b)==0:return dict(n=len(pairs),pearson=None,spearman=None)
    return dict(n=len(pairs),pearson=float(np.corrcoef(a,b)[0,1]),spearman=float(spearmanr(a,b).statistic))


def aggregate():
    resource('compact_aggregate_analysis',heavy=True)
    panel=read(OUT/'backbone_panel.json');align=table('structural_alignment_results.parquet');summary=table('designability_summary.parquet')
    latest=[b for b in panel['backbones'] if b['source_category']=='latest_best'];pairs=[]
    for b in latest:
        h=next(r for r in panel['backbones'] if r['source_category']=='historical_same_seed' and r['length']==b['length'])
        groups=[]
        for item in (h,b):
            rs=[r for r in align if r['backbone_id']==item['backbone_id'] and r['paired_four_design_endpoint']]
            vals=dict(backbone_id=item['backbone_id'],valid=len(rs),best_tm=max((r['target_tm_score'] for r in rs),default=None),median_tm=float(np.median([r['target_tm_score'] for r in rs])) if rs else None,D07=sum(r['strong_recovery'] for r in rs)/4 if len(rs)==4 else None,median_rmsd=float(np.median([r['aligned_ca_rmsd_angstrom'] for r in rs])) if rs else None,mean_plddt=float(np.mean([r['mean_plddt'] for r in rs])) if rs else None,mean_ptm=float(np.mean([r['ptm'] for r in rs])) if rs else None,mean_mpnn_score=float(np.mean([r['score'] for r in rs])) if rs else None)
            groups.append(vals)
        p=dict(length=b['length'],seed=b['seed'],historical_backbone=h['backbone_id'],latest_backbone=b['backbone_id'],matched_design_budget=4)
        for k in ('best_tm','median_tm','D07','median_rmsd','mean_plddt','mean_ptm','mean_mpnn_score'):
            p['historical_'+k]=groups[0][k];p['latest_'+k]=groups[1][k]
            p['change_'+k]=groups[1][k]-groups[0][k] if groups[1][k] is not None and groups[0][k] is not None else None
        pairs.append(p)
    pq.write_table(pa.Table.from_pylist(pairs),OUT/'same_seed_50_vs_latest.parquet')
    rng=np.random.default_rng(151502);boot={}
    for key in ('best_tm','median_tm','D07','median_rmsd'):
        ds=[r['change_'+key] for r in pairs if r['change_'+key] is not None]
        if len(ds)==5:
            samples=np.array(ds)[rng.integers(0,5,size=(5000,5))].mean(1)
            boot[key]=dict(mean_change=float(np.mean(ds)),interval95=np.quantile(samples,[.025,.975]).tolist(),length_pairs=5,interpretation='Bounded five-pair resampling; not evidence of broad statistical significance')
    write(OUT/'same_seed_bootstrap.json',boot)
    scores={b['backbone_id']:b.get('numerical_score') for b in panel['backbones']}
    for r in summary:r['numerical_score']=scores[r['backbone_id']]
    write(OUT/'correlations.json',dict(mpnn_score_vs_refold_tm=correlations(align,'score','target_tm_score'),plddt_vs_refold_tm=correlations(align,'mean_plddt','target_tm_score'),ptm_vs_refold_tm=correlations(align,'ptm','target_tm_score'),internal_quality_vs_designability=correlations([r for r in summary if r['source_category'].startswith('latest')],'numerical_score','D_0_7'),interpretation='Descriptive bounded-panel correlations; sequence-level observations are clustered within backbones, not independent biological replicates'))
    progress('AGGREGATE_ANALYSIS_COMPLETE')


def cards():
    resource('bounded_candidate_cards',heavy=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    panel=read(OUT/'backbone_panel.json')['backbones'];summary={r['backbone_id']:r for r in table('designability_summary.parquet')};alignment=table('structural_alignment_results.parquet');novelty={r['backbone_id']:r for r in table('novelty_summary.parquet')}
    def draw(ax,xyz,title,extent,color=None):
        if xyz is not None:
            pts=np.stack([xyz[:-1],xyz[1:]],axis=1)
            lc=Line3DCollection(pts,cmap='viridis',linewidths=1.5)
            if color is None:lc.set_array(np.linspace(0,1,len(pts)))
            else:lc.set_color(color)
            ax.add_collection3d(lc);ax.scatter(*xyz[0],color='navy',s=12);ax.scatter(*xyz[-1],color='gold',s=12)
        ax.set(xlim=(-extent,extent),ylim=(-extent,extent),zlim=(-extent,extent),title=title,xlabel='Å',ylabel='Å',zlabel='Å');ax.view_init(20,45)
    for b in panel:
        if b['source_category'] not in ('latest_best','real_heldout'):continue
        bid=b['backbone_id'];rs=sorted([r for r in alignment if r['backbone_id']==bid],key=lambda r:r['target_tm_score'],reverse=True)
        s=summary.get(bid,{});n=novelty.get(bid,{})
        x=ca(b['sanitized_pdb']);x-=x.mean(0);y=None
        if rs:
            y=ca(rs[0]['pdb']);y-=y.mean(0)
            u,v,vt=np.linalg.svd(y.T@x);y=y@(u@np.diag([1,1,np.linalg.det(u@vt)])@vt)
        extent=max(5,float(np.abs(x).max())*1.08,float(np.abs(y).max())*1.08 if y is not None else 0)
        fig=plt.figure(figsize=(14,10),layout='constrained');grid=fig.add_gridspec(3,3)
        draw(fig.add_subplot(grid[0,0],projection='3d'),x,'Cα target; neutral residue placeholders',extent)
        draw(fig.add_subplot(grid[0,1],projection='3d'),y,'Best single-sequence refold' if y is not None else 'Refold unavailable',extent)
        ax=fig.add_subplot(grid[0,2],projection='3d');draw(ax,x,'Index-matched overlay; target blue/refold red',extent,'steelblue')
        if y is not None:draw(ax,y,'Index-matched overlay; target blue/refold red',extent,'tomato')
        ax=fig.add_subplot(grid[1,0]);sc=read(rs[0]['score_json']) if rs else {}
        if 'pae' in sc:im=ax.imshow(sc['pae'],vmin=0,vmax=30,cmap='magma');fig.colorbar(im,ax=ax,label='PAE Å')
        else:ax.text(.5,.5,'PAE unavailable',ha='center',transform=ax.transAxes)
        ax.set_title('Predicted aligned error')
        ax=fig.add_subplot(grid[1,1]);ax.plot(sc.get('plddt',[]));ax.set(xlabel='Residue',ylabel='pLDDT',ylim=(0,100),title='Predictor confidence; not target agreement')
        ax=fig.add_subplot(grid[1,2]);ax.axis('off')
        lines=[f'N={b["length"]}; source={b["source_category"]}',f'Internal score={b.get("numerical_score")}',f'D0.7={s.get("D_0_7")}; D0.8={s.get("D_0_8")}',f'Classification={s.get("classification","INCONCLUSIVE")}',f'Novelty={n.get("classification","NOT_ASSESSED")}',f'Nearest PDB={n.get("nearest_pdb","not confirmed")}',f'qTM={n.get("query_tm_score","not measured")}',f'Successful sequence clusters={s.get("successful_clusters",0)}']
        if rs:lines.extend([f'Best target TM={rs[0]["target_tm_score"]:.3f}',f'Aligned RMSD={rs[0]["aligned_ca_rmsd_angstrom"]:.2f} Å',f'Coverage={rs[0]["target_coverage"]:.1%}',f'MPNN score={rs[0]["score"]:.3f}; cluster={rs[0]["identity80_cluster"]}'])
        ax.text(0,1,'\n'.join(lines),va='top',fontsize=9)
        ax=fig.add_subplot(grid[2,:]);ax.axis('off');ax.text(0,1,'No experimental stability, solubility or function claim.\nTargets are Cα-only; predictor structures carry designed sequences.\nA low PDB similarity without validated multi-sequence target recovery is not novelty evidence.\nMSA-assisted predictions and nearest-PDB overlays are shown only when completed and confirmed.',va='top')
        fig.suptitle(bid,fontsize=12)
        folder='positive_control_gallery' if b['source_category']=='real_heldout' else ('known_fold_like_gallery' if n.get('classification')=='KNOWN_FOLD_LIKE' else 'potentially_novel_gallery' if n.get('classification') in ('POTENTIALLY_NOVEL_RELATIVE_TO_PDB','HIGH_PRIORITY_NOVEL_CANDIDATE') else 'candidates')
        p=OUT/folder/(bid+'_card.png');fig.savefig(p,dpi=120);plt.close(fig)
    progress('CANDIDATE_CARDS_COMPLETE')


def report():
    panel=read(OUT/'backbone_panel.json');summary=table('designability_summary.parquet');align=table('structural_alignment_results.parquet');seqs=table('proteinmpnn_all_sequences.parquet');folds=table('colabfold_screening.parquet');tc=read(OUT/'toolchain_manifest.json');ctrl=read(OUT/'controls_summary.json') if (OUT/'controls_summary.json').exists() else dict(status='NOT_COMPLETED');novel=table('novelty_summary.parquet')
    primary=[r for r in summary if r['source_category']=='latest_best'];validated=ctrl['status']=='VALIDATED'
    strong=sum(r['classification']=='STRONGLY_DESIGNABLE' for r in primary);promising=sum(r['classification'] in ('STRONGLY_DESIGNABLE','PROMISINGLY_DESIGNABLE') for r in primary)
    tested=sum(r['source_category']!='real_heldout' for r in summary)
    verdict=dict(toolchain='VALIDATED' if validated else 'PARTIALLY_VALIDATED' if align else 'INVALID',designability='STRONG_EVIDENCE' if validated and strong>=3 else 'PROMISING_BEST_TAIL' if validated and promising else 'WEAK_EVIDENCE' if validated and any(r['strong_recovery'] for r in align if r['source_category']=='latest_best') else 'NO_EVIDENCE' if validated and len(primary)==5 and all(r['valid_predictions']==r['selected_denominator'] and r['selected_denominator'] for r in primary) else 'INCONCLUSIVE',structural_novelty='POTENTIALLY_NOVEL_CANDIDATES_FOUND' if any(r['classification'] in ('POTENTIALLY_NOVEL_RELATIVE_TO_PDB','HIGH_PRIORITY_NOVEL_CANDIDATE') for r in novel) else 'INCONCLUSIVE',same_seed_improvement='INCONCLUSIVE',practical_use='GENERATE_AND_FILTER_IS_JUSTIFIED' if validated and promising else 'MORE_MODEL_TRAINING_REQUIRED' if validated else 'INCONCLUSIVE')
    pairs=table('same_seed_50_vs_latest.parquet')
    changes=[r['change_D07'] for r in pairs if r.get('change_D07') is not None]
    if validated and len(changes)==5:verdict['same_seed_improvement']='DESIGNABILITY_IMPROVED' if np.mean(changes)>0 else 'REGRESSED' if np.mean(changes)<0 else 'NO_CLEAR_IMPROVEMENT'
    write(OUT/'verdict.json',verdict)
    counts=dict(e015_panel_backbones=12,real_controls=5,raw_sequences_generated=len(seqs),refolds_completed=sum(r['status']=='COMPLETE' for r in folds),latest_best_strong=strong,latest_best_at_least_promising=promising,latest_best_tested=len(primary),designable_yield=promising/5 if validated and len(primary)==5 else None,validated=validated)
    write(OUT/'model_level_yield.json',dict(**counts,interpretation='Five numerically selected best-tail backbones, enriched rather than randomly sampled. This yield is not the full model probability; remaining15 primary generated backbones have not all been externally tested.'))
    lines=['# E015 external computational designability assessment','',f'Evaluation: {panel["evaluation"]}',f'Checkpoint: {json.dumps(panel["checkpoint"])}','',f'Progress: {read(OUT/"progress.json")["phase"]}',f'Controls: {ctrl["status"]}',f'Counts: {json.dumps(counts)}','',f'Verdicts: {json.dumps(verdict)}','','| Backbone | N | Best TM | Median TM | D0.7 | D0.8 | Valid/selected | Classification |','|---|---:|---:|---:|---:|---:|---:|---|']
    for r in summary:lines.append('| '+ ' | '.join(str(r.get(k)) for k in ('backbone_id','length','best_target_tm_score','median_target_tm_score','D_0_7','D_0_8'))+f' | {r["valid_predictions"]}/{r["selected_denominator"]} | {r["classification"]} |')
    lines.extend(['','The hard82 screen cap is honored:40 latest-best,20 historical,20 real-control,1 median,1 failure. The stated per-category request would otherwise total84. All32 raw designs per backbone are retained. The fair same-start comparison uses a matched four-design endpoint; latest eight-design results are reported separately.','','Positive-control failure makes E015 designability inconclusive, rather than evidence of non-designability. Single-sequence screening uses no target coordinates, templates or homologous MSA input. Confidence does not replace target structural agreement.','','Novelty remains unassessed until the real-control gate passes and PDB neighbors are independently confirmed. No candidate is described as an experimentally stable, soluble or functional protein.','','Numerical data: structural_alignment_results.parquet, designability_summary.parquet, same_seed_50_vs_latest.parquet. FASTA: sequences/selected_sequences.fasta. Predictor outputs: colabfold/screening/. Candidate cards: candidates/ and classified galleries.','','Exact commands, upstream commits, dependency lock, weight hashes and tool help are in TOOLCHAIN.md, toolchain_manifest.json and logs/. Frozen allocation and structural gates: assessment_protocol.json. Same Gaussian starts: same_initial_gaussian_proof.json. Production pause and resume proofs remain separate from the scientific sampling checkpoint.'])
    text='\n'.join(lines)+'\n';(OUT/'REPORT.md').write_text(text);(OUT/'EXECUTIVE_SUMMARY.md').write_text('\n'.join(lines[:12])+'\n\nThe main multi-sequence refolding/novelty question is answered only by validated target agreement and confirmed PDB searches; sequence generation alone is not designability evidence.\n')
    # Empty result tables explicitly mean not executed, never fabricated hits.
    for name in ('colabfold_confirmation.parquet','foldseek_hits.parquet','confirmed_pdb_hits.parquet','novelty_summary.parquet'):
        if not (OUT/name).exists():pq.write_table(pa.table({'status':pa.array([],type=pa.string()),'reason':pa.array([],type=pa.string())}),OUT/name)
    (OUT/'TOOLCHAIN.md').write_text('# Pinned external toolchain\n\n'+json.dumps(tc,indent=2)+'\n\nOfficial upstream source only. Existing proteingen packages were not modified. New packages, model files, caches and outputs are on D:. Screening is single-sequence AF2-ptm, model1, three recycles, no templates or relaxation.\n')
    files=[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)) for p in OUT.iterdir() if p.is_file() and p.name not in ('MANIFEST.json','resource_monitor.jsonl','progress.json')]
    write(OUT/'MANIFEST.json',dict(status='COMPLETE' if validated and len(primary)==5 and all(r['valid_predictions']==r['selected_denominator'] for r in primary) else 'PARTIAL_OR_INCONCLUSIVE',counts=counts,verdict=verdict,files=files,panel_sha256=sha(OUT/'backbone_panel.json'),all_artifacts_on_d=True,original_e015_generations_immutable=True,other_experiments_modified=False))
    print(text)
