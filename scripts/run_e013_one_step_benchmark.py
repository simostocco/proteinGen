"""Bounded matched-exposure architecture pilot with immutable source audits."""
from run_e013_unconditional_equivariant_denoiser import ROOT, d_path, sha
import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from protein_distance_diffusion.models.e013_unconditional_equivariant_denoiser import E013UnconditionalEquivariantDenoiser
from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import EquivariantPairCoordinateUNet
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion, center_coordinates
from protein_distance_diffusion.training.e013_losses import coordinate_v_loss
from protein_distance_diffusion.evaluation.e013_one_step_benchmark import diagnostics, summarize, adjudicate

HIST = Path('/mnt/d/Simone/proteinGen/reports/experiments/E007_matrix_sequence_cogeneration')
DATA = Path('/mnt/d/Users/Simone Stocco/proteinGen_audits/e006_rich_geometry_sidecars_v2')


def verify(path, required):
    observed = sha(path)
    if observed != required:
        raise ValueError(f'provenance failure: {path}')
    return observed


def prepare_panel(out):
    hashes = {}
    expected = dict(protocol='653d1a88c4366401345287d6c3eea2267c363945bac429374db2058015913b8b',
                    schema='8d522a7c198c8b35317be41bb8f55ef7a52de118f015c88a374e221844afd329',
                    normalization='a54d862d756ca59a78503801a7ea1488ba5e701116db66545ce22e9af91b30fe',
                    vocabulary='b6e87d56aa6023679c7249e04ba709576c26038fef4588e1b83763c395d4e1db')
    for name, digest in expected.items():
        hashes[str(DATA/f'{name}.json')] = verify(DATA/f'{name}.json', digest)
    inventory = DATA/'shard_hashes.sha256'
    hashes[str(inventory)] = verify(inventory, 'a3331f8df4a15e42f10611395d41672f8c7e282050e7b77e10081436548884c0')
    manifest = HIST/'coordinate_real_loader_smoke_v1/selected_panel_manifest.json'
    hashes[str(manifest)] = verify(manifest, '2d06745398ba2accd432046042e4541a65cd19c6b97e834fa59a99cfd9809619')
    clean = HIST/'split_homology_audit_v1_publication_correction_v1/mmseqs/identity_30/clean_validation_manifest.parquet'
    hashes[str(clean)] = verify(clean, 'a847fbf36497df7a015dd4cb3c683f47ec6c2d3217b1b6eaebdddcf7af0adb11')
    clean_ids = set(pq.read_table(clean, columns=['sample_id'], use_threads=False).column(0).to_pylist())
    records = json.loads(manifest.read_text())['records']
    records = [r for r in records if r['split']=='train' or r['sample_id'] in clean_ids]
    # Preserve selected historical panel; remove known cross-split homologs.
    for split in ('train', 'validation'):
        if len({r['length_stratum'] for r in records if r['split']==split}) != 5:
            raise ValueError('selected panel lacks a required stratum')
    protocol = json.loads((DATA/'protocol.json').read_text())
    panel = []
    for record in records:
        index = record['dataset_index']
        for shard in protocol['shards']:
            if shard['dataset'] != record['split']:
                continue
            if index >= shard['row_count']:
                index -= shard['row_count']
                continue
            path = DATA/shard['path']
            if str(path) not in hashes:
                hashes[str(path)] = verify(path, shard['sha256'])
            # Only coordinate/provenance columns. Never read amino-acid identity.
            row = pq.read_table(path, columns=['sample_id', 'ca_coordinates', 'ca_mask', 'chain_continuity_mask',
                                              'source_sha256', 'npz_sha256'], use_threads=False).slice(index, 1).to_pylist()[0]
            x = np.asarray(row['ca_coordinates'], dtype=np.float32)
            if row['sample_id'] != record['sample_id'] or len(x) != record['sequence_length']:
                raise ValueError('source row identity contradiction')
            for key in ('source_sha256', 'npz_sha256'):
                if row[key] != record[key]:
                    raise ValueError('coordinate source provenance mismatch')
            if not all(row['ca_mask']) or not all(row['chain_continuity_mask']) or not np.isfinite(x).all():
                raise ValueError('non-contiguous/nonfinite source')
            digest = hashlib.sha256(np.ascontiguousarray(x, dtype='<f4').tobytes())
            digest.update(np.asarray(row['ca_mask'], dtype=np.uint8).tobytes())
            if digest.hexdigest() != record['coordinate_sha256']:
                raise ValueError('historical selected-coordinate hash mismatch')
            panel.append(dict(record=record, coordinates=torch.from_numpy(x.copy())))
            break
        else:
            raise ValueError('dataset index absent')
    evidence = dict(source_hashes=hashes, selected_identities=[p['record'] for p in panel],
                    validation_policy='historical_panel_intersect_30_percent_identity_80_percent_coverage_clean',
                    selected_shards_verified=True, full_dataset_rehash=False, sequence_read=False)
    (out/'panel_provenance.json').write_text(json.dumps(evidence, indent=2)+'\n')
    torch.save(panel, out/'coordinate_panel.pt')
    return panel, evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--updates', type=int, default=400)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfgpath = Path('configs/e013_unconditional_equivariant_denoiser_v1.yaml')
    cfg = yaml.safe_load(cfgpath.read_text())
    preflight = json.loads((ROOT/'preflight_cuda.json').read_text())
    if not preflight['passed'] or preflight['config_sha256'] != sha(cfgpath):
        raise ValueError('preflight prerequisite invalid')
    if verify(Path(cfg['normalization_artifact']), cfg['normalization_required_sha256']) != preflight['normalization_sha256']:
        raise ValueError('normalization boundary changed')
    out = d_path(ROOT/'one_step_v1')
    out.mkdir(exist_ok=False)
    panel, provenance = prepare_panel(out)
    scale = cfg['coordinate_scale_angstrom']
    for item in panel:
        item['x0'] = item['coordinates'].to('cuda')[None]/scale
        item['mask'] = torch.ones(item['x0'].shape[:2], device='cuda', dtype=torch.bool)
        item['x0'] = center_coordinates(item['x0'], item['mask'])
    train = [p for p in panel if p['record']['split']=='train']
    validation = [p for p in panel if p['record']['split']=='validation']
    oldcfg = yaml.safe_load(Path('configs/e007_coordinate_generator_v1.yaml').read_text())['model']
    diffusion = CoordinateVPDiffusion(500)
    models = {}
    for name, constructor, kwargs in [('e007_control', EquivariantPairCoordinateUNet, oldcfg),
                                      ('e013', E013UnconditionalEquivariantDenoiser, cfg['model'])]:
        torch.manual_seed(cfg['seed'])
        models[name] = constructor(**kwargs).to('cuda')
    def predict(name, batch, item):
        length = torch.tensor([item['x0'].shape[1]], device='cuda')
        model = models[name]
        if name=='e007_control':
            continuity = torch.ones(1, int(length[0])-1, device='cuda', dtype=torch.bool)
            return model(batch.noisy_coordinates, batch.timesteps, length, item['mask'], continuity)['v_prediction']
        return model(batch.noisy_coordinates, batch.timesteps, length, item['mask'])
    # Freeze tolerance algorithm before any validation inference. Calibration
    # uses training-only initial control noise-replicate v losses, not val/test.
    calibration = []
    for idx, item in enumerate(train):
        for rep in range(2):
            g = torch.Generator(device='cuda').manual_seed(13100+idx*2+rep)
            batch = diffusion.make_training_batch(item['x0'], item['mask'], timesteps=torch.tensor([250], device='cuda'), generator=g)
            with torch.no_grad():
                calibration.append(float(coordinate_v_loss(predict('e007_control', batch, item), batch.coordinate_v_target, item['mask'])))
    noise_sd = float(np.std(np.asarray(calibration).reshape(-1, 2)[:, 0]-np.asarray(calibration).reshape(-1, 2)[:, 1])/np.sqrt(2))
    tol = min(.10, max(.02, 2*noise_sd/(np.sqrt(len(train))*np.mean(calibration))))
    tolerances = dict(aggregate=tol, length=2*tol, improvement=.05, collapse_radius_ratio=.25)
    protocol = dict(git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    config_sha256=sha(cfgpath), normalization_sha256=cfg['normalization_required_sha256'],
                    panel_provenance_sha256=sha(out/'panel_provenance.json'), seed=cfg['seed'],
                    updates_per_model=args.updates, physical_batch_size=1, optimizer='AdamW',
                    learning_rate=.0002, weight_decay=.01, gradient_clip=1., objective='equal_protein_coordinate_v_mse',
                    geometry_auxiliaries=False, tolerances=tolerances, training_only_noise_sd=noise_sd,
                    prospective_validation_not_yet_evaluated=True, torch_version=torch.__version__,
                    cuda_version=torch.version.cuda, output_path=str(out), storage_compliant=True,
                    parameter_counts={k:sum(p.numel() for p in m.parameters()) for k,m in models.items()},
                    exposure_policy='identical_protein_order_timesteps_and_noise_seeds_no_validation_selection')
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    optimizers = {k:torch.optim.AdamW(m.parameters(), lr=.0002, weight_decay=.01) for k,m in models.items()}
    order = list(range(len(train)))
    random.Random(cfg['seed']).shuffle(order)
    log = open(out/'training.jsonl', 'x')
    start = time.perf_counter()
    for step in range(args.updates):
        item = train[order[step%len(order)]]
        generator = torch.Generator(device='cuda').manual_seed(cfg['seed']+100000+step)
        batch = diffusion.make_training_batch(item['x0'], item['mask'], generator=generator)
        row = dict(update=step+1, sample_id=item['record']['sample_id'], timestep=int(batch.timesteps[0]))
        for name, model in models.items():
            model.train()
            optimizers[name].zero_grad(set_to_none=True)
            pred = predict(name, batch, item)
            loss = coordinate_v_loss(pred, batch.coordinate_v_target, item['mask'])
            loss.backward()
            if not bool(torch.isfinite(loss)) or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise ValueError('training nonfinite; fail closed')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizers[name].step()
            row[name] = float(loss.detach())
        log.write(json.dumps(row)+'\n')
        log.flush()
        if (step+1)%20==0:
            print(json.dumps(row), flush=True)
    log.close()
    checkpoint_hashes = {}
    for name, model in models.items():
        model.eval()
        path = out/f'{name}_final.pt'
        torch.save(dict(model=model.state_dict(), optimizer=optimizers[name].state_dict(), protocol=protocol), path)
        checkpoint_hashes[name] = sha(path)
    rows = []
    with torch.no_grad():
        for split, items in [('train', train), ('validation', validation)]:
            for idx, item in enumerate(items):
                for t in cfg['stage_c']['noise_timesteps']:
                    g = torch.Generator(device='cuda').manual_seed(1300000+idx*500+t)
                    batch = diffusion.make_training_batch(item['x0'], item['mask'], timesteps=torch.tensor([t], device='cuda'), generator=g)
                    for name in models:
                        vhat = predict(name, batch, item)
                        xhat = diffusion.reconstruct_x0(batch.noisy_coordinates, batch.timesteps, vhat, item['mask'])
                        row = diagnostics(xhat[0], item['x0'][0], vhat[0], batch.coordinate_v_target[0], scale)
                        row.update(model=name, split=split, timestep=t, stratum=item['record']['length_stratum'], sample_id=item['record']['sample_id'])
                        rows.append(row)
    (out/'metrics.json').write_text(json.dumps(rows, indent=2)+'\n')
    control = [r for r in rows if r['model']=='e007_control' and r['split']=='validation']
    candidate = [r for r in rows if r['model']=='e013' and r['split']=='validation']
    decision = adjudicate(control, candidate, tolerances)
    fields = ['v_mse', 'x0_mse_angstrom2', 'aligned_rmsd_angstrom', 'local_mse_angstrom2', 'torsion_mse',
              'triple_inversion_fraction', 'clash_fraction', 'radius_of_gyration_angstrom', 'long_range_contact_density']
    summary = []
    for model in models:
        for split in ('train', 'validation'):
            selected = [r for r in rows if r['model']==model and r['split']==split]
            groups = [('aggregate', 'all', selected)]
            groups += [('noise', t, [r for r in selected if r['timestep']==t]) for t in cfg['stage_c']['noise_timesteps']]
            groups += [('length', s, [r for r in selected if r['stratum']==s]) for s in sorted({r['stratum'] for r in selected})]
            for dimension, value, group in groups:
                summary.append(dict(model=model, split=split, dimension=dimension, value=value, count=len(group),
                                    **{f:summarize(group, f) for f in fields}))
    report = dict(protocol=protocol, decision=decision, checkpoint_hashes=checkpoint_hashes, summaries=summary,
                  elapsed_seconds=time.perf_counter()-start, trained=True, evaluated=True,
                  stage_d='not_run', stage_e='not_run', de_novo_claim=False,
                  limitation='20 historical training identities, 17 homology-clean validation identities; 400-update feasibility pilot, not population-level convergence')
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(decision), flush=True)


if __name__=='__main__':
    main()
