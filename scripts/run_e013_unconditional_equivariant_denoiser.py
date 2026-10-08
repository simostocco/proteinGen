"""Plan and strict preflight; generated outputs restricted to physical D: paths."""
import os
from pathlib import Path

ROOT = Path('/mnt/d/Simone/proteinGen/e013_artifacts')
ROOT.mkdir(parents=True, exist_ok=True)
for key, subdir in [('TMPDIR', 'tmp'), ('XDG_CACHE_HOME', 'cache'), ('TORCH_HOME', 'cache/torch'),
                    ('CUDA_CACHE_PATH', 'cache/cuda'), ('TRITON_CACHE_DIR', 'cache/triton')]:
    path = ROOT/subdir
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import hashlib
import json
import subprocess
import time
import torch
import yaml
from protein_distance_diffusion.models.e013_unconditional_equivariant_denoiser import E013UnconditionalEquivariantDenoiser


def d_path(path):
    p = Path(path).resolve()
    if not p.is_relative_to('/mnt/d'):
        raise ValueError(f'generated path is not on D: {p}')
    return p


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['plan', 'preflight'], required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--config', default='configs/e013_unconditional_equivariant_denoiser_v1.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    output = d_path(cfg['output_dir'])
    for key in ('TMPDIR', 'XDG_CACHE_HOME', 'TORCH_HOME', 'CUDA_CACHE_PATH', 'TRITON_CACHE_DIR'):
        d_path(os.environ[key])
    torch.set_num_threads(1)
    torch.manual_seed(cfg['seed'])
    model = E013UnconditionalEquivariantDenoiser(**cfg['model'])
    normalization = Path(cfg['normalization_artifact'])
    if sha(normalization) != cfg['normalization_required_sha256']:
        raise ValueError('normalization hash mismatch')
    norm = json.loads(normalization.read_text())
    if norm['coordinate_scale_angstrom'] != cfg['coordinate_scale_angstrom'] or norm['fitting_split'] != 'train':
        raise ValueError('normalization provenance mismatch')
    report = dict(stage=args.stage, git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  config_sha256=sha(args.config), normalization_sha256=sha(normalization),
                  normalization_artifact=str(normalization), parameter_count=sum(p.numel() for p in model.parameters()),
                  seed=cfg['seed'], torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  output_path=str(output), storage_compliant=True, selected_identities=[],
                  sequence_input=False, target_derived_inference_features=False)
    if args.stage == 'preflight':
        device = args.device
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA prerequisite failed')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = model.to(device).eval()
        report['device'] = str(device)
        report['lengths'] = []
        def sync():
            if device == 'cuda':
                torch.cuda.synchronize()
        for n in (64, 128, 256, 384, 500):
            x = torch.randn(1, n, 3, device=device)
            t = torch.tensor([50], device=device)
            lengths = torch.tensor([n], device=device)
            mask = torch.ones(1, n, dtype=torch.bool, device=device)
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                model(x, t, lengths, mask)
                sync()
                start = time.perf_counter()
                for _ in range(5):
                    y = model(x, t, lengths, mask)
                sync()
                latency = (time.perf_counter()-start)/5
                r, _ = torch.linalg.qr(torch.randn(3, 3, device=device))
                if torch.linalg.det(r) < 0:
                    r[:, 0] *= -1
                yr = model(x@r+2, t, lengths, mask)
                delta = yr-y@r
                absolute = float(delta.abs().max())
                relative = float(delta.norm()/(y.norm()+1e-12))
                reflected_x = x*torch.tensor([-1., 1, 1], device=device)
                reflected_y = model(reflected_x, t, lengths, mask)
                parity = float((reflected_y-y*torch.tensor([-1., 1, 1], device=device)).norm())
            model.zero_grad(set_to_none=True)
            sync()
            start = time.perf_counter()
            prediction = model(x, t, lengths, mask)
            prediction.square().mean().backward()
            sync()
            backward_latency = time.perf_counter()-start
            finite = bool(torch.isfinite(prediction).all()) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            allocated = torch.cuda.max_memory_allocated() if device == 'cuda' else None
            reserved = torch.cuda.max_memory_reserved() if device == 'cuda' else None
            row = dict(length=n, forward_seconds=latency, forward_backward_seconds=backward_latency,
                       peak_allocated_bytes=allocated, peak_reserved_bytes=reserved,
                       equivariance_max_abs=absolute, equivariance_relative_l2=relative,
                       reflection_non_equivalence_l2=parity, finite=finite)
            report['lengths'].append(row)
            print(json.dumps(row), flush=True)
        gate = cfg['preflight']
        report['passed'] = all(r['finite'] and r['equivariance_max_abs'] <= gate['equivariance_atol']
                               and r['equivariance_relative_l2'] <= gate['equivariance_relative_l2']
                               and r['reflection_non_equivalence_l2'] > 1e-6
                               and (r['peak_reserved_bytes'] is None or r['peak_reserved_bytes'] <= gate['max_peak_reserved_bytes'])
                               for r in report['lengths'])
    path = output/f'{args.stage}_{args.device}.json'
    if path.exists():
        raise FileExistsError(f'will not overwrite prior boundary: {path}')
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    if report.get('passed') is False:
        raise SystemExit('preflight failed; training forbidden')


if __name__ == '__main__':
    main()
