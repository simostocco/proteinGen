"""Integrated microbatch engine and update-level PCGrad accumulator."""

import contextlib

import torch

from protein_distance_diffusion.models.e015_multiscale_parity_denoiser import E015MultiscaleParityDenoiser
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
from protein_distance_diffusion.training.e015_ema import make_ema, update_ema
from protein_distance_diffusion.training.e015_losses import global_scale_regularizer, objectives
from protein_distance_diffusion.training.e015_pcgrad import combine_gradients
from protein_distance_diffusion.training.e015_rollout_states import induced_state
from protein_distance_diffusion.training.e015_self_conditioning import make_self_condition
from protein_distance_diffusion.training.e015_snr import SNRBinSampler


class Engine:
    def __init__(self, cfg, calibration, device="cuda", model=None, precision=None):
        self.cfg, self.calibration, self.device = cfg, calibration, device
        self.model = (model or E015MultiscaleParityDenoiser(**cfg["model"])).to(device).float().train()
        self.model.parity_midpoint.fill_(calibration["midpoints"]["chirality"])
        self.ema = make_ema(self.model)
        self.diffusion = CoordinateVPDiffusion(cfg["diffusion_steps"])
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg["optimizer"]["learning_rate"], weight_decay=cfg["optimizer"]["weight_decay"]
        )
        self.precision = precision or (
            "bf16" if device == "cuda" and torch.cuda.is_bf16_supported() else "fp16" if device == "cuda" else "fp32"
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.precision == "fp16")
        self.sampler = SNRBinSampler(cfg["seed"])
        self.sc_rng = torch.Generator().manual_seed(cfg["seed"] + 1)
        self.rollout_rng = torch.Generator().manual_seed(cfg["seed"] + 2)
        self.noise_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 3)
        self.total_sc, self.total_rollout, self.total_batches = 0, 0, 0
        self.actual_t_hist = torch.zeros(500, dtype=torch.int64)
        self.gv = self.gs = None
        self.weights = torch.tensor(calibration["v_weights"], device=device)

    def autocast(self):
        if self.precision == "fp32":
            return contextlib.nullcontext()
        return torch.autocast(self.device, dtype=torch.bfloat16 if self.precision == "bf16" else torch.float16)

    def begin(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.gv = [None for _ in self.model.parameters()]
        self.gs = [None for _ in self.model.parameters()]

    def microbatch(
        self,
        x,
        lengths,
        mask,
        exposures,
        weight,
        force_sc=None,
        force_rollout=None,
        force_rollout_steps=None,
        force_sc_context=None,
    ):
        times, bins = self.sampler.sample(
            len(x), self.device, self.cfg["features"].get("snr_balancing", True), shared_bin=True
        )
        batch = self.diffusion.make_training_batch(x, mask, timesteps=times, generator=self.noise_rng)
        xt, target = batch.noisy_coordinates, batch.coordinate_v_target
        do_rollout = (
            bool(torch.rand((), generator=self.rollout_rng) < self.cfg["rollout_probability"])
            if force_rollout is None
            else force_rollout
        )
        do_rollout = (
            do_rollout
            and self.cfg["features"]["rollout_states"]
            and (exposures >= self.cfg["rollout_warmup_exposures"] or force_rollout is True)
        )
        steps = 0
        if do_rollout:
            steps = (
                int(torch.randint(1, 4, (), generator=self.rollout_rng))
                if force_rollout_steps is None
                else force_rollout_steps
            )
            # Keep nonterminal states; preserve the independently sampled bins.
            times = times.clamp_min(steps + 1)
            batch = self.diffusion.make_training_batch(x, mask, timesteps=times, generator=self.noise_rng)
            with self.autocast():
                xt, times, target = induced_state(
                    self.ema, self.diffusion, batch.noisy_coordinates, times, x, lengths, mask, steps
                )
            bins = torch.tensor(
                [next(b for b, ts in enumerate(self.sampler.bins) if int(t) in ts.tolist()) for t in times],
                device=self.device,
            )
        do_sc = (
            bool(torch.rand((), generator=self.sc_rng) < self.cfg["self_conditioning_probability"])
            if force_sc is None
            else force_sc
        )
        do_sc = do_sc and self.cfg["model"]["self_conditioning"]
        with self.autocast():
            previous, available = make_self_condition(self.model, self.diffusion, xt, times, lengths, mask, do_sc)
            if force_sc_context is not None:
                previous = force_sc_context.detach()
                available = torch.ones(len(x), device=x.device, dtype=torch.bool)
            prediction, components = self.model.with_components(xt, times, lengths, mask, previous, available)
        xhat = self.diffusion.reconstruct_x0(xt, times, prediction.float(), mask)
        logsnr = self.sampler.logsnr.to(self.device)[times]
        gates = {
            term: ((logsnr - mid) / self.calibration["gate_width_logsnr"]).sigmoid()
            for term, mid in self.calibration["midpoints"].items()
        }
        if not self.cfg["features"]["geometry_losses"]:
            gates = {name: value * 0 for name, value in gates.items()}
        weights = (
            self.weights[bins] if self.cfg["features"]["snr_weighting"] else torch.ones(len(x), device=self.device)
        )
        vl, sl, losses = objectives(prediction, target, xhat, x, mask, weights, gates, self.calibration["coefficients"])
        if self.cfg["features"].get("global_scale_regularizer", False):
            sl = sl + self.cfg["global_scale_coefficient"] * global_scale_regularizer(xhat, x, mask)
        if not torch.isfinite(vl) or not torch.isfinite(sl):
            raise FloatingPointError("Nonfinite E015 objective")
        params = tuple(self.model.parameters())
        scale = self.scaler.get_scale()
        gv = torch.autograd.grad(self.scaler.scale(vl), params, retain_graph=True, allow_unused=True)
        gs = torch.autograd.grad(self.scaler.scale(sl), params, allow_unused=True)
        micro_vnorm = float(sum(g.detach().float().square().sum() for g in gv if g is not None).sqrt() / scale)
        micro_snorm = float(sum(g.detach().float().square().sum() for g in gs if g is not None).sqrt() / scale)
        for destination, source in ((self.gv, gv), (self.gs, gs)):
            for i, g in enumerate(source):
                if g is None:
                    continue
                g = g.detach().float() * (weight / scale)
                if not torch.isfinite(g).all():
                    raise FloatingPointError("Nonfinite dual-objective gradient; no fallback")
                if destination[i] is None:
                    destination[i] = g
                else:
                    destination[i].add_(g)
        self.actual_t_hist += torch.bincount(times.detach().cpu(), minlength=500)
        self.total_sc += int(do_sc)
        self.total_rollout += int(do_rollout)
        self.total_batches += 1
        telemetry = dict(
            v_loss=float(vl.detach()),
            struct_loss=float(sl.detach()),
            self_conditioned=do_sc,
            rollout=do_rollout,
            rollout_steps=steps,
            snr_bins=bins.tolist(),
            v_gradient_norm=micro_vnorm,
            struct_gradient_norm=micro_snorm,
            timesteps=times.tolist(),
            state_family="induced" if do_rollout else "forward",
            geometry={name: float(value.detach().mean()) for name, value in losses.items()},
            branch_by_timestep=[
                dict(
                    timestep=int(times[b]),
                    length=int(lengths[b]),
                    polar_norm=float(c["global_v"].detach().norm()),
                    chiral_norm=float(c["chiral_effective"].detach().norm()),
                    local_norm=float(c["local_effective"].detach().norm()),
                    gate=float(c["gate"].detach().mean()),
                    reliability=float(c["reliability"].detach().mean()),
                    assessable=int(c["assessable"].sum()),
                )
                for b, c in enumerate(components)
            ],
            output_norms={
                name: float(torch.stack([c[key].detach().norm() for c in components]).mean())
                for name, key in [("polar", "global_v"), ("chiral", "chiral_effective"), ("local", "local_effective")]
            },
            parity_gate_mean=float(torch.stack([c["gate"].detach().mean() for c in components]).mean()),
        )
        return telemetry

    def finish(self):
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            raise RuntimeError(
                "Distributed PCGrad reduction/exposure contract not certified; refuse rank-local projection"
            )
        if self.cfg["features"]["pcgrad"]:
            gradients, telemetry = combine_gradients(self.gv, self.gs)
        else:
            gradients = [s if v is None else v if s is None else v + s for v, s in zip(self.gv, self.gs, strict=True)]
            telemetry = dict(conflict=False, disabled_by_config=True)
        scale = self.scaler.get_scale()
        for p, g in zip(self.model.parameters(), gradients, strict=True):
            if g is not None:
                p.grad = g.to(p.dtype) * scale
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg["optimizer"]["gradient_clip"], error_if_nonfinite=True
        )
        before = [p.detach().clone() for p in self.model.parameters()]
        self.scaler.step(self.optimizer)
        self.scaler.update()
        telemetry["optimizer_step_norm"] = float(
            sum(
                (p.detach() - old).square().sum() for p, old in zip(self.model.parameters(), before, strict=True)
            ).sqrt()
        )
        telemetry["parameter_norm"] = float(sum(p.detach().square().sum() for p in self.model.parameters()).sqrt())
        del before
        telemetry["clipped"] = bool(norm > self.cfg["optimizer"]["gradient_clip"])
        if self.cfg["features"]["ema"]:
            update_ema(self.ema, self.model, self.cfg["ema_decay"])
        else:
            self.ema.load_state_dict(self.model.state_dict())
        self.gv = self.gs = None
        telemetry["combined_preclip_norm"] = float(norm)
        self.optimizer.zero_grad(set_to_none=True)
        return telemetry

    def rng_state(self):
        return dict(
            sampler=self.sampler.state_dict(),
            actual_t_hist=self.actual_t_hist.clone(),
            sc=self.sc_rng.get_state(),
            rollout=self.rollout_rng.get_state(),
            noise=self.noise_rng.get_state(),
            total_sc=self.total_sc,
            total_rollout=self.total_rollout,
            total_batches=self.total_batches,
        )

    def restore_rng(self, state):
        self.sampler.load_state_dict(state["sampler"])
        self.actual_t_hist = state["actual_t_hist"].clone()
        self.sc_rng.set_state(state["sc"])
        self.rollout_rng.set_state(state["rollout"])
        self.noise_rng.set_state(state["noise"])
        self.total_sc, self.total_rollout, self.total_batches = (
            state["total_sc"],
            state["total_rollout"],
            state["total_batches"],
        )
