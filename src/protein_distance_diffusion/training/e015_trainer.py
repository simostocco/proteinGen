"""Complete-epoch E015 trainer with exact identity accounting and atomic saves."""

import json
import math
import shutil
import signal
import time

import numpy as np
import torch

from protein_distance_diffusion.evaluation.e014_evaluation import coordinate_batch
from protein_distance_diffusion.training.e014_runtime import (
    atomic_checkpoint,
    canonical_hash,
    epoch_schedule,
    load_checkpoint,
    restore_rng,
    rng_state,
    sha,
)
from protein_distance_diffusion.training.e015_engine import Engine
from protein_distance_diffusion.training.e015_replay import replay_snapshot
from protein_distance_diffusion.training.e015_runtime import GPULease, e007_active, status
from protein_distance_diffusion.training.e015_symmetry import autocast, error, require_symmetry, symmetry


def identity_schedule(rows, batch_sizes, seed, epoch):
    batches = epoch_schedule([r["length"] for r in rows], batch_sizes, seed, epoch)
    flat = [i for batch in batches for i in batch]
    ends = np.cumsum([len(batch) for batch in batches]).tolist()
    return flat, ends


def plan_update(rows, flat, ends, cursor, stop, residue_budget):
    groups = []
    residues = 0
    while cursor < stop and residues < residue_budget:
        end = next(end for end in ends if end > cursor)
        end = min(end, stop)
        ids = flat[cursor:end]
        groups.append(ids)
        residues += sum(rows[i]["length"] for i in ids)
        cursor = end
    return groups, cursor, residues


def smoke_folder(contract):
    return "preflight/training_smoke_" + canonical_hash(contract)[:12]


class Trainer:
    def __init__(self, runtime, mode="production", max_updates=None, resume=False):
        self.runtime, self.cfg, self.mode = runtime, runtime.cfg, mode
        self.lease = GPULease(runtime)
        self.calibration = json.loads(runtime.path("calibration/frozen.json").read_text())
        self.batching = json.loads(runtime.path("batch_autotuning.json").read_text())
        self.contract = runtime.contract(self.calibration)
        self.contract["batching_hash"] = canonical_hash(self.batching)
        self.folder = "production" if mode == "production" else smoke_folder(self.contract)
        self.directory = runtime.path(self.folder + "/checkpoints")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_updates = max_updates
        torch.manual_seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])
        import random

        random.seed(self.cfg["seed"])
        self.engine = Engine(self.cfg, self.calibration, precision=self.batching["selected"]["385-500"]["precision"])
        self.epoch, self.cursor, self.update, self.exposures = 1, 0, 0, 0
        self.seen = np.zeros(len(runtime.train), dtype=np.uint8)
        self.flat, self.ends = identity_schedule(
            runtime.train, self.batching["batch_sizes"], self.cfg["seed"], self.epoch
        )
        self.completed_evaluations = set()
        self.summaries = dict(bins={}, symmetry=[], accumulation=[], branch_gradient_max={}, ema_step_observed=False)
        self.conflicts = 0
        self.start = time.monotonic()
        self.last_save = self.start
        self.stopping = False
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.request_stop)
        if resume:
            saved = load_checkpoint(self.directory, self.contract)
            self.engine.model.load_state_dict(saved["model"])
            self.engine.ema.load_state_dict(saved["ema"])
            self.engine.optimizer.load_state_dict(saved["optimizer"])
            self.engine.scaler.load_state_dict(saved["scaler"])
            self.engine.restore_rng(saved["engine_rng"])
            restore_rng(saved["rng"])
            position = saved["position"]
            self.epoch, self.cursor, self.update, self.exposures = (
                position[k] for k in ("epoch", "cursor", "update", "exposures")
            )
            self.seen = np.asarray(saved["exposures"], dtype=np.uint8)
            self.completed_evaluations = set(saved["completed_evaluations"])
            self.conflicts = saved["conflicts"]
            self.summaries = saved["summaries"]
            self.flat, self.ends = identity_schedule(
                runtime.train, self.batching["batch_sizes"], self.cfg["seed"], self.epoch
            )
            if self.flat != saved["epoch_schedule"] or self.ends != saved["microbatch_ends"]:
                raise RuntimeError("Epoch schedule changed")
            self.audit()

    def request_stop(self, *_):
        self.stopping = True

    def audit(self):
        expected = np.full(len(self.seen), self.epoch - 1, dtype=np.uint8)
        expected[np.asarray(self.flat[: self.cursor], dtype=np.int64)] += 1
        if not np.array_equal(self.seen, expected) or int(self.seen.sum()) != self.exposures:
            raise RuntimeError("Exposure audit: skipped or duplicated TRAIN identities")

    def save(self, name, rolling=False):
        self.audit()
        self.runtime.disk_check(additional_bytes=600 * 1024**2)
        payload = dict(
            contract=self.contract,
            model=self.engine.model.state_dict(),
            ema=self.engine.ema.state_dict(),
            optimizer=self.engine.optimizer.state_dict(),
            scaler=self.engine.scaler.state_dict(),
            rng=rng_state(),
            engine_rng=self.engine.rng_state(),
            epoch_schedule=self.flat,
            microbatch_ends=self.ends,
            exposures=self.seen,
            position=dict(epoch=self.epoch, cursor=self.cursor, update=self.update, exposures=self.exposures),
            completed_evaluations=sorted(self.completed_evaluations),
            conflicts=self.conflicts,
            summaries=self.summaries,
            precision=self.engine.precision,
            fp32_geometry_islands=True,
            population_hash=self.cfg["population_hash"],
            architecture_hash=canonical_hash(self.cfg["model"]),
        )
        path = atomic_checkpoint(self.directory, name, payload, rolling)
        self.runtime.write(
            self.folder + "/COMMITTED.json", dict(path=str(path), sha256=sha(path), position=payload["position"])
        )
        if self.mode == "production":
            self.runtime.write("COMMITTED.json", dict(path=str(path), sha256=sha(path), position=payload["position"]))
        self.last_save = time.monotonic()
        return path

    def heartbeat(self, phase, telemetry=None, gradient=None, residues=0):
        elapsed = time.monotonic() - self.start
        info = status(
            phase,
            mode=self.mode,
            precision=self.engine.precision,
            fp32_geometry_islands=True,
            disk_policy="telemetry_only",
            schedule_cursor=self.cursor,
            epoch=self.epoch,
            update=self.update,
            exposures=self.exposures,
            completed_epochs=self.exposures // len(self.seen),
            identities_seen_this_epoch=self.cursor,
            total_identities=len(self.seen),
            loss_groups=telemetry,
            pcgrad=gradient,
            pcgrad_conflict_fraction=self.conflicts / max(self.update, 1),
            timestep_histogram=self.engine.actual_t_hist.tolist(),
            sampled_timestep_histogram=self.engine.sampler.timestep_histogram.tolist(),
            snr_bin_histogram=self.engine.sampler.histogram.tolist(),
            self_conditioning_fraction=self.engine.total_sc / max(self.engine.total_batches, 1),
            rollout_state_fraction=self.engine.total_rollout / max(self.engine.total_batches, 1),
            raw_and_ema=True,
            proteins_per_second=(self.exposures - self.initial_exposures) / max(elapsed, 1e-9),
            residues_per_second=self.processed_residues / max(elapsed, 1e-9),
            last_update_residues=residues,
            scientific_symmetry=self.summaries["symmetry"][-1] if self.summaries["symmetry"] else None,
            proper_rotation_diagnostic=self.summaries["symmetry"][-1]["raw"]["rotation"]
            if self.summaries["symmetry"]
            else None,
            reflection_gap_diagnostic=self.summaries["symmetry"][-1]["raw"]["reflection_gap"]
            if self.summaries["symmetry"]
            else None,
            accumulation=self.summaries["accumulation"][-1] if self.summaries["accumulation"] else None,
            last_microbatch_timing=self.engine.last_microbatch_timing,
            last_finish_timing=self.engine.last_finish_timing,
            gpu_allocated_bytes=torch.cuda.memory_allocated(),
            gpu_reserved_bytes=torch.cuda.memory_reserved(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
            gpu_free_bytes=torch.cuda.mem_get_info()[0],
            gpu_total_used_bytes=torch.cuda.get_device_properties(0).total_memory - torch.cuda.mem_get_info()[0],
            d_free_bytes=shutil.disk_usage(self.runtime.root).free,
        )
        self.runtime.write(self.folder + "/HEARTBEAT.json", info)
        self.runtime.write("HEARTBEAT.json", info)
        self.runtime.write(self.folder + "/STATE.json", info)
        if self.mode == "production":
            self.runtime.write("STATE.json", info)
        with open(self.runtime.path(self.folder + "/logs/training.jsonl"), "a") as stream:
            stream.write(json.dumps(info) + "\n")

    def evaluate(self, label, full=False):
        if label in self.completed_evaluations:
            return
        from protein_distance_diffusion.evaluation.e015_evaluation import evaluate_boundary

        self.heartbeat("evaluating_" + label)
        evaluate_boundary(self.runtime, self.engine, self.folder, label, full)
        self.completed_evaluations.add(label)

    def execute(self, groups, proteins, capture=False):
        self.engine.capture_replay = capture
        self.engine.begin()
        observations = []
        for ids in groups:
            rows = [self.runtime.train[i] for i in ids]
            x, n, m = coordinate_batch(rows, self.runtime.cache, self.cfg["coordinate_scale_angstrom"])
            # Smokes activate the existing rollout pathway early, with its frozen
            # 10% probability; exposure accounting itself is never overridden.
            context_exposures = (
                self.exposures
                if self.mode == "production"
                else max(self.exposures, self.cfg["rollout_warmup_exposures"])
            )
            observations.append(self.engine.microbatch(x, n, m, context_exposures, len(ids) / proteins))
        return observations, self.engine.finish()

    def record_observations(self, observations, gradient, groups, residues):
        self.summaries["ema_step_observed"] |= gradient["optimizer_step_norm"] > 0
        shape = dict(
            update=self.update,
            physical_proteins=[len(g) for g in groups],
            valid_residues=[sum(self.runtime.train[i]["length"] for i in g) for g in groups],
            accumulation_steps=len(groups),
            proteins=sum(map(len, groups)),
            residues=residues,
        )
        self.summaries["accumulation"].append(shape)
        self.summaries["accumulation"] = self.summaries["accumulation"][-1000:]
        for observation in observations:
            for branch, norm in observation["branch_gradient_norms"].items():
                self.summaries["branch_gradient_max"][branch] = max(
                    self.summaries["branch_gradient_max"].get(branch, 0), norm
                )
            for row in observation["protein_metrics"]:
                entry = self.summaries["bins"].setdefault(
                    str(row["snr_bin"]),
                    dict(
                        count=0,
                        v_loss=0.0,
                        struct_loss=0.0,
                        gate=0.0,
                        distance=0.0,
                        torsion=0.0,
                        chirality=0.0,
                        clash=0.0,
                        polar_norm=0.0,
                        chiral_norm=0.0,
                        local_norm=0.0,
                        gate_distance=0.0,
                        gate_torsion=0.0,
                        gate_chirality=0.0,
                        gate_clash=0.0,
                    ),
                )
                entry["count"] += 1
                for name in (
                    "v_loss",
                    "struct_loss",
                    "gate",
                    "distance",
                    "torsion",
                    "chirality",
                    "clash",
                    "polar_norm",
                    "chiral_norm",
                    "local_norm",
                ):
                    entry[name] += row[name]
                for name, value in row["gates"].items():
                    entry["gate_" + name] += value

    def check_symmetry(self):
        row = next(r for r in self.runtime.train if r["length"] == 64)
        x, n, m = coordinate_batch([row], self.runtime.cache, self.cfg["coordinate_scale_angstrom"])
        t = torch.tensor([50], device="cuda")
        generator = torch.Generator(device="cuda").manual_seed(30015)
        x = self.engine.diffusion.make_training_batch(x, m, timesteps=t, generator=generator).noisy_coordinates
        reflection = torch.diag(x.new_tensor([-1.0, 1.0, 1.0]))
        reflected_x = x @ reflection.T
        probe = {}
        for name, model in (("raw", self.engine.model), ("ema", self.engine.ema)):
            result = symmetry(model, x, t, n, m, None, None, self.engine.precision)
            require_symmetry(result, self.engine.precision)
            was_training = model.training
            model.eval()
            with torch.no_grad(), autocast(self.engine.precision, "cuda"):
                original = model(x, t, n, m)
                reflected = model(reflected_x, t, n, m)
            model.train(was_training)
            result["reflection_gap"] = error(reflected, original.float() @ reflection.T)
            probe[name] = result
        self.summaries["symmetry"].append(dict(update=self.update, **probe))
        points = [p["raw"]["rotation"]["relative_error"] for p in self.summaries["symmetry"][-4:]]
        if (
            len(points) == 4
            and all(a < b for a, b in zip(points, points[1:], strict=False))
            and points[-1] - points[0] > 0.005
        ):
            raise RuntimeError("Persistent material increase in smoke symmetry error")

    def finish_smoke(self, combined, gradient, residues):
        self.audit()
        fraction = self.engine.total_sc / max(self.engine.total_batches, 1)
        if not 0.3 <= fraction <= 0.7:
            raise RuntimeError("Smoke self-conditioning frequency inconsistent with0.5")
        if torch.any(self.engine.sampler.histogram == 0):
            raise RuntimeError("Smoke did not sample every SNR bin")
        if self.engine.total_rollout == 0 or not self.summaries["ema_step_observed"]:
            raise RuntimeError("Smoke did not exercise model-induced states/EMA updates")
        if any(self.summaries["branch_gradient_max"].get(name, 0) <= 0 for name in ("polar", "chiral", "local")):
            raise RuntimeError("Smoke parity/chiral/local gradient flow absent")
        if self.max_updates == 1000:
            if not self.runtime.path("preflight/smoke_100/update_101_replay.json").exists():
                raise RuntimeError("Exact smoke100-to101 resume proof missing")
            self.evaluate("smoke_1000", full=False)
        path = self.save(f"smoke_{self.max_updates}_complete")
        self.heartbeat("smoke_complete", combined, gradient, residues)
        report = dict(
            passed=True,
            contract=self.runtime.contract(self.calibration),
            updates=self.update,
            checkpoint=str(path),
            exposures=self.exposures,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            all_components=True,
            self_conditioning_fraction=fraction,
            rollout_fraction=self.engine.total_rollout / self.engine.total_batches,
            snr_histogram=self.engine.sampler.histogram.tolist(),
            pcgrad_conflict_fraction=self.conflicts / self.update,
            summaries=self.summaries,
            precision=self.engine.precision,
            fp32_geometry_islands=True,
            no_skipped_or_duplicate_identity=True,
            disk_reserve_gate=False,
        )
        if self.max_updates == 100:
            boundaries = [math.ceil(len(self.seen) * f) for f in self.cfg["checkpoint_fractions"]]
            stop = next(x for x in boundaries if x > self.cursor)
            groups, _, _ = plan_update(
                self.runtime.train, self.flat, self.ends, self.cursor, stop, self.cfg["residue_budget"]
            )
            try:
                observations, grad = self.execute(groups, sum(map(len, groups)), capture=True)
                expected = replay_snapshot(self.engine, observations, grad)
                self.runtime.write(
                    "preflight/smoke_100/expected_update_101.json", dict(groups=groups, snapshot=expected)
                )
            finally:
                # The replay preview is uncommitted. Restore canonical update100
                # even if a diagnostic write fails, preserving valid accounting.
                saved = load_checkpoint(self.directory, self.contract)
                self.engine.model.load_state_dict(saved["model"])
                self.engine.ema.load_state_dict(saved["ema"])
                self.engine.optimizer.load_state_dict(saved["optimizer"])
                self.engine.scaler.load_state_dict(saved["scaler"])
                self.engine.restore_rng(saved["engine_rng"])
                restore_rng(saved["rng"])
                self.engine.capture_replay = False
        self.runtime.write(f"preflight/smoke_{self.max_updates}/PASSED.json", report)

    def run(self):
        self.initial_exposures, self.processed_residues = self.exposures, 0
        if e007_active(self.cfg):
            raise RuntimeError("E007 active; E015 training forbidden")
        if self.update == 0:
            if not (self.directory / "initialization.pt").exists():
                self.save("initialization")
            if self.mode == "production":
                self.evaluate("initialization")
        if self.update == 0:
            self.check_symmetry()
        self.heartbeat("training_" + self.mode)
        while self.epoch <= self.cfg["maximum_epochs"]:
            boundaries = [math.ceil(len(self.seen) * f) for f in self.cfg["checkpoint_fractions"]]
            boundary = next((x for x in boundaries if x > self.cursor), len(self.seen))
            if self.mode == "production" and self.cursor in boundaries:
                label = f"epoch_{self.epoch}_fraction_{self.cursor / len(self.seen):.2f}"
                if label not in self.completed_evaluations:
                    self.evaluate(label, full=self.cursor == len(self.seen))
                    self.save("recovery_0", rolling=True)
            if self.cursor == len(self.seen):
                if self.epoch >= 2:
                    from protein_distance_diffusion.evaluation.e015_evaluation import continue_after_epoch

                    if self.mode == "production" and not continue_after_epoch(self.runtime, self.folder, self.epoch):
                        break
                if self.epoch >= self.cfg["maximum_epochs"]:
                    break
                self.epoch += 1
                self.cursor = 0
                self.flat, self.ends = identity_schedule(
                    self.runtime.train, self.batching["batch_sizes"], self.cfg["seed"], self.epoch
                )
                continue
            if self.stopping or e007_active(self.cfg):
                self.save(f"guard_stop_update_{self.update}")
                self.heartbeat("paused_guard")
                return
            # Owner amendment: free space is recorded, never an admission gate.
            self.runtime.disk_check()
            groups, new_cursor, residues = plan_update(
                self.runtime.train, self.flat, self.ends, self.cursor, boundary, self.cfg["residue_budget"]
            )
            proteins = sum(len(g) for g in groups)
            verify_101 = self.mode != "production" and self.update == 100
            observations, gradient = self.execute(groups, proteins, capture=verify_101)
            if verify_101:
                expected = json.loads(self.runtime.path("preflight/smoke_100/expected_update_101.json").read_text())
                actual = replay_snapshot(self.engine, observations, gradient)
                if expected["snapshot"] != actual or expected["groups"] != groups:
                    self.runtime.write(
                        "preflight/smoke_100/update_101_replay_failure.json",
                        dict(expected=expected, actual=actual, groups=groups),
                    )
                    raise RuntimeError("Smoke resume into update101 is not bitwise exact")
                self.runtime.write(
                    "preflight/smoke_100/update_101_replay.json", dict(passed=True, bitwise_exact=True, snapshot=actual)
                )
                self.engine.capture_replay = False
            self.conflicts += int(gradient["conflict"])
            self.update += 1
            for group in groups:
                self.seen[group] += 1
            self.cursor = new_cursor
            self.exposures += proteins
            self.processed_residues += residues
            self.audit()
            self.record_observations(observations, gradient, groups, residues)
            if self.update % 100 == 0:
                self.check_symmetry()
            free, _ = torch.cuda.mem_get_info()
            total = torch.cuda.get_device_properties(0).total_memory
            if (
                torch.cuda.max_memory_reserved() > self.cfg["preflight"]["hard_reserved_fraction"] * total
                or free < 512 * 1024**2
            ):
                self.save(f"memory_guard_update_{self.update}")
                self.heartbeat("paused_memory_guard")
                return
            combined = dict(observations[-1])
            combined["v_loss"] = sum(o["v_loss"] * len(g) / proteins for o, g in zip(observations, groups, strict=True))
            combined["struct_loss"] = sum(
                o["struct_loss"] * len(g) / proteins for o, g in zip(observations, groups, strict=True)
            )
            combined["total_loss"] = combined["v_loss"] + combined["struct_loss"]
            combined["geometry"] = {
                name: sum(o["geometry"][name] * len(g) / proteins for o, g in zip(observations, groups, strict=True))
                for name in combined["geometry"]
            }
            if self.update % 5 == 0:
                self.heartbeat("training_" + self.mode, combined, gradient, residues)
            if self.cursor == boundary:
                fraction = self.cursor / len(self.seen)
                label = f"epoch_{self.epoch}_fraction_{fraction:.2f}"
                self.save(label)
                if self.mode == "production":
                    self.evaluate(label, full=self.cursor == len(self.seen))
            elif time.monotonic() - self.last_save >= self.cfg["rolling_checkpoint_seconds"]:
                self.save("recovery_0", rolling=True)
            if self.max_updates and self.update >= self.max_updates:
                self.finish_smoke(combined, gradient, residues)
                return
        self.save(f"final_epoch_{self.epoch}_update_{self.update}")
        self.heartbeat("complete")
