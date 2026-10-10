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
from protein_distance_diffusion.training.e015_runtime import GPULease, e007_active, status


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


class Trainer:
    def __init__(self, runtime, mode="production", max_updates=None, resume=False):
        self.runtime, self.cfg, self.mode = runtime, runtime.cfg, mode
        self.lease = GPULease(runtime)
        self.calibration = json.loads(runtime.path("calibration/frozen.json").read_text())
        self.batching = json.loads(runtime.path("batch_autotuning.json").read_text())
        self.contract = runtime.contract(self.calibration)
        self.contract["batching_hash"] = canonical_hash(self.batching)
        self.folder = "production" if mode == "production" else f"preflight/smoke_{max_updates}"
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
            gpu_allocated_bytes=torch.cuda.memory_allocated(),
            gpu_reserved_bytes=torch.cuda.memory_reserved(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
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

    def run(self):
        self.initial_exposures, self.processed_residues = self.exposures, 0
        if e007_active(self.cfg):
            raise RuntimeError("E007 active; E015 training forbidden")
        if self.update == 0:
            if not (self.directory / "initialization.pt").exists():
                self.save("initialization")
            if self.mode == "production":
                self.evaluate("initialization")
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
            try:
                self.runtime.disk_check()
            except RuntimeError:
                # No new step is admitted. Existing 16 GiB budget bounds checkpoint size.
                self.save(f"disk_guard_update_{self.update}")
                self.heartbeat("paused_disk_reserve")
                return
            groups, new_cursor, residues = plan_update(
                self.runtime.train, self.flat, self.ends, self.cursor, boundary, self.cfg["residue_budget"]
            )
            proteins = sum(len(g) for g in groups)
            self.engine.begin()
            observations = []
            for ids in groups:
                rows = [self.runtime.train[i] for i in ids]
                x, n, m = coordinate_batch(rows, self.runtime.cache, self.cfg["coordinate_scale_angstrom"])
                observation = self.engine.microbatch(
                    x,
                    n,
                    m,
                    self.exposures,
                    len(ids) / proteins,
                    force_sc=True if self.mode != "production" else None,
                    force_rollout=True if self.mode != "production" else None,
                    force_rollout_steps=3 if self.mode != "production" else None,
                )
                observations.append(observation)
            gradient = self.engine.finish()
            self.conflicts += int(gradient["conflict"])
            self.update += 1
            for group in groups:
                self.seen[group] += 1
            self.cursor = new_cursor
            self.exposures += proteins
            self.processed_residues += residues
            self.audit()
            free, _ = torch.cuda.mem_get_info()
            total = torch.cuda.get_device_properties(0).total_memory
            if (
                torch.cuda.max_memory_reserved() > self.cfg["preflight"]["hard_reserved_fraction"] * total
                or free < 512 * 1024**2
            ):
                self.save(f"memory_guard_update_{self.update}")
                self.heartbeat("paused_memory_guard")
                return
            combined = observations[-1]
            combined["v_loss"] = sum(o["v_loss"] * len(g) / proteins for o, g in zip(observations, groups, strict=True))
            combined["struct_loss"] = sum(
                o["struct_loss"] * len(g) / proteins for o, g in zip(observations, groups, strict=True)
            )
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
                path = self.save(f"smoke_{self.max_updates}_complete")
                self.heartbeat("smoke_complete", combined, gradient, residues)
                self.runtime.write(
                    self.folder + "/PASSED.json",
                    dict(
                        passed=True,
                        updates=self.update,
                        checkpoint=str(path),
                        exposures=self.exposures,
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        all_components=True,
                    ),
                )
                return
        self.save(f"final_epoch_{self.epoch}_update_{self.update}")
        self.heartbeat("complete")
