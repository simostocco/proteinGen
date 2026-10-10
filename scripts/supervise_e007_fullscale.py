"""Detached durable workflow: epoch 1, diagnostics, epoch 2, matched adjudication."""

# ruff: noqa: I001, E501
from e007_fullscale_runtime import RUN, CORRECTED, CONFIG, MODEL_CONFIG, verified_inputs, atomic_json, sha, yaml
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def prepare():
    cfg, population, _, _ = verified_inputs()
    profile = json.loads((RUN / "compute_profile.json").read_text())
    assert profile["passed"]
    if not (RUN / "training_manifest.json").exists():
        from protein_distance_diffusion.models.e014_e007_control import E007ScaleupControl

        model = E007ScaleupControl(**yaml.safe_load(MODEL_CONFIG.read_text())["model"])
        assert sum(p.numel() for p in model.parameters()) == 7586505
        protected = {str(p): sha(p) for p in sorted(Path("src/protein_distance_diffusion").rglob("*.py"))}
        protected.update({str(p): sha(p) for p in (CONFIG, MODEL_CONFIG, Path("scripts/run_e014_scaleup.py"))})
        atomic_json(RUN / "PROTECTED_INPUT_HASHES.json", protected)
        manifest = dict(
            experiment="E007_matched_fullscale_v1",
            architecture="unchanged historical EquivariantPairCoordinateUNet",
            parameters=7586505,
            initialization="from scratch",
            seed=14014,
            train_population=population["train_sample_id_hash"],
            TRAIN=230132,
            development_panel=population["development_sample_id_hash"],
            normalization=12.22820347644835,
            normalization_hash=cfg["normalization_sha256"],
            diffusion_steps=500,
            parameterization="centered coordinate-v",
            config_sha256=sha(CONFIG),
            model_config_sha256=sha(MODEL_CONFIG),
            minimum_epochs=2,
            maximum_epochs=2,
            primary_exposures=460264,
            optimizer=cfg["optimizer"],
            betas=[0.9, 0.999],
            epsilon=1e-8,
            scheduler=None,
            EMA=False,
            AMP=False,
            TF32=False,
            physical_batches=[4, 2, 1, 1, 1],
            accumulation=1,
            equal_protein_loss=True,
            training_noise_namespace="exact e014-train-epoch-{epoch}",
            optimizer_matching="Frozen matched protocol overrides historical E007 lr0.0003/coordinate-pooled batch loss; matched E014 lr0.0002/equal-protein loss",
            allocator="expandable_segments:True",
            old_primary_completion_gate_overridden_by_explicit_user_instruction=True,
            old_5GiB_comfort_gate=False,
            actual_512MiB_headroom_preflight_passed=True,
            prospective_allowed=False,
            full_sampling_allowed=False,
            sequence_state=str(CORRECTED / "production/STATE.json"),
            cpu_threads=1,
            nice=10,
            checkpoint_fractions=[0.25, 0.5, 0.75, 1.0],
            rolling_seconds=900,
            startup_extra_checkpoint_update=20,
            git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            classification_rule="Familywise clear geometry plus chirality advantage: A, or C if E014 wins v-MSE. E014 clear v/geometry/chirality wins: B. D/mixed cases require scientific review; nonsignificance is not equivalence.",
        )
        atomic_json(RUN / "training_manifest.json", manifest)
        (RUN / "REPORT.md").write_text(
            "# Matched full-data E007 control\n\nStatus: authorized and CUDA-preflight passed; scientific results pending two complete epochs.\n\nThe historical model is unchanged (7,586,505 parameters). TRAIN hash: "
            + population["train_sample_id_hash"]
            + ". Normalization: 12.22820347644835 Å. Frozen shared AdamW lr0.0002, no scheduler/EMA/AMP, clip1, batches4/2/1/1/1, accumulation1. All outputs are on D:. E014/E012 scientific inputs are read-only.\n\nWorkflow: epoch1 and all fraction validations; frozen epoch1 field/marginal/Gaussian diagnostics; exact resume to epoch2; epoch2 diagnostics and 5,000-draw paired comparison. No epoch3, prospective holdout or t=0 generation.\n\nThe old final-stop gate is superseded by the owner's explicit control instruction. Default allocator failed the unchanged 512 MiB headroom check; expandable segments passed without changing scientific semantics. Both profiles are preserved.\n"
        )


def workflow(resume):
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    lock = open(RUN / "SUPERVISOR_OWNER.lock", "a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if resume and (RUN / "STATE.json").exists():
        state = json.loads((RUN / "STATE.json").read_text())
        if state.get("phase", "").startswith("paused_"):
            state["phase"] = "explicit_resume_requested"
            atomic_json(RUN / "STATE.json", state)
    logdir = RUN / "logs"
    logdir.mkdir(exist_ok=True)
    while True:
        state = json.loads((RUN / "STATE.json").read_text()) if (RUN / "STATE.json").exists() else {}
        phase = state.get("phase", "new")
        if phase == "complete_epoch2_evaluated":
            return
        if phase.startswith("paused_"):
            atomic_json(
                RUN / "SUPERVISOR.json",
                dict(pid=os.getpid(), phase=phase, training_active=False, requires_explicit_resume=True),
            )
            return
        completed = (
            2
            if phase == "epoch2_training_complete_awaiting_comparison"
            else 1
            if phase == "epoch1_complete_awaiting_diagnostics"
            else None
        )
        if completed:
            command = [sys.executable, "-B", "scripts/evaluate_e007_fullscale.py", str(completed)]
        else:
            command = [sys.executable, "-B", "scripts/run_e007_fullscale.py"]
            if resume or (RUN / "checkpoints/LATEST.json").exists():
                command.append("--resume")
        with open(logdir / f"workflow_{time.time_ns()}.log", "a", buffering=1) as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            atomic_json(
                RUN / "SUPERVISOR.json",
                dict(
                    pid=os.getpid(),
                    child_pid=process.pid,
                    command=command,
                    log=log.name,
                    phase="diagnostics" if completed else "training",
                    timestamp_unix=time.time(),
                ),
            )
            result = process.wait()
        if result:
            atomic_json(
                RUN / "SUPERVISOR.json",
                dict(
                    pid=os.getpid(), phase="stopped_child_error", returncode=result, log=log.name, training_active=False
                ),
            )
            return
        if completed == 2:
            state.update(phase="complete_epoch2_evaluated", timestamp_unix=time.time())
            atomic_json(RUN / "STATE.json", state)
            atomic_json(
                RUN / "SUPERVISOR.json",
                dict(pid=os.getpid(), phase="complete", training_active=False, epoch3_started=False),
            )
            with open(RUN / "REPORT.md", "a") as report:
                report.write(
                    "\nTwo complete epochs and matched evaluation finished. See [epoch-2 comparison](epoch2_comparison/REPORT.md).\n"
                )
            return
        if completed == 1:
            # Production resumes the committed end-of-epoch checkpoint, then
            # advances epoch/cursor only after proving complete coverage.
            with open(logdir / f"resume_epoch2_{time.time_ns()}.log", "a", buffering=1) as log:
                p = subprocess.Popen(
                    [sys.executable, "-B", "scripts/run_e007_fullscale.py", "--resume"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                atomic_json(
                    RUN / "SUPERVISOR.json",
                    dict(pid=os.getpid(), child_pid=p.pid, phase="training_epoch2", log=log.name),
                )
                result = p.wait()
                if result:
                    atomic_json(
                        RUN / "SUPERVISOR.json", dict(phase="stopped_child_error", returncode=result, log=log.name)
                    )
                    return
        resume = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    prepare()
    if args.launch:
        (RUN / "logs").mkdir(exist_ok=True)
        with open(RUN / "logs/supervisor.log", "a", buffering=1) as log:
            command = [sys.executable, "-B", str(Path(__file__).resolve())]
            if args.resume:
                command.append("--resume")
            p = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        atomic_json(
            RUN / "LAUNCH.json", dict(supervisor_pid=p.pid, root=str(RUN), command=command, timestamp_unix=time.time())
        )
        print(json.dumps(dict(supervisor_pid=p.pid, root=str(RUN))))
    else:
        workflow(args.resume)
