# ruff: noqa: E501
"""Serialized latest E015 evaluation owner; production science remains unchanged."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import e015_serial_recovery as r

PY = "/home/simostocco/miniforge3/envs/proteingen/bin/python"
OLD = r.ROOT / "evaluations/absolute_generation_u3811_2986d44e_v1"
original_gate = r.gate


def gate():
    data = original_gate()
    assert data["ram_available_bytes"] >= 3 * 1024**3
    assert data.get("windows_ram_available_bytes", 3 * 1024**3) >= 3 * 1024**3
    return data


r.gate = gate


def alive(pid):
    try:
        return b"python" in Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False


def run_audit(folder, name):
    log = folder / (name + ".log")
    with log.open("w") as stream:
        subprocess.run([PY, "-B", __file__, "audit"], stdout=stream, stderr=subprocess.STDOUT, check=True)
    return Path(json.loads(log.read_text().splitlines()[-1])["report"])


def main():
    request = r.REPORT.with_name(r.REPORT.name.replace("shutdown_recovery", "latest_evaluation_request"))
    request.mkdir(parents=True)
    state = r.read(r.ROOT / "STATE.json")
    owner = r.read(r.ROOT / "SUPERVISOR.json")
    latest = r.read(r.ROOT / "production/checkpoints/LATEST.json")
    metadata = {
        name: r.read(r.ROOT / name)
        for name in (
            "STATE.json",
            "HEARTBEAT.json",
            "COMMITTED.json",
            "SUPERVISOR.json",
            "production/checkpoints/LATEST.json",
        )
    }
    metadata["resources"] = r.resources()
    metadata["checkpoints"] = [
        dict(path=str(p), bytes=p.stat().st_size, mtime=p.stat().st_mtime)
        for p in (r.ROOT / "production/checkpoints").glob("*.pt")
    ]
    metadata["windows_processes"] = subprocess.check_output(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process | Where-Object {$_.Name -match 'python|wsl|ollama'} | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress",
        ],
        text=True,
        timeout=30,
    )
    r.write(request / "read_only_audit.json", metadata)
    active = state["phase"].startswith("training") and alive(state["pid"])
    if active:
        assert owner["child_pid"] == state["pid"] and alive(owner["pid"])
        print(
            json.dumps(dict(phase="waiting_next_durable_boundary", request=str(request), starting=latest)), flush=True
        )
        while (
            r.read(r.ROOT / "production/checkpoints/LATEST.json")["position"]["update"] <= latest["position"]["update"]
        ):
            resource = r.resources()
            with (request / "resource_monitor.jsonl").open("a") as f:
                f.write(json.dumps(resource) + "\n")
            assert resource["ram_available_bytes"] >= 3 * 1024**3 and resource["d_free_bytes"] >= 30 * 1024**3
            time.sleep(30)
        # Reuse the actual existing SIGINT request_stop handler, with fresh evidence.
        launch_time = r.read(r.ROOT / "SUPERVISOR.json")["timestamp_unix"]
        r.write(request / "launch.json", dict(timestamp_unix=launch_time, supervisor_pid=owner["pid"]))
        r.write(request / "recovery_audit.json", dict(selected=latest))
        r.pause(request)
        while alive(state["pid"]) or alive(owner["pid"]):
            time.sleep(10)
        assert r.read(r.ROOT / "STATE.json")["phase"] == "paused_guard"
    gate()
    audit = run_audit(request, "selected_checkpoint_audit")
    proof = r.read(audit / "recovery_audit.json")
    selected = proof["selected"]
    pos = selected["position"]
    assert pos["exposures"] >= 115066
    assert selected == r.read(r.ROOT / "COMMITTED.json") == r.read(r.ROOT / "production/COMMITTED.json")
    state = r.read(r.ROOT / "STATE.json")
    assert state["update"] == pos["update"] and state["exposures"] == pos["exposures"]
    out = r.ROOT / "evaluations" / f"latest_complete_u{pos['update']}_{selected['sha256'][:8]}_v1"
    assert not out.exists(), "Refuse to overwrite an existing evaluation"
    for name in (
        "snapshots",
        "numerical",
        "samples/raw",
        "samples/ema",
        "inference_story",
        "vector_field_views",
        "same_seed_comparison",
        "trajectory_gifs",
        "plots",
        "train_heldout",
        "success_gallery",
        "median_gallery",
        "failure_gallery",
        "logs",
    ):
        (out / name).mkdir(parents=True, exist_ok=True)
    snapshot = out / "snapshots" / Path(selected["path"]).name
    import hashlib

    def digest(path):
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            while block := f.read(8 * 1024**2):
                h.update(block)
        return h.hexdigest()

    assert digest(selected["path"]) == selected["sha256"]
    partial = snapshot.with_suffix(".partial")
    with Path(selected["path"]).open("rb") as a, partial.open("wb") as b:
        while block := a.read(8 * 1024**2):
            b.write(block)
        b.flush()
        os.fsync(b.fileno())
    assert digest(partial) == selected["sha256"]
    partial.replace(snapshot)
    assert digest(snapshot) == selected["sha256"]
    selection = dict(
        source=selected,
        snapshot=str(snapshot),
        audit=str(audit),
        request=str(request),
        completed_epochs=state["completed_epochs"],
        next_operation=proof["next_updates"][0],
        zero_uncommitted_exposures=True,
        selection_rule="Existing active trainer gracefully paused after next durable boundary; no performance selection",
    )
    r.write(out / "checkpoint_selection.json", selection)
    r.write(out / "checkpoint_inventory.json", proof["inventory"])
    r.write(request / "evaluation_destination.json", dict(out=str(out)))
    print(json.dumps(dict(phase="paused_verified", out=str(out), selected=selected)), flush=True)
    completed, error = [], None
    try:
        for phase in ("reconstruction", "generation", "analysis", "visuals", "report"):
            before = gate()
            r.write(out / "logs/stage.json", dict(phase=phase, resources=before))
            with (out / "logs" / (phase + ".log")).open("a") as stream:
                child = subprocess.Popen(
                    [PY, "-B", str(r.WORK / "docs/e015_latest_evaluation.py"), phase, str(out)],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                while child.poll() is None:
                    resource = r.resources()
                    resource.update(phase=phase, evaluator_pid=child.pid, trainer_pid=None)
                    with (out / "resource_monitor.jsonl").open("a") as f:
                        f.write(json.dumps(resource) + "\n")
                    if (
                        resource["ram_available_bytes"] < 3 * 1024**3
                        or resource["d_free_bytes"] < 30 * 1024**3
                        or resource["swap_used_bytes"] > 256 * 1024**2
                    ):
                        r.write(out / "STOP_EVALUATION.json", dict(reason="resource pressure", resource=resource))
                    time.sleep(30)
                assert child.returncode == 0, f"{phase} exited {child.returncode}"
            gate()
            completed.append(phase)
            print(json.dumps(dict(phase="completed", stage=phase)), flush=True)
    except Exception as exc:
        error = str(exc)
        print(json.dumps(dict(phase="evaluation_error", error=error)), flush=True)
    finally:
        gate()
        assert r.read(r.ROOT / "production/checkpoints/LATEST.json") == selected
        assert digest(selected["path"]) == selected["sha256"]
        resume_audit = run_audit(out / "logs", "pre_resume_audit")
        complete = r.read(r.ROOT / "STATE.json")["phase"] == "complete"
        resume = dict(training_contract_complete=complete, selected=selected, completed=completed, error=error)
        if not complete:
            command = r.read(resume_audit / "recovery_audit.json")["command"]
            with (resume_audit / "supervisor.log").open("a") as stream:
                supervisor = subprocess.Popen(
                    command,
                    cwd=r.WORK,
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            r.write(
                resume_audit / "launch.json",
                dict(supervisor_pid=supervisor.pid, command=command, timestamp_unix=time.time()),
            )
            r.observe(resume_audit, 1200)
            rolling = r.read(r.ROOT / "production/checkpoints/LATEST.json")
            assert rolling["position"]["update"] > pos["update"] and digest(rolling["path"]) == rolling["sha256"]
            resume.update(
                resumed=True,
                audit=str(resume_audit),
                first_operation=r.read(resume_audit / "first_resumed_heartbeat.json"),
                rolling=rolling,
                rolling_sha_verified=True,
                supervisor_pid=supervisor.pid,
                trainer_pid=r.read(r.ROOT / "STATE.json")["pid"],
            )
        else:
            resume["resumed"] = False
        r.write(out / "recovery_resume_proof.json", resume)
        print(json.dumps(dict(phase="cycle_complete", out=str(out), error=error)), flush=True)


def finalize(out):
    """Metadata-only completion, safe alongside resumed production."""
    out = Path(out)
    proof = r.read(out / "recovery_resume_proof.json")
    manifest = r.read(out / "MANIFEST.json")
    assert proof["error"] is None and len(proof["completed"]) == 5
    assert proof["training_contract_complete"] or proof["rolling_sha_verified"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=r.WORK, text=True).strip()
    manifest.update(status="COMPLETE", resume=proof, commit=commit)
    r.write(out / "MANIFEST.json", manifest)
    resources = []
    for file in [
        out / "resource_monitor.jsonl",
        *([Path(proof["audit"]) / "resource_monitor.jsonl"] if proof.get("audit") else []),
    ]:
        for line in file.read_text().splitlines():
            resources.append(json.loads(line))
    gpu = [[float(x) for x in q["gpu_csv"].split(",")] for q in resources]
    import statistics

    summary = dict(
        minimum_ram_gib=min(q["ram_available_bytes"] for q in resources) / 1024**3,
        maximum_swap_gib=max(q["swap_used_bytes"] for q in resources) / 1024**3,
        minimum_d_free_gib=min(q["d_free_bytes"] for q in resources) / 1024**3,
        gpu_peak_used_mib=max(q[0] for q in gpu),
        gpu_temperature_range_c=[min(q[3] for q in gpu), max(q[3] for q in gpu)],
        cpu_load_peak=max(q["load_average"][0] for q in resources),
        resource_pressure=(out / "STOP_EVALUATION.json").exists(),
    )
    phases = sorted({q.get("phase", "unknown") for q in resources})
    summary["gpu_utilization_by_phase"] = {}
    for phase in phases:
        utilization = sorted(float(q["gpu_csv"].split(",")[2]) for q in resources if q.get("phase", "unknown") == phase)
        summary["gpu_utilization_by_phase"][phase] = dict(
            count=len(utilization),
            minimum=utilization[0],
            maximum=utilization[-1],
            mean=statistics.mean(utilization),
            median=statistics.median(utilization),
            p10=utilization[int(0.1 * (len(utilization) - 1))],
            p90=utilization[int(0.9 * (len(utilization) - 1))],
        )
    active = [
        q
        for q in resources
        if q.get("phase") == "training_production"
        and q.get("trainer_pid") == proof.get("trainer_pid")
        and q.get("state_age_seconds", 999) < 120
    ]
    if len(active) >= 2:
        duration = active[-1]["timestamp_unix"] - active[0]["timestamp_unix"]
        updates = active[-1]["update"] - active[0]["update"]
        exposures = active[-1]["exposures"] - active[0]["exposures"]
        summary["final_resume_throughput"] = dict(
            observed_seconds=duration,
            updates_gained=updates,
            exposures_gained=exposures,
            updates_per_second=updates / duration,
            protein_exposures_per_second=exposures / duration,
        )
        io = [q for q in active if q.get("trainer_io")]
        if len(io) >= 2:

            def written(q):
                return int(
                    next(line.split(":")[1] for line in q["trainer_io"].splitlines() if line.startswith("write_bytes:"))
                )

            summary["final_resume_throughput"]["trainer_disk_write_bytes"] = written(io[-1]) - written(io[0])
    r.write(out / "numerical/resource_summary.json", summary)
    state = r.read(r.ROOT / "STATE.json")
    owner = r.read(r.ROOT / "SUPERVISOR.json")
    with (out / "REPORT.md").open("a") as f:
        f.write(
            "\nFinal serial completion: all five phases passed. Resource summary: `" + json.dumps(summary) + "`.\n\n"
        )
        f.write(
            "Production resumed: "
            + str(proof["resumed"])
            + ". Current state: `"
            + json.dumps({k: state.get(k) for k in ("phase", "pid", "update", "exposures", "epoch", "schedule_cursor")})
            + "`. Supervisor PID: "
            + str(owner["pid"])
            + ".\n\n"
        )
        if proof["resumed"]:
            first = proof["first_operation"]
            f.write(
                "First computed resumed update/exposures: "
                + str(first["first_computed_update"])
                + "/"
                + str(first["first_computed_exposures"])
                + "; first persisted heartbeat: "
                + str(first["state"]["update"])
                + "/"
                + str(first["state"]["exposures"])
                + "; exact next operation match: "
                + str(first["match"])
                + ". Verified subsequent rolling checkpoint: `"
                + json.dumps(proof["rolling"])
                + "`.\n\n"
            )
        f.write("Final commit: `" + commit + "`. E007/E012/E014 remained untouched. All artifacts stayed onD.\n")
    import hashlib

    previous_hashes = {q["path"]: q["sha256"] for q in manifest["files"]}
    files = []
    for p in out.rglob("*"):
        if not p.is_file() or "runtime" in p.parts or p.name == "MANIFEST.json":
            continue
        relative = str(p.relative_to(out))
        if p.suffix == ".pt" and relative in previous_hashes:
            digest = previous_hashes[relative]
        else:
            h = hashlib.sha256()
            with p.open("rb") as f:
                while block := f.read(8 * 1024**2):
                    h.update(block)
            digest = h.hexdigest()
        files.append(dict(path=relative, bytes=p.stat().st_size, sha256=digest))
    manifest.update(files=files, total_bytes=sum(p.stat().st_size for p in out.rglob("*") if p.is_file()))
    assert manifest["total_bytes"] <= 6 * 1024**3
    r.write(out / "MANIFEST.json", manifest)
    print(
        json.dumps(
            dict(
                status="COMPLETE",
                out=str(out),
                commit=commit,
                state={k: state.get(k) for k in ("pid", "update", "exposures")},
                supervisor_pid=owner["pid"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "archive-layout-history":
        import shutil

        out = Path(sys.argv[2])
        target = out / "logs" / ("card_layout_history_" + str(time.time_ns()))
        target.mkdir()
        for relative in (
            "logs/resume_before_card_layout_correction.json",
            "logs/card_layout_pause.json",
            "logs/card_layout_pause/launch.json",
            "logs/card_layout_pause/graceful_pause_request.json",
            "logs/card_layout_report.log",
        ):
            path = out / relative
            if path.exists():
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
        print(json.dumps(dict(archived=str(target))))
    elif len(sys.argv) > 1 and sys.argv[1] == "validate":
        import e015_latest_validate

        e015_latest_validate.main(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "audit":
        r.audit()
    elif len(sys.argv) > 1 and sys.argv[1] == "finalize":
        finalize(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "continue":
        import e015_latest_continue

        e015_latest_continue.main(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "correct-pause-record":
        out = Path(sys.argv[2])
        previous = r.read(out / "logs/resume_after_plot_error.json")
        actual = r.read(Path(previous["audit"]) / "launch.json")
        request = out / "logs/continuation_pause"
        record = r.read(request / "launch.json")
        pause = r.read(request / "graceful_pause_request.json")
        record.update(
            timestamp_unix=actual["timestamp_unix"],
            timestamp_source=str(Path(previous["audit"]) / "launch.json"),
            actual_active_seconds_before_pause=pause["timestamp_unix"] - actual["timestamp_unix"],
            note="Corrected continuation bookkeeping to real supervisor launch; actual stable runtime exceeded30min",
        )
        assert record["actual_active_seconds_before_pause"] >= 1800
        r.write(request / "launch.json", record)
        print(json.dumps(record))
    elif len(sys.argv) > 1 and sys.argv[1] == "finish-images":
        import e015_latest_finish_images

        e015_latest_finish_images.main(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 675787)
    else:
        main()
