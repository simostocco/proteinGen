# ruff: noqa: E501
"""One final serialized card-layout correction; no scientific recomputation."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

import e015_latest_control as c

r = c.r


def main(out, previous_owner=675787):
    out = Path(out)
    boundary = r.read(r.ROOT / "production/checkpoints/LATEST.json")
    while c.alive(previous_owner):
        time.sleep(10)
    before = r.read(out / "recovery_resume_proof.json")
    assert len(before["completed"]) == 5
    r.write(out / "logs/resume_before_card_layout_correction.json", before)
    owner = r.read(r.ROOT / "SUPERVISOR.json")
    boundary = before["selected"]
    trainer = owner["child_pid"]
    while True:
        latest = r.read(r.ROOT / "production/checkpoints/LATEST.json")
        eligible = (
            Path(latest["path"]).name.startswith("epoch_")
            or Path(latest["path"]).stat().st_mtime >= owner["timestamp_unix"] + 1800
        )
        if (
            time.time() - owner["timestamp_unix"] >= 1800
            and latest["position"]["update"] > boundary["position"]["update"]
            and eligible
        ):
            break
        time.sleep(15)
    request = out / "logs/card_layout_pause"
    request.mkdir(exist_ok=True)
    r.write(request / "launch.json", dict(supervisor_pid=owner["pid"], timestamp_unix=owner["timestamp_unix"]))
    r.write(request / "recovery_audit.json", dict(selected=boundary))
    r.pause(request)
    while c.alive(trainer) or c.alive(owner["pid"]):
        time.sleep(5)
    c.gate()
    audit = c.run_audit(out / "logs", "card_layout_pause_audit")
    pause = r.read(audit / "recovery_audit.json")["selected"]
    r.write(
        out / "logs/card_layout_pause.json", dict(selected=pause, zero_uncommitted_exposures=True, audit=str(audit))
    )
    error = None
    try:
        with (out / "logs/card_layout_report.log").open("a") as log:
            subprocess.run(
                [c.PY, "-B", str(r.WORK / "docs/e015_latest_evaluation.py"), "report", str(out)],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        assert (out / "logs/cards_layout_verified.json").exists()
        print(json.dumps(dict(card_layout_complete=True)), flush=True)
    except Exception as exc:
        error = str(exc)
    finally:
        c.gate()
        resume_audit = c.run_audit(out / "logs", "card_layout_resume_audit")
        assert r.read(r.ROOT / "production/checkpoints/LATEST.json") == pause
        command = r.read(resume_audit / "recovery_audit.json")["command"]
        with (resume_audit / "supervisor.log").open("a") as log:
            supervisor = subprocess.Popen(
                command,
                cwd=r.WORK,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        r.write(
            resume_audit / "launch.json",
            dict(supervisor_pid=supervisor.pid, command=command, timestamp_unix=time.time()),
        )
        r.observe(resume_audit, 1200)
        latest = r.read(r.ROOT / "production/checkpoints/LATEST.json")
        h = hashlib.sha256()
        with Path(latest["path"]).open("rb") as f:
            while block := f.read(8 * 1024**2):
                h.update(block)
        assert latest["position"]["update"] > pause["position"]["update"] and h.hexdigest() == latest["sha256"]
        r.write(
            out / "recovery_resume_proof.json",
            dict(
                selected=pause,
                primary_evaluation=r.read(out / "checkpoint_selection.json"),
                resumed=True,
                training_contract_complete=False,
                audit=str(resume_audit),
                first_operation=r.read(resume_audit / "first_resumed_heartbeat.json"),
                rolling=latest,
                rolling_sha_verified=True,
                supervisor_pid=supervisor.pid,
                trainer_pid=r.read(r.ROOT / "STATE.json")["pid"],
                completed=before["completed"],
                error=error,
                card_layout_corrected=error is None,
            ),
        )
        print(json.dumps(dict(phase="final_card_cycle_complete", error=error)), flush=True)
