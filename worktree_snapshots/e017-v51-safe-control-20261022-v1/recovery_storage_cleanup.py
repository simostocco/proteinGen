"""Narrow, receipt-backed removal of installed-package HTTP wheel downloads only."""
# Report/receipt prose keeps complete explanations on one source line.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import time
import zipfile
from pathlib import Path

RUN = Path("/mnt/d/Simone/proteinGen_data/e017_v51_safe_control_20261022_v1")
OUT = RUN / "recovery_storage_20261022"
OWNER = Path(
    "/mnt/d/Simone/proteinGen/e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1/external_designability_loop_v1"
)
CACHE = OWNER / "cache/pip_cache_dir/http-v2"
SITE = OWNER / "tools/colabfold_env/lib/python3.12/site-packages"


def sha(path):
    path = Path(path)
    if path.stat().st_size >= 16 * 1024**2:
        assert str(path).startswith("/mnt/d/")
        native = "D:\\" + str(path)[7:].replace("/", "\\")
        command = (
            "(Get-FileHash -LiteralPath '" + native.replace("'", "''") + "' -Algorithm SHA256).Hash.ToLowerInvariant()"
        )
        value = subprocess.check_output(["powershell.exe", "-NoProfile", "-Command", command], text=True).strip()
        assert len(value) == 64 and all(x in "0123456789abcdef" for x in value)
        return value
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def save(name, value):
    p = OUT / name
    tmp = p.with_suffix(".pending")
    with tmp.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def free():
    s = os.statvfs("/mnt/d")
    return s.f_bavail * s.f_frsize


def identity(p):
    assert p.is_relative_to(CACHE)
    for ancestor in [p, *p.parents]:
        assert not ancestor.is_symlink(), str(ancestor)
        if ancestor == CACHE:
            break
    st = p.lstat()
    assert stat.S_ISREG(st.st_mode) and st.st_nlink == 1
    return {
        "path": str(p),
        "device": st.st_dev,
        "inode": st.st_ino,
        "size": st.st_size,
        "allocated_bytes_drvfs": st.st_blocks * 512,
        "nlink": st.st_nlink,
        "mtime_ns": st.st_mtime_ns,
        "sha256": sha(p),
    }


def idle():
    # The inventory process is allowed; no research or package-install worker is.
    rows = subprocess.check_output(["ps", "-eo", "pid,ppid,comm,args"], text=True).splitlines()
    conflicts = []
    excluded = {os.getpid(), os.getppid()}
    for line in rows[1:]:
        parts = line.split(None, 3)
        if len(parts) < 4 or int(parts[0]) in excluded:
            continue
        if parts[2] in ("python", "python3", "pip", "pip3", "mmseqs", "conda"):
            if "unattended-upgrade" not in parts[3]:
                conflicts.append(line)
    assert not conflicts, conflicts
    native = subprocess.check_output(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            '@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match "python|pip|conda|mmseqs|colab|fold"} | Select-Object ProcessId,Name,CommandLine) | ConvertTo-Json -Compress',
        ],
        text=True,
    ).strip()
    assert not json.loads(native or "[]"), native
    return {"unix_time": time.time(), "research_or_install_conflicts": conflicts, "native_conflicts": []}


def plan():
    OUT.mkdir(exist_ok=True)
    toolchain = json.loads((OWNER / "toolchain_manifest.json").read_text())
    assert Path(toolchain["ColabFold"]["environment"]) == SITE.parents[2]
    manifest = json.loads((OWNER / "MANIFEST.json").read_text())
    assert not any(x["path"].startswith("cache/pip_cache_dir/") for x in manifest["files"])
    evidence = idle()
    receipts = []
    skipped = []
    for body in sorted(CACHE.rglob("*.body")):
        if body.stat().st_size < 1024**2:
            continue
        try:
            with zipfile.ZipFile(body) as z:
                metas = [n for n in z.namelist() if n.endswith(".dist-info/METADATA")]
                if len(metas) != 1:
                    raise ValueError("not a single wheel distribution")
                installed = SITE / metas[0]
                metadata = z.read(metas[0])
                if not installed.is_file() or metadata != installed.read_bytes():
                    raise ValueError("no byte-identical installed distribution metadata")
                record = installed.with_name("RECORD")
                if not record.is_file():
                    raise ValueError("installed distribution RECORD absent")
                detail = {
                    "wheel_distribution": metas[0].split("/")[0],
                    "installed_metadata": str(installed),
                    "installed_metadata_sha256": sha(installed),
                    "installed_RECORD": str(record),
                    "installed_RECORD_sha256": sha(record),
                    "reason": "Completed HTTP wheel download cache; same distribution installed in preserved environment. Not a model/checkpoint/dataset or certified output. Regenerable by pinned package download; download cache is not retained offline-install evidence.",
                }
            receipts.append(identity(body) | detail)
            header = body.with_suffix("")
            if header.is_file():
                receipts.append(
                    identity(header)
                    | detail
                    | {"reason": "HTTP metadata paired with the disposable installed-package wheel download"}
                )
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            skipped.append({"path": str(body), "reason": str(exc)})
    assert receipts
    result = {
        "created_unix": time.time(),
        "owner": str(OWNER),
        "cache_root": str(CACHE),
        "idle_evidence": evidence,
        "manifest_sha256": sha(OWNER / "MANIFEST.json"),
        "dependency_lock": str(OWNER / "logs/colabfold_freeze.log"),
        "dependency_lock_sha256": sha(OWNER / "logs/colabfold_freeze.log"),
        "files": receipts,
        "skipped": skipped,
        "expected_reclaimed_drvfs_bytes": sum(x["allocated_bytes_drvfs"] for x in receipts),
        "scope": "Only these individually hashed cache files. Installed environments, model downloads, MSA/cache outputs, logs, datasets and all checkpoints are excluded.",
    }
    save("cleanup_plan.json", result)
    print(
        json.dumps(
            {
                "files": len(receipts),
                "expected_bytes": result["expected_reclaimed_drvfs_bytes"],
                "skipped": len(skipped),
            }
        )
    )


def execute():
    plan = json.loads((OUT / "cleanup_plan.json").read_text())
    assert plan["cache_root"] == str(CACHE)
    assert sha(OWNER / "MANIFEST.json") == plan["manifest_sha256"]
    assert sha(plan["dependency_lock"]) == plan["dependency_lock_sha256"]
    evidence = idle()
    # Validate the entire fixed list before the first unlink.
    for r in plan["files"]:
        assert identity(Path(r["path"])) == {k: r[k] for k in identity_keys()}
        assert sha(r["installed_metadata"]) == r["installed_metadata_sha256"]
        assert sha(r["installed_RECORD"]) == r["installed_RECORD_sha256"]
    before = free()
    ledger = OUT / "deletions.jsonl"
    assert not ledger.exists(), "Never replay a partially executed deletion ledger"
    with ledger.open("x") as f:
        for r in plan["files"]:
            p = Path(r["path"])
            st = p.lstat()
            assert (st.st_dev, st.st_ino, st.st_size, st.st_nlink, st.st_mtime_ns) == (
                r["device"],
                r["inode"],
                r["size"],
                1,
                r["mtime_ns"],
            )
            f.write(json.dumps({"phase": "verified_intent", **r}, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
            p.unlink()
            assert not p.exists()
            f.write(json.dumps({"phase": "unlinked", "path": str(p), "time": time.time()}, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
    assert sha(OWNER / "MANIFEST.json") == plan["manifest_sha256"]
    assert sha(plan["dependency_lock"]) == plan["dependency_lock_sha256"]
    after = free()
    result = {
        "completed_unix": time.time(),
        "deleted_files": len(plan["files"]),
        "idle_evidence": evidence,
        "expected_reclaimed_drvfs_bytes": plan["expected_reclaimed_drvfs_bytes"],
        "free_before_bytes": before,
        "free_after_bytes": after,
        "observed_free_increase_bytes": after - before,
        "ledger_sha256": sha(ledger),
        "protected_manifest_and_environment_lock_unchanged": True,
        "note": "Observed free change can include unrelated host writes; NTFS allocation and link identity are recorded separately by the native inventory.",
    }
    save("cleanup_result.json", result)
    print(json.dumps(result, indent=2))


def identity_keys():
    return ["path", "device", "inode", "size", "allocated_bytes_drvfs", "nlink", "mtime_ns", "sha256"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["plan", "execute"])
    a = p.parse_args()
    if a.stage == "plan":
        plan()
    else:
        execute()
