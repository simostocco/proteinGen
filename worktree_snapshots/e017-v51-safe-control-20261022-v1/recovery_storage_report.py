"""Compact recovery/storage handoff from existing receipts; no model execution."""
# Embedded report paragraphs intentionally exceed the source line limit.
# ruff: noqa: E501

from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

RUN = Path("/mnt/d/Simone/proteinGen_data/e017_v51_safe_control_20261022_v1")
OUT = RUN / "recovery_storage_20261022"
GIB = 2**30


def read(name):
    return json.loads((OUT / name).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def save(name, value):
    path = OUT / name
    temp = path.with_suffix(path.suffix + ".pending")
    with temp.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def main():
    inv = read("storage_inventory.json")
    recovery = read("experiment_recovery.json")
    cleanup = read("cleanup_result.json")
    plan = read("cleanup_plan.json")
    dirs = {r["path"]: r for r in inv["directories"]}
    assert sha(OUT / "deletions.jsonl") == cleanup["ledger_sha256"]
    for r in plan["files"]:
        assert not Path(r["path"]).exists()
        assert sha(r["installed_metadata"]) == r["installed_metadata_sha256"]
        assert sha(r["installed_RECORD"]) == r["installed_RECORD_sha256"]
    assert sha(Path(plan["owner"]) / "MANIFEST.json") == plan["manifest_sha256"]
    assert sha(plan["dependency_lock"]) == plan["dependency_lock_sha256"]
    for arm, e in recovery["arms"].items():
        assert json.loads((RUN / f"results/{arm}_STOP.json").read_text())["reason"] == e["stop_reason"]
        assert json.loads((RUN / f"checkpoints/{arm}/INDEX.json").read_text())["latest"] == e["latest"]["record"]
        for role in ["latest", "best"]:
            record = e[role]["record"]
            assert Path(record["path"]).stat().st_size == record["bytes"]
    ps = subprocess.check_output(["ps", "-eo", "pid,ppid,comm,args"], text=True)
    active = []
    for line in ps.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) < 4 or int(parts[0]) in (os.getpid(), os.getppid()):
            continue
        if parts[2] in ["python", "python3", "mmseqs", "pip", "pip3"] and "unattended-upgrade" not in parts[3]:
            active.append(line)
    assert not active, active
    cuda = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True
    ).strip()
    assert not cuda, cuda
    native = subprocess.check_output(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            '@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match "python|pip|conda|mmseqs|colab|fold"} | Select-Object ProcessId,Name,CommandLine) | ConvertTo-Json -Compress',
        ],
        text=True,
    ).strip()
    assert not json.loads(native or "[]")
    leftovers = []
    for p in sorted(Path("/tmp").glob("*v51*")):
        if p.is_file():
            leftovers.append(
                {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p), "executed_in_this_recovery": False}
            )
    extra = Path("/tmp/v51_final_extras.py")
    if extra.exists():
        dest = OUT / "unexecuted_v51_final_extras.py"
        if not dest.exists():
            shutil.copyfile(extra, dest)
        assert sha(dest) == sha(extra)
    major_specs = [
        (
            "/mnt/d/Simone/proteinGen/outputs",
            "Historical E000–E010 model outputs; legacy model checkpoints dominate.",
            "Checkpoint roles and model-selection dependencies unresolved; retain.",
        ),
        (
            "/mnt/d/Simone/proteinGen/data",
            "Original PDB and distinct processed/recovery populations.",
            "Protected inputs to E006/E007/E016/E017 and governed splits; retain.",
        ),
        (
            "/mnt/d/Simone/proteinGen/reports",
            "Scientific reports, homology/pairing SQLite audit states and provenance.",
            "Reproduction and split lineage evidence; retain.",
        ),
        (
            "/mnt/d/Simone/proteinGen/e015_artifacts",
            "E015 production/preflight/recovery, installed designability tools and outputs.",
            "All scientific/installed artifacts retained; only separately verified wheel-download cache retired.",
        ),
        (
            "/mnt/d/Simone/proteinGen_data/sequence_foundation",
            "UniRef filtering/screening/certification plus completed E012 run.",
            "Certified sources, reservoir including WAL/SHM and model evidence protected; no removals.",
        ),
        (
            "/mnt/d/Simone/proteinGen_data/e016_gpu_unified_pdb_20261015_v1",
            "Unified PDB typed tables, release and homology metadata.",
            "Governed E016/E017 dataset dependencies; retain.",
        ),
        (
            "/mnt/d/Simone/proteinGen_data/pdb_only_v3_20261013",
            "PDB-only governed sequence/geometry data and mapping.",
            "Protected research data, not interchangeable with other processed populations.",
        ),
        (
            "/mnt/d/Simone/proteinGen_data/pdb_supervised_split_50id80cov_20261016_v1",
            "Frozen supervised split manifests, homology and tables.",
            "Scientific split provenance and training dependencies; retain.",
        ),
        (
            str(RUN),
            "V5.1 latest+best complete checkpoints, matched100 inference copies, results and frozen source.",
            "All recovery/selected states and evidence retained; no resumption after scientific stops.",
        ),
        (
            "/mnt/d/Simone/state_data_v2",
            "Other project data.",
            "Outside ProteinGen scientific dependency audit; no deletions.",
        ),
        (
            "/mnt/d/Users/Simone Stocco/HMM_transformer",
            "Other project models/data.",
            "Ownership and dependencies unresolved; no deletions.",
        ),
        ("/mnt/d/Simone/rl_dual", "Other project data/models.", "Ownership and dependencies unresolved; no deletions."),
        (
            "/mnt/d/Simone/nq_hft_run",
            "Other project feature/model data.",
            "Ownership and dependencies unresolved; no deletions.",
        ),
        (
            "/mnt/d/Users/Simone Stocco/VSDatabase",
            "Other project data.",
            "Ownership and dependencies unresolved; no deletions.",
        ),
        ("/mnt/d/Downloads", "Mixed downloads.", "No file contents read and no deletions."),
        (
            "/mnt/d/Users/Simone Stocco/anaconda3",
            "Installed Windows environment/package cache.",
            "Environment and archive roles not retired; no deletions.",
        ),
        ("/mnt/d/Users/Simone Stocco/proteinGen_audits", "Historical audit evidence.", "Retain scientific evidence."),
    ]
    major = []
    for path, purpose, dependency in major_specs:
        r = dirs[path]
        deleted = sum(f["allocated_bytes_drvfs"] for f in plan["files"] if f["path"].startswith(path + "/"))
        major.append(
            {
                "path": path,
                "allocated_at_snapshot_bytes": r["allocated"],
                "unique_allocated_attributed_bytes": r["unique_allocated"],
                "logical_bytes": r["logical"],
                "files": r["files"],
                "known_removed_drvfs_allocation_bytes": deleted,
                "purpose": purpose,
                "dependency_and_reproducibility": dependency,
                "newest_mtime_utc": datetime.datetime.fromtimestamp(r["newest_unix"], datetime.UTC).isoformat(),
                "last_use_evidence": "Newest file mtime only; mtime/atime does not establish scientific obsolescence. Dependency manifests and checkpoints take precedence.",
            }
        )
    proposal_specs = [
        (
            "/mnt/d/Simone/proteinGen/outputs/e010_student_direction_observability_v1/neighbors",
            "Derived nearest-neighbor IDs/distances and tensor exports.",
            "Review exact input panel, representation hashes, scripts, seeds and final selected outputs. Regeneration/archival availability unverified; removal would affect direct reproduction.",
        ),
        (
            "/mnt/d/Simone/proteinGen/outputs/e010_student_direction_observability_v1/representations",
            "Derived immutable representation exports.",
            "Retain bound model checkpoints, source panel, masks, hashes and published statistics; first prove deterministic regeneration. Do not assume representations are interchangeable.",
        ),
        (
            "/mnt/d/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization/exports",
            "Historical derived candidate payload/key/sort exports.",
            "Material dependency for historical screening/replays. Preserve frozen sources, reservoir/WAL/SHM and receipts; establish exact replay requirements before any separate decision.",
        ),
        (
            "/mnt/d/Simone/proteinGen/outputs/recovered_full_b2_v_axial_e001_epoch1/checkpoints",
            "Historical full-state model checkpoint history.",
            "Historical selection/analysis/resume roles unresolved. Inventory named milestones, protected selections and complete-state dependencies; no current retirement authorization.",
        ),
    ]
    proposals = [
        {
            "path": p,
            "allocated_bytes": dirs[p]["allocated"],
            "unique_allocated_bytes": dirs[p]["unique_allocated"],
            "purpose": purpose,
            "required_review": review,
            "approval_required": True,
            "deleted": False,
            "guaranteed_reclaimable_now_bytes": 0,
        }
        for p, purpose, review in proposal_specs
    ]
    save(
        "review_required_proposals.json",
        {
            "not_authorized_for_deletion": True,
            "nonoverlapping_proposals": proposals,
            "protected_exclusions": "No original PDB/certified source, frozen J1, E016/E017 report/log/provenance or required recovery state is proposed for removal.",
        },
    )
    admission = {
        "actual_checkpoint_bytes": max(e["latest"]["record"]["bytes"] for e in recovery["arms"].values()),
        "checkpoint_hint_bytes": 220 * 2**20,
        "near_growth_bytes": 32 * 2**20,
        "operational_headroom_bytes": 2 * GIB,
    }
    admission["existing_policy_next_update_required_free_bytes"] = (
        admission["checkpoint_hint_bytes"] + admission["near_growth_bytes"] + admission["operational_headroom_bytes"]
    )
    admission["conservative_fresh_latest_best_incoming_required_free_bytes"] = (
        3 * admission["checkpoint_hint_bytes"]
        + admission["near_growth_bytes"]
        + admission["operational_headroom_bytes"]
    )
    result = {
        "snapshot_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "experiment_status": "COMPLETED_WITH_SCIENTIFIC_STOPS",
        "can_resume": False,
        "reason": "A/C: geometry KL>1; B/D: H4 validity retention drop>5pp, all at250. Controller refuses stopped arms. Space recovery does not authorize continuation.",
        "latest_verified_checkpoints": {arm: e["latest"]["record"] for arm, e in recovery["arms"].items()},
        "selected_checkpoints": {arm: e["best"]["record"] for arm, e in recovery["arms"].items()},
        "total_scientific_optimizer_updates_across_arms": 1000,
        "active_research_processes": active,
        "cuda_compute_owners": [],
        "campaign_and_final_audit_complete": True,
        "scientific_final_writeup_completed": False,
        "optional_final_extras_executed": False,
        "monitoring_gap": "LIVE_STATUS remained TRAINING/D250 after termination. RESOURCE_OBSERVER.jsonl is absent; observer outputs were suppressed, so its failure reason is unknown. FINISHED/STOP/INDEX and campaign/final-audit receipts are authoritative.",
        "initial_repo_tracked_changes": [],
        "new_audit_tool_files": subprocess.check_output(
            ["git", "status", "--short"], cwd=RUN / "source", text=True
        ).splitlines(),
        "unexecuted_temporary_sources": leftovers,
        "major_directory_usage": major,
        "directory_overlap_warning": "Parent and child rows overlap. Unique allocation is attributed to the first enumerated hardlink path; do not sum overlapping rows.",
        "inventory": {
            "path": str(OUT / "storage_inventory.json"),
            "sha256": sha(OUT / "storage_inventory.json"),
            "method": inv["method"],
            "files": inv["files"],
            "started_utc": inv["started_utc"],
            "finished_utc": inv["finished_utc"],
            "unique_allocated_bytes": inv["unique_allocated"],
            "hardlink_path_overcount_bytes": inv["allocated_paths"] - inv["unique_allocated"],
            "errors": inv["errors"],
            "excluded_directories": len(inv["exclusions"]),
            "limits": "Not atomic. No payload reads or TEST directory traversal. Recovery access denied and system/recycle/reparse/TEST exclusions plus NTFS metadata account for unclassified used space; these areas are not declared disposable.",
        },
        "cleanup": cleanup,
        "deleted_allocated_bytes_drvfs": plan["expected_reclaimed_drvfs_bytes"],
        "net_observed_free_increase_bytes": cleanup["observed_free_increase_bytes"],
        "allocation_difference_explained_by_new_deletion_ledger_bytes": (OUT / "deletions.jsonl").stat().st_blocks
        * 512,
        "admission": admission,
        "cleanup_target_15_to20_GiB_met": False,
        "larger_cleanup_requires_separate_review": proposals,
        "next_action": "Keep all V5.1 arms stopped; review the completed matched+100/final audit evidence and +250 safety failures. Review the derived-export proposal separately if larger storage recovery is desired. No training resume procedure is authorized.",
        "no_architecture_threshold_or_sampling_change": True,
        "TEST_accessed": False,
        "new_training_or_evaluation_started": False,
        "all_new_artifacts_on_D": True,
    }
    lines = [
        "# E017 V5.1 recovery and D: cleanup",
        "",
        f"Status: **{result['experiment_status']}**. No worker or supervisor is active. All four arms stopped at+250; final audit completed. LIVE_STATUS is stale and was preserved.",
        "",
        "| Arm | Latest verified update | Stop gate | Eligible selected checkpoint |",
        "|---|---:|---|---:|",
    ]
    for arm, e in recovery["arms"].items():
        lines.append(f"| {arm} | 250 | {e['stop_reason']} | +{e['best']['record']['update']} |")
    lines += [
        "",
        "All eight latest/selected checkpoints passed SHA256, CPU deserialization and complete-state semantic digest checks. Model/Adam/scheduler/sampler/RNG/dual/config/parent/J1 provenance are retained. Stop checkpoints are explicitly ineligible for promotion. No checkpoints were deleted.",
        "",
        "The conversation interruption affected visibility, not completion of the worker campaign. A final scientific writeup was not found. The staged `/tmp/v51_final_extras.py` was not executed; its source was preserved here for review. No new evaluation or optimizer update was run. The absent resource-observer log is a monitoring gap, not evidence of a storage or optimizer failure.",
        "",
        f"Deleted **{cleanup['deleted_files']}** individually verified HTTP wheel-cache files; installed package METADATA/RECORD, dependency lock and scientific manifest remain unchanged. Deleted allocation: **{plan['expected_reclaimed_drvfs_bytes']:,} bytes ({plan['expected_reclaimed_drvfs_bytes'] / GIB:.3f} GiB)**. Observed free-space gain after writing the deletion ledger: **{cleanup['observed_free_increase_bytes']:,} bytes**. The **{(OUT / 'deletions.jsonl').stat().st_blocks * 512:,}-byte** ledger accounts exactly for the difference. Exact paths, hashes, file identities and reasons are in `cleanup_plan.json` and `deletions.jsonl`.",
        "",
        "## Measured major storage consumers",
        "",
        "Native NTFS allocation/file IDs were measured; file contents were not scanned for inventory. Snapshot values precede cleanup; known deletions affect only E015 cache and its ancestors. Parent/child rows overlap. Last-write times and dependency assessments are in REPORT.json; last modification is not proof of last use.",
        "",
        "| Directory | Unique allocated GiB | Role / retention |",
        "|---|---:|---|",
    ]
    for row in major:
        lines.append(
            f"| `{row['path']}` | {row['unique_allocated_attributed_bytes'] / GIB:.3f} | {row['purpose']} {row['dependency_and_reproducibility']} |"
        )
    lines += [
        "",
        f"Global hardlink overcount avoided: **{(inv['allocated_paths'] - inv['unique_allocated']) / GIB:.3f} GiB**. The main SQL/WAL audit and preservation paths share physical files; unlinking one name frees zero. Reservoirs remain protected. Inventory coverage: {inv['files']:,} files; {len(inv['errors'])} errors; {len(inv['exclusions'])} excluded directories. TEST, recycle/system areas and reparse targets were excluded, so not all D: usage is attributed. No excluded area is classified as safe to delete.",
        "",
        "## Larger proposals — no deletion authorization exercised",
        "",
        "| Exact directory | Allocated GiB | Required review |",
        "|---|---:|---|",
    ]
    for p in proposals:
        lines.append(f"| `{p['path']}` | {p['allocated_bytes'] / GIB:.3f} | {p['required_review']} |")
    lines += [
        "",
        "The 15–20 GiB ideal was not met without affecting protected/unresolved historical evidence. Review candidates are upper bounds, not certified disposable bytes. Original PDB sources, governed datasets, certified UniRef material, frozen J1, all E016/E017 evidence and all latest/best recovery states remain untouched. Do not compress/copy these large trees into D: staging while near capacity.",
        "",
        "## Future growth controls",
        "",
        "- Use no-download-cache package installation for future isolated tool environments; preserve pinned lock/install receipts. Leave existing installed environments intact.",
        "- Share frozen datasets and immutable feature artifacts by verified manifest/content identity. Avoid complete conversions/copies unless semantics require them; track hardlinks by file ID. Prefix-dependent J1 features cannot be reused across changing prefixes.",
        "- Keep run-owned rolling latest/best states and crash-safe, verified atomic publication. Retire only same-run superseded files after the new index commits; retain scientific stop, named and selected evidence under explicit policy.",
        "- Keep one live replay and share unchanged immutable snapshots. V5.1 has no active RL replay, so there is no large replay buffer to purge.",
        "- Keep compact scalar telemetry; prevent per-step tensor dumps or duplicated evaluation exports. Preserve existing scientific logs and reports.",
        "- Forecast retained parents plus latest/best/incoming checkpoint and near-term growth, then add the approved2GiB headroom. Recheck actual serialized bytes before staging; space admission never overrides scientific gates.",
        "",
        "## Handoff",
        "",
        "V5.1 cannot safely resume under its sealed protocol: all arms have scientific STOP records, and the entrypoint explicitly refuses them. No resume command is recommended. Next action: review completed scientific results and the stop failures; handle any larger cleanup proposal separately. No architecture, schedule, sampling, threshold or TEST changes were made.",
        "",
        "Full machine-readable recovery/checkpoint hashes: `experiment_recovery.json` and `REPORT.json`. Allocation inventory: `storage_inventory.json`. Exact deletion evidence: `cleanup_plan.json`, `deletions.jsonl`, `cleanup_result.json`. Proposed reviewed cleanup: `review_required_proposals.json`.",
    ]
    # Publish first, then capture final free space after the report's own allocation.
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    save("REPORT.json", result)
    now = shutil.disk_usage("/mnt/d")
    result["final_D_free_bytes"] = now.free
    result["D_total_bytes"] = now.total
    result["unattributed_used_bytes_at_inventory"] = (
        inv["total_bytes"] - inv["free_after_bytes"] - inv["unique_allocated"]
    )
    admission["storage_admitted"] = now.free >= admission["conservative_fresh_latest_best_incoming_required_free_bytes"]
    admission["scientific_admitted"] = False
    save("REPORT.json", result)
    (OUT / "REPORT.md").write_text(
        "\n".join(lines)
        + f"\n\nFinal free-space snapshot: **{now.free:,} bytes ({now.free / GIB:.3f} GiB)**. Checkpoint storage admission passes; scientific continuation remains blocked.\n"
    )
    print(
        json.dumps(
            {
                "status": result["experiment_status"],
                "D_free_bytes": shutil.disk_usage("/mnt/d").free,
                "reclaimed_allocated_bytes": plan["expected_reclaimed_drvfs_bytes"],
                "observed_free_gain": cleanup["observed_free_increase_bytes"],
                "can_resume": False,
                "report": str(OUT / "REPORT.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
