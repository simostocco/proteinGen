"""Validation-blind expanding/binary physical-batch probes in fresh processes."""

import argparse
import json
import subprocess
import sys

import e015_bootstrap

from protein_distance_diffusion.training.e014_runtime import canonical_hash
from protein_distance_diffusion.training.e015_runtime import Runtime, e007_active, status

CONFIG = e015_bootstrap.WORKTREE / "configs/e015_integrated_multiscale_parity_selfconditioned_v1.yaml"
BUCKETS = (("20-64", 64), ("65-128", 128), ("129-256", 256), ("257-384", 384), ("385-500", 500))


def tune(runtime):
    if e007_active(runtime.cfg):
        raise RuntimeError("E007 active; do not autotune concurrently")
    if not runtime.path("calibration/frozen.json").exists():
        raise RuntimeError("Freeze TRAIN calibration before autotuning")
    calibration = json.loads(runtime.path("calibration/frozen.json").read_text())
    precision = json.loads(runtime.path("precision_selection.json").read_text())["precision"]
    probe_contract_hash = canonical_hash(dict(contract=runtime.contract(calibration), precision=precision))
    provisional = {"20-64": 88, "65-128": 39, "129-256": 18, "257-384": 12, "385-500": 9}
    selected = {}
    all_probes = []
    for bucket, n in BUCKETS:
        outcomes = {}

        def probe(batch, repeats=1, final=False, n=n, bucket=bucket, outcomes=outcomes):
            name = f"autotune_v2_{probe_contract_hash[:12]}_n{n}_b{batch}" + ("_final" if final else "")
            target = runtime.path(f"preflight/{name}.json")
            if target.exists() and json.loads(target.read_text()).get("probe_contract_hash") == probe_contract_hash:
                result = json.loads(target.read_text())
            else:
                runtime.write("HEARTBEAT.json", status("autotuning", bucket=bucket, physical_batch=batch))
                command = [
                    sys.executable,
                    "-B",
                    str(e015_bootstrap.WORKTREE / "scripts/profile_e015_capacity.py"),
                    "--n",
                    str(n),
                    "--batch",
                    str(batch),
                    "--repeats",
                    str(repeats),
                    "--name",
                    name,
                ]
                with open(runtime.path(f"logs/{name}.log"), "w") as stream:
                    child = subprocess.run(
                        command, cwd=e015_bootstrap.WORKTREE, stdout=stream, stderr=subprocess.STDOUT, timeout=600
                    )
                if child.returncode or not target.exists():
                    result = dict(
                        n=n,
                        physical_batch=batch,
                        passed=False,
                        subprocess_returncode=child.returncode,
                        log_path=str(runtime.path(f"logs/{name}.log")),
                    )
                    runtime.write(f"preflight/{name}.json", result)
                else:
                    result = json.loads(target.read_text())
            all_probes.append(result)
            if not final:
                outcomes[batch] = result
            return result

        maximum = runtime.cfg["preflight"]["maximum_microbatch"]
        lower, upper = 0, provisional[bucket]
        while upper <= maximum:
            result = probe(upper)
            if not result["passed"]:
                break
            lower = upper
            upper = min(maximum + 1, upper + max(1, upper // 4))
        upper = min(upper, maximum + 1)
        while upper - lower > 1:
            middle = (lower + upper) // 2
            if probe(middle)["passed"]:
                lower = middle
            else:
                upper = middle
        if lower == 0:
            raise RuntimeError(f"E015 cannot safely fit {bucket} at batch 1")
        confirmed = probe(lower, repeats=runtime.cfg["preflight"]["repetitions"], final=True)
        while not confirmed["passed"] and lower > 1:
            lower -= 1
            confirmed = probe(lower, repeats=runtime.cfg["preflight"]["repetitions"], final=True)
        if not confirmed["passed"]:
            raise RuntimeError("Repeated worst-path probe failed at batch 1")
        # Measure the adjacent smaller physical batch with the same repeated path.
        if lower > 1:
            smaller = probe(lower - 1, repeats=runtime.cfg["preflight"]["repetitions"], final=True)
            confirmed["next_smaller_comparison"] = smaller
            confirmed["throughput_improves_vs_next_smaller"] = confirmed["proteins_per_second"] > smaller.get(
                "proteins_per_second", 0
            )
            while (
                not confirmed["throughput_improves_vs_next_smaller"]
                and lower > provisional[bucket]
                and smaller["passed"]
            ):
                lower -= 1
                confirmed = smaller
                smaller = probe(lower - 1, repeats=runtime.cfg["preflight"]["repetitions"], final=True)
                confirmed["next_smaller_comparison"] = smaller
                confirmed["throughput_improves_vs_next_smaller"] = confirmed["proteins_per_second"] > smaller.get(
                    "proteins_per_second", 0
                )
        selected[bucket] = confirmed
        runtime.write("batch_autotuning.partial.json", dict(selected=selected, probes=all_probes))
    output = dict(
        selected=selected,
        probe_contract_hash=probe_contract_hash,
        precision=precision,
        profile_version=4,
        fp32_geometry_islands=True,
        probes=all_probes,
        batch_sizes=[selected[b]["physical_batch"] for b, _ in BUCKETS],
        target_residue_budget=runtime.cfg["residue_budget"],
        validation_used=False,
        selection="largest tested integer microbatch satisfying reserved<=90% and free>=768 MiB; repeated three times",
        note="85-90% occupancy is a target subordinate to actual free-memory safety; no artificial reservation",
    )
    runtime.write("batch_autotuning.json", output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    tune(Runtime(CONFIG))


if __name__ == "__main__":
    main()
