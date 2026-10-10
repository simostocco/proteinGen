"""Layerwise controlled CUDA symmetry diagnosis, including fixed-topology replay."""

import argparse
import contextlib
import hashlib
import json
import math

import e015_bootstrap
import torch

import protein_distance_diffusion.models.e015_multiscale_parity_denoiser as architecture
from protein_distance_diffusion.evaluation.e014_evaluation import coordinate_batch
from protein_distance_diffusion.training.e015_engine import Engine
from protein_distance_diffusion.training.e015_runtime import GPULease, Runtime

CONFIG = e015_bootstrap.WORKTREE / "configs/e015_integrated_multiscale_parity_selfconditioned_v1.yaml"


def metric(value, expected):
    value, expected = value.float(), expected.float()
    error = (value - expected).norm()
    return dict(
        absolute_error=float(error),
        relative_error=float(error / expected.norm().clamp_min(1e-12)),
        reference_norm=float(expected.norm()),
    )


def cpu(t):
    return t.detach().float().cpu().clone()


class Trace:
    def __init__(self, model, frozen=None):
        self.model, self.frozen, self.values, self.graphs = model, frozen, {}, []
        self.hooks = []
        self.original_graph = architecture.graph
        self.original_pool = architecture.pool_state
        self.original_prepare = model.prepare_single
        self.original_head = model.output_head

    def state(self, name, state):
        for key in ("h", "v", "p", "a"):
            self.values[name + "." + key] = cpu(getattr(state, key))

    def graph(self, x, previous, k, k_self, dense=False):
        observed = self.original_graph(x, previous, k, k_self, dense)
        if self.frozen is not None:
            reference = self.frozen[len(self.graphs)]
            i, j = (reference[z].to(x.device) for z in ("i", "j"))
            delta = x[j] - x[i]
            d = delta.norm(dim=-1)
            pd = torch.zeros_like(d) if previous is None else (previous[j] - previous[i]).norm(dim=-1)
            flags = reference["features"][:, 32:].to(x.device)
            features = torch.cat([architecture.radial_basis(d), architecture.radial_basis(pd), flags], -1)
            current_unit = delta / d[:, None].clamp_min(1e-6)
            sc_unit = (
                torch.zeros_like(current_unit)
                if previous is None
                else (previous[j] - previous[i]) / pd[:, None].clamp_min(1e-6)
            )
            observed = i, j, features, torch.stack([current_unit, sc_unit], 1)
        i, j, features, _ = observed
        self.graphs.append(
            dict(
                n=len(x),
                i=i.detach().cpu(),
                j=j.detach().cpu(),
                features=cpu(features),
                coordinates=cpu(x),
                previous=None if previous is None else cpu(previous),
            )
        )
        return observed

    def prepare(self, x, t, previous=None):
        self.values["preprocess.v"] = cpu(x - x.mean(0))
        result = self.original_prepare(x, t, previous)
        self.state("initial", result[0])
        self.values["initial_frame.v"] = cpu(result[1].transpose(-1, -2))
        self.values["initial_reliability.h"] = cpu(result[2])
        self.values["initial_gate.h"] = cpu(result[3])
        return result

    def pool(self, state):
        result = self.original_pool(state)
        self.state("fine_to_medium" if self.pool_count == 0 else "medium_to_coarse", result)
        self.pool_count += 1
        return result

    def head(self, state, frame, reliability, gate, shapes):
        h, v, p, a = state.h, state.v, state.p, state.a
        polar = (self.model.polar_v(h)[..., None] * v).sum(1) / math.sqrt(self.model.widths[1])
        polar += (self.model.polar_p(p)[..., None] * a).sum(1) / math.sqrt(self.model.widths[3])
        chiral = (self.model.chiral_a(h)[..., None] * a).sum(1) / math.sqrt(self.model.widths[3])
        chiral += (self.model.chiral_p(p)[..., None] * v).sum(1) / math.sqrt(self.model.widths[1])
        local = torch.einsum("nac,nc->na", frame, self.model.local(h).float())
        for name, value in [
            ("polar_head", polar),
            ("chiral_head", chiral),
            ("local_head", local),
            ("combined_before_center", polar + gate[:, None] * (chiral + local)),
        ]:
            self.values[name + ".v"] = cpu(value)
        return self.original_head(state, frame, reliability, gate, shapes)

    def __enter__(self):
        architecture.graph, architecture.pool_state = self.graph, self.pool
        self.model.prepare_single, self.model.output_head = self.prepare, self.head
        self.pool_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, architecture.TypedBlock):

                def hook(module, args, output, name=name):
                    self.state(name, architecture.State(*output))

                self.hooks.append(module.register_forward_hook(hook))
            if isinstance(module, architecture.Broadcast):

                def hook(module, args, output, name=name):
                    self.state(name, output)

                self.hooks.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, *_):
        architecture.graph, architecture.pool_state = self.original_graph, self.original_pool
        self.model.prepare_single, self.model.output_head = self.original_prepare, self.original_head
        for hook in self.hooks:
            hook.remove()


def edge_stats(reference, changed):
    output = []
    for scale, (a, b) in zip(("fine", "medium", "coarse"), zip(reference, changed, strict=True), strict=True):
        record = dict(scale=scale, n=a["n"], types={})
        for kind, column in [("all", None), ("polymer", 33), ("current_knn", 36), ("self_knn", 37)]:

            def edges(g, column=column, kind=kind):
                flag = (
                    torch.ones(len(g["i"]), dtype=torch.bool)
                    if column is None
                    else (g["features"][:, 33:36].any(-1) if kind == "polymer" else g["features"][:, column].bool())
                )
                return set(zip(g["i"][flag].tolist(), g["j"][flag].tolist(), strict=True))

            ea, eb = edges(a), edges(b)
            diff = ea ^ eb
            record["types"][kind] = dict(
                exact_identity=ea == eb,
                reference_edges=len(ea),
                changed_edges=len(eb),
                symmetric_difference=len(diff),
                jaccard=len(ea & eb) / max(1, len(ea | eb)),
                mismatches_by_residue={str(i): sum(u == i for u, _ in diff) for i in sorted({u for u, _ in diff})},
            )

        def digest(g):
            return hashlib.sha256(torch.stack([g["i"], g["j"]]).numpy().tobytes()).hexdigest()

        record["reference_edge_hash"], record["changed_edge_hash"] = digest(a), digest(b)
        record["near_ties"] = {}
        for kind, coords, k in [("current", a["coordinates"], 24), ("self", a["previous"], 16)]:
            if coords is None or scale == "coarse":
                continue
            delta = coords.double()[:, None] - coords.double()[None]
            distances = delta.square().sum(-1).sqrt()
            distances.fill_diagonal_(float("inf"))
            sorted_distances = distances.sort(stable=True).values
            k = min(k, len(coords) - 2)
            if k < 1:
                continue
            gap = sorted_distances[:, k] - sorted_distances[:, k - 1]
            tied = gap <= 1e-5 * sorted_distances[:, k - 1].clamp_min(1)
            record["near_ties"][kind] = dict(
                count=int(tied.sum()),
                minimum_gap=float(gap.min()),
                residues=tied.nonzero().flatten().tolist(),
                rule="relative1e-5",
            )
        output.append(record)
    return output


def diagnose(runtime, label):
    lease = GPULease(runtime)
    cal = json.loads(runtime.path("calibration/frozen.json").read_text())
    torch.manual_seed(runtime.cfg["seed"])
    engine = Engine(runtime.cfg, cal)
    model = engine.model.eval()
    row = next(r for r in runtime.train if r["length"] == 500)
    x, n, mask = coordinate_batch([row], runtime.cache, runtime.cfg["coordinate_scale_angstrom"])
    t = torch.tensor([20], device="cuda")
    previous, available = x * 0.8, torch.tensor([True], device="cuda")
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, device="cuda"))
    rotation[:, 0] *= torch.det(rotation)
    report = dict(
        label=label,
        parameter_count=sum(p.numel() for p in model.parameters()),
        input_fixture="exact original certification N500/t20/x*.8 context; geometry test only",
        sample_id=row["sample_id"],
        architecture_baseline="3d143e76bc95689ae053248f5c90f314f83ca35a",
        seed=runtime.cfg["seed"],
        disk_policy="telemetry_only",
        precisions={},
    )

    def forward(precision, coordinates, context, frozen=None):
        ac = contextlib.nullcontext() if precision == "fp32" else torch.autocast("cuda", dtype=torch.bfloat16)
        with torch.no_grad(), ac, Trace(model, frozen) as trace:
            output, components = model.with_components(coordinates, t, n, mask, context, available)
            trace.values["final_centered.v"] = cpu(output[0])
        return cpu(output[0]), trace, components

    for precision in ("fp32", "bf16"):
        a, first, components = forward(precision, x, previous)
        b, second, _ = forward(precision, x @ rotation.T + 3, previous @ rotation.T - 4)
        translated, _, _ = forward(precision, x + 3, previous - 4)
        fixed, fixed_trace, _ = forward(precision, x @ rotation.T + 3, previous @ rotation.T - 4, first.graphs)
        matrix = rotation.cpu()
        layers = {
            name: metric(second.values[name], value @ matrix.T if name.endswith((".v", ".a")) else value)
            for name, value in first.values.items()
        }
        fixed_layers = {
            name: metric(fixed_trace.values[name], value @ matrix.T if name.endswith((".v", ".a")) else value)
            for name, value in first.values.items()
        }
        reflection = torch.diag(x.new_tensor([-1.0, 1.0, 1.0]))
        reflected, _, _ = forward(precision, x @ reflection.T, previous @ reflection.T)
        model.chiral_enabled = False
        off, _, _ = forward(precision, x, previous)
        off_reflected, _, _ = forward(precision, x @ reflection.T, previous @ reflection.T)
        model.chiral_enabled = True
        report["precisions"][precision] = dict(
            rotation=metric(b, a @ matrix.T),
            fixed_graph_rotation=metric(fixed, a @ matrix.T),
            translation=metric(translated, a),
            output_norm=float(a.norm()),
            branch_norms={
                key: float(components[0][key].norm()) for key in ["global_v", "chiral_effective", "local_effective"]
            },
            reflection_gap=metric(reflected, a @ reflection.cpu().T),
            disabled_head_reflection_control=metric(off_reflected, off @ reflection.cpu().T),
            graph_comparison=edge_stats(first.graphs, second.graphs),
            layers=layers,
            fixed_graph_layers=fixed_layers,
            first_layer_over_1pct=next((name for name, v in layers.items() if v["relative_error"] > 0.01), None),
        )
        runtime.write(f"continuation_v2/symmetry_{label}.json", report)
        print(precision, report["precisions"][precision]["rotation"], flush=True)
    # Reproduce the historical test exactly: input rotation was performed
    # *inside* autocast, so it was not an orthogonal FP32 input transform.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        legacy_coordinates = x @ rotation.T + 3
        legacy_context = previous @ rotation.T - 4
        legacy_a = model(x, t, n, mask, previous, available)
        legacy_b = model(legacy_coordinates, t, n, mask, legacy_context, available)
        legacy_error = float((legacy_b - legacy_a @ rotation.T).norm() / legacy_a.norm().clamp_min(1e-8))
    exact_coordinates = x @ rotation.T + 3
    exact_context = previous @ rotation.T - 4
    _, exact_trace, _ = forward("bf16", exact_coordinates, exact_context)
    _, legacy_trace, _ = forward("bf16", legacy_coordinates, legacy_context)
    legacy_layers = {
        name: metric(legacy_trace.values[name], value @ rotation.cpu().T if name.endswith((".v", ".a")) else value)
        for name, value in first.values.items()
    }
    legacy_fixed, _, _ = forward("bf16", legacy_coordinates, legacy_context, first.graphs)
    report["legacy_autocast_input_transform"] = dict(
        layers=legacy_layers,
        first_layer_over_1pct=next(
            (name for name, value in legacy_layers.items() if value["relative_error"] > 0.01), None
        ),
        fixed_graph_rotation=metric(legacy_fixed, cpu(legacy_a) @ rotation.cpu().T),
        historical_relative_error=legacy_error,
        coordinates_dtype=str(legacy_coordinates.dtype),
        coordinates_rounding=metric(cpu(legacy_coordinates), cpu(exact_coordinates)),
        context_rounding=metric(cpu(legacy_context), cpu(exact_context)),
        graph_comparison=edge_stats(exact_trace.graphs, legacy_trace.graphs),
        note="Rotation/translation fixture arithmetic must be FP32 outside model autocast; tolerance unchanged",
    )
    runtime.write(f"continuation_v2/symmetry_{label}.json", report)
    print("legacy", legacy_error, flush=True)
    lease.stream.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    diagnose(Runtime(CONFIG), args.label)
