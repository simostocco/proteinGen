"""Predeclared observational rules; these never alter scientific training gates."""

import numpy as np

HIGH = (350, 375, 400, 420, 440, 450, 460, 470, 480, 485, 490, 492, 494, 496, 498)
LOW_MID = (5, 10, 20, 50, 100, 150, 200, 250, 300, 350, 400)
GRID = tuple(sorted(set(HIGH + LOW_MID)))
MARGINAL_GRID = tuple(sorted(set(HIGH + (5, 50, 250))))
ENDPOINTS = (490, 480, 470, 460, 450, 440, 420, 400)
LABELS = ("initialization", "epoch_1_fraction_0.25", "epoch_1_fraction_0.50", "epoch_1_fraction_0.75")
STRATA = ("20-64", "65-128", "129-256", "257-384", "385-500")


def schedule_values(diffusion, timestep):
    if not 0 <= timestep < diffusion.timesteps:
        raise ValueError("timestep outside frozen schedule")
    a2 = float(diffusion.alphas_cumprod[timestep])
    snr = a2 / (1 - a2)
    return dict(alpha=np.sqrt(a2), sigma=np.sqrt(1 - a2), snr=snr, log_snr=np.log(snr))


def marginal_status(metrics, rg_ratio, trace_ratio):
    values = {m["metric"]: m["model_minus_noop_ci"] for m in metrics}
    scale_ok = 0.8 <= rg_ratio <= 1.2 and 0.8 <= trace_ratio <= 1.2
    if values["energy"][1] < 0 and any(values[k][1] < 0 for k in ("mmd2", "sliced_wasserstein")) and scale_ok:
        return "resolved_positive"
    if values["energy"][0] > 0 and any(values[k][0] > 0 for k in ("mmd2", "sliced_wasserstein")):
        return "resolved_negative"
    return "unresolved"


def resolution_front(rows):
    rows = sorted(rows, key=lambda r: r["timestep"])
    useful = [r["timestep"] for r in rows if r["resolution_status"] == "resolved_positive"]
    contiguous = []
    for row in rows:
        if row["resolution_status"] != "resolved_positive":
            break
        contiguous.append(row["timestep"])
    return dict(
        highest_resolved_timestep=max(useful) if useful else None,
        resolved_timesteps=useful,
        contiguous_sampled_prefix_front=max(contiguous) if contiguous else None,
        nonmonotone=bool(useful and (not contiguous or max(useful) > max(contiguous))),
        sampled_grid=[r["timestep"] for r in rows],
    )


def collapse_onset(rows):
    rows = sorted(rows, key=lambda r: r["timestep"])
    return dict(
        first_median_rg_below_half=next((r["timestep"] for r in rows if r["direct_rg_median"] < 0.5), None),
        first_historical_collapse_fraction_over_half=next(
            (r["timestep"] for r in rows if r["direct_collapse_fraction"] > 0.5), None
        ),
        sampled_grid=[r["timestep"] for r in rows],
    )


def noisy_scale_status(rg_ratio, trace_ratio):
    return dict(rg_below_08=rg_ratio < 0.8, trace_below_08=trace_ratio < 0.8)


def gaussian_seed(index, length, noise_seed):
    return noise_seed("e014-marginal-gaussian-v1", f"sample-{index}-N{length}", 499)


def prefix_steps():
    # Predict at499 through401, arriving at400. No t=0 calls.
    return tuple(range(499, 400, -1))
