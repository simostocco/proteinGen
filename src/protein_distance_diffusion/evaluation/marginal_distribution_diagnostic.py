"""Invariant, equally weighted protein-level empirical distribution diagnostics."""

import numpy as np

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
SEPARATIONS = ((1, 1), (2, 2), (3, 3), (4, 8), (9, 32), (33, None))
NAMES = ["rg", "eig1", "eig2", "eig3", "trace", "anisotropy1", "anisotropy3", "component_rms"]
NAMES += ["pair_mean", "pair_std"] + [f"pair_p{int(q * 100):02}" for q in QUANTILES]
for low, high in SEPARATIONS:
    NAMES += [f"sep_{low}_{high}_{stat}" for stat in ("mean", "std")]
NAMES += ["norm_mean", "norm_std"] + [f"norm_p{int(q * 100):02}" for q in QUANTILES]
PRIMARY = ("mean_error", "covariance_error", "energy", "mmd2", "sliced_wasserstein")


def descriptor(coordinates):
    x = np.asarray(coordinates, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3 or len(x) < 4 or not np.isfinite(x).all():
        raise ValueError("invalid coordinate descriptor input")
    x = x - x.mean(0)
    eigen = np.linalg.eigvalsh(x.T @ x / len(x))[::-1]
    trace = eigen.sum()
    values = [
        np.sqrt(trace),
        *eigen,
        trace,
        eigen[0] / max(trace, 1e-12),
        eigen[2] / max(trace, 1e-12),
        np.sqrt(trace / 3),
    ]
    i, j = np.triu_indices(len(x), 1)
    distances = np.linalg.norm(x[j] - x[i], axis=1)
    values += [distances.mean(), distances.std(), *np.quantile(distances, QUANTILES)]
    separation = j - i
    for low, high in SEPARATIONS:
        selected = distances[(separation >= low) & ((separation <= high) if high is not None else True)]
        values += [selected.mean(), selected.std()] if len(selected) else [np.nan, np.nan]
    norms = np.linalg.norm(x, axis=1)
    values += [norms.mean(), norms.std(), *np.quantile(norms, QUANTILES)]
    assert len(values) == len(NAMES) == 34
    return np.asarray(values)


class ReferenceStandardizer:
    """Fit only reference B; keep columns available in both independent references."""

    def __init__(self, reference_a, reference_b, columns=None):
        both = np.concatenate([reference_a, reference_b])
        valid = np.isfinite(both).all(0)
        available = np.flatnonzero(valid)
        available = available[np.ptp(both[:, available], axis=0) > 1e-8]
        self.columns = available if columns is None else np.asarray(columns)
        if not set(self.columns).issubset(available):
            raise ValueError("unavailable reference descriptor")
        b = reference_b[:, self.columns]
        self.location = np.median(b, axis=0)
        self.scale = np.maximum((np.quantile(b, 0.75, axis=0) - np.quantile(b, 0.25, axis=0)) / 1.349, 1e-8)

    def transform(self, x):
        result = (x[:, self.columns] - self.location) / self.scale
        if not np.isfinite(result).all():
            raise ValueError("nonfinite standardized descriptor")
        return result


def distances(a, b):
    # Euclidean distance divided by sqrt(descriptor dimension).
    # Direct differences avoid cancellation at nearly identical descriptor rows.
    # Fixed component order also makes the reference-permutation check stable.
    square = np.zeros((len(a), len(b)), dtype=np.float64)
    for channel in range(a.shape[1]):
        delta = a[:, channel, None] - b[None, :, channel]
        square += delta * delta
    return np.sqrt(square / a.shape[1])


def bandwidth(reference_a, reference_b):
    pooled = np.concatenate([reference_a, reference_b])
    d = distances(pooled, pooled)
    positive = d[np.triu_indices(len(pooled), 1)]
    positive = positive[positive > 1e-12]
    return float(np.median(positive)) if len(positive) else 1.0


def projections(dimension, seed=14014014, count=64):
    p = np.random.Generator(np.random.PCG64(seed)).normal(size=(dimension, count))
    return p / np.linalg.norm(p, axis=0)


def w1(a, b):
    if len(a) != len(b):
        raise ValueError("empirical W1 requires equal-sized populations here")
    return float(np.abs(np.sort(a, axis=0) - np.sort(b, axis=0)).mean())


def ks(a, b):
    grid = np.sort(np.concatenate([a, b]))
    return float(
        np.max(
            np.abs(
                np.searchsorted(np.sort(a), grid, side="right") / len(a)
                - np.searchsorted(np.sort(b), grid, side="right") / len(b)
            )
        )
    )


def calibrated_gain(model, noop, floor):
    return float(1 - (model - floor) / (noop - floor)) if noop - floor > 1e-12 else None


class Comparison:
    """Cache pair matrices; bootstrap proteins, never individual residue pairs."""

    def __init__(self, a, b, raw_a, raw_b, h, projection, spectrum_columns):
        self.a, self.b, self.raw_a, self.raw_b = a, b, raw_a, raw_b
        self.aa, self.bb, self.ab = distances(a, a), distances(b, b), distances(a, b)
        self.kaa, self.kbb, self.kab = [np.exp(-v * v / (2 * h * h)) for v in (self.aa, self.bb, self.ab)]
        self.pa, self.pb = a @ projection, b @ projection
        self.spectrum_columns = spectrum_columns

    def metrics(self, indices=None):
        n = len(self.a)
        idx = np.arange(n) if indices is None else indices
        weights = np.bincount(idx, minlength=n).astype(float) / n
        a, b = self.a[idx], self.b[idx]
        ca, cb = np.cov(a, rowvar=False, ddof=1), np.cov(b, rowvar=False, ddof=1)
        return dict(
            mean_error=float(np.linalg.norm(a.mean(0) - b.mean(0)) / np.sqrt(a.shape[1])),
            covariance_error=float(np.linalg.norm(ca - cb) / max(np.linalg.norm(cb), 1e-12)),
            energy=float(
                max(2 * weights @ self.ab @ weights - weights @ self.aa @ weights - weights @ self.bb @ weights, 0)
            ),
            mmd2=float(
                max(weights @ self.kaa @ weights + weights @ self.kbb @ weights - 2 * weights @ self.kab @ weights, 0)
            ),
            sliced_wasserstein=w1(self.pa[idx], self.pb[idx]),
            rg_wasserstein=w1(self.raw_a[idx, 0], self.raw_b[idx, 0]),
            rg_ks=ks(self.raw_a[idx, 0], self.raw_b[idx, 0]),
            covariance_spectrum_w1=w1(a[:, self.spectrum_columns], b[:, self.spectrum_columns]),
        )


def compare_populations(populations, raw, columns, seed, strata_sizes, replicates=500):
    a, b = populations["ref_a"], populations["ref_b"]
    h = bandwidth(a, b)
    p = projections(a.shape[1])
    spectrum = [
        i for i, c in enumerate(columns) if NAMES[c] in ("eig1", "eig2", "eig3", "trace", "anisotropy1", "anisotropy3")
    ]
    comparisons = {
        label: Comparison(populations[label], b, raw[label], raw["ref_b"], h, p, spectrum)
        for label in ("ref_a", "noop", "model")
    }
    points = {label: comparison.metrics() for label, comparison in comparisons.items()}
    rng = np.random.Generator(np.random.PCG64(seed))
    boundaries = np.cumsum([0, *strata_sizes])
    draws = {label: {metric: [] for metric in points[label]} for label in points}
    differences = {metric: [] for metric in points["model"]}
    gains = {metric: [] for metric in points["model"]}
    for _ in range(replicates):
        indices = np.concatenate(
            [
                rng.integers(low, high, size=high - low)
                for low, high in zip(boundaries[:-1], boundaries[1:], strict=True)
            ]
        )
        values = {label: comparison.metrics(indices) for label, comparison in comparisons.items()}
        for metric in differences:
            for label in points:
                draws[label][metric].append(values[label][metric])
            differences[metric].append(values["model"][metric] - values["noop"][metric])
            gain = calibrated_gain(values["model"][metric], values["noop"][metric], values["ref_a"][metric])
            if gain is not None:
                gains[metric].append(gain)
    output = []
    for metric in differences:
        delta = np.quantile(differences[metric], [0.025, 0.975]).tolist()
        output.append(
            dict(
                metric=metric,
                ref_ref=points["ref_a"][metric],
                noop_ref=points["noop"][metric],
                model_ref=points["model"][metric],
                gain=calibrated_gain(points["model"][metric], points["noop"][metric], points["ref_a"][metric]),
                ref_ref_ci=np.quantile(draws["ref_a"][metric], [0.025, 0.975]).tolist(),
                noop_ref_ci=np.quantile(draws["noop"][metric], [0.025, 0.975]).tolist(),
                model_ref_ci=np.quantile(draws["model"][metric], [0.025, 0.975]).tolist(),
                model_minus_noop_ci=delta,
                bootstrap_gain_valid_count=len(gains[metric]),
                gain_ci=np.quantile(gains[metric], [0.025, 0.975]).tolist() if gains[metric] else None,
            )
        )
    primary = [r for r in output if r["metric"] in PRIMARY]
    improves = sum(r["model_minus_noop_ci"][1] < 0 for r in primary)
    worsens = sum(r["model_minus_noop_ci"][0] > 0 for r in primary)
    status = (
        "improves" if improves >= 3 and worsens == 0 else "worsens" if worsens >= 3 and improves == 0 else "unresolved"
    )
    # Point metrics must be reference-order invariant, including independent normalizer.
    reverse = np.arange(len(b))[::-1]
    perm = Comparison(populations["model"], b[reverse], raw["model"], raw["ref_b"][reverse], h, p, spectrum).metrics()
    permutation_error = max(abs(perm[k] - points["model"][k]) for k in perm)
    if permutation_error > 1e-10:
        raise ValueError(f"permutation invariance failed: {permutation_error}")
    return dict(
        sample_count=len(a),
        descriptor_columns=[NAMES[c] for c in columns],
        bandwidth=h,
        bootstrap_replicates=replicates,
        status=status,
        metrics=output,
        permutation_max_error=permutation_error,
        _model_draws=draws["model"],
    )
