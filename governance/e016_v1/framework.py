"""Preparation-only governance validator and synthetic diagnostics. No real-data reader or allocator."""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import math
import re

VERSION = "e016-governance/1"
MODES = {"HARD_CLOSURE", "AUDITED_CONTEXT_RESTRICTION", "UNRESOLVED"}
KINDS = {"exact_identity", "similarity", "fragment", "construct", "source_context", "observation_ownership"}
BANDS = ("20-64", "65-128", "129-256", "257-384", "385-500")
PARTITIONS = {"TRAIN", "VALIDATION", "TEST", "CALIBRATION"}
ROLES = {"ORDINARY", "REWARD_DEVELOPMENT", "FINAL_CONTROL", "TRAIN_EXCLUSION"}
MINIMA = {"unique_sequences", "independent_groups", "joint_observations", "joint_identities"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keys(value, expected, context):
    require(isinstance(value, dict) and set(value) == set(expected), f"{context}: exact keys required")


def validate_policy(p):
    keys(p, {"schema_version", "policy_id", "stage", "base_revision", "evidence", "relationships", "shadow_policy", "controls", "partitions", "evaluation", "guarantees"}, "policy")
    require(p["schema_version"] == VERSION and p["stage"] == "PREPARATION_ONLY", "unsupported version/stage")
    require(isinstance(p["policy_id"], str) and bool(p["policy_id"]), "policy ID")
    require(isinstance(p["base_revision"], str) and re.fullmatch(r"[0-9a-f]{40}", p["base_revision"]), "base revision")
    require(isinstance(p["evidence"], list) and bool(p["evidence"]), "input provenance required")
    seen = set()
    for item in p["evidence"]:
        keys(item, {"path", "sha256", "role"}, "evidence")
        require(isinstance(item["path"], str) and item["path"].startswith("/mnt/d/") and ".." not in item["path"].split("/"), "D-only lexical provenance path")
        require(item["path"] not in seen, "duplicate evidence path")
        seen.add(item["path"])
        require(isinstance(item["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]), "evidence SHA256")
        require(isinstance(item["role"], str) and bool(item["role"]), "evidence role")
    keys(p["relationships"], KINDS, "relationships")
    for kind, rule in p["relationships"].items():
        keys(rule, {"mode", "predicate", "justification"}, kind)
        require(rule["mode"] in MODES, "relationship mode")
        require(isinstance(rule["justification"], str) and bool(rule["justification"]), "justification")
        if kind in {"exact_identity", "observation_ownership"}:
            require(rule["mode"] == "HARD_CLOSURE" and rule["predicate"] == "IDENTITY_INVARIANT_V1", "identity/ownership invariant")
        else:
            require(rule["mode"] == "UNRESOLVED" and rule["predicate"] is None, "preparation cannot approve biological/context predicates")
    require(p["shadow_policy"] == "UNRESOLVED", "shadow participation requires separate approval")
    keys(p["controls"], {"reservation_scope", "role_separation", "selected_control_identities_preserved"}, "controls")
    require(p["controls"]["reservation_scope"] == "UNRESOLVED" and p["controls"]["role_separation"] == "REQUIRED" and p["controls"]["selected_control_identities_preserved"] is True, "control decisions")
    keys(p["partitions"], {"target_ratios", "target_semantics", "denominator", "seed"}, "partitions")
    ratios = p["partitions"]["target_ratios"]
    keys(ratios, {"TRAIN", "VALIDATION", "TEST"}, "ratios")
    require(all(type(v) in (int, float) and math.isfinite(v) and 0 < v < 1 for v in ratios.values()) and abs(sum(ratios.values()) - 1) < 1e-12, "finite positive ratios sum to one")
    require(p["partitions"]["target_semantics"] == "SOFT_OBJECTIVE" and p["partitions"]["denominator"] == "UNRESOLVED" and type(p["partitions"]["seed"]) is int, "allocation decisions")
    keys(p["evaluation"], {"minima", "required_length_bands", "required_methods", "required_contexts"}, "evaluation")
    keys(p["evaluation"]["minima"], MINIMA, "evaluation minima")
    require(all(v is None for v in p["evaluation"]["minima"].values()), "statistical acceptance not approved")
    require(all(p["evaluation"][k] is None for k in ["required_length_bands", "required_methods", "required_contexts"]), "coverage acceptance not approved")
    require(p["guarantees"] == {"real_data_status": "NOT_EVALUATED", "remote_homology_eliminated": False, "production_admission": False}, "no real-data certification in preparation")
    return {"status": "VALID_PREPARATION_SCHEMA", "real_data_ready": False,
            "blockers": ["BIOLOGICAL_AND_CONTEXT_PREDICATES", "SHADOW_POLICY", "CONTROL_RESERVATION_SCOPE", "ALLOCATION_DENOMINATOR", "EVALUATION_ACCEPTANCE"]}


@dataclass(frozen=True)
class ToyNode:
    identity: str
    eligible: bool = True
    joint_observations: int = 0
    band: str = "20-64"
    method: str = "X-RAY"

    def __post_init__(self):
        require(isinstance(self.identity, str) and re.fullmatch(r"toy-[A-Za-z0-9_-]+", self.identity), "synthetic identifiers only")
        require(type(self.eligible) is bool, "eligible bool")
        require(type(self.joint_observations) is int and self.joint_observations >= 0, "joint count")
        require(self.eligible or self.joint_observations == 0, "shadow cannot count as eligible joint observation")
        require(self.band in BANDS and self.method in {"X-RAY", "EM", "NMR"}, "toy stratum")


@dataclass(frozen=True)
class ToyEdge:
    a: str
    b: str
    kind: str


@dataclass(frozen=True)
class ToyReservation:
    node: str
    partition: str
    role: str = "ORDINARY"


def toy_components(nodes, edges, modes, *, shadow_mode, scope):
    require(scope == "SYNTHETIC_ONLY", "real data prohibited")
    require(shadow_mode in {"BRIDGE", "EXCLUDE", "UNRESOLVED"}, "toy shadow mode")
    nodes = tuple(nodes)
    require(bool(nodes) and all(type(n) is ToyNode for n in nodes), "toy nodes required")
    by_id = {n.identity: n for n in nodes}
    require(len(by_id) == len(nodes), "duplicate identity")
    require(isinstance(modes, dict) and all(k in KINDS and v in MODES for k, v in modes.items()), "toy edge modes")
    for k in {"exact_identity", "observation_ownership"} & modes.keys():
        require(modes[k] == "HARD_CLOSURE", "identity and ownership remain hard")
    edges = tuple(edges)
    for e in edges:
        require(type(e) is ToyEdge and e.a in by_id and e.b in by_id and e.kind in KINDS, "edge identity/type")
        require(e.kind in modes and modes[e.kind] != "UNRESOLVED", "unresolved edge cannot silently disappear")
    require(shadow_mode != "UNRESOLVED" or all(n.eligible for n in nodes), "unresolved shadow participation")
    active = {i for i, n in by_id.items() if n.eligible or shadow_mode == "BRIDGE"}
    parent = {i: i for i in active}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for e in edges:
        if e.a in active and e.b in active and modes[e.kind] == "HARD_CLOSURE":
            a, b = find(e.a), find(e.b)
            if a != b:
                parent[max(a, b)] = min(a, b)
    result = {}
    for i in sorted(active):
        result.setdefault(find(i), []).append(i)
    return tuple(tuple(v) for _, v in sorted(result.items()))


def diagnose_reservations(nodes, components, reservations, ratios, *, hard_bounds=None):
    """Necessary pre-allocation checks, never a general feasibility solver or allocator."""
    nodes = tuple(nodes)
    require(bool(nodes) and all(type(n) is ToyNode for n in nodes), "synthetic nodes only")
    by_id = {n.identity: n for n in nodes}
    require(len(by_id) == len(nodes), "duplicate identity")
    flat = [i for c in components for i in c]
    require(len(flat) == len(set(flat)) and all(i in by_id for i in flat), "component ownership")
    require({i for i, n in by_id.items() if n.eligible} <= set(flat), "all eligible nodes represented")
    require(set(ratios) == {"TRAIN", "VALIDATION", "TEST"} and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in ratios.values()) and abs(sum(ratios.values()) - 1) < 1e-12, "toy ratios")
    owner = {i: k for k, c in enumerate(components) for i in c}
    fixed = {}; errors = []; overshoot = []; n = sum(x.eligible for x in nodes)
    for r in reservations:
        require(type(r) is ToyReservation and r.node in owner and r.partition in PARTITIONS and r.role in ROLES, "reservation")
        if (r.role == "REWARD_DEVELOPMENT" and r.partition in {"TRAIN", "TEST"}) or (r.role == "FINAL_CONTROL" and r.partition != "TEST") or (r.role == "TRAIN_EXCLUSION" and r.partition == "TRAIN"):
            errors.append("CONTROL_ROLE_CONFLICT")
        k = owner[r.node]
        if k in fixed and fixed[k] != r.partition:
            errors.append("COMPONENT_RESERVATION_CONFLICT")
        else:
            fixed[k] = r.partition
    reserved = Counter()
    for k, split in fixed.items():
        reserved[split] += sum(by_id[i].eligible for i in components[k])
    for split, fraction in ratios.items():
        if reserved[split] > fraction * n + 1e-12:
            overshoot.append({"partition": split, "reserved": reserved[split], "target": fraction * n})
    if hard_bounds is not None:
        require(set(hard_bounds) == PARTITIONS, "explicit bounds for every role partition")
        for split, bounds in hard_bounds.items():
            require(len(bounds) == 2 and all(type(v) is int for v in bounds) and 0 <= bounds[0] <= bounds[1], "integer bounds")
            if reserved[split] > bounds[1]:
                errors.append("RESERVATION_EXCEEDS_HARD_MAX")
        if sum(v[0] for v in hard_bounds.values()) > n or sum(v[1] for v in hard_bounds.values()) < n:
            errors.append("INCOMPATIBLE_TOTAL_BOUNDS")
        for k, c in enumerate(components):
            weight = sum(by_id[i].eligible for i in c)
            if k not in fixed and not any(weight <= hi - reserved[s] for s, (_, hi) in hard_bounds.items()):
                errors.append("INDIVISIBLE_COMPONENT_EXCEEDS_REMAINING_CAPACITIES")
    return {"scope": "SYNTHETIC_ONLY", "status": "INFEASIBLE_NECESSARY_CONDITION" if errors else "NO_CONTRADICTION_FOUND_NOT_FEASIBILITY_PROOF", "errors": sorted(set(errors)), "soft_target_overshoot": overshoot, "denominator": n, "reserved_counts": dict(reserved), "sufficient_feasibility_proof": False}


def evaluate_toy_population(nodes, components, assignments, roles, minima=None):
    """Describe supplied synthetic assignments. Never generates assignments."""
    nodes = tuple(nodes)
    require(bool(nodes) and all(type(n) is ToyNode for n in nodes), "synthetic nodes only")
    by_id = {n.identity: n for n in nodes}
    require(len(by_id) == len(nodes), "duplicate identity")
    flat = [i for c in components for i in c]
    require(len(flat) == len(set(flat)) and set(flat) <= by_id.keys() and {i for i,n in by_id.items() if n.eligible} <= set(flat), "complete unique eligible ownership")
    require(set(assignments) == set(flat) and set(roles) == set(flat), "complete explicit assignments and roles")
    require(all(v in PARTITIONS for v in assignments.values()) and all(v in ROLES for v in roles.values()), "assignment/role enum")
    for c in components:
        require(len({assignments[i] for i in c}) == 1, "cannot split a hard component")
    if minima is not None:
        require(set(minima) == MINIMA and all(type(v) is int and v > 0 for v in minima.values()), "explicit positive scientific minima")
    report = {}
    for split in ["VALIDATION", "TEST"]:
        chosen = {i for i in flat if assignments[i] == split and roles[i] == "ORDINARY" and by_id[i].eligible}
        # Count any component containing nonordinary controls separately: never certify it as ordinary independent evaluation.
        contaminated = any(chosen.intersection(c) and any(roles[i] != "ORDINARY" for i in c) for c in components)
        counts = {"unique_sequences": len(chosen), "independent_groups": sum(bool(chosen.intersection(c)) for c in components), "joint_observations": sum(by_id[i].joint_observations for i in chosen), "joint_identities": sum(by_id[i].joint_observations > 0 for i in chosen)}
        issues = []
        if counts["independent_groups"] < 2:
            issues.append("EMPTY_OR_SINGLE_GROUP_EVALUATION")
        if not counts["joint_observations"]:
            issues.append("NO_JOINT_EVALUATION")
        if contaminated:
            issues.append("CONTROL_COMPONENT_MIXED_WITH_ORDINARY_EVALUATION")
        if minima is None:
            issues.append("STATISTICAL_ACCEPTANCE_UNSET")
        else:
            issues.extend("BELOW_MIN_" + k.upper() for k,v in minima.items() if counts[k] < v)
        bands = {band: {"sequences": sum(by_id[i].band == band for i in chosen), "joint_observations": sum(by_id[i].joint_observations for i in chosen if by_id[i].band == band)} for band in BANDS}
        report[split] = {"counts": counts, "length_bands": bands, "methods": dict(Counter(by_id[i].method for i in chosen)), "issues": issues}
    return {"scope": "SYNTHETIC_ONLY", "real_data_certified": False, "coverage_acceptance": "UNRESOLVED", "partitions": report}
