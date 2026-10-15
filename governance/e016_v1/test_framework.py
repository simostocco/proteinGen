"""Isolated stdlib tests. All graph identities and assignments are hand-authored toys."""
import copy
import json
from pathlib import Path
import unittest
from framework import (ToyNode as N, ToyEdge as E, ToyReservation as R, toy_components, diagnose_reservations, evaluate_toy_population, validate_policy)


def graph(nodes, edges=(), modes=None, shadow="BRIDGE"):
    return toy_components(nodes, edges, modes or {}, shadow_mode=shadow, scope="SYNTHETIC_ONLY")


class PreparationSchema(unittest.TestCase):
    def setUp(self):
        self.p = json.loads(Path(__file__).with_name("policy.proposed.json").read_text())

    def test_proposal_is_valid_but_not_ready(self):
        out = validate_policy(self.p)
        self.assertEqual(out["status"], "VALID_PREPARATION_SCHEMA")
        self.assertFalse(out["real_data_ready"])
        self.assertEqual(len(out["blockers"]), 5)

    def test_unknown_fields_rejected(self):
        self.p["unexpected"] = True
        with self.assertRaises(ValueError): validate_policy(self.p)

    def test_biological_predicate_cannot_be_approved(self):
        self.p["relationships"]["similarity"].update(mode="HARD_CLOSURE", predicate="invented")
        with self.assertRaises(ValueError): validate_policy(self.p)

    def test_context_restriction_cannot_be_silently_approved(self):
        self.p["relationships"]["source_context"]["mode"] = "AUDITED_CONTEXT_RESTRICTION"
        with self.assertRaises(ValueError): validate_policy(self.p)

    def test_identity_cannot_be_softened(self):
        self.p["relationships"]["exact_identity"]["mode"] = "UNRESOLVED"
        with self.assertRaises(ValueError): validate_policy(self.p)

    def test_numeric_adequacy_cannot_be_approved(self):
        self.p["evaluation"]["minima"]["independent_groups"] = 12
        with self.assertRaises(ValueError): validate_policy(self.p)

    def test_control_evidence_is_explicit(self):
        self.assertTrue(any(e["role"] == "preserved_selected_native_control_identities_and_historical_evidence" for e in self.p["evidence"]))

    def test_nonfinite_ratio_rejected(self):
        self.p["partitions"]["target_ratios"]["TRAIN"] = float("nan")
        with self.assertRaises(ValueError): validate_policy(self.p)

    def test_bad_hash_rejected(self):
        self.p["evidence"][0]["sha256"] = "unknown"
        with self.assertRaises(ValueError): validate_policy(self.p)

    def test_duplicate_evidence_rejected(self):
        self.p["evidence"].append(copy.deepcopy(self.p["evidence"][0]))
        with self.assertRaises(ValueError): validate_policy(self.p)

    def test_non_D_provenance_rejected(self):
        self.p["evidence"][0]["path"] = "/tmp/not-allowed"
        with self.assertRaises(ValueError): validate_policy(self.p)

    def test_preparation_cannot_certify(self):
        self.p["guarantees"]["production_admission"] = True
        with self.assertRaises(ValueError): validate_policy(self.p)


class SyntheticClosure(unittest.TestCase):
    def setUp(self): self.nodes = [N("toy-a"), N("toy-b"), N("toy-c")]

    def test_transitive_context_bridge(self):
        e = [E("toy-a", "toy-b", "source_context"), E("toy-b", "toy-c", "source_context")]
        self.assertEqual(len(graph(self.nodes, e, {"source_context":"HARD_CLOSURE"})), 1)
        self.assertEqual(len(graph(self.nodes, e, {"source_context":"AUDITED_CONTEXT_RESTRICTION"})), 3)
        # Context mode only changes toy closure; it does not prove any safeguard passes.

    def test_shadow_bridge_is_explicit(self):
        nodes = [N("toy-a"), N("toy-shadow", False), N("toy-c")]
        e = [E("toy-a", "toy-shadow", "similarity"), E("toy-shadow", "toy-c", "similarity")]
        self.assertEqual(len(graph(nodes, e, {"similarity":"HARD_CLOSURE"})), 1)
        self.assertEqual(len(graph(nodes, e, {"similarity":"HARD_CLOSURE"}, shadow="EXCLUDE")), 2)
        with self.assertRaises(ValueError): graph(nodes, e, {"similarity":"HARD_CLOSURE"}, shadow="UNRESOLVED")

    def test_unresolved_edge_is_not_dropped(self):
        with self.assertRaises(ValueError): graph(self.nodes, [E("toy-a", "toy-b", "fragment")], {"fragment":"UNRESOLVED"})

    def test_ownership_stays_hard(self):
        e = [E("toy-a", "toy-b", "observation_ownership")]
        self.assertEqual(len(graph(self.nodes, e, {"observation_ownership":"HARD_CLOSURE"})), 2)
        with self.assertRaises(ValueError): graph(self.nodes, e, {"observation_ownership":"AUDITED_CONTEXT_RESTRICTION"})

    def test_real_identifier_and_scope_rejected(self):
        with self.assertRaises(ValueError): N("3FM1")
        with self.assertRaises(ValueError): toy_components(self.nodes, [], {}, shadow_mode="BRIDGE", scope="REAL_DATA")

    def test_order_determinism_and_duplicate_edge(self):
        e = [E("toy-a", "toy-b", "exact_identity"), E("toy-b", "toy-c", "exact_identity")]
        modes = {"exact_identity":"HARD_CLOSURE"}
        self.assertEqual(graph(self.nodes, e, modes), graph(list(reversed(self.nodes)), e[::-1]+e, modes))

    def test_duplicate_nodes_and_unknown_endpoints_rejected(self):
        with self.assertRaises(ValueError): graph([N("toy-a"), N("toy-a")])
        with self.assertRaises(ValueError): graph(self.nodes, [E("toy-a", "toy-missing", "fragment")], {"fragment":"HARD_CLOSURE"})


class SyntheticFeasibility(unittest.TestCase):
    ratios = {"TRAIN":.8,"VALIDATION":.1,"TEST":.1}

    def test_conflicting_component_reservations(self):
        nodes = [N("toy-a"), N("toy-b")]
        c = graph(nodes, [E("toy-a","toy-b","exact_identity")], {"exact_identity":"HARD_CLOSURE"})
        r = diagnose_reservations(nodes, c, [R("toy-a","CALIBRATION","REWARD_DEVELOPMENT"),R("toy-b","TEST","FINAL_CONTROL")], self.ratios)
        self.assertIn("COMPONENT_RESERVATION_CONFLICT", r["errors"])

    def test_reward_control_cannot_be_training_or_test(self):
        nodes = [N("toy-a")]
        for split in ["TRAIN","TEST"]:
            r = diagnose_reservations(nodes, graph(nodes), [R("toy-a",split,"REWARD_DEVELOPMENT")], self.ratios)
            self.assertIn("CONTROL_ROLE_CONFLICT", r["errors"])

    def test_soft_target_overshoot_not_false_hard_failure(self):
        nodes = [N(f"toy-{i}") for i in range(10)]
        reserved = [R(n.identity,"VALIDATION","REWARD_DEVELOPMENT") for n in nodes[:8]]
        r = diagnose_reservations(nodes, graph(nodes), reserved, self.ratios)
        self.assertEqual(r["soft_target_overshoot"], [{"partition":"VALIDATION","reserved":8,"target":1.0}])
        self.assertEqual(r["errors"], [])
        self.assertFalse(r["sufficient_feasibility_proof"])

    def test_reservation_violates_explicit_hard_bounds(self):
        nodes = [N(f"toy-{i}") for i in range(10)]
        r = diagnose_reservations(nodes, graph(nodes), [R(n.identity,"VALIDATION") for n in nodes[:8]], self.ratios, hard_bounds={"TRAIN":(8,8),"VALIDATION":(1,1),"TEST":(1,1),"CALIBRATION":(0,0)})
        self.assertIn("RESERVATION_EXCEEDS_HARD_MAX", r["errors"])

    def test_component_cannot_fit_remaining_capacity(self):
        nodes = [N(f"toy-{i}") for i in range(4)]
        c = graph(nodes, [E("toy-0","toy-1","similarity"),E("toy-1","toy-2","similarity")], {"similarity":"HARD_CLOSURE"})
        r = diagnose_reservations(nodes,c,[],self.ratios,hard_bounds={s:(0,1) for s in ["TRAIN","VALIDATION","TEST","CALIBRATION"]})
        self.assertIn("INDIVISIBLE_COMPONENT_EXCEEDS_REMAINING_CAPACITIES",r["errors"])

    def test_no_contradiction_is_not_feasibility_proof(self):
        nodes = [N("toy-a")]
        r = diagnose_reservations(nodes,graph(nodes),[],self.ratios)
        self.assertEqual(r["status"],"NO_CONTRADICTION_FOUND_NOT_FEASIBILITY_PROOF")

    def test_missing_eligible_nodes_rejected(self):
        with self.assertRaises(ValueError): diagnose_reservations([N("toy-a")],[],[],self.ratios)


class SyntheticEvaluation(unittest.TestCase):
    def test_singleton_and_empty_evaluation_rejected(self):
        nodes = [N("toy-a",joint_observations=2)]
        r = evaluate_toy_population(nodes,graph(nodes),{"toy-a":"TEST"},{"toy-a":"ORDINARY"})
        for split in ["VALIDATION","TEST"]:
            self.assertIn("EMPTY_OR_SINGLE_GROUP_EVALUATION",r["partitions"][split]["issues"])
            self.assertIn("STATISTICAL_ACCEPTANCE_UNSET",r["partitions"][split]["issues"])
        self.assertFalse(r["real_data_certified"])

    def test_observation_multiplicity_is_not_independent_groups(self):
        nodes = [N("toy-a",joint_observations=100)]
        r = evaluate_toy_population(nodes,graph(nodes),{"toy-a":"TEST"},{"toy-a":"ORDINARY"})
        self.assertEqual(r["partitions"]["TEST"]["counts"]["independent_groups"],1)

    def test_calibration_not_ordinary_validation(self):
        nodes = [N("toy-a",joint_observations=3)]
        r = evaluate_toy_population(nodes,graph(nodes),{"toy-a":"VALIDATION"},{"toy-a":"REWARD_DEVELOPMENT"})
        self.assertEqual(r["partitions"]["VALIDATION"]["counts"]["joint_observations"],0)

    def test_split_component_rejected(self):
        nodes = [N("toy-a"),N("toy-b")]
        c = graph(nodes,[E("toy-a","toy-b","exact_identity")],{"exact_identity":"HARD_CLOSURE"})
        with self.assertRaises(ValueError): evaluate_toy_population(nodes,c,{"toy-a":"TRAIN","toy-b":"TEST"},{n.identity:"ORDINARY" for n in nodes})

    def test_coverage_and_joint_identities_are_separate(self):
        nodes = [N("toy-a",joint_observations=4),N("toy-b",joint_observations=0,band="385-500",method="NMR")]
        r = evaluate_toy_population(nodes,graph(nodes),{n.identity:"TEST" for n in nodes},{n.identity:"ORDINARY" for n in nodes})
        c = r["partitions"]["TEST"]
        self.assertEqual(c["counts"],{"unique_sequences":2,"independent_groups":2,"joint_observations":4,"joint_identities":1})
        self.assertEqual(c["length_bands"]["385-500"]["joint_observations"],0)
        self.assertEqual(r["coverage_acceptance"],"UNRESOLVED")

    def test_numeric_minima_only_synthetic_illustration(self):
        nodes = [N("toy-a",joint_observations=1),N("toy-b",joint_observations=1)]
        r = evaluate_toy_population(nodes,graph(nodes),{n.identity:"TEST" for n in nodes},{n.identity:"ORDINARY" for n in nodes},{k:3 for k in ["unique_sequences","independent_groups","joint_observations","joint_identities"]})
        self.assertIn("BELOW_MIN_INDEPENDENT_GROUPS",r["partitions"]["TEST"]["issues"])
        self.assertFalse(r["real_data_certified"])

    def test_mixed_control_component_cannot_be_ordinary_evaluation(self):
        nodes = [N("toy-a",joint_observations=1),N("toy-b",joint_observations=1)]
        c = graph(nodes,[E("toy-a","toy-b","similarity")],{"similarity":"HARD_CLOSURE"})
        r = evaluate_toy_population(nodes,c,{n.identity:"VALIDATION" for n in nodes},{"toy-a":"ORDINARY","toy-b":"REWARD_DEVELOPMENT"})
        self.assertIn("CONTROL_COMPONENT_MIXED_WITH_ORDINARY_EVALUATION",r["partitions"]["VALIDATION"]["issues"])
