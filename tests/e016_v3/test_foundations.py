import copy, json, math, hashlib
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from protein_joint_sac.contracts import *
from protein_joint_sac.geometry.ca_builder import append, inverse, replay, mask_for_step, action_from_positions
from protein_joint_sac.geometry.local_frames import features
from protein_joint_sac.chemistry.reference_adapter import load, reference_features
from protein_joint_sac.data.bc_prefix_sampler import diagnostic_prefix, diagnostic_target
from protein_joint_sac.models.hybrid_distributions import GeometryDistribution
from protein_joint_sac.models.dense_geometry_attention import ModelConfig, DenseBlock
from protein_joint_sac.models.joint_actor import JointActor
from protein_joint_sac.models.sequence_only_actor import SequenceOnlyActor
from protein_joint_sac.models.build_critic import TwinCritics
from protein_joint_sac.models.rope import apply_rope
from protein_joint_sac.rl.soft_targets import *
from protein_joint_sac.rl.hybrid_sac_build import HybridSAC
from protein_joint_sac.oracle.mock_oracle import MockOracle
from protein_joint_sac.oracle.queue import EvaluationQueue
from protein_joint_sac.rl.delayed_episode_replay import EpisodeReplay
from protein_joint_sac.rl.checkpointing import save_atomic, load_checked
from protein_joint_sac.env.edit_env import propose, PendingEdit
from protein_joint_sac.models.edit_actor import EditActor, EditCritic
from protein_joint_sac.search.diverse_segment_beam import SearchBudget, toy_beam

P = Path("/mnt/d/Simone/proteinGen_data/pdb_only_v3_20261013")


def trace(N=20):
    out = []
    for i in range(N):
        m = mask_for_step(i)
        g = torch.tensor(
            [3.8 if m[0] else 0.0, 1.9 if m[1] else 0.0, 0.7 * math.sin(i) + 0.2 if m[2] else 0.0], dtype=torch.float64
        )
        out.append(BuildAction(AA[i % 20], g, m))
    return tuple(out)


def state(t=5):
    return replay(20, trace()[:t])


def protocol():
    return OracleProtocol("MOCK_UNIT_TEST", "synthetic_v1", (), "none", "none", (7,), 0, "SYNTHETIC_ONLY")


def protein():
    return GeneratedProtein(replay(20, trace()))


@pytest.mark.parametrize("N", [20, 64, 128, 256, 500])
def test_builder_api_all_lengths(N):
    actions = trace(N)
    s = replay(N, actions)
    assert s.t == N and torch.isfinite(s.ca).all()
    for i in [0, 1, 2, 3, N - 1]:
        g, m = inverse(s.ca[: i + 1])
        torch.testing.assert_close(g, actions[i].geometry, atol=2e-11, rtol=2e-11)
        assert m == actions[i].mask


def test_bootstrap_gauge():
    s = state(3)
    torch.testing.assert_close(s.ca[0], torch.zeros(3, dtype=torch.float64))
    assert s.ca[1, 0] > 0 and s.ca[1, 1] == s.ca[1, 2] == 0
    assert s.ca[2, 2] == 0 and s.ca[2, 1] > 0


def rotation():
    q, _ = torch.linalg.qr(torch.tensor([[0.3, 0.7, 0.2], [0.9, 0.2, -0.5], [0.1, -0.6, 0.8]], dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_builder_SO3_translation_and_chirality():
    s = state()
    a = trace()[5]
    R = rotation()
    shift = torch.tensor([10.0, -2.0, 4.0], dtype=torch.float64)
    transformed = BuildState(s.residues, s.ca @ R.T + shift, s.N)
    x = append(transformed, a).ca[-1]
    torch.testing.assert_close(x, append(s, a).ca[-1] @ R.T + shift, atol=1e-10, rtol=1e-10)
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    g = inverse(s.ca)[0]
    gm = inverse(s.ca @ reflection)[0]
    torch.testing.assert_close(g[:2], gm[:2])
    torch.testing.assert_close(g[2], -gm[2])


def test_geometry_gradcheck():
    s = state()
    g = trace()[5].geometry.clone().requires_grad_()
    assert torch.autograd.gradcheck(
        lambda v: append(s, BuildAction("G", v, (True, True, True))).ca[-1], (g,), eps=1e-6, atol=1e-5
    )


def test_degenerate_builder_rejected():
    s = BuildState(
        ("A", "G", "P"), torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64), 20
    )
    with pytest.raises(ValueError, match="degenerate"):
        append(s, trace()[3])
    f = features(s)
    assert not f["frame_valid"].any()
    assert torch.isfinite(f["right_handed_frame"]).all()


def test_closed_state_missing_tokens_plans():
    with pytest.raises(ValueError):
        BuildState(("A",), torch.zeros((2, 3)), 20)
    for a in ["MSE", "UNK", "BOS", "EOS", "PAD"]:
        with pytest.raises(ValueError):
            BuildAction(a, torch.zeros(3), (False, False, False))
    with pytest.raises(ValueError):
        PlanRef("TEACHER_PLAN_ONLY", "a" * 64)
    with pytest.raises(TypeError):
        BuildState((), torch.zeros((0, 3)), 20, oracle_result="future")


@pytest.mark.parametrize("entry,chain", [("3fm1", "A"), ("7z53", "W"), ("9qex", "E"), ("6zk9", "b"), ("6zkc", "b")])
def test_actual_mapping_14_feature_equivalence(entry, chain):
    parsed = json.loads((P / "tests" / f"{entry}_actual.json").read_text())
    o = next(o for o in parsed["observations"] if chain in o["auth_asym_ids"])
    s = diagnostic_prefix(o, 16, parsed["source_sha256"])
    ref = reference_features(s)
    f = features(s)
    assert len(load().APPROVED_NAMES) == 14
    for k in [
        "chemistry",
        "identity_onehot",
        "CA_centered_A",
        "CA_valid",
        "right_handed_frame",
        "frame_valid",
        "trailing_CA_bond_A",
        "CA_bond_valid",
        "trailing_CA_bend_rad",
        "CA_bend_valid",
        "trailing_CA_virtual_torsion_rad",
        "CA_torsion_valid",
        "prefix_crowding",
        "prefix_exposure_proxy",
        "prefix_nonpolar_packing_proxy",
        "generated_terminal_flags",
        "generated_plan_values",
    ]:
        np.testing.assert_allclose(f[k].detach().numpy(), ref[k], atol=1e-7, rtol=1e-7, err_msg=k)
    pairs = ref["pair_indices"]
    np.testing.assert_allclose(f["pair_distance_A"].detach().numpy()[pairs[:, 0], pairs[:, 1]], ref["pair_distance_A"])
    ori = np.concatenate(
        [
            f["pair_local_direction"].detach().numpy()[pairs[:, 0], pairs[:, 1]],
            f["pair_local_direction"].detach().numpy()[pairs[:, 1], pairs[:, 0]],
        ],
        -1,
    )
    np.testing.assert_allclose(ori, ref["pair_local_orientations"], atol=1e-7)
    if entry in ["6zk9", "6zkc"]:
        assert not s.valid.any()
        with pytest.raises(ValueError, match="missing"):
            diagnostic_target(o, 16, parsed["source_sha256"])


@pytest.mark.parametrize("mode", ["BC", "SAC", "INFERENCE"])
def test_same_chemistry_definition_modes(mode):
    ref = reference_features(state(), mode)
    base = reference_features(state(), "INFERENCE")
    assert load().fingerprint(ref) == load().fingerprint(base)


def test_suffix_swap_actual_actor_no_future():
    d = json.loads((P / "tests/7z53_actual.json").read_text())
    o = next(o for o in d["observations"] if "W" in o["auth_asym_ids"])
    m = copy.deepcopy(o)
    t = 12
    m["canonical_full_sequence"] = o["canonical_full_sequence"][:t] + "W" * (len(o["canonical_full_sequence"]) - t)
    for row in m["residue_mapping"][t:]:
        row["ca"] = [500.0, -200.0, 30.0]
    m["native_DSSP"] = "POISON"
    m["native_full_SASA"] = "POISON"
    m["observation_id"] = "DIFFERENT_ORIGIN"
    m["alternate_selection_ambiguous"] = True
    torch.manual_seed(1)
    actor = JointActor()
    s = diagnostic_prefix(o, t, d["source_sha256"])
    s2 = diagnostic_prefix(m, t, d["source_sha256"])
    torch.testing.assert_close(actor.logits(s), actor.logits(s2), atol=0, rtol=0)
    with pytest.raises(ValueError):
        actor.logits(m)
    assert features(s)["pair_distance_A"].shape == (t, t)


def test_prefix_normalization_and_local_alt_mask():
    d = json.loads((P / "tests/7z53_actual.json").read_text())
    o = next(o for o in d["observations"] if "W" in o["auth_asym_ids"])
    s = diagnostic_prefix(o, 3, d["source_sha256"])
    torch.testing.assert_close(
        features(s)["CA_centered_A"].mean(0), torch.zeros(3, dtype=torch.float64), atol=1e-12, rtol=0
    )
    m = copy.deepcopy(o)
    m["backbone_alternates"]["1"] = {"ambiguous_occupancy_tie": True}
    assert not diagnostic_prefix(m, 3, d["source_sha256"]).valid[0]


def test_chemistry_gly_pro_cys_and_modified():
    m = load()
    assert m.LUT["G"][4] == 0 and m.LUT["P"][4] == 3 and m.LUT["C"][3] == 1
    with pytest.raises(ValueError):
        BuildState(("MSE",), torch.zeros((1, 3)), 20)


def test_rope_norm_relative_position_and_even_dimension():
    x = torch.randn((2, 5, 8), dtype=torch.float64)
    pos = torch.arange(5)
    y = apply_rope(x, pos)
    torch.testing.assert_close(x.square().sum(-1), y.square().sum(-1))
    q = torch.randn((1, 5, 8), dtype=torch.float64)
    k = torch.randn_like(q)
    a = apply_rope(q, pos) @ apply_rope(k, pos).transpose(-1, -2)
    b = apply_rope(q, pos + 11) @ apply_rope(k, pos + 11).transpose(-1, -2)
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError):
        ModelConfig(width=30, heads=2)


def test_prefix_bidirectional_and_SDPA_fallback():
    torch.manual_seed(4)
    c = ModelConfig(layers=1)
    a = DenseBlock(c).double()
    b = DenseBlock(replace(c, backend="sdpa")).double()
    b.load_state_dict(a.state_dict())
    x = torch.randn((4, 32), dtype=torch.float64, requires_grad=True)
    bias = torch.randn((4, 4, 4), dtype=torch.float64) * 0.1
    ya = a(x, bias)
    yb = b(x, bias)
    torch.testing.assert_close(ya, yb, atol=1e-10, rtol=1e-10)
    gradient = torch.autograd.grad(ya[0].sum(), x)[0]
    assert gradient[-1].abs().sum() > 0  # last present token affects first present token


def test_actor_SO3_and_sequence_only_geometry_removal():
    torch.manual_seed(3)
    a = JointActor()
    s = state()
    R = rotation()
    s2 = BuildState(s.residues, s.ca @ R.T + 8.0, s.N)
    torch.testing.assert_close(a.logits(s), a.logits(s2), atol=2e-6, rtol=2e-6)
    b = SequenceOnlyActor()
    scrambled = BuildState(s.residues, torch.randn_like(s.ca), s.N, torch.zeros(s.t, dtype=torch.bool))
    torch.testing.assert_close(b(s), b(scrambled), atol=0, rtol=0)


def test_AA_conditioning_masks_and_seed_replay():
    torch.manual_seed(2)
    a = JointActor()
    for t in [0, 1, 2, 3]:
        s = state(t)
        g, la, lg = a.sample(s)
        assert g.mask == mask_for_step(t)
        assert torch.isfinite(la + lg)
        torch.testing.assert_close(a.geometry_distribution(s, g.residue).log_prob(g.geometry), lg, atol=1e-5, rtol=1e-5)
    s = state()
    assert not torch.equal(a.geometry_distribution(s, "A").mu, a.geometry_distribution(s, "W").mu)
    torch.manual_seed(77)
    x = a.sample(s)
    torch.manual_seed(77)
    y = a.sample(s)
    assert x[0].residue == y[0].residue
    torch.testing.assert_close(x[0].geometry, y[0].geometry)


def test_candidate_Q_gradients_and_independent_targets():
    torch.manual_seed(8)
    q = TwinCritics()
    s = state()
    g = trace()[5].geometry.clone().requires_grad_()
    a = BuildAction("W", g, (True, True, True))
    v = q.q1(s, a)
    grad = torch.autograd.grad(v, g)[0]
    assert torch.isfinite(grad).all() and (grad.abs() > 1e-9).all()
    assert not ({p.data_ptr() for p in q.q1.parameters()} & {p.data_ptr() for p in q.q2.parameters()})
    assert all(not p.requires_grad for p in q.target1.parameters())
    old = [p.clone() for p in q.target1.parameters()]
    with torch.no_grad():
        for p in q.q1.parameters():
            p.add_(1.0)
    q.polyak(0.25)
    for p, o in zip(q.target1.parameters(), old):
        torch.testing.assert_close(p, o + 0.25)


@pytest.mark.parametrize("mask", [mask_for_step(t) for t in range(4)])
def test_distribution_jacobian_and_logprob(mask):
    mu = torch.tensor([0.3, -0.4, 0.2], dtype=torch.float64, requires_grad=True)
    dist = GeometryDistribution(mu, torch.zeros(3, dtype=torch.float64), mask)
    u = torch.tensor([0.2, 0.4, -0.5], dtype=torch.float64, requires_grad=True)
    g = dist.transform(u)
    jac = torch.autograd.functional.jacobian(dist.transform, u)
    m = torch.tensor(mask)
    if m.any():
        torch.testing.assert_close(torch.diagonal(jac)[m].abs().log(), dist.log_jacobian(u)[m])
    lp = dist.log_prob(g)
    expected = torch.where(m, dist.base.log_prob(u) - dist.log_jacobian(u), torch.zeros_like(u)).sum()
    torch.testing.assert_close(lp, expected)
    lp.backward()
    assert torch.isfinite(mu.grad).all()


def test_torsion_seam_explicit_and_density_normalization():
    d = GeometryDistribution(
        torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), (False, False, True)
    )
    grid = torch.linspace(-math.pi + 1e-6, math.pi - 1e-6, 10000, dtype=torch.float64)
    v = grid / math.pi
    u = torch.atanh(v)
    density = torch.exp(torch.distributions.Normal(0.0, 1.0).log_prob(u) - (math.log(math.pi) + torch.log1p(-v * v)))
    assert abs(float(torch.trapezoid(density, grid)) - 1) < 1e-4
    for tau in [-math.pi, math.pi]:
        assert torch.isneginf(d.wrapped_log_prob(torch.tensor([0.0, 0.0, tau], dtype=torch.float64)))
    assert torch.isfinite(d.log_prob(torch.tensor([0.0, 0.0, math.pi - 1e-5], dtype=torch.float64)))


def test_bellman_terminal_and_entropy_and_nstep():
    r = torch.tensor([1.0, 1.0])
    done = torch.tensor([False, True])
    v = soft_target(
        r, done, torch.tensor([5.0, 5.0]), torch.tensor([-1.0, -1.0]), torch.tensor([-2.0, -2.0]), 0.9, 0.2, 0.3
    )
    torch.testing.assert_close(v, torch.tensor([6.22, 1.0]))
    assert monte_carlo_returns([0.0, 0.0, 4.0], 0.5) == [1.0, 2.0, 4.0]
    assert n_step_target([1.0, 2.0], 0.5, 8.0, False) == 4.0
    assert n_step_target([1.0, 2.0], 0.5, 8.0, True) == 2.0
    T = Temperatures(2.0, -0.5)
    loss = T.loss(torch.tensor(-0.1), torch.tensor(-0.1), False)
    loss.backward()
    assert T.log_aa.grad < 0 and T.log_geom.grad is None
    with pytest.raises(ValueError):
        Temperatures(4.0, 0.0)


def test_discrete_exact_gradient_equals_expected_score_estimator():
    logits = torch.tensor([0.3, -0.4, 0.8], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)
    exact = discrete_objective(logits, q, 0.0)
    g = torch.autograd.grad(exact, logits, retain_graph=True)[0]
    sampled = torch.stack(
        [
            torch.autograd.grad(sampled_discrete_objective(logits, i, q[i]), logits, retain_graph=True)[0]
            * logits.softmax(0)[i].detach()
            for i in range(3)
        ]
    ).sum(0)
    torch.testing.assert_close(g, sampled)


def test_known_discrete_continuous_toy_optimum():
    torch.manual_seed(12)
    logits = torch.nn.Parameter(torch.zeros(20))
    mu = torch.nn.Parameter(torch.tensor(-1.0))
    opt = torch.optim.Adam([logits, mu], lr=0.08)
    values = torch.zeros(20)
    values[7] = 3.0
    for _ in range(160):
        loss = discrete_objective(logits, values, 0.2) + (mu - 1.25).square()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert logits.softmax(0)[7] > 0.98 and abs(float(mu.detach()) - 1.25) < 0.005


def test_cpu_hybrid_SAC_update_reference():
    torch.manual_seed(3)
    a = JointActor(ModelConfig(width=16, heads=2, layers=1))
    q = TwinCritics(ModelConfig(width=16, heads=2, layers=1))
    s = state(3)
    action = trace()[3]
    next_s = append(s, action)
    alg = HybridSAC(a, q)
    metrics = alg.update_one(s, action, 0.1, next_s, False)
    assert all(math.isfinite(metrics[k]) for k in ["critic_loss", "actor_loss", "temperature_loss"])


def test_mock_queue_cache_distinct_geometry_and_budget(d_tmp):
    p = protein()
    proto = protocol()
    oracle = MockOracle(proto)
    path = d_tmp / "queue.sqlite"
    q = EvaluationQueue(path, max_calls=1)
    q.enqueue("one", p, proto, 7)
    q.enqueue("two", GeneratedProtein(BuildState(p.state.residues, p.state.ca + 2, p.state.N)), proto, 7)
    assert q.claim("one")
    result = oracle.evaluate(p, 7)
    q.finish("one", result)
    assert q.claim("two") is False
    assert q.row("two")["status"] == "SUCCEEDED" and q.row("one")["geometry_sha"] != q.row("two")["geometry_sha"]
    assert q.calls_used == oracle.calls == 1
    q.close()
    q = EvaluationQueue(path, max_calls=1)
    q.enqueue("new-seq", GeneratedProtein(BuildState(("A",) * 20, p.state.ca, 20)), proto, 7)
    with pytest.raises(ValueError, match="cap"):
        q.claim("new-seq")
    with pytest.raises(ValueError):
        q.enqueue("one", GeneratedProtein(BuildState(("A",) * 20, p.state.ca, 20)), proto, 7)
    q.close()


def test_queue_resume_interrupted_no_zero(d_tmp):
    path = d_tmp / "q.sqlite"
    q = EvaluationQueue(path)
    q.enqueue("x", protein(), protocol(), 7)
    q.claim("x")
    q.close()
    q = EvaluationQueue(path)
    assert q.recover_running() == 1
    assert q.row("x")["status"] == "TIMEOUT" and q.calls_used == 1
    q.close()


def test_delayed_replay_reward_attachment(d_tmp):
    path = d_tmp / "r.sqlite"
    r = EpisodeReplay(path)
    e = BuildEpisode("x", 20, trace())
    r.store(e)
    assert r.resolved_ids() == []
    with pytest.raises(ValueError, match="UNKNOWN"):
        r.oracle_return("x")
    r.close()
    r = EpisodeReplay(path)
    p = GeneratedProtein(replay(20, r.episode("x").actions))
    torch.testing.assert_close(p.state.ca, protein().state.ca, atol=0, rtol=0)
    card = RewardCard(
        "synthetic_v1",
        p.sequence_sha256,
        p.geometry_sha256,
        protocol().protocol_sha256,
        "MOCK_UNIT_TEST",
        {"synthetic_terminal": 3.0},
    )
    r.attach("x", card)
    assert r.resolved_ids() == ["x"]
    with pytest.raises(ValueError):
        r.attach("x", replace(card, components={"synthetic_terminal": 4.0}))
    r.close()
    with pytest.raises(ValueError):
        BuildEpisode("beam", 20, trace(), "BEAM_SELECTED")


def test_edit_separation_STOP_noop_delta_and_call_budget():
    p = protein()
    oracle = MockOracle(protocol())
    result = oracle.evaluate(p, 7)
    s = EditState(p, result, 1)
    assert propose(s, EditAction(None, None, True)) is s
    with pytest.raises(ValueError):
        propose(s, EditAction(0, p.sequence[0]))
    pending = propose(s, EditAction(0, "W"))
    assert isinstance(pending, PendingEdit)
    next_result = oracle.evaluate(pending.protein, 7)
    n, reward = pending.resolve(next_result, 1.0, 1.5, 0.05, 0.1)
    assert reward == pytest.approx(0.35) and n.remaining_calls == 0 and len(n.history) == 1
    with pytest.raises(ValueError):
        propose(n, EditAction(0, "A"))
    with pytest.raises(ValueError):
        JointActor().logits(s)
    logits = EditActor()(s)
    assert len(logits) == 401 and torch.isneginf(logits[AA.index(p.sequence[0])])
    assert EditCritic()(s, EditAction(None, None, True)).isfinite()
    with pytest.raises(ValueError):
        EditState(p, replace(result, sequence_sha256="f" * 64), 1)


def test_mock_guard_atomic_checkpoint_rng(d_tmp):
    with pytest.raises(ValueError):
        MockOracle(replace(protocol(), backend="AlphaFold", scope="PLANNED_UNCALIBRATED"))
    with pytest.raises(ValueError):
        replace(MockOracle(protocol()).evaluate(protein(), 7), plddt=(90.0,) * 20)
    torch.manual_seed(5)
    rng = torch.get_rng_state()
    path = d_tmp / "toy.pt"
    save_atomic(path, {"rng": rng, "tensor": torch.tensor([1.0, 2.0])})
    v = load_checked(path)
    torch.testing.assert_close(v["rng"], rng)
    with path.open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError):
        load_checked(path)


def test_beam_budget_and_E015_gated():
    b = SearchBudget(10, 2)
    out = toy_beam("", lambda s, n: [s + "A", s + "C"], lambda s: s.count("A"), 2, 2, b)
    assert b.proposals == 6 and b.completed == 2 and out[0] == "AA"
    with pytest.raises(ValueError):
        toy_beam("", lambda s, n: [s + "A", s + "C"], len, 5, 2, SearchBudget(2, 2))
    from protein_joint_sac.planner.e015_optional import E015PlanAdapter

    with pytest.raises(RuntimeError):
        E015PlanAdapter().sample()


def test_real_PDB_roundtrip_after_gauge_alignment():
    d = json.loads((P / "tests/7z53_actual.json").read_text())
    o = next(o for o in d["observations"] if "W" in o["auth_asym_ids"])
    s = diagnostic_prefix(o, 16, d["source_sha256"])
    actions = tuple(action_from_positions(s.residues[i], s.ca[: i + 1]) for i in range(16))
    built = replay(s.N, actions)
    X = built.ca - built.ca.mean(0)
    Y = s.ca - s.ca.mean(0)
    u, _, vh = torch.linalg.svd(X.T @ Y)
    D = torch.eye(3, dtype=X.dtype)
    D[-1, -1] = torch.det(u @ vh)
    torch.testing.assert_close(X @ (u @ D @ vh), Y, atol=1e-9, rtol=1e-9)


def test_real_NMR_models_remain_observations():
    import gzip

    j = [json.loads(l) for l in (P / "DOWNLOAD_JOURNAL.jsonl").read_text().splitlines()]
    row = next(r for r in j if r.get("entry_id") == "9RPN" and r["status"] == "DONE")
    with gzip.open(row["parsed_path"], "rt") as f:
        data = json.load(f)
    for model in [1, 25]:
        o = next(o for o in data["observations"] if o["model_num"] == model)
        s = diagnostic_prefix(o, 8, data["source_sha256"])
        assert features(s)["pair_distance_A"].shape == (8, 8)


def test_completed_agreement_recomputed_for_each_proposal():
    from protein_joint_sac.oracle.fold_agreement import compare_completed

    p = protein()
    result = MockOracle(protocol()).evaluate(p, 7)
    first = compare_completed(p, result)
    xyz = p.state.ca.clone()
    xyz[5] += torch.tensor([20.0, -12.0, 10.0], dtype=xyz.dtype)
    changed = GeneratedProtein(BuildState(p.state.residues, xyz, 20))
    second = compare_completed(changed, result)
    assert (
        first["candidate_geometry_sha256"] != second["candidate_geometry_sha256"]
        and first["fixed_mapping_Kabsch_RMSD_A"] != second["fixed_mapping_Kabsch_RMSD_A"]
    )
    assert first["optimized_TM_score"] is None
    with pytest.raises(ValueError):
        compare_completed(state(5), result)


def test_supervision_next_label_not_actor_input():
    from protein_joint_sac.rl.bc import diagnostic_bc_loss

    s = state(3)
    a = JointActor()
    before = a.logits(s).clone()
    target = BCPrefixTarget(s, trace()[3], "a" * 64, 4)
    loss, metrics = diagnostic_bc_loss(a, target)
    assert loss.isfinite()
    torch.testing.assert_close(a.logits(s), before, atol=0, rtol=0)
    assert metrics["scope"] == "DIAGNOSTIC_ONLY_NOT_PRODUCTION_TRAINING"


def test_uncertified_TRAIN_statistics_and_heldout_rejected(d_tmp):
    from protein_joint_sac.data.pdb_v3_adapter import require_certified, inspect

    assert inspect(P)["training_certified"] is False
    with pytest.raises(ValueError):
        require_certified(DatasetRef(str(P), "a" * 64, "b" * 64, "PROVISIONAL"))
    m = {
        "status": "SEALED",
        "scope": "synthetic_unit_test",
        "splits": {"TRAIN": ["tr1", "tr2"], "VALIDATION": ["va"], "TEST": ["te"]},
        "homology_groups": {"tr1": "g1", "tr2": "g2", "va": "g3", "te": "g4"},
    }
    path = d_tmp / "membership.json"
    path.write_text(json.dumps(m))
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        load().fit_train_angle_prior(
            [{"sequence_id": "va", "residue": "A", "angle": 0.3}], path, expected_manifest_sha256=h
        )
    with pytest.raises(ValueError):
        load().fit_train_angle_prior([], path, expected_manifest_sha256="f" * 64)


def test_sampled_soft_value_target_only_and_terminal_features():
    alg = HybridSAC(
        JointActor(ModelConfig(width=16, heads=2, layers=1)), TwinCritics(ModelConfig(width=16, heads=2, layers=1))
    )
    with pytest.raises(ValueError, match="target-only"):
        alg.soft_value(state(3), estimator="sampled")
    with torch.no_grad():
        assert alg.soft_value(state(3), targets=True, estimator="sampled").isfinite()
    partial = features(state(5))["generated_terminal_flags"]
    complete = features(protein().state)["generated_terminal_flags"]
    assert partial[-1, 2] and not partial[-1, 1] and complete[-1, 1] and not complete[-1, 2]


def test_terminal_target_ignores_unavailable_next_value():
    out = soft_target(
        torch.tensor(1.0),
        torch.tensor(True),
        torch.tensor(float("nan")),
        torch.tensor(float("nan")),
        torch.tensor(float("nan")),
        0.99,
        0.2,
        0.2,
    )
    assert out == 1 and out.isfinite()


def test_chemistry_ablation_disables_pair_compatibility():
    from protein_joint_sac.geometry.dense_pair_bias import DensePairBias

    f = features(state(6))
    bias = DensePairBias(2, chemistry=False).double()
    before = bias(f)
    altered = dict(f)
    altered["chemistry"] = f["chemistry"] * 1000
    torch.testing.assert_close(bias(altered), before, atol=0, rtol=0)


def test_queue_budget_frozen_across_resume(d_tmp):
    path = d_tmp / "frozen_budget.sqlite"
    q = EvaluationQueue(path, max_requests=3, max_calls=1)
    q.close()
    with pytest.raises(ValueError, match="budget changed"):
        EvaluationQueue(path, max_requests=3, max_calls=2)
