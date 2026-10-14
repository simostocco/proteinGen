import os, sys, json, subprocess, copy
from pathlib import Path
import pytest
import torch
from protein_joint_sac.contracts import BuildState, BuildAction, GeneratedProtein, BuildEpisode, RewardCard
from protein_joint_sac.geometry.ca_builder import replay, mask_for_step, append
from protein_joint_sac.models.joint_actor import JointActor
from protein_joint_sac.models.sequence_only_actor import SequenceOnlyActor
from protein_joint_sac.models.build_critic import TwinCritics
from protein_joint_sac.models.dense_geometry_attention import ModelConfig
from protein_joint_sac.geometry.local_frames import features
from protein_joint_sac.rl.hybrid_sac_build import HybridSAC
from protein_joint_sac.rl.delayed_episode_replay import EpisodeReplay

BASE = Path("/mnt/d/Simone/proteinGen_data/e016_af_guided_v3_foundations_20261014_v1/source")


def actions(N=20):
    return tuple(
        BuildAction(
            "A",
            torch.tensor([3.8 if i else 0.0, 1.9 if i >= 2 else 0.0, 0.4 if i >= 3 else 0.0], dtype=torch.float64),
            mask_for_step(i),
        )
        for i in range(N)
    )


def state(t=5):
    return replay(20, actions()[:t])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_coordinate_dtypes_and_masks(dtype):
    s = state()
    s = BuildState(s.residues, s.ca.to(dtype), s.N)
    out = features(s)
    assert all(v.device == s.ca.device for v in out.values() if isinstance(v, torch.Tensor))
    assert out["CA_valid"].dtype == torch.bool
    assert torch.isfinite(JointActor().logits(s)).all()
    assert BuildState.empty(20, device="cpu", dtype=dtype).ca.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_geometry_rejected(dtype):
    with pytest.raises(ValueError):
        BuildState(("A",), torch.zeros((1, 3), dtype=dtype), 20)
    with pytest.raises(ValueError):
        BuildAction("A", torch.zeros(3, dtype=dtype), (False, False, False))


def test_device_mismatch_and_unapproved_devices():
    s = state()
    with pytest.raises(ValueError):
        BuildState(s.residues, s.ca, 20, torch.ones(s.t, dtype=torch.bool, device="meta"))
    with pytest.raises(ValueError):
        BuildState.empty(20, device="meta")
    model = JointActor().to("meta")
    with pytest.raises(ValueError, match="device mismatch"):
        model.logits(s)


def test_cpu_transfer_preserves_gradients():
    s = state()
    g = actions()[5].geometry.clone().requires_grad_()
    next_s = append(s, BuildAction("A", g, (True, True, True))).to("cpu")
    next_s.ca[-1].sum().backward()
    assert torch.isfinite(g.grad).all()


def test_foundation_checkpoint_and_output_compatibility(d_tmp):
    checkpoint = d_tmp / "foundation.pt"
    program = """import sys,json,torch
from protein_joint_sac.models.joint_actor import JointActor
from protein_joint_sac.models.build_critic import TwinCritics
from protein_joint_sac.contracts import BuildAction
from protein_joint_sac.geometry.ca_builder import replay,mask_for_step
torch.manual_seed(21);a=JointActor();q=TwinCritics();acts=tuple(BuildAction('A',torch.tensor([3.8 if i else 0.,1.9 if i>=2 else 0.,.4 if i>=3 else 0.],dtype=torch.float64),mask_for_step(i)) for i in range(5));s=replay(20,acts);torch.save({'actor':a.state_dict(),'critics':q.state_dict(),'logits':a.logits(s).detach()},sys.argv[1])
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE / "src")
    subprocess.run([sys.executable, "-B", "-c", program, str(checkpoint)], env=env, check=True)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    a = JointActor()
    q = TwinCritics()
    a.load_state_dict(saved["actor"], strict=True)
    q.load_state_dict(saved["critics"], strict=True)
    torch.testing.assert_close(a.logits(state()), saved["logits"], atol=0, rtol=0)


def test_streamed_exact_actor_matches_eager_gradient():
    torch.manual_seed(43)
    c = ModelConfig(width=16, heads=2, layers=1)
    actor = JointActor(c)
    q = TwinCritics(c)
    alg = HybridSAC(actor, q)
    s = state(3)
    for p in list(q.q1.parameters()) + list(q.q2.parameters()):
        p.requires_grad_(False)
    torch.manual_seed(44)
    loss = alg.actor_loss(s)
    loss.backward()
    expected = {n: p.grad.clone() if p.grad is not None else None for n, p in actor.named_parameters()}
    alg.actor_optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(44)
    value = alg.backward_actor_exact_streamed(s)
    assert value == pytest.approx(float(loss.detach()), abs=2e-6)
    for n, p in actor.named_parameters():
        if expected[n] is not None:
            torch.testing.assert_close(p.grad, expected[n], atol=3e-6, rtol=3e-5)


def test_exact_proposal_replay_backward_compatibility(d_tmp):
    r = EpisodeReplay(d_tmp / "replay.sqlite")
    e = BuildEpisode("new", 20, actions())
    p = GeneratedProtein(replay(20, e.actions))
    r.store(e, p)
    card = RewardCard("test", p.sequence_sha256, p.geometry_sha256, "a" * 64, "MOCK_UNIT_TEST", {"value": 1.0})
    r.attach("new", card)
    assert r.resolved_ids() == ["new"]
    old = BuildEpisode("old", 20, actions())
    r.store(old)
    r.attach("old", card)
    assert r.resolved_ids() == ["new", "old"]
    r.close()


# CUDA tests are supplied for the explicitly approved benchmark gate; never triggered by auto-detection.
@pytest.mark.skipif(
    os.environ.get("E016_CUDA_TEST_APPROVED") != "explicit", reason="CUDA execution not authorized in this task"
)
@pytest.mark.parametrize("N", [64, 128, 256, 500])
def test_approved_cuda_reconstruction(N):
    s = replay(N, actions(N), device="cuda")
    cpu = replay(N, actions(N))
    torch.testing.assert_close(s.ca.cpu(), cpu.ca, atol=1e-8, rtol=1e-8)


def test_benchmark_estimates_and_approval_guard(d_tmp):
    import importlib.util

    root = Path(os.environ["E016_ARTIFACT_ROOT"])
    p = root / "source/scripts/e016_v3/benchmark_readiness.py"
    spec = importlib.util.spec_from_file_location("readiness_benchmark", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    streamed = m.estimate(500, 128, 4, 4, "streamed")
    eager = m.estimate(500, 128, 4, 4, "eager")
    assert streamed["SAC_peak_budget_bytes"] < eager["SAC_peak_budget_bytes"] and streamed["status"].startswith(
        "INFERRED"
    )
    env = os.environ.copy()
    output = d_tmp / "forbidden.json"
    result = subprocess.run(
        [sys.executable, "-B", str(p), "--execute", "--output", str(output)],
        cwd=root / "source",
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "BLOCKED_EXPLICIT_APPROVAL" in result.stderr and not output.exists()


@pytest.mark.skipif(os.environ.get("E016_CUDA_TEST_APPROVED") != "explicit", reason="CUDA not authorized")
@pytest.mark.parametrize("N", [64, 128, 256, 500])
def test_approved_cuda_neural_parity_and_Q_gradient(N):
    torch.manual_seed(55)
    cpu_actor = JointActor()
    gpu_actor = copy.deepcopy(cpu_actor).to("cuda")
    cpu = replay(N, actions(N)[:-1])
    gpu = cpu.to("cuda")
    torch.testing.assert_close(gpu_actor.logits(gpu).cpu(), cpu_actor.logits(cpu), atol=5e-5, rtol=5e-5)
    assert all(v.device.type == "cuda" for v in features(gpu).values() if isinstance(v, torch.Tensor))
    q = TwinCritics().to("cuda")
    g = actions(N)[-1].geometry.to("cuda").requires_grad_()
    value = q.q1(gpu, BuildAction("A", g, (True, True, True)))
    gradient = torch.autograd.grad(value, g)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
