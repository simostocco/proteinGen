"""CPU only allocator contract for the Phase 3I.2 pilot lifecycle."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from protein_distance_diffusion.training.e007_local_backbone_repair import CudaMemoryTelemetry

MIB = 2**20


class MockCuda:
    def __init__(self) -> None:
        self.current = (0, 0, 0)
        self.peak = (0, 0)
        self.resets = 0

    def get_device_properties(self, _device):
        return SimpleNamespace(total_memory=8151 * MIB)

    def synchronize(self, _device):
        pass

    def memory_allocated(self, _device):
        return self.current[0] * MIB

    def memory_reserved(self, _device):
        return self.current[1] * MIB

    def memory_stats(self, _device):
        return {"active_bytes.all.current": self.current[2] * MIB}

    def max_memory_allocated(self, _device):
        return self.peak[0] * MIB

    def max_memory_reserved(self, _device):
        return self.peak[1] * MIB

    def reset_peak_memory_stats(self, _device):
        self.resets += 1
        self.peak = self.current[:2]


def test_separate_phase_peaks_and_two_arms_take_maximum() -> None:
    cuda = MockCuda()
    telemetry = CudaMemoryTelemetry(cuda, "cuda:0")
    assert cuda.resets == 1
    for allocated, reserved in ((5000, 6284), (3500, 4022), (4700, 6100)):
        cuda.current = (100, 128, 90)
        cuda.peak = (allocated, reserved)
        observed = telemetry.end_phase()
        assert observed["current_cuda_reserved_mib"] == 128
        assert observed["current_cuda_active_bytes"] == 90 * MIB
    assert observed["run_peak_cuda_reserved_mib"] == 6284
    assert observed["run_peak_cuda_allocated_mib"] == 5000
    assert observed["run_peak_cuda_reserved_mib"] != 10306
    assert cuda.resets == 4
    other_arm = CudaMemoryTelemetry(cuda, "cuda:0")
    other_arm.run_allocated = telemetry.run_allocated
    other_arm.run_reserved = telemetry.run_reserved
    cuda.current = (200, 256, 180)
    cuda.peak = (5100, 6500)
    assert other_arm.end_phase()["run_peak_cuda_reserved_mib"] == 6500
    assert other_arm.run_allocated == 5100


def test_bounded_pilot_startup_audits_evaluation_and_25_updates_without_cuda() -> None:
    cuda = MockCuda()
    telemetry = CudaMemoryTelemetry(cuda, "cuda:0")
    observed_boundaries = []
    # Pilot startup evaluates update zero, then audits zero through ten.
    phases = [("evaluation_0", 4022)] + [(f"audit_{index}", 6284 if index == 0 else 3900) for index in range(11)]
    phases += [(f"training_{index}", 4100) for index in range(1, 26)]
    phases.append(("audit_25", 4000))
    for name, reserved in phases:
        cuda.current = (100, 128, 90)
        cuda.peak = (min(reserved - 100, 5871), reserved)
        observed_boundaries.append((name, telemetry.end_phase()))
    assert len(observed_boundaries) == 38
    assert observed_boundaries[-1][1]["run_peak_cuda_reserved_mib"] == 6284
    assert max(row["phase_peak_cuda_reserved_mib"] for _, row in observed_boundaries) == 6284
    assert cuda.resets == len(phases) + 1


def test_prepared_v3_smoke_paths_and_authorizations_are_separate() -> None:
    config = yaml.safe_load(Path("configs/e007_local_backbone_repair_pilot_phase3i2_reviewed_v6_v3.yaml").read_text())
    v2 = yaml.safe_load(Path("configs/e007_local_backbone_repair_pilot_phase3i2_reviewed_v6_v2.yaml").read_text())
    assert config["output_dir"].endswith("reviewed_v6_v3")
    assert config["lifecycle_smoke_output_dir"].endswith("lifecycle_smoke_v3")
    assert config["lifecycle_smoke_output_dir"] != config["output_dir"]
    assert config["pilot_authorized"] is False
    assert all(value is False for value in config["authorization"].values())
    for key in ("output_dir", "lifecycle_smoke_output_dir", "reviewed_v6_decision_path", "pilot_authorized"):
        config.pop(key, None)
        v2.pop(key, None)
    config["authorization"].pop("pilot_authorized")
    v2["authorization"].pop("pilot_authorized")
    assert config == v2


@pytest.mark.parametrize(
    ("current", "peak"),
    [
        ((100, 120, 90), (100, 10306)),
        ((100, 80, 90), (100, 120)),
        ((100, 120, 121), (100, 120)),
        ((100, 120, 90), (float("nan"), 120)),
        ((-1, 120, 0), (100, 120)),
    ],
)
def test_invalid_direct_observations_fail_closed(current, peak) -> None:
    cuda = MockCuda()
    telemetry = CudaMemoryTelemetry(cuda, "cuda:0")
    cuda.current, cuda.peak = current, peak
    with pytest.raises(MemoryError):
        telemetry.end_phase()
    assert cuda.resets == 1
