"""Closed build/edit schemas. All physical coordinates are C-alpha Angstroms."""

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import torch

AA = "ACDEFGHIKLMNPQRSTVWY"
GEOMETRY_VERSION = "ca_internal_signed_v1"


def digest(obj):
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().tolist()
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(value):
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("SHA256 required")


def d_path(path):
    p = Path(path).resolve()
    if not str(p).startswith("/mnt/d/"):
        raise ValueError("D-only path required")
    return p


@dataclass(frozen=True)
class DatasetRef:
    root: str
    manifest_sha256: str
    split_sha256: str
    status: str

    def __post_init__(self):
        d_path(self.root)
        sha(self.manifest_sha256)
        sha(self.split_sha256)
        if self.status not in ["CERTIFIED", "PROVISIONAL", "SYNTHETIC_ONLY"]:
            raise ValueError("explicit certification state required")


@dataclass(frozen=True)
class ChemistryRegistryRef:
    root: str
    registry_sha256: str
    implementation_sha256: str

    def __post_init__(self):
        d_path(self.root)
        sha(self.registry_sha256)
        sha(self.implementation_sha256)


@dataclass(frozen=True)
class PlanRef:
    kind: str = "NO_PLAN"
    source_sha256: str | None = None

    def __post_init__(self):
        if self.kind != "NO_PLAN":
            raise ValueError("native/E015 plans not approved in this foundation")
        if self.source_sha256 is not None:
            raise ValueError("NO_PLAN has no source encoding")


@dataclass(frozen=True)
class BuildState:
    residues: tuple[str, ...]
    ca: torch.Tensor
    N: int
    valid: torch.Tensor | None = None
    plan: PlanRef = field(default_factory=PlanRef)
    geometry_version: str = GEOMETRY_VERSION

    def __post_init__(self):
        if type(self.N) is not int or not 20 <= self.N <= 500:
            raise ValueError("N20..500 required")
        if type(self.residues) is not tuple or any(a not in AA or len(a) != 1 for a in self.residues):
            raise ValueError("canonical20 immutable prefix only")
        if len(self.residues) > self.N or self.ca.shape != (len(self.residues), 3):
            raise ValueError("literal prefix; no future/padded coordinates")
        if (
            self.ca.device.type not in ["cpu", "cuda"]
            or self.ca.dtype not in [torch.float32, torch.float64]
            or not torch.isfinite(self.ca).all()
        ):
            raise ValueError("finite FP32/FP64 coordinates on CPU/CUDA required")
        if self.valid is None:
            object.__setattr__(self, "valid", torch.ones(self.t, dtype=torch.bool, device=self.ca.device))
        if self.valid.shape != (self.t,) or self.valid.dtype != torch.bool or self.valid.device != self.ca.device:
            raise ValueError("explicit aligned validity")
        if type(self.plan) is not PlanRef or self.plan.kind != "NO_PLAN" or self.geometry_version != GEOMETRY_VERSION:
            raise ValueError("unapproved plan/geometry version")

    @property
    def t(self):
        return len(self.residues)

    @classmethod
    def empty(cls, N, *, device="cpu", dtype=torch.float64):
        return cls((), torch.empty((0, 3), dtype=dtype, device=device), N)

    def to(self, device):
        return BuildState(
            self.residues, self.ca.to(device), self.N, self.valid.to(device), self.plan, self.geometry_version
        )


@dataclass(frozen=True)
class BuildAction:
    residue: str
    geometry: torch.Tensor
    mask: tuple[bool, bool, bool]
    units: tuple[str, str, str] = ("angstrom", "radian", "radian")

    def __post_init__(self):
        if self.residue not in AA or len(self.residue) != 1:
            raise ValueError("chemical action canonical20 only")
        if (
            self.geometry.shape != (3,)
            or self.geometry.device.type not in ["cpu", "cuda"]
            or self.geometry.dtype not in [torch.float32, torch.float64]
            or not torch.isfinite(self.geometry).all()
        ):
            raise ValueError("three finite geometry slots required")
        if self.mask not in [(False, False, False), (True, False, False), (True, True, False), (True, True, True)]:
            raise ValueError("bootstrap geometry mask")
        if self.units != ("angstrom", "radian", "radian"):
            raise ValueError("units fixed")
        if any(float(self.geometry[i].detach()) != 0 for i in range(3) if not self.mask[i]):
            raise ValueError("undefined components must be zero")


@dataclass(frozen=True)
class GeneratedProtein:
    state: BuildState

    def __post_init__(self):
        if self.state.t != self.state.N or not self.state.valid.all():
            raise ValueError("complete generated protein required")

    @property
    def sequence(self):
        return "".join(self.state.residues)

    @property
    def sequence_sha256(self):
        return hashlib.sha256(self.sequence.encode()).hexdigest()

    @property
    def geometry_sha256(self):
        return digest(self.state.ca)


@dataclass(frozen=True)
class CandidateAction:
    state: BuildState
    action: BuildAction


@dataclass(frozen=True)
class BCPrefixTarget:
    state: BuildState
    next_action: BuildAction
    source_sha256: str
    label_seq_id: int
    role: str = "SUPERVISION_ONLY"

    def __post_init__(self):
        sha(self.source_sha256)
        if self.role != "SUPERVISION_ONLY" or self.label_seq_id != self.state.t + 1:
            raise ValueError("offline next target mapping")


@dataclass(frozen=True)
class ActorObservation:
    state: BuildState

    def __post_init__(self):
        if type(self.state) is not BuildState:
            raise ValueError("actor only BuildState")


@dataclass(frozen=True)
class QObservation:
    candidate: CandidateAction

    def __post_init__(self):
        if type(self.candidate) is not CandidateAction:
            raise ValueError("Q only state+candidate")


class OracleStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class OracleProtocol:
    backend: str
    code_revision: str
    weights_sha256: tuple[str, ...]
    msa_mode: str
    templates_mode: str
    seeds: tuple[int, ...]
    recycles: int
    scope: str

    def __post_init__(self):
        if self.scope not in ["SYNTHETIC_ONLY", "PLANNED_UNCALIBRATED", "CALIBRATED_EXTERNAL"]:
            raise ValueError("oracle scope")
        if self.scope == "SYNTHETIC_ONLY" and self.backend != "MOCK_UNIT_TEST":
            raise ValueError("mock must be explicitly named")
        if self.scope == "CALIBRATED_EXTERNAL" and (self.backend == "MOCK_UNIT_TEST" or not self.weights_sha256):
            raise ValueError("real calibrated backend and weights required")
        if not self.seeds or self.recycles < 0:
            raise ValueError("settings explicit")
        for h in self.weights_sha256:
            sha(h)

    @property
    def protocol_sha256(self):
        return digest(self.__dict__)


@dataclass(frozen=True)
class OracleResult:
    sequence_sha256: str
    protocol_sha256: str
    status: OracleStatus
    seed: int
    evidence_type: str
    predictor_output_sha256: str | None = None
    predicted_ca: tuple | None = None
    plddt: tuple | None = None
    ptm: float | None = None
    pae: tuple | None = None
    synthetic_score: float | None = None
    failure_reason: str | None = None

    def __post_init__(self):
        sha(self.sequence_sha256)
        sha(self.protocol_sha256)
        if self.evidence_type not in ["MOCK_UNIT_TEST", "EXTERNAL_PREDICTION"]:
            raise ValueError("explicit oracle evidence")
        if type(self.status) is not OracleStatus:
            raise ValueError("typed oracle status required")
        if self.predicted_ca is not None and (
            not self.predicted_ca or any(len(x) != 3 or not all(math.isfinite(v) for v in x) for x in self.predicted_ca)
        ):
            raise ValueError("finite ordered predicted CA mapping")
        if self.plddt is not None and (
            any(not math.isfinite(v) or not 0 <= v <= 100 for v in self.plddt)
            or self.predicted_ca is None
            or len(self.plddt) != len(self.predicted_ca)
        ):
            raise ValueError("confidence mapped to predictor positions, range0..100")
        if self.ptm is not None and (not math.isfinite(self.ptm) or not 0 <= self.ptm <= 1):
            raise ValueError("pTM range0..1")
        if self.predictor_output_sha256 is not None:
            sha(self.predictor_output_sha256)
        if self.evidence_type == "MOCK_UNIT_TEST" and any(x is not None for x in [self.plddt, self.ptm, self.pae]):
            raise ValueError("mock not AlphaFold confidence")
        if self.status == OracleStatus.SUCCEEDED and self.predictor_output_sha256 is None:
            raise ValueError("successful output must be hashed")
        if (
            self.status in [OracleStatus.FAILED, OracleStatus.TIMEOUT, OracleStatus.UNSUPPORTED]
            and not self.failure_reason
        ):
            raise ValueError("failure class required")


@dataclass(frozen=True)
class RewardCard:
    score_version: str
    sequence_sha256: str
    candidate_geometry_sha256: str
    oracle_protocol_sha256: str
    evidence_type: str
    components: dict
    calibrated: bool = False

    def __post_init__(self):
        for h in [self.sequence_sha256, self.candidate_geometry_sha256, self.oracle_protocol_sha256]:
            sha(h)
        if self.evidence_type not in ["MOCK_UNIT_TEST", "EXTERNAL_PREDICTION"]:
            raise ValueError("reward provenance")
        if self.calibrated and self.evidence_type == "MOCK_UNIT_TEST":
            raise ValueError("synthetic card not scientifically calibrated")


@dataclass(frozen=True)
class BuildEpisode:
    episode_id: str
    N: int
    actions: tuple[BuildAction, ...]
    behavior: str = "STOCHASTIC_POLICY"

    def __post_init__(self):
        if not self.episode_id or not 20 <= self.N <= 500 or len(self.actions) > self.N:
            raise ValueError("episode scope")
        if self.behavior != "STOCHASTIC_POLICY":
            raise ValueError("beam data not ordinary SAC replay")


@dataclass(frozen=True)
class ReplayRecord:
    episode: BuildEpisode
    reward_card: RewardCard | None
    stream: str

    def __post_init__(self):
        if self.stream not in ["CHEAP_ONLY", "ORACLE_RESOLVED"]:
            raise ValueError("explicit replay stream")
        if self.stream == "ORACLE_RESOLVED" and self.reward_card is None:
            raise ValueError("pending never terminal zero")


@dataclass(frozen=True)
class EditState:
    protein: GeneratedProtein
    oracle: OracleResult
    remaining_calls: int
    history: tuple = ()

    def __post_init__(self):
        if self.oracle.status != OracleStatus.SUCCEEDED or self.oracle.sequence_sha256 != self.protein.sequence_sha256:
            raise ValueError("edit needs already evaluated matching complete sequence")
        if self.oracle.predicted_ca is None or len(self.oracle.predicted_ca) != self.protein.state.N:
            raise ValueError("edit requires full mapped prediction coordinates")
        if self.remaining_calls < 0:
            raise ValueError("remaining budget nonnegative")


@dataclass(frozen=True)
class EditAction:
    position: int | None
    replacement: str | None
    stop: bool = False

    def __post_init__(self):
        if self.stop:
            if self.position is not None or self.replacement is not None:
                raise ValueError("STOP no mutation")
        elif type(self.position) is not int or self.position < 0 or self.replacement not in AA:
            raise ValueError("one substitution")


@dataclass(frozen=True)
class EditEpisode:
    episode_id: str
    states: tuple[EditState, ...]
    actions: tuple[EditAction, ...]

    def __post_init__(self):
        if len(self.actions) > len(self.states):
            raise ValueError("evaluated edit-state history required")
