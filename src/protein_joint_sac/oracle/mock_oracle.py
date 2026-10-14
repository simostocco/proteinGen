"""Unit-test mock, never an AlphaFold substitute."""

from ..contracts import OracleResult, OracleStatus, digest


class MockOracle:
    def __init__(self, protocol):
        if protocol.backend != "MOCK_UNIT_TEST" or protocol.scope != "SYNTHETIC_ONLY":
            raise ValueError("mock cannot impersonate AF")
        self.protocol = protocol
        self.calls = 0

    def evaluate(self, protein, seed):
        self.calls += 1
        score = protein.sequence.count("A") / len(protein.sequence)
        coords = tuple((float(i) * 3.8, float(seed % 3), 0.0) for i in range(len(protein.sequence)))
        h = digest({"ca": coords, "score": score, "seed": seed})
        return OracleResult(
            protein.sequence_sha256,
            self.protocol.protocol_sha256,
            OracleStatus.SUCCEEDED,
            seed,
            "MOCK_UNIT_TEST",
            h,
            coords,
            synthetic_score=score,
        )
