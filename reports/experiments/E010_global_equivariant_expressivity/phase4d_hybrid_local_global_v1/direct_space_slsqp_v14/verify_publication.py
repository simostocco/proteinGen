import json
import tempfile
from pathlib import Path

from protein_distance_diffusion.training.e010_phase4d_diagnostic import file_hash
from scripts import report_e010_slsqp_v14 as report
from scripts import run_e010_slsqp_v14 as runner

original = runner.OUT

def forbidden(*args, **kwargs):
    raise AssertionError("Optimizer invocation forbidden in publication verification")

runner.sqp.minimize = forbidden
runner.sqp.direct.solve = forbidden
before = {k: file_hash(original / k) for k in ["result.json", "RESULTS.md", "publication_verification.json"]}
with tempfile.TemporaryDirectory(prefix="e010_v14_publication_") as name:
    folder = Path(name)
    for item in ["execution_contract.json", "preflight_complete.json", "untracked_states"]:
        (folder / item).symlink_to(original / item)
    runner.OUT = folder
    report.main()
    assert before == {k: file_hash(folder / k) for k in before}
assert before == {k: file_hash(original / k) for k in before}
assert not list((original / "attempts").glob("*.json"))
proof = dict(
    reports_byte_exact=True,
    fixed_preflight_state_certificates_exact=True,
    historical_and_input_hashes_verified=True,
    scientific_panel_optimizer_invocations=0,
    verification_optimizer_invocations=0,
    optimizer_functions_forbidden=True,
    original_reports_unchanged=True,
    helper_sha256=file_hash(__file__),
)
(original / "independent_publication_verification.json").write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
print("Independent publication byte-exact; original records unchanged; zero optimizer invocations.")
