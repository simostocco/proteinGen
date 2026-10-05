from __future__ import annotations

import io
import json
import tarfile
import urllib.request
from pathlib import Path

import pytest
import yaml

import protein_distance_diffusion.evaluation.e007_pretrained_sequence_prior_smoke as smoke

CONFIG_PATH = Path("configs/e007_pretrained_sequence_prior_smoke_v1.yaml")
V2_CONFIG_PATH = Path("configs/e007_pretrained_sequence_prior_smoke_v2.yaml")


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def _archive_bytes(*, duplicate: bool = False, traversal: bool = False, symlink: bool = False) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        members = [("progen2-small/config.json", b"{}"), ("progen2-small/pytorch_model.bin", b"weights")]
        if duplicate:
            members.append(("progen2-small/config.json", b"duplicate"))
        if traversal:
            members.append(("../escape", b"bad"))
        for name, content in members:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        if symlink:
            info = tarfile.TarInfo("progen2-small/link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/tmp/escape"
            archive.addfile(info)
    return buffer.getvalue()


def _custom_archive_bytes(entries: list[dict]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for entry in entries:
            info = tarfile.TarInfo(entry["name"])
            kind = entry.get("kind", "file")
            if kind == "file":
                content = entry.get("content", b"x")
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
            elif kind == "directory":
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = entry.get("target", "target")
                archive.addfile(info)
            elif kind == "hardlink":
                info.type = tarfile.LNKTYPE
                info.linkname = entry.get("target", "target")
                archive.addfile(info)
            elif kind == "fifo":
                info.type = tarfile.FIFOTYPE
                archive.addfile(info)
            else:
                raise AssertionError(kind)
    return buffer.getvalue()


def _temporary_config(tmp_path: Path) -> Path:
    config = _config()
    config["artifact_cache_root"] = str(tmp_path / "cache")
    config["artifact_lock_output"] = str(tmp_path / "artifact_lock")
    config["smoke_output"] = str(tmp_path / "smoke")
    config["sources"]["esm2_150m"]["artifacts"] = [
        {"path": "esm2_150m/config.json", "url": "https://example.test/esm/config.json", "sha256": None}
    ]
    config["sources"]["progen2_151m"]["artifacts"] = [
        {
            "path": "progen2_151m/progen2-small.tar.gz",
            "url": "https://example.test/progen/progen2-small.tar.gz",
            "sha256": None,
            "archive": "tar_gz",
            "extract_to": "progen2_151m/checkpoint",
            "expected_members": ["config.json", "pytorch_model.bin"],
        }
    ]
    config["sources"]["proteinmpnn_ca_only"]["artifacts"] = [
        {
            "path": "proteinmpnn_ca_only/ca_model_weights/v_48_020.pt",
            "url": "https://example.test/mpnn/v_48_020.pt",
            "sha256": None,
        }
    ]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _fake_download(spec: dict, target: Path, _config: dict) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    content = _archive_bytes() if spec.get("archive") == "tar_gz" else f"fixture:{spec['path']}".encode()
    target.write_bytes(content)
    return {
        "size_bytes": len(content),
        "sha256": smoke.sha256_file(target),
        "resolved_url": spec["url"],
        "final_host": "example.test",
        "redirect_chain": [
            {
                "status_code": 200,
                "source_scheme": "https",
                "source_host": "example.test",
                "destination_scheme": None,
                "destination_host": None,
                "destination_path_category": "other",
                "signed_delivery_parameters": False,
                "policy": "fixture",
                "cross_host": False,
                "authorization_headers_forwarded": False,
            }
        ],
    }


def _redirect_settings() -> dict:
    return _config()["acquisition"]


def _redirect(source: str, destination: str) -> dict:
    return smoke._validate_redirect(
        source,
        destination,
        origin_url=source,
        status_code=302,
        settings=_redirect_settings(),
    )


def _fake_worker(_config_path: str, candidate: str, result_path: str) -> None:
    Path(result_path).write_text(
        json.dumps(
            {
                "candidate": candidate,
                "status": "passed",
                "optimizer_updates": 0,
                "peak_rss_mib": 128.0,
                "peak_cuda_allocated_mib": 256.0,
                "peak_cuda_reserved_mib": 512.0,
                "parameter_sha256_before": f"{candidate}-same",
                "parameter_sha256_after": f"{candidate}-same",
                "downloaded_code_executed": False,
                **smoke.NON_AUTHORIZING,
            }
        )
    )


def _fake_environment_verifier(_config_path: str | Path) -> dict:
    return {"status": "verified_fixture", "runtime_fingerprint": "fixture"}


def test_pinned_sources_and_ca_only_contract() -> None:
    config = smoke._load_config(CONFIG_PATH)
    assert config["sources"]["esm2_150m"]["model_commit"] == "a695f6045e2e32885fa60af20c13cb35398ce30c"
    assert config["sources"]["progen2_151m"]["source_commit"] == "9b4d4fb5ec19c9e55c4bb06305c0e613e46c1cf5"
    assert config["sources"]["proteinmpnn_ca_only"]["source_commit"] == ("8907e6671bfbfc92303b5f79c4b5e6ce47cdef57")
    mpnn = config["sources"]["proteinmpnn_ca_only"]
    assert mpnn["checkpoint_path"] == "ca_model_weights/v_48_020.pt"
    assert mpnn["full_backbone_checkpoint_forbidden"] is True
    assert all(source["trust_remote_code"] is False for source in config["sources"].values())


@pytest.mark.parametrize("host", ["us.aws.cdn.hf.co", "cdn-lfs.hf.co"])
def test_huggingface_redirect_policy_accepts_official_boundary_hosts(host: str) -> None:
    row = _redirect(
        "https://huggingface.co/org/model/resolve/" + "a" * 40 + "/model.safetensors",
        f"https://{host}/xet-bridge/object?Policy=p&Signature=s&Key-Pair-Id=k",
    )
    assert row["destination_host"] == host
    assert row["destination_path_category"] == "huggingface_signed_delivery"
    assert row["signed_delivery_parameters"] is True
    assert row["authorization_headers_forwarded"] is False


@pytest.mark.parametrize("host", ["evil-hf.co", "hf.co.attacker.example"])
def test_huggingface_redirect_policy_rejects_deceptive_hosts(host: str) -> None:
    with pytest.raises(ValueError, match="cross-host redirect refused"):
        _redirect("https://huggingface.co/pinned", f"https://{host}/object")


def test_redirect_policy_rejects_downgrade_credentials_and_cross_source() -> None:
    with pytest.raises(ValueError, match="HTTPS redirect downgrade"):
        _redirect("https://huggingface.co/pinned", "http://us.aws.cdn.hf.co/object")
    with pytest.raises(ValueError, match="credentials refused"):
        _redirect("https://huggingface.co/pinned", "https://user:secret@us.aws.cdn.hf.co/object")
    with pytest.raises(ValueError, match="cross-host redirect refused"):
        _redirect("https://raw.githubusercontent.com/org/repo/commit/file", "https://us.aws.cdn.hf.co/object")


def test_redirect_handler_strips_sensitive_headers_and_bounds_chain() -> None:
    settings = _redirect_settings()
    source = "https://huggingface.co/pinned"
    request = urllib.request.Request(
        source,
        headers={"Authorization": "Bearer secret", "Cookie": "session=secret"},
        unverifiable=True,
    )
    request.add_unredirected_header("Proxy-Authorization", "proxy-secret")
    handler = smoke._SecureRedirectHandler(settings, source)
    redirected = handler.redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "https://us.aws.cdn.hf.co/xet-bridge/object?Policy=p",
    )
    assert redirected is not None
    headers = {name.lower(): value for name, value in redirected.header_items()}
    assert "authorization" not in headers
    assert "cookie" not in headers
    assert "proxy-authorization" not in headers
    assert len(handler.redirect_chain) == 1
    assert handler.redirect_chain[0]["status_code"] == 302
    handler.redirect_chain = [{}] * int(settings["maximum_redirects"])
    with pytest.raises(ValueError, match="maximum redirect count exceeded"):
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://us.aws.cdn.hf.co/another",
        )


class _DownloadResponse:
    def __init__(self, url: str, payload: bytes, *, fail_after_first_read: bool = False) -> None:
        self.url = url
        self.payload = payload
        self.fail_after_first_read = fail_after_first_read
        self.read_count = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def geturl(self) -> str:
        return self.url

    def getcode(self) -> int:
        return 200

    def read(self, _size: int) -> bytes:
        self.read_count += 1
        if self.fail_after_first_read and self.read_count > 1:
            raise OSError("interrupted fixture transfer")
        if self.read_count == 1:
            return self.payload
        return b""


class _DownloadOpener:
    def __init__(self, response: _DownloadResponse) -> None:
        self.response = response

    def open(self, _request, timeout: float):  # noqa: ANN001, ARG002
        return self.response


def test_hash_mismatch_and_interrupted_partial_download_never_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    source = config["sources"]["esm2_150m"]["artifacts"][0]["url"]
    target = tmp_path / "config.json"
    monkeypatch.setattr(
        smoke.urllib.request,
        "build_opener",
        lambda *_args: _DownloadOpener(_DownloadResponse(source, b"payload")),
    )
    with pytest.raises(ValueError, match="SHA-256 contradiction"):
        smoke._download_file({"url": source, "path": "config.json", "sha256": "0" * 64}, target, config)
    assert not target.exists()
    assert not (tmp_path / ".config.json.download.inprogress").exists()

    monkeypatch.setattr(
        smoke.urllib.request,
        "build_opener",
        lambda *_args: _DownloadOpener(_DownloadResponse(source, b"partial", fail_after_first_read=True)),
    )
    with pytest.raises(OSError, match="interrupted fixture transfer"):
        smoke._download_file({"url": source, "path": "config.json", "sha256": None}, target, config)
    assert not target.exists()
    assert not (tmp_path / ".config.json.download.inprogress").exists()


def test_ordinary_retry_replaces_unjournaled_failed_staging_artifact(tmp_path: Path) -> None:
    config_path = _temporary_config(tmp_path)
    staging = tmp_path / ".cache.acquisition.inprogress" / "esm2_150m"
    staging.mkdir(parents=True)
    stale = staging / "config.json"
    stale.write_bytes(b"failed-attempt-uncommitted")
    result = smoke.acquire_artifacts(config_path, download_file=_fake_download)
    assert result["status"] == "completed_immutable_artifact_acquisition"
    assert (tmp_path / "cache" / "esm2_150m" / "config.json").read_bytes() != b"failed-attempt-uncommitted"


def test_ordinary_retry_reuses_valid_journaled_downloads_without_network(
    tmp_path: Path,
) -> None:
    config_path = _temporary_config(tmp_path)
    config = smoke._load_config(config_path)
    staging = tmp_path / ".cache.acquisition.inprogress"
    staging.mkdir()
    journal = {"version": smoke.VERSION, "committed": {}}
    for spec in smoke._artifact_specs(config):
        target = smoke._contained(staging, spec["path"])
        metadata = _fake_download(spec, target, config)
        journal["committed"][spec["path"]] = metadata
    smoke._atomic_json(staging / ".acquisition_journal.json", journal)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a valid journaled artifact must not be downloaded again")

    result = smoke.acquire_artifacts(config_path, download_file=forbidden)
    assert result["status"] == "completed_immutable_artifact_acquisition"
    assert (tmp_path / "cache" / "progen2_151m" / "checkpoint" / "pytorch_model.bin").is_file()


def test_plan_only_has_no_network_cache_or_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _temporary_config(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("network access is forbidden in plan-only")

    monkeypatch.setattr(smoke.urllib.request, "urlopen", forbidden)
    result = smoke.plan(config_path)
    assert result["network_accessed"] is False
    assert result["cache_written"] is False
    assert result["package_installed"] is False
    assert result["model_created"] is False
    assert result["cuda_used"] is False
    assert result["dataset_scanned"] is False
    assert result["smoke_case_count"] == 60
    assert len(result["unresolved_artifact_hashes"]) == 3
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "artifact_lock").exists()
    assert not (tmp_path / "smoke").exists()


def test_safe_archive_extraction_and_inventory(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.tar.gz"
    archive.write_bytes(_archive_bytes())
    rows = smoke.safe_extract_tar(
        archive,
        tmp_path / "out",
        expected_members=["config.json", "pytorch_model.bin"],
        maximum_total_bytes=1024,
    )
    assert [row["path"] for row in rows] == ["config.json", "pytorch_model.bin"]
    assert (tmp_path / "out" / "config.json").read_bytes() == b"{}"


def test_archive_root_directory_markers_are_noops_and_leading_dot_is_normalized(tmp_path: Path) -> None:
    archive = tmp_path / "root-markers.tar.gz"
    archive.write_bytes(
        _custom_archive_bytes(
            [
                {"name": ".", "kind": "directory"},
                {"name": "./", "kind": "directory"},
                {"name": "./weights/model.pt", "content": b"weights"},
            ]
        )
    )
    rows = smoke.safe_extract_tar(
        archive,
        tmp_path / "out",
        expected_members=["model.pt"],
        maximum_members=4,
        maximum_file_bytes=32,
        maximum_total_bytes=32,
    )
    assert rows[0]["path"] == "model.pt"
    assert (tmp_path / "out" / "model.pt").read_bytes() == b"weights"
    assert not (tmp_path / "out" / ".").is_file()


@pytest.mark.parametrize("kind", ["file", "symlink", "hardlink"])
def test_archive_root_marker_must_be_a_directory(tmp_path: Path, kind: str) -> None:
    archive = tmp_path / f"root-{kind}.tar.gz"
    archive.write_bytes(_custom_archive_bytes([{"name": ".", "kind": kind}]))
    with pytest.raises(ValueError, match="root marker is not a directory"):
        smoke.safe_extract_tar(archive, tmp_path / "out", expected_members=[])
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "name",
    ["../x", "a/../../x", "/absolute", "C:/windows", "//server/share", "a\\..\\x"],
)
def test_archive_path_ambiguities_are_rejected_before_extraction(tmp_path: Path, name: str) -> None:
    archive = tmp_path / "unsafe-name.tar.gz"
    archive.write_bytes(_custom_archive_bytes([{"name": name}]))
    with pytest.raises(ValueError, match="archive"):
        smoke.safe_extract_tar(archive, tmp_path / "out", expected_members=["x"])
    assert not (tmp_path / "out").exists()


def test_archive_nul_is_rejected() -> None:
    with pytest.raises(ValueError, match="NUL"):
        smoke._normalized_member_name("weights/evil\x00.pt")


@pytest.mark.parametrize(
    "entries",
    [
        [{"name": "model.pt"}, {"name": "model.pt"}],
        [{"name": "model.pt"}, {"name": "./model.pt"}],
        [{"name": "Model.pt"}, {"name": "model.pt"}],
    ],
)
def test_archive_duplicate_and_case_colliding_paths_are_rejected(tmp_path: Path, entries: list[dict]) -> None:
    archive = tmp_path / "duplicates.tar.gz"
    archive.write_bytes(_custom_archive_bytes(entries))
    with pytest.raises(ValueError, match="duplicate|case-colliding"):
        smoke.safe_extract_tar(archive, tmp_path / "out", expected_members=["model.pt"])
    assert not (tmp_path / "out").exists()


def test_archive_special_entries_and_explicit_limits_are_rejected(tmp_path: Path) -> None:
    special = tmp_path / "special.tar.gz"
    special.write_bytes(_custom_archive_bytes([{"name": "pipe", "kind": "fifo"}]))
    with pytest.raises(ValueError, match="non-regular"):
        smoke.safe_extract_tar(special, tmp_path / "special-out", expected_members=[])

    archive = tmp_path / "limits.tar.gz"
    archive.write_bytes(
        _custom_archive_bytes(
            [
                {"name": "a", "content": b"aa"},
                {"name": "b", "content": b"bb"},
            ]
        )
    )
    with pytest.raises(ValueError, match="member count"):
        smoke.safe_extract_tar(archive, tmp_path / "count", expected_members=["a", "b"], maximum_members=1)
    with pytest.raises(ValueError, match="member exceeds size"):
        smoke.safe_extract_tar(
            archive,
            tmp_path / "file-size",
            expected_members=["a", "b"],
            maximum_file_bytes=1,
        )
    with pytest.raises(ValueError, match="exceeds size limit"):
        smoke.safe_extract_tar(
            archive,
            tmp_path / "total-size",
            expected_members=["a", "b"],
            maximum_total_bytes=3,
        )
    assert not any((tmp_path / name).exists() for name in ("count", "file-size", "total-size"))


def test_complete_preflight_precedes_writes_and_failed_extraction_is_not_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unsafe = tmp_path / "late-unsafe.tar.gz"
    unsafe.write_bytes(_custom_archive_bytes([{"name": "model.pt"}, {"name": "../late"}]))
    with pytest.raises(ValueError, match="archive"):
        smoke.safe_extract_tar(unsafe, tmp_path / "never-created", expected_members=["model.pt"])
    assert not (tmp_path / "never-created").exists()

    valid = tmp_path / "valid.tar.gz"
    valid.write_bytes(_custom_archive_bytes([{"name": "model.pt", "content": b"weights"}]))
    original_contained = smoke._contained

    def fail_during_extraction(root: Path, relative: str) -> Path:
        if root.name.endswith("extract.inprogress"):
            raise OSError("fixture extraction interruption")
        return original_contained(root, relative)

    monkeypatch.setattr(smoke, "_contained", fail_during_extraction)
    with pytest.raises(OSError, match="fixture extraction interruption"):
        smoke.safe_extract_tar(valid, tmp_path / "not-committed", expected_members=["model.pt"])
    assert not (tmp_path / "not-committed").exists()
    assert not list(tmp_path.glob(".*.extract.inprogress"))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"duplicate": True}, "duplicate archive member"),
        ({"traversal": True}, "unsafe archive member"),
        ({"symlink": True}, "archive link refused"),
    ],
)
def test_unsafe_archives_are_rejected(tmp_path: Path, kwargs: dict, message: str) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    archive.write_bytes(_archive_bytes(**kwargs))
    with pytest.raises(ValueError, match=message):
        smoke.safe_extract_tar(
            archive,
            tmp_path / "out",
            expected_members=["config.json", "pytorch_model.bin"],
            maximum_total_bytes=4096,
        )
    assert not (tmp_path / "escape").exists()


def test_unexpected_and_missing_archive_members_are_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.tar.gz"
    archive.write_bytes(_archive_bytes())
    with pytest.raises(ValueError, match="unexpected archive member"):
        smoke.safe_extract_tar(
            archive,
            tmp_path / "out",
            expected_members=["config.json"],
            maximum_total_bytes=4096,
        )
    with pytest.raises(ValueError, match="inventory contradiction"):
        smoke.safe_extract_tar(
            archive,
            tmp_path / "other",
            expected_members=["config.json", "pytorch_model.bin", "missing"],
            maximum_total_bytes=4096,
        )


def test_acquisition_offline_verification_and_tamper_refusal(tmp_path: Path) -> None:
    config_path = _temporary_config(tmp_path)
    result = smoke.acquire_artifacts(config_path, download_file=_fake_download)
    assert result["status"] == "completed_immutable_artifact_acquisition"
    assert result["artifact_count"] == 5
    lock = tmp_path / "artifact_lock"
    assert {path.name for path in lock.iterdir()} == {
        "artifact_lock.json",
        "source_inventory.json",
        "environment_lock.json",
        "report.json",
        "protocol.json",
        "artifact_inventory.json",
        "heartbeat.json",
    }
    verified = smoke.verify_artifacts_offline(config_path)
    assert verified["artifact_count"] == 5
    assert verified["network_accessed"] is False
    target = tmp_path / "cache" / "esm2_150m" / "config.json"
    original = target.read_bytes()
    target.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ValueError, match="hash contradiction"):
        smoke.verify_artifacts_offline(config_path)


def test_acquisition_refuses_existing_cache_and_never_executes_downloaded_code(tmp_path: Path) -> None:
    config_path = _temporary_config(tmp_path)
    smoke.acquire_artifacts(config_path, download_file=_fake_download)
    lock = json.loads((tmp_path / "artifact_lock" / "artifact_lock.json").read_text())
    assert lock["downloaded_repository_code_executed"] is False
    with pytest.raises(FileExistsError, match="cache already exists"):
        smoke.acquire_artifacts(config_path, download_file=_fake_download)


def test_resume_after_atomic_cache_promotion_publishes_lock(tmp_path: Path) -> None:
    config_path = _temporary_config(tmp_path)
    config = smoke._load_config(config_path)
    staging = Path(config["artifact_cache_root"]).with_name(".cache.acquisition.inprogress")
    staging.mkdir()
    rows = []
    for spec in smoke._artifact_specs(config):
        target = smoke._contained(staging, spec["path"])
        metadata = _fake_download(spec, target, config)
        rows.append((spec, metadata))
        if spec.get("archive"):
            smoke.safe_extract_tar(
                target,
                smoke._contained(staging, spec["extract_to"]),
                expected_members=spec["expected_members"],
                maximum_total_bytes=4096,
            )
    staging.replace(Path(config["artifact_cache_root"]))
    result = smoke.acquire_artifacts(config_path, download_file=_fake_download, resume=True)
    assert result["resumed_after_cache_promotion"] is True
    assert (tmp_path / "artifact_lock" / "artifact_lock.json").is_file()


def test_fake_isolated_smoke_is_atomic_zero_update_and_non_authorizing(tmp_path: Path) -> None:
    config_path = _temporary_config(tmp_path)
    smoke.acquire_artifacts(config_path, download_file=_fake_download)
    result = smoke.run_smoke(
        config_path,
        worker_target=_fake_worker,
        environment_verifier=_fake_environment_verifier,
    )
    assert result["status"] == "completed_non_authorizing"
    assert result["decision"] == "esm2_and_progen2_advance"
    output = tmp_path / "smoke"
    assert output.is_dir() and not (tmp_path / ".smoke.inprogress").exists()
    report = json.loads((output / "report.json").read_text())
    assert report["optimizer_updates"] == 0
    assert report["training_performed"] is False
    assert report["sampling_performed"] is False
    assert report["final_conditioning_architecture_selected"] is False
    assert all(report[field] is False for field in smoke.NON_AUTHORIZING)


def test_smoke_resume_hash_verifies_committed_candidate(tmp_path: Path) -> None:
    config_path = _temporary_config(tmp_path)
    smoke.acquire_artifacts(config_path, download_file=_fake_download)
    staging = tmp_path / ".smoke.inprogress"
    staging.mkdir()
    esm = staging / "esm2_150m.json"
    _fake_worker(str(config_path), "esm2_150m", str(esm))
    original = smoke.sha256_file(esm)
    (staging / "journal.json").write_text(json.dumps({"committed": {"esm2_150m": original}}))
    result = smoke.run_smoke(
        config_path,
        resume=True,
        worker_target=_fake_worker,
        environment_verifier=_fake_environment_verifier,
    )
    assert result["status"] == "completed_non_authorizing"
    assert smoke.sha256_file(tmp_path / "smoke" / "esm2_150m.json") == original


def test_smoke_resume_rejects_changed_committed_result(tmp_path: Path) -> None:
    config_path = _temporary_config(tmp_path)
    smoke.acquire_artifacts(config_path, download_file=_fake_download)
    staging = tmp_path / ".smoke.inprogress"
    staging.mkdir()
    result = staging / "esm2_150m.json"
    _fake_worker(str(config_path), "esm2_150m", str(result))
    (staging / "journal.json").write_text(json.dumps({"committed": {"esm2_150m": "0" * 64}}))
    with pytest.raises(ValueError, match="result hash contradiction"):
        smoke.run_smoke(
            config_path,
            resume=True,
            worker_target=_fake_worker,
            environment_verifier=_fake_environment_verifier,
        )


def test_decision_keeps_loading_success_separate_from_final_architecture() -> None:
    assert (
        smoke._decision(
            [
                {"candidate": "esm2_150m", "status": "passed"},
                {"candidate": "progen2_151m", "status": "failed"},
                {"candidate": "proteinmpnn_ca_only", "status": "baseline_unavailable_reviewed_local_loader_required"},
            ]
        )
        == "esm2_only_advances"
    )


def test_fake_model_state_hash_detects_mutation() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(4, 3)
    before = smoke._parameter_sha(model)
    with torch.no_grad():
        model.weight[0, 0] += 1
    assert smoke._parameter_sha(model) != before


def test_environment_lock_is_single_isolated_and_does_not_downgrade_project_torch() -> None:
    config = smoke._load_config(CONFIG_PATH)
    lock = yaml.safe_load(Path(config["environment_lock"]["path"]).read_text())
    assert lock["strategy"] == "single_isolated_environment_with_builtin_transformers_esm_and_progen"
    assert lock["base_project_environment_mutated"] is False
    assert lock["constraints"]["project_torch_downgrade_forbidden"] is True
    assert lock["constraints"]["trust_remote_code"] is False


def test_artifact_lock_records_candidate_licenses_and_provenance(tmp_path: Path) -> None:
    config_path = _temporary_config(tmp_path)
    smoke.acquire_artifacts(config_path, download_file=_fake_download)
    inventory = json.loads((tmp_path / "artifact_lock" / "source_inventory.json").read_text())
    assert inventory["candidate_sources"]["esm2_150m"]["license"] == "MIT"
    assert inventory["candidate_sources"]["progen2_151m"]["license"] == "BSD-3-Clause"
    assert inventory["candidate_sources"]["proteinmpnn_ca_only"]["loader"] == ("reviewed_local_proteinmpnn_ca")
    direct = [row for row in inventory["sources"] if not str(row["url"]).startswith("archive:")]
    assert all(row["redirect_chain"] for row in direct)
    assert all(row["redirect_chain"][-1]["status_code"] == 200 for row in direct)


def test_v2_production_plan_does_not_import_torch_or_create_paths(tmp_path: Path) -> None:
    payload = yaml.safe_load(V2_CONFIG_PATH.read_text())
    payload["smoke_output"] = str(tmp_path / "planned-smoke")
    config_path = tmp_path / "v2-plan.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    config = smoke._load_config(config_path)
    assert Path(config["artifact_cache_root"]).is_dir()
    assert Path(config["artifact_lock_output"]).is_dir()
    assert not Path(config["smoke_output"]).exists()
    result = smoke.plan(config_path)
    assert result["artifact_count"] == 14
    assert result["unresolved_artifact_hashes"] == []
    assert result["artifact_lock_status"] == "verified_offline"
    assert result["isolated_candidate_process_count"] == 3
    assert result["output_created"] is False
    assert "torch" not in smoke.__dict__


def test_path_traversal_in_config_is_rejected(tmp_path: Path) -> None:
    config = _config()
    config["sources"]["esm2_150m"]["artifacts"][0]["path"] = "../escape"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="unsafe relative path"):
        smoke._load_config(path)


def test_mutable_source_url_is_rejected(tmp_path: Path) -> None:
    config = _config()
    config["sources"]["proteinmpnn_ca_only"]["artifacts"][0]["url"] = (
        "https://raw.githubusercontent.com/dauparas/ProteinMPNN/main/ca_model_weights/v_48_020.pt"
    )
    path = tmp_path / "mutable.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="mutable GitHub artifact URL"):
        smoke._load_config(path)


class _FakeEsmTokenizer:
    cls_token_id = 0
    pad_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    mask_token_id = 4

    def __init__(self, *, substitute: bool = False) -> None:
        self.substitute = substitute
        self.mapping = {residue: index + 5 for index, residue in enumerate(smoke.CANONICAL)}
        self.mapping.update({residue: index + 25 for index, residue in enumerate("BOUXZ")})

    def convert_tokens_to_ids(self, residue: str) -> int:
        if residue == "J":
            return self.unk_token_id
        return self.mapping[residue]

    def __call__(self, sequence: str, *, add_special_tokens: bool = True) -> dict:
        ids = [self.convert_tokens_to_ids(residue) for residue in sequence]
        return {"input_ids": [self.cls_token_id, *ids, self.eos_token_id] if add_special_tokens else ids}

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        reverse = {value: key for key, value in self.mapping.items()}
        decoded = "".join(reverse.get(value, "") for value in ids if value not in {0, 1, 2, 3, 4})
        if self.substitute and decoded:
            return "A" + decoded[1:]
        return decoded


def test_tokenizer_contract_accepts_explicit_unknown_but_rejects_substitution() -> None:
    result = smoke._tokenizer_checks("esm2_150m", _FakeEsmTokenizer(), [64, 128, 256, 384, 500])
    assert result["length_500_encoded_token_count"] == 502
    assert result["ambiguous_residue_round_trips"]["J"]["status"] == "explicit_unknown"
    with pytest.raises(ValueError, match="substitution detected"):
        smoke._tokenizer_checks("esm2_150m", _FakeEsmTokenizer(substitute=True), [500])


def test_no_test_network_or_cache_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = Path(_config()["artifact_cache_root"])
    before = sorted((path.relative_to(cache), path.stat().st_size) for path in cache.rglob("*") if path.is_file())
    monkeypatch.setattr(
        smoke.urllib.request, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError)
    )
    smoke._load_config(CONFIG_PATH)
    after = sorted((path.relative_to(cache), path.stat().st_size) for path in cache.rglob("*") if path.is_file())
    assert after == before
