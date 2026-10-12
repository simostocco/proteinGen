"""Durable operational epoch-one hold and bounded resource monitor; no science changes."""

import ast, importlib.util, json, os, pathlib, subprocess, sys, time

W = pathlib.Path("/mnt/d/Simone/proteinGen/.e014_worktree")
R = W / "e007_matched_fullscale_v1"
C = pathlib.Path("/mnt/d/Simone/proteinGen/e007_e015_epoch1_comparison_v1")
PY = "/home/simostocco/miniforge3/envs/proteingen/bin/python"


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + ".partial")
    with q.open("w") as f:
        json.dump(x, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    q.replace(p)


def read(p):
    return json.loads(p.read_text())


def helper():
    spec = importlib.util.spec_from_file_location(
        "switch", pathlib.Path(__file__).with_name("e015_to_e007_switchover.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for key in [
        "TMPDIR",
        "TMP",
        "TEMP",
        "XDG_CACHE_HOME",
        "MPLCONFIGDIR",
        "TORCH_HOME",
        "HF_HOME",
        "CUDA_CACHE_PATH",
        "TRITON_CACHE_DIR",
    ]:
        value = os.environ.get(key)
        if value:
            path = pathlib.Path(value)
            assert path.is_relative_to("/mnt/d")
            path.mkdir(parents=True, exist_ok=True)
    import tempfile

    tempfile.tempdir = os.environ["TMPDIR"]
    return m


def install():
    p = W / "scripts/supervise_e007_fullscale.py"
    s = p.read_text()
    marker = "        if completed == 1:\n"
    replacement = """        if completed == 1 and (RUN / "EPOCH1_OPERATIONAL_HOLD.json").exists():
            request = json.loads((RUN / "EPOCH1_OPERATIONAL_HOLD.json").read_text())
            assert request["target_exposures"] == 230132 and request["active"]
            committed = json.loads((RUN / "COMMITTED.json").read_text())
            assert committed["exposures"] == 230132 and committed["epoch"] == 1
            assert (RUN / "EPOCH_1_DIAGNOSTICS_COMPLETE.json").exists()
            state = json.loads((RUN / "STATE.json").read_text())
            state.update(phase="paused_epoch1_comparative_evaluation", timestamp_unix=time.time())
            atomic_json(RUN / "STATE.json", state)
            atomic_json(RUN / "SUPERVISOR.json", dict(pid=os.getpid(), phase=state["phase"], training_active=False, requires_explicit_resume=True))
            return
        if completed == 1:
"""
    if "EPOCH1_OPERATIONAL_HOLD.json" not in s:
        assert s.count(marker) == 1
        s = s.replace(marker, replacement)
        ast.parse(s)
    # AST regression: execute only the operational branch with fake filesystem.
    tree = ast.parse(s)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "workflow")
    branch = next(n for n in ast.walk(fn) if isinstance(n, ast.If) and "EPOCH1_OPERATIONAL_HOLD" in ast.unparse(n.test))
    assert isinstance(branch.body[-1], ast.Return)
    assert "subprocess" not in ast.unparse(branch)
    # Execute the actual hold branch in a mocked RUN for normal, restart and wrong-count cases.
    import tempfile

    with tempfile.TemporaryDirectory(dir=C) as td:
        rr = pathlib.Path(td)
        write(rr / "EPOCH1_OPERATIONAL_HOLD.json", dict(active=True, target_exposures=230132))
        write(rr / "COMMITTED.json", dict(exposures=230132, epoch=1))
        write(rr / "EPOCH_1_DIAGNOSTICS_COMPLETE.json", dict(complete=True))
        write(rr / "STATE.json", dict(phase="epoch1_complete_awaiting_diagnostics"))
        testfn = ast.FunctionDef(
            name="test_hold",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[branch],
            decorator_list=[],
        )
        module = ast.fix_missing_locations(ast.Module(body=[testfn], type_ignores=[]))
        scope = dict(RUN=rr, completed=1, json=json, os=os, time=time, atomic_json=write)
        exec(compile(module, "operational_hold_test", "exec"), scope)
        scope["test_hold"]()
        assert read(rr / "STATE.json")["phase"] == "paused_epoch1_comparative_evaluation"
        scope["test_hold"]()  # restart: still holds, no epoch2 dispatch
        write(rr / "COMMITTED.json", dict(exposures=230133, epoch=1))
        try:
            scope["test_hold"]()
        except AssertionError:
            pass
        else:
            raise AssertionError("wrong exposure must fail closed")
    assert s.index("EPOCH1_OPERATIONAL_HOLD.json") < s.index("# Production resumes the committed")
    # Keep certified supervisor byte-for-byte unchanged; operational wrapper is separate.
    original = subprocess.check_output(["git", "show", "HEAD:scripts/supervise_e007_fullscale.py"], cwd=W, text=True)
    assert original.replace(marker, replacement) == s
    p.write_text(original)
    wrapper = pathlib.Path(__file__).with_name("supervise_e007_epoch1_hold.py")
    s = s.replace(
        '    if resume and (RUN / "STATE.json").exists():\n',
        '    if (RUN / "EPOCH1_OPERATIONAL_HOLD.json").exists() and (RUN / "COMMITTED.json").exists():\n        held = json.loads((RUN / "COMMITTED.json").read_text())\n        if held["exposures"] >= 230132 and (RUN / "EPOCH_1_DIAGNOSTICS_COMPLETE.json").exists():\n            phase = "paused_epoch1_comparative_evaluation" if held["exposures"] == 230132 else "paused_after_epoch1_boundary_crossed"\n            state = json.loads((RUN / "STATE.json").read_text())\n            state.update(phase=phase, timestamp_unix=time.time())\n            atomic_json(RUN / "STATE.json", state)\n            atomic_json(RUN / "SUPERVISOR.json", dict(pid=os.getpid(), phase=phase, training_active=False, requires_explicit_resume=True))\n            return\n    if resume and (RUN / "STATE.json").exists():\n',
        1,
    )
    wrapper.write_text("import sys\nsys.path.insert(0, " + repr(str(W / "scripts")) + ")\n" + s)
    write(
        R / "EPOCH1_OPERATIONAL_HOLD.json",
        dict(
            active=True,
            target_exposures=230132,
            reason="User requested exact epoch-1 comparative evaluation; no automatic epoch 2",
            timestamp=time.time(),
        ),
    )
    write(
        C / "progress.json",
        dict(
            stage="RECOVERY",
            epoch1_hold_persisted=True,
            training_science_unchanged=True,
            automatic_epoch2=False,
            automatic_E015_training=False,
            recovery_command=PY + " -B " + str(pathlib.Path(__file__).resolve()) + " resume",
        ),
    )
    print("Persistent epoch-one supervisor hold installed; AST dispatch regression passed.")


def audit():
    m = helper()
    m.host(True)
    cp = read(R / "COMMITTED.json")
    latest = read(R / "checkpoints/LATEST.json")
    assert cp["path"] == latest["path"] and cp["sha256"] == latest["sha256"]
    original_write = m.write
    m.write = lambda p, x: original_write(
        C / "CURRENT_CPU_AUDIT.json" if pathlib.Path(p).name == "FIRST_NEW_ROLLING_EXCLUSIVE_VERIFICATION.json" else p,
        x,
    )
    if cp["exposures"] >= 230132:
        if cp["exposures"] > 230132:
            write(
                C / "BOUNDARY_ALREADY_CROSSED.json",
                dict(
                    committed=cp,
                    training_not_extended=True,
                    primary_evaluation="Named epoch1 checkpoint; current state is not exactly epoch1",
                ),
            )
            raise RuntimeError(
                "Durable epoch1 boundary already crossed; no training launched. Named epoch1 evaluation requires recovery review."
            )
        epoch_verify((R / "EPOCH_1_DIAGNOSTICS_COMPLETE.json").exists())
        write(
            C / "RECOVERY_AUDIT.json",
            dict(
                checkpoint=cp["path"],
                sha256=cp["sha256"],
                exposures=230132,
                kind="EPOCH1_BOUNDARY_DIAGNOSTICS_ONLY",
                exact_next_operation=read(C / "E007_EPOCH1_CHECKPOINT.json")["next_operation"],
            ),
        )
        return
    m.e007_audit(cp["path"], False)
    result = read(C / "CURRENT_CPU_AUDIT.json")
    assert result["sha256"] == cp["sha256"]
    result["uncommitted_exposures_to_recompute"] = read(R / "STATE.json")["total_exposures"] - cp["exposures"]
    write(C / "RECOVERY_AUDIT.json", result)
    print(json.dumps(result["exact_next_operation"]))


def resume():
    audit()  # Always select current durable metadata, not a stale recovery record.
    m = helper()
    baseline = m.host(True)
    a = read(C / "RECOVERY_AUDIT.json")
    assert a["exposures"] <= 230132
    if a["exposures"] == 230132:
        if (R / "EPOCH_1_DIAGNOSTICS_COMPLETE.json").exists():
            print("Epoch1 is durably held; no training launched.")
            return
        state = read(R / "STATE.json")
        state["phase"] = (
            "explicit_resume_requested"
            if read(C / "E007_EPOCH1_CHECKPOINT.json")["pending_boundary"]
            else "epoch1_complete_awaiting_diagnostics"
        )
        write(R / "STATE.json", state)
    assert (R / "EPOCH1_OPERATIONAL_HOLD.json").exists()
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["PYTHONPATH"] = str(W / "src")
    command = [PY, "-B", str(pathlib.Path(__file__).with_name("supervise_e007_epoch1_hold.py")), "--launch", "--resume"]
    # Certified runner already nice+10; ionice affects scheduling only, not RNG/arithmetic.
    command = ["ionice", "-c", "2", "-n", "7"] + command
    out = subprocess.check_output(command, cwd=W, env=env, text=True)
    write(
        C / "LAUNCH.json",
        dict(
            command=command,
            result=out,
            baseline=baseline,
            expected_next_operation=a["exact_next_operation"],
            started=time.time(),
        ),
    )
    with (C / "logs/monitor.log").open("a") as log:
        p = subprocess.Popen(
            [PY, "-B", str(pathlib.Path(__file__).resolve()), "monitor"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(out, "MONITOR_PID", p.pid)


def epoch_verify(require_diagnostics=True):
    m = helper()
    m.host(True)
    sys.path[:0] = [str(W / "scripts"), str(W / "src")]
    os.chdir(W)
    import random, numpy as np, torch, e007_fullscale_runtime as rt
    from protein_distance_diffusion.models.e014_e007_control import E007ScaleupControl
    from protein_distance_diffusion.training.e014_runtime import canonical_hash, noise_seed

    torch.set_num_threads(1)
    cfg, pop, train, dev = rt.verified_inputs()
    inventory = read(R / "checkpoint_inventory.json")["checkpoints"]
    entry = next(x for x in inventory if x["label"] == "epoch_1_fraction_1.00")
    named = pathlib.Path(entry["path"])
    named_digest = m.sha(named)
    assert named_digest == entry["sha256"]
    latest = read(R / "COMMITTED.json")
    cp = pathlib.Path(latest["path"])
    digest = m.sha(cp)
    assert digest == latest["sha256"]
    saved = torch.load(cp, map_location="cpu", weights_only=False)
    assert saved["epoch"] == 1 and int(saved["exposures"].sum()) == 230132 and bool((saved["exposures"] == 1).all())
    for path, h in saved["contract"]["source_hashes"].items():
        assert m.sha(W / path) == h
    schedule, exp = rt.expected_exposures(
        [x["length"] for x in train], 1, saved["position"], cfg["batch_sizes"], cfg["seed"]
    )
    assert (
        saved["position"] == len(schedule)
        and np.array_equal(exp, saved["exposures"])
        and canonical_hash(schedule) == saved["schedule_hash"]
    )
    model = E007ScaleupControl(**rt.yaml.safe_load(rt.MODEL_CONFIG.read_text())["model"])
    model.load_state_dict(saved["model"], strict=True)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["optimizer"]["learning_rate"], weight_decay=cfg["optimizer"]["weight_decay"]
    )
    opt.load_state_dict(saved["optimizer"])
    assert all(int(v["step"]) == saved["updates"] for v in opt.state.values())
    for group in opt.param_groups:
        for parameter in group["params"]:
            state = opt.state.get(parameter, {})
            for key in ["exp_avg", "exp_avg_sq"]:
                if key in state:
                    assert (
                        state[key].shape == parameter.shape
                        and state[key].dtype == parameter.dtype
                        and bool(torch.isfinite(state[key]).all())
                    )

    torch.set_rng_state(saved["rng"]["torch"])
    np.random.set_state(saved["rng"]["numpy"])
    random.setstate(saved["rng"]["python"])
    assert all(t.dtype == torch.uint8 and t.device.type == "cpu" for t in saved["rng"]["cuda"])
    nextschedule, _ = rt.expected_exposures([x["length"] for x in train], 2, 0, cfg["batch_sizes"], cfg["seed"])
    ids = nextschedule[0]
    named_saved = torch.load(named, map_location="cpu", weights_only=False)
    assert named_saved["model_hash"] == saved["model_hash"] and named_saved["updates"] == saved["updates"]
    assert int(named_saved["exposures"].sum()) == 230132
    del named_saved
    if require_diagnostics:
        assert (R / "EPOCH_1_DIAGNOSTICS_COMPLETE.json").exists()
        state = read(R / "STATE.json")
        write(C / "PRE_HOLD_STATE.json", state)
        state.update(
            phase="paused_epoch1_comparative_evaluation",
            epoch=1,
            update=saved["updates"],
            schedule_cursor=saved["position"],
            total_exposures=230132,
            timestamp_unix=time.time(),
        )
        write(R / "STATE.json", state)
        write(R / "SUPERVISOR.json", dict(phase=state["phase"], training_active=False, requires_explicit_resume=True))
    write(
        C / "E007_EPOCH1_CHECKPOINT.json",
        dict(
            checkpoint=str(named),
            sha256=named_digest,
            resume_checkpoint=str(cp),
            resume_sha256=digest,
            epoch=1,
            exposures=230132,
            update=saved["updates"],
            cursor=saved["position"],
            zero_uncommitted=True,
            diagnostics_completed=(R / "EPOCH_1_DIAGNOSTICS_COMPLETE.json").exists(),
            pending_boundary=saved["pending_boundary"],
            strict_model_optimizer_rng=True,
            source_contract_unchanged=True,
            exposure_vector_exact=True,
            next_operation=dict(
                epoch=2,
                update=saved["updates"] + 1,
                exposures=230132 + len(ids),
                batch=[
                    dict(
                        identity=train[i]["sample_id"],
                        length=train[i]["length"],
                        noise_seed=noise_seed("e014-train-epoch-2", train[i]["sample_id"], 0),
                        noise_seed_exact_decimal=str(noise_seed("e014-train-epoch-2", train[i]["sample_id"], 0)),
                    )
                    for i in ids
                ],
            ),
            automatic_epoch2=False,
        ),
    )


def after_epoch():
    import fcntl

    lease = (C / "PIPELINE_OWNER.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    m = helper()
    # Wait for existing supervisor to finish frozen epoch diagnostics and hold.
    while True:
        state = read(R / "STATE.json")
        if state["phase"] == "paused_epoch1_comparative_evaluation":
            break
        if state["phase"].startswith("paused_"):
            raise RuntimeError("Training stopped before epoch1: " + state["phase"])
        sup = read(R / "SUPERVISOR.json")
        if sup.get("phase") == "stopped_child_error":
            raise RuntimeError("Training child error; no evaluation launched")
        time.sleep(60)
    for attempt in range(20):
        try:
            m.host(True)
            break
        except Exception:
            time.sleep(3)
    else:
        raise RuntimeError("GPU ownership not released")
    subprocess.run([PY, "-B", str(pathlib.Path(__file__).resolve()), "epoch-verify"], check=True)
    ext = pathlib.Path(
        "/mnt/d/Simone/proteinGen/e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1/external_designability_loop_v1"
    )
    script = pathlib.Path(__file__).with_name("e015_external_designability.py")
    for stage in ["screen", "align", "aggregate"]:
        m.host(True)
        with (C / "logs" / ("E015_" + stage + ".log")).open("a") as log:
            subprocess.run(
                [PY, "-B", str(script), stage, "--release-hold"], stdout=log, stderr=subprocess.STDOUT, check=True
            )
    controls = read(ext / "controls_summary.json")
    pending = []
    if controls["status"] == "VALIDATED":
        for stage in ["confirm", "confirm-align", "novelty"]:
            try:
                m.host(True)
                with (C / "logs" / ("E015_" + stage + ".log")).open("a") as log:
                    subprocess.run([PY, "-B", str(script), stage], stdout=log, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as e:
                pending.append(dict(stage=stage, error=str(e)))
    for stage in ["aggregate", "cards", "report"]:
        m.host(True)
        with (C / "logs" / ("E015_" + stage + ".log")).open("a") as log:
            subprocess.run([PY, "-B", str(script), stage], stdout=log, stderr=subprocess.STDOUT, check=True)
    write(
        C / "E015_EXTERNAL_COMPLETION.json",
        dict(root=str(ext), controls=controls, pending=pending, production_training_resumed=False),
    )
    for stage in ["reconstruction", "generation"]:
        m.host(True)
        with (C / "logs" / ("E007_" + stage + ".log")).open("a") as log:
            subprocess.run(
                [PY, "-B", str(pathlib.Path(__file__).with_name("e007_epoch1_evaluation.py")), stage],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    for stage in ["analysis", "external-panel"]:
        m.host(True)
        with (C / "logs" / ("E007_" + stage + ".log")).open("a") as log:
            subprocess.run(
                [PY, "-B", str(pathlib.Path(__file__).with_name("e007_epoch1_analysis.py")), stage],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    for stage in ["mpnn", "sequences", "screen", "align"]:
        m.host(True)
        with (C / "logs" / ("E007_external_" + stage + ".log")).open("a") as log:
            subprocess.run(
                [PY, "-B", str(pathlib.Path(__file__).with_name("e007_external_designability.py")), stage],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    control7 = read(C / "e007_inverse_folding/controls_summary.json")
    if control7["status"] == "VALIDATED":
        for stage in ["confirm", "confirm-align", "novelty"]:
            try:
                m.host(True)
                with (C / "logs" / ("E007_external_" + stage + ".log")).open("a") as log:
                    subprocess.run(
                        [PY, "-B", str(pathlib.Path(__file__).with_name("e007_external_designability.py")), stage],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            except subprocess.CalledProcessError as e:
                pending.append(dict(stage="E007_" + stage, error=str(e)))
    subprocess.run(
        [PY, "-B", str(pathlib.Path(__file__).with_name("e007_external_designability.py")), "summary"], check=True
    )
    subprocess.run(
        [PY, "-B", str(pathlib.Path(__file__).with_name("e007_external_designability.py")), "cards"], check=True
    )
    try:
        m.host(True)
        with (C / "logs/E015_matched_epoch1_generation.log").open("a") as log:
            subprocess.run(
                [PY, "-B", str(pathlib.Path(__file__).with_name("e015_matched_epoch1_generation.py"))],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    except subprocess.CalledProcessError as e:
        pending.append(dict(stage="E015_matched_epoch1_generation", error=str(e)))
    if (
        (C / "e015_epoch1_context/MANIFEST.json").exists()
        and controls["status"] == "VALIDATED"
        and control7["status"] == "VALIDATED"
    ):
        subprocess.run(
            [PY, "-B", str(pathlib.Path(__file__).with_name("e007_epoch1_analysis.py")), "matched-external-panel"],
            check=True,
        )
        env = os.environ.copy()
        env["COMPARISON_EXTERNAL_OUT"] = str(C / "e015_epoch1_inverse_folding")
        env["COMPARISON_SOURCE_MODEL"] = "E015_epoch1"
        for stage in ["mpnn", "sequences", "screen", "align"]:
            m.host(True)
            with (C / "logs" / ("E015_matched_external_" + stage + ".log")).open("a") as log:
                subprocess.run(
                    [PY, "-B", str(pathlib.Path(__file__).with_name("e007_external_designability.py")), stage],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        for stage in ["confirm", "confirm-align", "novelty", "summary", "cards"]:
            try:
                m.host(True)
                with (C / "logs" / ("E015_matched_external_" + stage + ".log")).open("a") as log:
                    subprocess.run(
                        [PY, "-B", str(pathlib.Path(__file__).with_name("e007_external_designability.py")), stage],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            except subprocess.CalledProcessError as e:
                pending.append(dict(stage="E015_matched_" + stage, error=str(e)))
    write(C / "COMPUTATION_STATUS.json", dict(pending=pending, automatic_epoch2=False, automatic_E015_training=False))
    m.host(True)
    subprocess.run([PY, "-B", str(pathlib.Path(__file__).with_name("e007_e015_comparison_report.py"))], check=True)


def queue_after_epoch():
    if (C / "AFTER_EPOCH_CONTROLLER.json").exists():
        old = read(C / "AFTER_EPOCH_CONTROLLER.json")["pid"]
        if pathlib.Path(f"/proc/{old}/cmdline").exists():
            assert b"e007_epoch1_mission.py" not in pathlib.Path(f"/proc/{old}/cmdline").read_bytes(), (
                "Continuation controller already active"
            )
    with (C / "logs/after_epoch.log").open("a") as log:
        p = subprocess.Popen(
            [PY, "-B", str(pathlib.Path(__file__).resolve()), "after-epoch"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write(
        C / "AFTER_EPOCH_CONTROLLER.json",
        dict(
            pid=p.pid,
            command=[PY, "-B", str(pathlib.Path(__file__).resolve()), "after-epoch"],
            serialized=True,
            no_training_auto_resume=True,
        ),
    )
    print("Serialized continuation controller", p.pid)


def preserve_historical_proof():
    record = read(R / "E015_TO_E007_SWITCHOVER.json")["first_new_rolling_strict_verification"]
    assert record["sha256"] == "72bb82ff54cc58257007b81171fce5b2e883235fd98752f5f97a50ced9031363"
    write(R / "FIRST_NEW_ROLLING_EXCLUSIVE_VERIFICATION.json", record)
    write(
        C / "AUDIT_RECORD_PRESERVATION.json",
        dict(
            original_proof_restored_from_durable_switchover_record=True,
            original_sha256=record["sha256"],
            new_recovery_audits="RECOVERY_AUDIT.json and POST_RECOVERY_ROLLING_CHECKPOINT.json",
            scientific_checkpoints_unchanged=True,
        ),
    )


def record_methods():
    m = helper()
    sources = [
        "e007_epoch1_mission.py",
        "supervise_e007_epoch1_hold.py",
        "e007_comparison_metrics.py",
        "e007_epoch1_evaluation.py",
        "e007_epoch1_analysis.py",
        "e007_external_designability.py",
        "e007_e015_comparison_report.py",
        "e015_matched_epoch1_generation.py",
        "e015_external_designability.py",
        "e015_external_reporting.py",
    ]
    hashes = {name: m.sha(pathlib.Path(__file__).with_name(name)) for name in sources}
    write(
        C / "METHODS.json",
        dict(
            version="matched_epoch1_v1",
            source_hashes=hashes,
            primary_exposures=230132,
            comparison_context_exposures=314256,
            reconstruction=dict(
                panel="Existing frozen100TRAIN+100heldout",
                timesteps=[5, 50, 250, 400, 490],
                noise_family="e015-absolute-matched-noise-v1",
                bootstrap_draws=5000,
            ),
            generation=dict(
                lengths=[48, 96, 192, 320, 448],
                seeds=[15015000, 15015001, 15015002, 15015003],
                samples=20,
                native_steps=500,
                initial_state="Original50%X499 tensors, normalized Gaussian recovered and bitwise checked",
            ),
            external=dict(
                protocol_reference=str(
                    pathlib.Path(
                        "/mnt/d/Simone/proteinGen/e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1/external_designability_loop_v1/assessment_protocol.json"
                    )
                ),
                positive_controls="Same five predeclared real controls; reuse hash-compatible results",
                best_of_four="Five independent per-length numeric winners; cross-model winning seeds may differ",
                matched_external="Additional exact-epoch1E015 candidates only if original positive-control gate passes and resource admission succeeds",
            ),
            automatic_epoch2=False,
            automatic_E015_training=False,
            recovery_commands=dict(
                training=PY + " -B " + str(pathlib.Path(__file__).resolve()) + " resume",
                continuation=PY + " -B " + str(pathlib.Path(__file__).resolve()) + " queue-after-epoch",
            ),
        ),
    )
    print("Versioned methods and recovery commands recorded")


def rolling_proof():
    m = helper()
    committed = read(R / "COMMITTED.json")
    assert committed["updates"] > 90413
    original_write = m.write
    m.write = lambda p, x: original_write(
        C / "CURRENT_CPU_AUDIT.json" if pathlib.Path(p).name == "FIRST_NEW_ROLLING_EXCLUSIVE_VERIFICATION.json" else p,
        x,
    )
    m.e007_audit(committed["path"], False)
    record = read(C / "CURRENT_CPU_AUDIT.json")
    assert record["sha256"] == committed["sha256"]
    write(C / "POST_RECOVERY_ROLLING_CHECKPOINT.json", record)


def first_proof():
    audit = read(C / "RECOVERY_AUDIT.json")
    expected = audit["exact_next_operation"]
    path = R / "logs/TRAINING.jsonl"
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 5 * 1024**2))
        lines = f.read().decode(errors="replace").splitlines()
    matches = []
    for line in lines:
        try:
            row = json.loads(line)
            if row.get("update") == expected["update"]:
                matches.append(row)
        except ValueError:
            pass
    assert matches
    row = matches[-1]
    assert row["sample_ids"] == [r["identity"] for r in expected["batch"]] and row["identity_count"] == len(
        expected["batch"]
    )
    import math

    assert math.isfinite(row["loss"])
    write(
        C / "FIRST_RESUMED_OPERATION.json",
        dict(
            expected=expected,
            observed=row,
            exposure_count_basis="Verified durable exposure vector plus logged identity count",
            noise_basis="Frozen runner uses exact identity/epoch deterministic seed; not a logged seed field",
            no_durable_skip_duplicate=audit["no_skipped_or_duplicate_durable_exposure"],
        ),
    )
    report = """# E007 recovery and exact epoch-one hold

Recovered from update90,413 /115,067 committed exposures after a CUDA unknown-error exit. Stale live work had reached120,387 exposures;5,320 exposures require recomputation. CPU strict model/optimizer/RNG and exact schedule/exposure vector passed; all frozen source hashes match.

The certified supervisor remains byte-for-byte unchanged. A separate wrapper stops after the normal epoch-one diagnostics, requiring230,132 committed exposures and the diagnostics completion marker. Focused execution tests cover exact boundary, restart, and an incorrect exposure count. The original two-epoch contract is preserved; epoch2 and E015 production will not auto-resume.

Scheduling only: certified nice+10 remains; ionice best-effort class2 priority7 lowers I/O contention. No thread, affinity, batch, precision, allocator, objective or deterministic noise changes. The prior CUDA error alone does not establish whether host freezes arose from the driver, CPU contention, host memory, swap, or NTFS I/O. Windows/WSL memory, swap, GPU and D:reserve are monitored; pressure invokes the existing graceful checkpoint/stop handler.

Progress and remaining-time telemetry:progress.json and resource_monitor.jsonl. Durable recovery:RECOVERY_AUDIT.json. First operation:FIRST_RESUMED_OPERATION.json. Exact commands:LAUNCH.json and AFTER_EPOCH_CONTROLLER.json. Every subsequent heavy stage waits for ownership release. Previously completed E015 predictions are hash-checked and reused. Scientific conclusions remain pending.
"""
    (C / "RECOVERY_REPORT.md").write_text(report)
    write(
        C / "MANIFEST.json",
        dict(
            status="IN_PROGRESS",
            recovery_audit="RECOVERY_AUDIT.json",
            first_operation="FIRST_RESUMED_OPERATION.json",
            training_contract=dict(
                epoch_exposures=230132, total_exposures=460264, temporary_epoch1_hold=True, automatic_epoch2=False
            ),
            E015_production_paused=True,
            continuation="AFTER_EPOCH_CONTROLLER.json",
            scientific_comparison_complete=False,
            all_artifacts_on_d=True,
        ),
    )


def refresh_continuation():
    import signal

    old = read(C / "AFTER_EPOCH_CONTROLLER.json")["pid"]
    cmd = pathlib.Path(f"/proc/{old}/cmdline").read_bytes()
    assert b"e007_epoch1_mission.py" in cmd and b"after-epoch" in cmd
    assert not pathlib.Path(f"/proc/{old}/task/{old}/children").read_text().strip(), (
        "Cannot refresh during an active atomic evaluation unit"
    )
    os.kill(old, signal.SIGINT)  # Waiting metadata-only controller; never scientific worker.
    time.sleep(1)
    queue_after_epoch()


def refresh_monitor():
    import signal

    old = read(C / "progress.json")["monitor_pid"]
    cmd = pathlib.Path(f"/proc/{old}/cmdline").read_bytes()
    assert b"e007_epoch1_mission.py" in cmd and b"monitor" in cmd
    os.kill(old, signal.SIGINT)  # Standard Python interrupt for metadata-only monitor; never trainer.
    time.sleep(1)
    with (C / "logs/monitor.log").open("a") as log:
        p = subprocess.Popen(
            [PY, "-B", str(pathlib.Path(__file__).resolve()), "monitor"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print("Replacement resource monitor", p.pid)


def graceful_stop(reason):
    import ctypes, signal

    pid = read(R / "SUPERVISOR.json").get("child_pid") or read(R / "STATE.json")["pid"]
    path = pathlib.Path(f"/proc/{pid}/cmdline")
    cmd = path.read_bytes()
    assert b"run_e007_fullscale.py" in cmd or b"evaluate_e007_fullscale.py" in cmd
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(434, pid, 0)
    assert fd >= 0
    try:
        assert path.read_bytes() == cmd
        assert libc.syscall(424, fd, int(signal.SIGINT), 0, 0) == 0
    finally:
        os.close(fd)
    write(C / "GRACEFUL_RESOURCE_STOP.json", dict(reason=reason, pid=pid, time=time.time()))


def monitor():
    m = helper()
    started = time.time()
    first = None
    pressure = 0
    previous_swap = None
    while True:
        s = read(R / "STATE.json")
        committed = read(R / "COMMITTED.json")
        if (C / "STOP_MONITOR.json").exists():
            return
        if s.get("phase") == "training" and time.time() - s.get("timestamp_unix", 0) > 180:
            write(C / "progress.json", dict(stage="STALE_TRAINING_REQUIRES_RECOVERY", state=s, committed=committed))
            return
        try:
            h = m.host(False)
            vm = {
                line.split()[0]: int(line.split()[1])
                for line in pathlib.Path("/proc/vmstat").read_text().splitlines()
                if line.startswith(("pswpin ", "pswpout "))
            }
            swaps = vm.get("pswpin", 0) + vm.get("pswpout", 0)
            now = time.time()
            h["swap_pages_per_second"] = (swaps - previous_swap[0]) / (now - previous_swap[1]) if previous_swap else 0
            previous_swap = (swaps, now)
            assert h["swap_pages_per_second"] < 1024, "Sustained swap pressure:4MiB/s admission threshold"
            assert h["load_average"][0] < max(4, len(os.sched_getaffinity(0)) - 4), "Sustained host scheduling/IO load"
            owners = [int(row.split(",")[0]) for row in h["cuda_compute_apps"].splitlines() if row.strip()]
            allowed = {s["pid"], read(R / "SUPERVISOR.json").get("child_pid")}
            assert len(owners) <= 1 and set(owners) <= allowed, "Competing CUDA owner; gracefully release E007"

            pressure = 0
        except Exception as e:
            pressure += 1
            h = {"resource_error": str(e)}
            if pressure >= 2:
                graceful_stop(str(e))
                write(C / "progress.json", dict(stage="RESOURCE_GRACEFUL_STOP", error=str(e)))
                return
        io = pathlib.Path("/proc/" + str(s["pid"]) + "/io")
        row = dict(
            timestamp=time.time(),
            state=s,
            committed=committed,
            resource=h,
            disk_io_counters=io.read_text() if io.exists() else None,
            runnable_count=pathlib.Path("/proc/loadavg").read_text().split()[3],
        )
        with (C / "resource_monitor.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        if first is None and s.get("timestamp_unix", 0) > started:
            first = row
        delta = time.time() - first["timestamp"] if first else 0
        rate = (s["total_exposures"] - first["state"]["total_exposures"]) / delta if delta > 0 else None
        write(
            C / "progress.json",
            dict(
                stage=s["phase"],
                update=s["update"],
                exposures=s["total_exposures"],
                committed_exposures=committed["exposures"],
                proteins_s=rate,
                remaining=230132 - s["total_exposures"],
                eta_seconds=(230132 - s["total_exposures"]) / rate if rate and rate > 0 else None,
                epoch1_hold=True,
                monitor_pid=os.getpid(),
                automatic_epoch2=False,
            ),
        )
        if delta >= 900:
            write(
                C / "E007_RESOURCE_OBSERVATION.json",
                dict(
                    first=first,
                    last=row,
                    seconds=delta,
                    proteins_s=rate,
                    updates_s=(s["update"] - first["state"]["update"]) / delta,
                ),
            )
        if s["phase"] == "paused_epoch1_comparative_evaluation":
            write(
                C / "E007_EPOCH1_CHECKPOINT.json",
                dict(committed=committed, verification_pending=True, stage="EPOCH1_PAUSED_GPU_RELEASE_VERIFY_REQUIRED"),
            )
            return
        if s["phase"].startswith("paused_"):
            return
        time.sleep(60)


if __name__ == "__main__":
    C.mkdir(exist_ok=True)
    for n in [
        "logs",
        "e007_samples",
        "e007_trajectories",
        "e007_galleries",
        "e007_inverse_folding",
        "e007_refolding",
        "e007_foldseek",
        "comparison_plots",
        "comparison_galleries",
    ]:
        (C / n).mkdir(exist_ok=True)
    try:
        {
            "install": install,
            "audit": audit,
            "resume": resume,
            "monitor": monitor,
            "refresh-monitor": refresh_monitor,
            "epoch-verify": epoch_verify,
            "after-epoch": after_epoch,
            "queue-after-epoch": queue_after_epoch,
            "refresh-continuation": refresh_continuation,
            "first-proof": first_proof,
            "rolling-proof": rolling_proof,
            "record-methods": record_methods,
            "preserve-historical-proof": preserve_historical_proof,
        }[sys.argv[1]]()
    except Exception as error:
        write(
            C / "LAST_CONTROLLER_ERROR.json",
            dict(
                stage=sys.argv[1],
                error=repr(error),
                timestamp=time.time(),
                completed_artifacts_preserved=True,
                recovery_command=PY + " -B " + str(pathlib.Path(__file__).resolve()) + " " + sys.argv[1],
                automatic_epoch2=False,
                automatic_E015_training=False,
            ),
        )
        raise
