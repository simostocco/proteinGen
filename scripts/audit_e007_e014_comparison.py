"""Inventory matching evidence and freeze model choices before new metrics."""

# ruff: noqa: I001, E501, F403, F405
from e007_e014_comparison_protocol import *  # noqa: F403,F405
import argparse
import random
import zipfile
import hashlib


def historical_train_population():
    """Reconstruct identity coverage from the recorded unchanged rank stream."""
    train = pq.read_table(ROOT / "population/train.parquet", use_threads=False).to_pylist()
    assert len(train) == len({r["sample_id"] for r in train}) == 230132
    original = json.loads(
        (
            HIST.parents[1]
            / "reports/experiments/E007_matrix_sequence_cogeneration/coordinate_real_pilot_v1/panel_manifest.json"
        ).read_text()
    )
    continuation = json.loads((HIST / "panel_manifest.json").read_text())
    ranked = {}
    for stratum in STRATA:
        ranked[stratum] = sorted(
            [r for r in train if r["length_stratum"] == stratum],
            key=lambda r: (
                int.from_bytes(
                    hashlib.sha256(f"7361|train_schedule_and_evaluation|{r['sample_id']}".encode()).digest(), "big"
                ),
                r["sample_id"],
            ),
        )
        count = dict(zip(STRATA, (800, 400, 200, 200, 200), strict=True))[stratum]
        ids = [r["sample_id"] for r in ranked[stratum][:count]]
        assert ids == original["train"][stratum][:count]
        assert canonical_hash(ids) == continuation["training_prefix_sha256_by_stratum"][stratum]
    results = []
    for step in (9000, 10000):
        sampled = [
            dict(step=step, length_stratum=stratum, sample_id=r["sample_id"], length=r["length"])
            for stratum, size in zip(STRATA, (4, 2, 1, 1, 1), strict=True)
            for r in ranked[stratum][: step // 5 * size]
        ]
        assert len(sampled) == len({r["sample_id"] for r in sampled}) == step * 9 // 5
        table(f"historical_E007_train_stream_{step}.parquet", sampled)
        results.append(
            dict(
                step=step,
                exposures=len(sampled),
                unique_identities=len(sampled),
                fraction_of_eligible_TRAIN=len(sampled) / 230132,
                identity_sha256=canonical_hash([r["sample_id"] for r in sampled]),
                sampled_by_stratum={s: sum(r["length_stratum"] == s for r in sampled) for s in STRATA},
                eligible_by_stratum={s: len(ranked[s]) for s in STRATA},
            )
        )
    write(
        "historical_train_population.json",
        dict(
            reconstructed_from_frozen_rank_stream=True,
            original_1000_step_prefix_verified=True,
            records=results,
            full_train_sha256=sha(ROOT / "population/train.parquet"),
        ),
    )
    assessment = json.loads((OUT / "matching_assessment.json").read_text())
    assessment["historical_actual_identity_coverage"] = results
    assessment["same_training_loss_aggregation"] = False
    assessment["training_loss_aggregation"] = {
        "E007": "all valid coordinate components pooled across a batch (length weighting)",
        "E014": "coordinate mean within protein, then equal protein mean across batch",
    }
    assessment["training_backend_note"] = (
        "Historical E007 training enforces deterministic algorithms; E014 runner does not enforce them. Current comparison uses identical FP32/TF32-off inference settings and documents separate replay envelopes."
    )
    write("matching_assessment.json", assessment)
    inventory = json.loads((OUT / "checkpoint_inventory.json").read_text())
    for record in inventory["checkpoints"] + inventory["selected"]:
        record["unique_TRAIN_identities"] = record["exposures"] if record["model"] == "E007" else 230132
    write("checkpoint_inventory.json", inventory)


def audit():
    torch.set_num_threads(1)
    panel, subset = population()
    cfg = yaml.safe_load(Path("configs/e007_coordinate_real_pilot_v1.yaml").read_text())
    assert cfg["coordinate_scale_angstrom"] == SCALE and cfg["diffusion_steps"] == 500
    assert cfg["objective"]["name"] == "uniform_valid_coordinate_v_mse"
    assert cfg["objective"]["auxiliary_losses"] == []
    e014cfg = yaml.safe_load(Path("configs/e014_unconditional_coordinate_v_scaleup_v1.yaml").read_text())
    assert e014cfg["coordinate_scale_angstrom"] == SCALE and e014cfg["diffusion_steps"] == 500
    assert cfg["coordinate_normalization"]["artifact_sha256"] == e014cfg["normalization_sha256"]
    assert cfg["dataset"]["protocol_sha256"] == e014cfg["dataset_protocol_sha256"]
    assert cfg["dataset"]["shard_inventory_sha256"] == e014cfg["dataset_inventory_sha256"]
    assert cfg["clean_validation"]["manifest_sha256"] == e014cfg["clean_validation_sha256"]
    assert sha(e014cfg["normalization_artifact"]) == e014cfg["normalization_sha256"]
    source = json.loads((HIST / "report.json").read_text())
    records = []
    for path in sorted((HIST / "checkpoints").glob("step-*.pt")):
        state = torch.load(path, map_location="cpu", weights_only=False)
        digest = sha(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        assert digest == meta["sha256"]
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
        assert state["optimizer_update"] == state["sampler_cursor"] == meta["optimizer_update"]
        assert state["successful_optimizer_boundary"]
        assert state["scheduler"]["last_epoch"] == state["optimizer_update"]
        assert state["scheduler"]["_last_lr"] == [0.0003]
        assert state["continuation_configuration_sha256"] == sha(
            "configs/e007_coordinate_real_continuation_to_10000_v1.yaml"
        )
        model = E007ScaleupControl(
            **yaml.safe_load(Path("configs/e007_coordinate_generator_v1.yaml").read_text())["model"]
        )
        model.core.load_state_dict(state["model"], strict=True)
        assert sum(p.numel() for p in model.parameters()) == 7586505
        assert all(torch.isfinite(v).all() for v in state["model"].values())
        params = list(model.parameters())
        opt = state["optimizer"]
        assert len(opt["state"]) == len(params)
        for i, item in opt["state"].items():
            assert int(item["step"]) == state["optimizer_update"]
            assert item["exp_avg"].shape == item["exp_avg_sq"].shape == params[i].shape
            assert torch.isfinite(item["exp_avg"]).all() and torch.isfinite(item["exp_avg_sq"]).all()
        rng = state["rng_state"]
        random.Random().setstate(rng["python"])
        np.random.RandomState().set_state(rng["numpy"])
        torch.Generator().set_state(rng["torch"])
        assert isinstance(rng["cuda"], list) and all(v.dtype == torch.uint8 for v in rng["cuda"])
        assert sha(path) == digest
        records.append(
            dict(
                model="E007",
                label=f"e007_step_{state['optimizer_update']:05d}",
                path=str(path),
                sha256=digest,
                parameters=7586505,
                updates=state["optimizer_update"],
                exposures=state["samples_processed"],
                complete_full_data_epochs=0,
                config_hash=state["continuation_configuration_sha256"],
                optimizer=opt["param_groups"][0],
                EMA=False,
                code_commit=None,
                code_commit_note="Not recorded in checkpoint or continuation protocol; current source hashes preserved separately.",
                normalization=SCALE,
                diffusion_steps=500,
                objective="uniform centered-coordinate v-MSE; no auxiliaries",
                eligible_population=230132,
                actual_sampling="balanced stratum stream prefixes; no complete population epoch",
                load_hash_crc_optimizer_rng_verified=True,
            )
        )
        del model, state
    oldinv = json.loads((E014 / "checkpoint_inventory.json").read_text())
    for cp in oldinv["selected"][1:]:
        path = Path(cp.get("immutable_copy", cp["path"]))
        assert sha(path) == cp["sha256"]
        records.append(
            dict(
                model="E014",
                label=cp["label"],
                path=str(path),
                sha256=cp["sha256"],
                parameters=5119069,
                updates=cp["updates"],
                exposures=cp["exposures"],
                complete_full_data_epochs=cp["completed_epochs"],
                config_hash=cp["contract"]["config_hash"],
                optimizer=e014cfg["optimizer"],
                EMA=False,
                normalization=SCALE,
                diffusion_steps=500,
                code_commit=cp["contract"].get("git_sha"),
                previously_audited=True,
            )
        )
    assert next(x for x in records if x["label"] == "recovery_0")["sha256"] == NEWEST_SHA
    selected = [next(x for x in records if x["label"] == label) for label in (*E007_LABELS, *E014_LABELS)]
    prerequisites = dict(
        primary_final_report_exists=(ROOT / "e014/FINAL_REPORT.json").exists(),
        resource_approval_exists=(ROOT / "MATCHED_CONTROL_RESOURCE_APPROVED.json").exists(),
        prepared_control_config_sha256=sha("configs/e014_matched_e007_control_v1.yaml"),
        previous_N500_reserved_bytes=7254048768,
        frozen_comfortable_reserved_bytes=5368709120,
    )
    assessment = dict(
        classification="M2",
        usable_historical_context=True,
        architecture_attribution_allowed=False,
        same_eligible_train_universe=True,
        exact_actual_exposure_population_matched=False,
        same_normalization=True,
        same_diffusion=True,
        same_v_target=True,
        same_clean_validation_universe=True,
        same_new_evaluation_panel=True,
        training_optimizer_family_same=True,
        training_learning_rates=[0.0003, 0.0002],
        stopping_rules_same=False,
        exposures={x["label"]: x["exposures"] for x in selected},
        reasons=[
            "E007 endpoint has only 18,000 exposures, zero complete TRAIN epochs; E014 primary has 460,264 exposures / two complete epochs.",
            "Historical balanced stratum prefix stream differs from full-population deterministic epoch scheduling.",
            "AdamW learning rate differs; original validation panel and stopping rule differ.",
        ],
        matched_control_predeclared=True,
        matched_control_prerequisites=prerequisites,
        matched_control_action="Prepare strict matched manifest; do not bypass missing frozen primary-completion/resource prerequisites or modify E014 to create them.",
        primary_comparison="E007 fixed 10,000-update review endpoint vs E014 epoch 2; explicitly nonmatched training",
        secondary_comparisons="Previously published E007 step 9000 and E014 newest; no selection using new metrics",
        causal_classification="C5",
    )
    write("matching_assessment.json", assessment)
    write(
        "checkpoint_inventory.json",
        dict(
            checkpoints=records,
            selected=selected,
            historical_preselected_9000=json.loads(
                Path(
                    "/mnt/d/Simone/proteinGen/reports/experiments/E007_matrix_sequence_cogeneration/coordinate_checkpoint_selection_v1/selected_checkpoint.json"
                ).read_text()
            ),
            original_training_report_sha256=sha(HIST / "report.json"),
            protected_training_inputs=source["protected_input_hashes"],
        ),
    )
    write(
        "PROTOCOL.json",
        dict(
            panel_hash=PANEL_SHA,
            subset_hash=SUBSET_SHA,
            normalization=SCALE,
            diffusion_steps=500,
            reference_sha256=sha(OLD / "shared_reference_descriptors.parquet"),
            selected=selected,
            primary_family=list(PRIMARY),
            scalar_bootstraps=5000,
            secondary_distribution_bootstraps=500,
            primary_distribution_bootstraps=5000,
            confidence="95% paired intervals plus Bonferroni 99.1667% intervals for six primary tests",
            direction="E014 minus E007; negative favors E014 for errors/distances, positive for cosine/gain",
            full_sampling=False,
            training=False,
            retrospective_training_confounded=True,
            gaussian_endpoints=list(ENDPOINTS),
        ),
    )
    protected = {
        str(p): sha(p)
        for p in [
            *Path("src").rglob("*.py"),
            *Path("configs").glob("*e007*.yaml"),
            Path("configs/e014_unconditional_coordinate_v_scaleup_v1.yaml"),
            Path("scripts/run_e014_scaleup.py"),
            HIST / "report.json",
            HIST / "protocol.json",
            E014 / "REPORT.md",
            E014 / "replay_numerics.json",
        ]
    }
    write("PROTECTED_INPUT_HASHES.json", protected)
    write(
        "MATCHED_CONTROL_PREPARATION.json",
        dict(
            architecture="unchanged historical EquivariantPairCoordinateUNet",
            parameters=7586505,
            shared_training_config_sha256=sha("configs/e014_unconditional_coordinate_v_scaleup_v1.yaml"),
            train_identity_hash=json.loads((ROOT / "population/population.json").read_text())["train_sample_id_hash"],
            TRAIN=230132,
            minimum_epochs=2,
            primary_exposures=460264,
            normalization=SCALE,
            diffusion_steps=500,
            same_epoch_schedule=True,
            initialization="From scratch; frozen shared seed; no historical E007 warm start",
            shared_seed=e014cfg["seed"],
            same_batches=True,
            same_noise_streams=True,
            optimizer=e014cfg["optimizer"],
            same_checkpoint_cadence=True,
            same_panels=True,
            launched=False,
            prerequisites=prerequisites,
        ),
    )
    historical_train_population()
    pilot_inventory()
    print(json.dumps(assessment, indent=2), flush=True)


def pilot_inventory():
    inventory = json.loads((OUT / "checkpoint_inventory.json").read_text())
    pilot = (
        HIST.parents[1] / "reports/experiments/E007_matrix_sequence_cogeneration/coordinate_real_pilot_v1/checkpoints"
    )
    existing = {r["path"] for r in inventory["checkpoints"]}
    for path in sorted(pilot.glob("step-*.pt")):
        if str(path) in existing:
            continue
        state = torch.load(path, map_location="cpu", weights_only=False)
        digest = sha(path)
        assert digest == json.loads(path.with_suffix(".json").read_text())["sha256"]
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
        config = yaml.safe_load(Path("configs/e007_coordinate_generator_v1.yaml").read_text())["model"]
        model = E007ScaleupControl(**config)
        model.core.load_state_dict(state["model"], strict=True)
        assert state["optimizer_update"] == state["sampler_cursor"] == state["scheduler"]["last_epoch"]
        assert state["configuration_sha256"] == sha("configs/e007_coordinate_real_pilot_v1.yaml")
        assert state["successful_optimizer_boundary"] and all(torch.isfinite(v).all() for v in state["model"].values())
        params = list(model.parameters())
        assert len(state["optimizer"]["state"]) == len(params)
        for i, item in state["optimizer"]["state"].items():
            assert int(item["step"]) == state["optimizer_update"]
            assert item["exp_avg"].shape == item["exp_avg_sq"].shape == params[i].shape
            assert torch.isfinite(item["exp_avg"]).all() and torch.isfinite(item["exp_avg_sq"]).all()
        rng = state["rng_state"]
        random.Random().setstate(rng["python"])
        np.random.RandomState().set_state(rng["numpy"])
        torch.Generator().set_state(rng["torch"])
        inventory["checkpoints"].append(
            dict(
                model="E007",
                label=f"e007_step_{state['optimizer_update']:05d}",
                path=str(path),
                sha256=digest,
                parameters=7586505,
                updates=state["optimizer_update"],
                exposures=state["samples_processed"],
                unique_TRAIN_identities=state["samples_processed"],
                complete_full_data_epochs=0,
                eligible_population=230132,
                config_hash=state["configuration_sha256"],
                optimizer=state["optimizer"]["param_groups"][0],
                EMA=False,
                code_commit=None,
                normalization=SCALE,
                diffusion_steps=500,
                load_hash_crc_optimizer_rng_verified=True,
                origin="Original source pilot; inventoried, not selected by new performance",
            )
        )
        del state, model
    write("checkpoint_inventory.json", inventory)


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    audit()
