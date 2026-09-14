"""Status labels distinguish verified assets, warmup checks and requested devices."""
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import demucs_server as ds


MODEL = "bs_roformer_sw"


def health(*, warmup="skipped", verified=True, device="cpu", model=MODEL):
    return {"status": "ok", "device": device, "gpu": True, "demucs_model": model,
            "warmup": {model: warmup}, "runtime": {
                "schema_version": 1, "managed": True, "capabilities": ["verified_models_v1"],
                "models": {model: {"verified": verified}}}}


def presentation(payload, *, model=MODEL, present=None, running=True, requested="", started=""):
    return ds._status_presentation(payload, present or {}, model, {"device": started}, running, requested)


def test_verified_skipped_model_is_installed_on_demand_not_claimed_warm():
    result = presentation(health(), present={MODEL: True})
    assert result["model"] == {"state": "on_demand", "label": "Installed and verified — loads when needed",
                               "present": True, "verified": True, "warmup_state": "skipped", "ready": True}
    assert result["settled"]


@pytest.mark.parametrize("present", [False, True])
def test_legacy_disk_presence_is_not_verification_or_startup_readiness(present):
    result = presentation({"warmup": {MODEL: "skipped"}}, present={MODEL: present})
    assert result["model"]["label"] == "Not checked at startup"
    assert result["model"]["verified"] is None
    assert result["model"]["present"] is present
    assert result["settled"]


@pytest.mark.parametrize("payload", [
    {}, {"warmup": {MODEL: "unrecognized"}}, {"runtime": {"schema_version": 99}},
    {"runtime": {"schema_version": 1, "managed": True, "capabilities": ["verified_models_v1"], "models": []}},
])
def test_unknown_status_is_not_called_loading_or_warm(payload):
    result = presentation(payload)
    assert result["model"]["state"] == "unknown"
    assert result["model"]["label"] == "Status not reported"
    assert result["settled"]


@pytest.mark.parametrize("warmup", ["failed", "failed: exit 1", "FAILED: CUDA init", "error"])
def test_failed_prefix_is_terminal_even_with_verified_files(warmup):
    result = presentation(health(warmup=warmup))
    assert result["model"]["state"] == "failed"
    assert not result["model"]["ready"]
    assert result["settled"]


@pytest.mark.parametrize("warmup,present,label", [("downloading", False, "Preparing model…"),
                                                   ("downloading", True, "Loading model…"),
                                                   ("pending", False, "Preparing model…")])
def test_ambiguous_legacy_downloading_is_preparation_not_confirmed_network(warmup, present, label):
    result = presentation({"warmup": {MODEL: warmup}}, present={MODEL: present})
    assert result["model"]["state"] == "loading"
    assert result["model"]["label"] == label
    assert not result["settled"]


@pytest.mark.parametrize("reported,selected,state", [("htdemucs_6s", "htdemucs_6s", "ready"),
                                                     ("htdemucs_6s", MODEL, "unknown"),
                                                     (MODEL, "htdemucs_6s", "unknown")])
def test_demucs_alias_is_applied_only_to_actual_reported_selected_model(reported, selected, state):
    result = presentation({"demucs_model": reported, "warmup": {"demucs": "ready"}}, model=selected)
    assert result["model"]["state"] == state
    if selected != reported:
        assert result["features"]["demucs"]["state"] == "not_selected"


def test_direct_selected_state_wins_over_alias():
    result = presentation({"demucs_model": MODEL, "warmup": {MODEL: "failed: exit 1", "demucs": "ready"}})
    assert result["model"]["state"] == "failed"


def test_optional_features_do_not_block_selected_stem_readiness():
    payload = health()
    payload["warmup"].update(whisperx="failed: missing weights", crepe="pending", demucs="ready")
    result = presentation(payload)
    assert result["model"]["ready"]
    assert result["features"]["whisperx"]["state"] == "failed"
    assert result["features"]["crepe"]["state"] == "loading"
    assert not result["settled"]  # still refresh genuinely active secondary work


def test_unused_alternate_failed_state_remains_visible():
    payload = health(model="htdemucs_6s")
    payload["warmup"][MODEL] = "failed: previous initialization"
    result = presentation(payload, model="htdemucs_6s")
    assert result["features"][MODEL]["state"] == "failed"
    assert result["model"]["ready"]


def test_managed_missing_model_is_explicit_and_does_not_trust_disk_presence():
    payload = health(model="htdemucs_6s")
    result = presentation(payload, present={MODEL: True})
    assert result["model"]["state"] == "missing"
    assert result["model"]["verified"] is False
    assert not result["model"]["ready"]


@pytest.mark.parametrize("schema,verified", [(True, True), ("1", True), (1, "true")])
def test_verification_metadata_rejects_coerced_truth_values(schema, verified):
    payload = health(verified=verified)
    payload["runtime"]["schema_version"] = schema
    assert presentation(payload)["model"]["verified"] is None


@pytest.mark.parametrize("requested,started,effective,needs_restart", [
    ("cpu", "cpu", "cpu", False),
    ("cpu", "cuda", "cuda", True),
    ("cuda", "cpu", "cpu", True),
    ("", "", "cuda", False),
    ("", "", "cpu", False),
    ("cpu", "", "cpu", True),
    ("", "cpu", "cpu", True),
    ("cuda:1", "cuda:1", "cuda:1", False),
])
def test_requested_mode_is_distinct_from_effective_device_and_auto_resolution(requested, started, effective, needs_restart):
    result = presentation(health(device=effective), requested=requested, started=started)
    assert result["effective_device"] == effective
    assert result["requested_device"] == (requested or "auto")
    assert result["restart_required"] is needs_restart


def test_stopped_server_has_no_effective_device_or_restart_requirement():
    result = presentation(health(device="cuda"), requested="cpu", started="cuda", running=False)
    assert result["effective_device"] is None
    assert not result["restart_required"]
    assert result["model"]["state"] == "stopped"
    assert result["settled"]


def test_real_status_adds_projection_without_network_or_extra_presence_scans(tmp_path):
    payload = health(device="cpu")  # gpu=True reports availability, never current use.
    with mock.patch.object(ds, "is_running", return_value=(True, 19861)), \
            mock.patch.object(ds, "_read_state", return_value={"port": 19861, "device": "cpu"}), \
            mock.patch.object(ds, "server_health", return_value=(True, payload)) as probe, \
            mock.patch.object(ds, "_has_roformer", return_value=True) as roformer, \
            mock.patch.object(ds, "_has_whisper", return_value=False) as whisper, \
            mock.patch.object(ds, "_has_aligner", return_value=False) as aligner, \
            mock.patch.object(ds, "detect_nvidia_gpu", return_value=None), \
            mock.patch.object(ds, "can_manage", return_value=(True, "")), \
            mock.patch.object(ds, "_server_disk_bytes", return_value=123):
        result = ds.server_status(tmp_path, requested_device="cpu")
    assert result["health"] is payload
    assert result["presentation"]["effective_device"] == "cpu"
    assert result["models_ready"] is True  # unchanged legacy public meaning for skipped
    assert result["presentation"]["model"]["state"] == "on_demand"
    probe.assert_called_once_with(ds.url_for(19861), timeout=2.0)
    for scanner in (roformer, whisper, aligner):
        scanner.assert_called_once()


@pytest.mark.parametrize("managed", [False, True])
def test_evicted_model_is_on_demand_not_failed_or_stuck_loading(managed):
    payload = health(warmup="evicted") if managed else {"warmup": {MODEL: "evicted"}}
    result = presentation(payload)
    assert result["model"]["state"] == "on_demand"
    assert result["model"]["label"] == "Starts when needed"
    assert result["settled"]


def test_unused_demucs_status_remains_visible_when_bs_is_selected():
    payload = health()
    payload["warmup"]["demucs"] = "ready"
    result = presentation(payload)
    assert result["features"]["demucs"]["state"] == "not_selected"
    assert result["model"]["state"] == "on_demand"


def test_bs_never_borrows_generic_demucs_readiness_even_if_health_default_matches():
    result = presentation({"demucs_model": MODEL, "warmup": {"demucs": "ready"}})
    assert result["model"]["state"] == "unknown"
    assert result["features"]["demucs"]["state"] == "not_selected"


@pytest.mark.parametrize("raw,detail", [("FAILED: CUDA Init on GPU 1", "CUDA Init on GPU 1"),
                                        ("failed: exit 1", "exit 1"),
                                        ("failed", "Server reported a failure without an error message.")])
def test_failure_details_preserve_server_error_without_replacing_neutral_label(raw, detail):
    result = presentation(health(warmup=raw))
    assert result["model"]["label"] == "Preparation failed"
    assert result["model"]["detail"] == detail


@pytest.mark.parametrize("model", ["htdemucs_ft", "mdx_extra", "mdx_q", "custom-demucs-model"])
def test_matching_extensible_legacy_demucs_model_uses_generic_alias(model):
    result = presentation({"demucs_model": model, "warmup": {"demucs": "ready"}}, model=model)
    assert result["model"]["state"] == "ready"
    assert "demucs" not in result["features"]


@pytest.mark.parametrize("model,engine", [("bs_roformer_sw", None), ("custom_roformer", None),
                                         ("custom-model", "audio-separator"), ("custom-model", "roformer")])
def test_known_roformer_name_or_engine_never_borrows_demucs_readiness(model, engine):
    payload = {"demucs_model": model, "warmup": {"demucs": "ready"},
               "runtime": {"models": {model: {"engine": engine}}}}
    result = presentation(payload, model=model)
    assert result["model"]["state"] == "unknown"
    assert result["features"]["demucs"]["state"] == "not_selected"


def test_per_language_aligner_failure_and_loading_remain_visible_and_keep_polling():
    payload = health()
    payload["warmup"]["whisperx_aligners"] = {"en": "ready", "fr": "failed: Model missing", "sv": "loading"}
    result = presentation(payload, present={"whisperx_aligners": True})
    assert "whisperx_aligners" not in result["features"]
    assert result["features"]["whisperx_aligners:en"]["state"] == "ready"
    assert result["features"]["whisperx_aligners:fr"]["state"] == "failed"
    assert result["features"]["whisperx_aligners:fr"]["detail"] == "Model missing"
    assert result["features"]["whisperx_aligners:sv"]["state"] == "loading"
    # Generic disk presence does not prove any specific language's files exist.
    assert result["features"]["whisperx_aligners:sv"]["label"] == "Preparing model…"
    assert result["model"]["ready"]
    assert not result["settled"]


def test_terminal_language_aligners_settle_and_skipped_stays_unverified():
    payload = health()
    payload["warmup"]["whisperx_aligners"] = {"en": "ready", "fr": "failed: missing", "sv": "skipped"}
    result = presentation(payload)
    assert result["features"]["whisperx_aligners:sv"]["state"] == "on_demand"
    assert result["features"]["whisperx_aligners:sv"]["verified"] is None
    assert result["features"]["whisperx_aligners:sv"]["label"] == "Not checked at startup"
    assert result["settled"]


def test_aligner_projection_is_one_level_and_malformed_language_state_is_unknown():
    payload = health()
    payload["warmup"]["whisperx_aligners"] = {"en": {"nested": "loading"}, "fr": None}
    result = presentation(payload)
    assert result["features"]["whisperx_aligners:en"]["state"] == "unknown"
    assert result["features"]["whisperx_aligners:fr"]["state"] == "unknown"
    assert result["settled"]
