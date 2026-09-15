"""Updater contracts: real temp directories, fake downloads/processes, no ML imports."""
import hashlib
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime_update as ru
import demucs_server as ds


DATA = b"verified test weights"
CONFIG = b"inference: {num_overlap: 2}\n"
FFMPEG = b"test ffmpeg executable"
FFPROBE = b"test ffprobe executable"


def asset(name, data):
    return {"path": name, "url": "https://models.example.test/" + name,
            "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def manifest():
    target = sys.platform if sys.platform in {"win32", "linux", "darwin"} else "linux"
    architecture = ru._normalized_architecture()
    suffix = ".exe" if target == "win32" else ""
    archive_digest = hashlib.sha256(b"test media archive").hexdigest()
    return {"schema_version": 2,
            "profiles": {"test-profile": {"python": ">=3.10,<3.14", "platforms": ["win32", "linux", "darwin"],
                "backends": ["cpu", "cuda"], "requirements": ["torch>=2.8,<2.9"],
                "no_deps": ["audio-separator>=0.47,<0.48"],
                "cuda_torch": {"torch": "2.8.0", "torchaudio": "2.8.0", "tags": ["cu126", "cu128"]}}},
            "media_tools": {"test-media-tools": {"revision": "test-tools-v1", "platform": target,
                "architecture": architecture, "profiles": ["test-profile"], "license": "GPL-3.0-or-later",
                "source_url": "https://media.example.test/source", "archives": [{
                    "url": "https://media.example.test/tools.zip", "size": len(b"test media archive"),
                    "sha256": archive_digest, "format": "zip",
                    "members": {"ffmpeg" + suffix: "ffmpeg" + suffix,
                                "ffprobe" + suffix: "ffprobe" + suffix}}]}},
            "models": {"bs_roformer_sw": {"engine": "audio-separator", "revision": "weights-v1-config-v1",
                "entrypoint": "BS-Roformer-SW.ckpt", "stems": ["guitar", "other"], "profiles": ["test-profile"],
                "assets": [asset("BS-Roformer-SW.ckpt", DATA), asset("BS-Roformer-SW.yaml", CONFIG)]}}}


def legacy(cfg):
    base = ru._base(cfg)
    (base / "src").mkdir(parents=True)
    (base / "src" / "server.py").write_text("OLD_SERVER")
    (base / "pylibs" / "torch-2.8.0.dist-info").mkdir(parents=True)
    (base / "pylibs" / "torch-2.8.0.dist-info" / "METADATA").write_text("Name: torch\nVersion: 2.8.0\n")
    (base / "pylibs" / "old-sentinel").write_text("DO_NOT_CHANGE")
    ru._atomic_json(base / "source.json", {"commit": "a" * 40, "ref": "main"})
    ru._atomic_json(base / "install.json", {"gpu": False, "torch": "2.8.0"})
    cached = base / "cache" / "_roformer-models"
    cached.mkdir(parents=True)
    (cached / "BS-Roformer-SW.ckpt").write_bytes(DATA)
    (cached / "BS-Roformer-SW.yaml").write_bytes(CONFIG)
    return base


def make_plan(cfg, spec=None):
    data = spec or manifest()
    with mock.patch.object(ds, "_resolve_commit", return_value="b" * 40), \
            mock.patch.object(ru, "_load_manifest", return_value=data), \
            mock.patch.object(ru, "_candidate_versions", return_value={"audio-separator": {"version": "0.47.0"}}):
        return ru.check_updates(cfg)


def fake_source(cfg, root, plan, cancel):
    source = root / "src"
    source.mkdir()
    (source / "server.py").write_text("NEW_SERVER")
    ru._atomic_json(source / ru.MANIFEST_NAME, plan["manifest"])
    ru._atomic_json(root / "source.json", {"commit": plan["source_commit"], "ref": plan["ref"]})


def fake_install(cfg, root, plan, cancel, callback):
    (root / "pylibs").mkdir()
    (root / "pylibs" / "candidate-sentinel").write_text("NEW_LIBRARIES")
    return {"torch": "2.8.0", "audio-separator": "0.47.0"}


def fake_media_tools(cfg, root, plan, cancel, callback):
    target = root / "tools" / "bin"
    target.mkdir(parents=True)
    names = ru._canonical_media_names(plan["media_tool_spec"]["platform"])
    files = []
    for name, payload in zip(names, (FFMPEG, FFPROBE)):
        path = target / name
        path.write_bytes(payload)
        path.chmod(0o755)
        files.append({"path": name, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    return {"set_id": plan["media_tool_spec"]["set_id"], "revision": plan["media_tool_spec"]["revision"],
            "platform": plan["media_tool_spec"]["platform"],
            "architecture": plan["media_tool_spec"]["architecture"], "asset_dir": "tools/bin",
            "archives": [{"sha256": row["sha256"], "size": row["size"]}
                         for row in plan["media_tool_spec"]["archives"]], "files": files}


@pytest.fixture
def sandbox(tmp_path):
    cfg = tmp_path / "profile"
    base = legacy(cfg)
    yield cfg, base


@pytest.fixture
def fake_runtime():
    with mock.patch.object(ru, "_disk_budget"), \
            mock.patch.object(ru, "_stage_source", side_effect=fake_source), \
            mock.patch.object(ru, "_stage_media_tools", side_effect=fake_media_tools), \
            mock.patch.object(ru, "_validate_media_tools"), \
            mock.patch.object(ru, "_install_dependencies", side_effect=fake_install), \
            mock.patch.object(ru, "_validate_candidate"), \
            mock.patch.object(ru, "_inference_smoke"), \
            mock.patch.object(ds, "is_running", return_value=(False, None)), \
            mock.patch.object(ds, "stop_server"):
        yield


def test_check_is_read_only_except_plan_and_never_claims_transitives_current(sandbox):
    cfg, base = sandbox
    before = (base / "pylibs" / "old-sentinel").read_bytes()
    with mock.patch.object(ru, "_install_dependencies") as install, mock.patch.object(ru, "_download_file") as download:
        plan = make_plan(cfg)
    assert plan["can_update"]
    assert plan["libraries_state"] == "refresh_available"
    assert "transitive" in plan["dependency_graph"]
    assert plan["installed"]["audio_separator"] is None
    assert (base / "pylibs" / "old-sentinel").read_bytes() == before
    assert not (base / "active.json").exists()
    install.assert_not_called()
    download.assert_not_called()


def test_offline_is_unknown_and_cannot_apply(sandbox):
    cfg, _ = sandbox
    with mock.patch.object(ds, "_resolve_commit", return_value=None):
        plan = ru.check_updates(cfg)
    assert not plan["can_update"]
    assert plan["state"] != "current"
    assert "immutable" in plan["reason"]


@pytest.mark.parametrize("backend,platform_name,python_version,tag,works", [
    (False, "win32", "3.12.10", "cu128", True),
    (False, "linux", "3.11.4", "cu128", True),
    (True, "win32", "3.12.10", "cu126", True),
    (True, "win32", "3.12.10", "cu128", True),
    (True, "darwin", "3.12.10", "cu128", False),
    (True, "linux", "3.12.10", "cu121", False),
    (False, "win32", "3.14.0", "cu128", False),
])
def test_profile_selection_depends_on_capabilities_not_gpu_product_name(backend, platform_name, python_version, tag, works):
    with mock.patch.object(ru.sys, "platform", platform_name), mock.patch.object(ru.platform, "python_version", return_value=python_version):
        if works:
            assert ru._select_profile(manifest(), backend, tag)[0] == "test-profile"
        else:
            with pytest.raises(ValueError, match="No supported"):
                ru._select_profile(manifest(), backend, tag)


def test_unknown_model_not_advertised_verified(sandbox):
    cfg, _ = sandbox
    data = manifest()
    with mock.patch.object(ds, "_resolve_commit", return_value="b" * 40), mock.patch.object(ru, "_load_manifest", return_value=data):
        plan = ru.check_updates(cfg, model="arbitrary_legacy_demucs")
    assert not plan["can_update"]
    assert "No verified update catalog" in plan["reason"]


def test_demucs_catalog_entrypoint_names_a_verified_bag():
    data = manifest()
    data["models"]["htdemucs_6s"] = {"engine": "demucs", "revision": "v1", "entrypoint": "htdemucs_6s",
        "assets": [asset("htdemucs_6s.yaml", CONFIG), asset("model.th", DATA)]}
    ru._validate_manifest(data)


@pytest.mark.parametrize("path", ["../escape.ckpt", "/outside.ckpt", "C:\\outside.ckpt", "x/../../escape"])
def test_catalog_paths_cannot_escape(path):
    data = manifest()
    data["models"]["bs_roformer_sw"]["assets"][0]["path"] = path
    with pytest.raises(ValueError):
        ru._validate_manifest(data)


def test_install_promotes_complete_generation_and_preserves_legacy_assets(sandbox, fake_runtime):
    cfg, base = sandbox
    plan = make_plan(cfg)
    with mock.patch.object(ru, "_download_file") as dl:
        result = ru.apply_update(cfg, plan["plan_id"])
    assert result["state"] == "active"
    assert not result["active"]
    assert result["rollback_available"]
    root = ru.active_root(cfg)
    assert root != base
    assert (base / "src" / "server.py").read_text() == "OLD_SERVER"
    assert (base / "pylibs" / "old-sentinel").read_text() == "DO_NOT_CHANGE"
    assert (base / "cache" / "_roformer-models" / "BS-Roformer-SW.ckpt").read_bytes() == DATA
    assert ds.src_dir(cfg) == root / "src"
    assert ds.pylibs_dir(cfg) == root / "pylibs"
    assert ru.inventory(cfg)["audio_separator"] == "0.47.0"
    dl.assert_not_called()  # legacy weights reused only after real hash validation


def test_contained_media_tools_are_verified_before_dependencies_or_models(sandbox):
    cfg, _ = sandbox
    plan = make_plan(cfg)
    order = []

    def source(*args):
        order.append("source")
        return fake_source(*args)

    def media(*args):
        order.append("media")
        return fake_media_tools(*args)

    def verify_media(*args):
        order.append("verify_media")

    def libraries(*args):
        order.append("libraries")
        return fake_install(*args)

    with mock.patch.object(ru, "_disk_budget"), mock.patch.object(ru, "_stage_source", side_effect=source), \
            mock.patch.object(ru, "_stage_media_tools", side_effect=media), \
            mock.patch.object(ru, "_validate_media_tools", side_effect=verify_media), \
            mock.patch.object(ru, "_install_dependencies", side_effect=libraries), \
            mock.patch.object(ru, "_stage_models", side_effect=lambda *args: order.append("models") or {}), \
            mock.patch.object(ru, "_validate_candidate", side_effect=lambda *args: order.append("candidate")):
        ru._stage_candidate(cfg, plan, lambda: None, None)
    assert order == ["source", "media", "verify_media", "libraries", "models", "candidate"]


def test_media_archive_is_included_in_conservative_disk_budget(tmp_path):
    plan = {"gpu": False, "components": [], "model_specs": {},
            "media_tool_spec": {"archives": [{"size": 100}]}}
    # Base headroom plus 299 bytes would pass if the 100-byte archive were not
    # budgeted three times for retained archive, extraction and staging headroom.
    with mock.patch.object(ru.shutil, "disk_usage", return_value=mock.Mock(free=512 * 1024**2 + 299)):
        with pytest.raises(RuntimeError, match="Not enough disk space"):
            ru._disk_budget(tmp_path, plan)


@pytest.mark.parametrize("failure_stage", ["_stage_source", "_stage_media_tools", "_validate_media_tools",
                                            "_install_dependencies", "_validate_candidate", "_inference_smoke"])
def test_candidate_failure_never_replaces_live_tree(sandbox, fake_runtime, failure_stage):
    cfg, base = sandbox
    plan = make_plan(cfg)
    with mock.patch.object(ru, failure_stage, side_effect=RuntimeError("candidate failed")):
        with pytest.raises(RuntimeError, match="candidate failed"):
            ru.apply_update(cfg, plan["plan_id"])
    assert ru.active_root(cfg) == base
    assert not (base / "active.json").exists()
    assert (base / "src" / "server.py").read_text() == "OLD_SERVER"
    assert (base / "pylibs" / "old-sentinel").read_text() == "DO_NOT_CHANGE"
    assert ru.status(cfg)["state"] == "failed"


def test_oom_candidate_keeps_previous_device_and_can_be_discarded(sandbox, fake_runtime):
    cfg, base = sandbox
    plan = make_plan(cfg)
    with mock.patch.object(ru, "_inference_smoke", side_effect=RuntimeError("CUDA out of memory")):
        with pytest.raises(RuntimeError, match="out of memory"):
            ru.apply_update(cfg, plan["plan_id"])
    candidate = ru._generation(cfg, ru.status(cfg)["generation_id"])
    ru.discard_candidate(cfg)
    assert not candidate.exists()
    assert ru.active_root(cfg) == base
    assert (base / "install.json").is_file()


def test_cancel_before_activation_preserves_active_runtime(sandbox, fake_runtime):
    cfg, base = sandbox
    plan = make_plan(cfg)
    def before_activate():
        assert ru.cancel_update(cfg)["cancel_accepted"]
    with pytest.raises(ru.UpdateCancelled):
        ru.apply_update(cfg, plan["plan_id"], before_activate=before_activate)
    assert ru.active_root(cfg) == base
    assert ru.status(cfg)["state"] == "canceled"


def test_legacy_running_server_is_not_force_stopped(sandbox, fake_runtime):
    cfg, base = sandbox
    plan = make_plan(cfg)
    with mock.patch.object(ds, "is_running", return_value=(True, 9123)), \
            mock.patch.object(ru, "_managed_identity", return_value={}), mock.patch.object(ds, "stop_server") as stop, \
            mock.patch.object(ru, "_inference_smoke") as smoke:
        result = ru.apply_update(cfg, plan["plan_id"], port=7865)
    assert result["pending_activation"]
    assert ru.active_root(cfg) == base
    stop.assert_not_called()
    smoke.assert_not_called()
    result = ru.activate_pending(cfg)
    assert result["state"] == "active"


def test_new_server_start_failure_rolls_back_even_after_pointer_switch(sandbox, fake_runtime):
    cfg, base = sandbox
    plan = make_plan(cfg)
    seen = []
    def start(cfg, identity, port, device, model, callback):
        seen.append((identity, ru._active_id(cfg)))
        if identity != "legacy":
            raise RuntimeError("new service failed readiness")
    with mock.patch.object(ru, "_start_verified", side_effect=start):
        with pytest.raises(RuntimeError, match="readiness"):
            ru.apply_update(cfg, plan["plan_id"], start_after=True)
    assert seen[0][0] == seen[0][1]  # checked after actual pointer switch
    assert ru.active_root(cfg) == base
    assert (base / "pylibs" / "old-sentinel").exists()


def test_rollback_restores_legacy_tree_without_deleting_candidate(sandbox, fake_runtime):
    cfg, base = sandbox
    plan = make_plan(cfg)
    ru.apply_update(cfg, plan["plan_id"])
    candidate = ru.active_root(cfg)
    result = ru.rollback(cfg)
    assert result["state"] == "rolled_back"
    assert ru.active_root(cfg) == base
    assert candidate.is_dir()


def test_stopped_server_rollback_rejects_damaged_v2_media_tools_before_pointer_switch(sandbox, fake_runtime):
    cfg, base = sandbox
    ru.apply_update(cfg, make_plan(cfg)["plan_id"])
    previous = ru.active_root(cfg)
    ru.apply_update(cfg, make_plan(cfg)["plan_id"])
    active = ru.active_root(cfg)
    pointer_before = (base / "active.json").read_bytes()
    media_name = ru._canonical_media_names(sys.platform)[0]
    damaged = previous / "tools" / "bin" / media_name
    damaged.write_bytes(b"X" * damaged.stat().st_size)
    with mock.patch.object(ds, "stop_server") as stop, pytest.raises(ValueError, match="receipt verification"):
        ru.rollback(cfg, start_after=False)
    stop.assert_not_called()
    assert (base / "active.json").read_bytes() == pointer_before
    assert ru.active_root(cfg) == active


def test_active_and_rollback_generations_cannot_be_discarded(sandbox, fake_runtime):
    cfg, _ = sandbox
    ru.apply_update(cfg, make_plan(cfg)["plan_id"])
    with pytest.raises(ValueError, match="cannot be discarded"):
        ru.discard_candidate(cfg)


def test_interrupted_stage_is_reported_and_not_automatically_promoted(sandbox):
    cfg, base = sandbox
    ru._persist(cfg, active=True, state="installing", generation_id="g-interrupted")
    assert ru.status(cfg)["needs_recovery"]
    assert not ru.status(cfg)["active"]
    with pytest.raises(RuntimeError, match="recovery"):
        with ru._operation(cfg, "downloading"):
            pass
    assert ru.active_root(cfg) == base


def test_bad_active_pointer_fails_closed(sandbox):
    cfg, base = sandbox
    (base / "active.json").write_text("{broken")
    with pytest.raises(RuntimeError, match="damaged"):
        ru.active_root(cfg)
    assert ru.inventory(cfg)["state"] == "damaged"


def test_missing_asset_never_reports_generation_models_ready(sandbox, fake_runtime):
    cfg, _ = sandbox
    ru.apply_update(cfg, make_plan(cfg)["plan_id"])
    assert ds.models_downloaded(cfg)
    (ru.active_root(cfg) / "models" / "bs_roformer_sw" / "BS-Roformer-SW.yaml").unlink()
    assert not ds.models_downloaded(cfg)
    assert "bs_roformer_sw" in ds.missing_models(cfg)


def test_install_context_does_not_change_other_threads_or_active_pointer(sandbox):
    import concurrent.futures
    cfg, base = sandbox
    candidate = base / "generations" / "candidate"
    with ds.installation_context(candidate):
        assert ds.src_dir(cfg) == candidate / "src"
        with concurrent.futures.ThreadPoolExecutor(1) as executor:
            assert executor.submit(ds.src_dir, cfg).result() == base / "src"
    assert ds.src_dir(cfg) == base / "src"


def test_legacy_install_api_cannot_clear_live_tree(sandbox):
    cfg, base = sandbox
    (base / "pylibs" / "uvicorn").mkdir()
    with pytest.raises(RuntimeError, match="already installed"):
        ds.install_server(cfg)
    assert (base / "pylibs" / "old-sentinel").read_text() == "DO_NOT_CHANGE"


def test_seal_must_succeed_before_drain_finishes(sandbox):
    cfg, _ = sandbox
    replies = [(200, {"idle": True}), (409, {}), (200, {"idle": True}), (200, {"idle": True, "sealed": True})]
    with mock.patch.object(ru, "_lifecycle", side_effect=replies) as life, mock.patch.object(ru.time, "sleep"):
        ru._drain(cfg, 7865, "gen", lambda: None, None)
    assert life.call_count == 4
    assert life.call_args.kwargs["seal"] is True


def test_candidate_model_write_cannot_change_rollback_weights(sandbox, fake_runtime):
    cfg, base = sandbox
    ru.apply_update(cfg, make_plan(cfg)["plan_id"])
    candidate = ru.active_root(cfg) / "models" / "bs_roformer_sw" / "BS-Roformer-SW.ckpt"
    candidate.write_bytes(b"broken candidate")
    assert (base / "cache" / "_roformer-models" / "BS-Roformer-SW.ckpt").read_bytes() == DATA


def test_changed_yaml_is_downloaded_without_redownloading_unchanged_weights(sandbox, fake_runtime):
    cfg, base = sandbox
    ru.apply_update(cfg, make_plan(cfg)["plan_id"])
    previous = ru.active_root(cfg)
    changed = manifest()
    updated = CONFIG + b"# changed defaults\n"
    changed["models"]["bs_roformer_sw"].update(revision="weights-v1-config-v2")
    changed["models"]["bs_roformer_sw"]["assets"][1] = asset("BS-Roformer-SW.yaml", updated)
    plan = make_plan(cfg, changed)
    def download(url, target, cancel, **kwargs):
        assert url.endswith("BS-Roformer-SW.yaml")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(updated)
    with mock.patch.object(ru, "_download_file", side_effect=download) as dl:
        ru.apply_update(cfg, plan["plan_id"])
    assert dl.call_count == 1
    assert (previous / "models" / "bs_roformer_sw" / "BS-Roformer-SW.yaml").read_bytes() == CONFIG
    assert (ru.active_root(cfg) / "models" / "bs_roformer_sw" / "BS-Roformer-SW.yaml").read_bytes() == updated


def test_model_setting_change_invalidates_plan_before_any_download(sandbox):
    cfg, _ = sandbox
    plan = make_plan(cfg)
    with mock.patch.object(ru, "_stage_candidate") as stage, pytest.raises(ValueError, match="model changed"):
        ru.apply_update(cfg, plan["plan_id"], model="htdemucs_6s")
    stage.assert_not_called()


def test_pending_candidate_cannot_be_silently_replaced(sandbox, fake_runtime):
    cfg, _ = sandbox
    plan = make_plan(cfg)
    with mock.patch.object(ds, "is_running", return_value=(True, 9123)), mock.patch.object(ru, "_managed_identity", return_value={}):
        ru.apply_update(cfg, plan["plan_id"])
    pending = ru.status(cfg)["generation_id"]
    with pytest.raises(RuntimeError, match="Activate or discard"):
        ru.apply_update(cfg, plan["plan_id"])
    assert ru.status(cfg)["generation_id"] == pending


def test_pypi_metadata_does_not_select_python_incompatible_or_prerelease(sandbox):
    profile = {"requirements": ["example>=1,<3"], "no_deps": []}
    data = {"releases": {"1.9": [{"requires_python": ">=3.10"}], "2.0": [{"requires_python": ">=3.14"}],
                         "2.1rc1": [{"requires_python": ">=3.10"}], "2.2": [{"yanked": True}]}}
    with mock.patch.object(ru, "_json_get", return_value=data), mock.patch.object(ru.platform, "python_version", return_value="3.12.10"):
        result = ru._candidate_versions(profile, False, "cu128")
    assert result["example"]["version"] == "1.9"
    assert result["example"]["newer_unsupported"]


def test_actual_driver_bootstrap_imports_sibling_with_isolated_search_path(tmp_path):
    import subprocess
    source = tmp_path / "src"
    source.mkdir()
    (source / "managed_runtime.py").write_text("IDENTITY='sibling-import-works'\n")
    driver = source / "run_demucs.py"
    driver.write_text(ds._BOOTSTRAP_TEMPLATE.format(pylibs=str(tmp_path / "libs")) + "from managed_runtime import IDENTITY\nprint(IDENTITY)\n")
    # -I removes script/cwd and PYTHONPATH like the packaged isolated runtime.
    proc = subprocess.run([sys.executable, "-I", "-B", str(driver)], capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "sibling-import-works"


def test_failed_stop_preserves_handle_and_state(sandbox):
    cfg, base = sandbox
    child = mock.Mock(pid=9999999)
    child.poll.return_value = None
    child.wait.side_effect = TimeoutError("still alive")
    state = ds.state_file(cfg)
    state.write_text('{"pid":9999999,"port":9123}')
    with mock.patch.object(ds, "_proc", child), mock.patch.object(ds, "_pid_alive", return_value=True), \
            mock.patch.object(ds.subprocess, "run"), mock.patch.object(ds, "_posix_kill_tree"):
        with pytest.raises(RuntimeError, match="did not stop"):
            ds.stop_server(cfg)
        assert ds._proc is child
    assert state.exists()


def test_progress_callback_failure_reaps_actual_pip_process(tmp_path):
    import engine_install
    import subprocess
    original = subprocess.Popen
    spawned = []
    def launch(_args, **kwargs):
        if _args[0] != sys.executable:
            return original(_args, **kwargs)
        proc = original([sys.executable, "-u", "-c", "import time;print('Collecting example');time.sleep(30)"], **kwargs)
        spawned.append(proc)
        return proc
    def progress(event):
        if event.get("line", "").startswith("Collecting"):
            raise ru.UpdateCancelled("user canceled while pip is writing")
    with mock.patch.object(engine_install.subprocess, "Popen", side_effect=launch), \
            pytest.raises(ru.UpdateCancelled, match="user canceled"):
        engine_install.stream_pip(sys.executable, ["example"], "test", progress, 0, 1, cancel_cb=lambda: None)
    assert len(spawned) == 1
    assert spawned[0].poll() is not None


def test_os_operation_lock_blocks_second_updater_and_distinguishes_live_process(sandbox):
    cfg, _ = sandbox
    handle = ru._acquire_operation_lock(cfg)
    try:
        ru._persist(cfg, active=True, state="installing")
        state = ru.status(cfg)
        assert state["active"]
        assert not state["needs_recovery"]
        with pytest.raises(RuntimeError, match="Another application"):
            with ru._operation(cfg, "downloading"):
                pass
    finally:
        ru._lock_file(handle, unlock=True)
        handle.close()
    assert ru.status(cfg)["needs_recovery"]


def test_archive_must_contain_runtime_contract_and_match_checked_manifest(sandbox, tmp_path):
    import zipfile
    cfg, _ = sandbox
    plan = make_plan(cfg)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    def archive(_url, target, _cancel):
        with zipfile.ZipFile(target, "w") as z:
            for name in ds.SOURCE_FILES:
                z.writestr("repo/" + name, "source")
    with mock.patch.object(ru, "_download_file", side_effect=archive), pytest.raises(ValueError, match="contract files"):
        ru._stage_source(cfg, candidate, plan, lambda: None)
    assert not (candidate / "source.zip").exists()


def test_cancel_prepared_candidate_restores_normal_start_and_retains_discard(sandbox, fake_runtime):
    cfg, base = sandbox
    plan = make_plan(cfg)
    with mock.patch.object(ds, "is_running", return_value=(True, 9123)), mock.patch.object(ru, "_managed_identity", return_value={}):
        ru.apply_update(cfg, plan["plan_id"])
    result = ru.cancel_update(cfg)
    assert result["cancel_accepted"]
    assert result["state"] == "canceled"
    assert not result["pending_activation"]
    assert ru.active_root(cfg) == base
    ru.discard_candidate(cfg)


def test_status_retains_reviewable_plan_without_reapply_or_private_manifest(sandbox, fake_runtime):
    cfg, _ = sandbox
    plan = make_plan(cfg)
    with mock.patch.object(ds, "is_running", return_value=(True, 9123)), mock.patch.object(ru, "_managed_identity", return_value={}):
        ru.apply_update(cfg, plan["plan_id"])
    summary = ru.status(cfg)["checked_plan"]
    assert summary["plan_id"] == plan["plan_id"]
    assert summary["available"] == plan["available"]
    assert not summary["can_update"]
    assert "manifest" not in summary
    assert "model_specs" not in summary
    assert "asset_dir" not in json.dumps(summary)


def test_managed_pip_worker_exits_if_application_parent_dies(tmp_path):
    import engine_install
    import os
    import signal
    import subprocess
    import time
    marker = tmp_path / "owned-worker.pid"
    # Exercise the exact watchdog with a disposable sleeping payload, no pip or
    # package operations. The child is its own process group, like stream_pip.
    payload = f"from pathlib import Path;Path({str(marker)!r}).write_text(str(os.getpid()));time.sleep(30)"
    wrapper = engine_install._MANAGED_PIP_WRAPPER.replace("runpy.run_module('pip', run_name='__main__')", payload)
    group = {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {"start_new_session": True}
    parent_code = (
        "import os,subprocess,sys\n"
        "env=dict(os.environ);env['STEM_SPLITTER_INSTALL_PARENT_PID']=str(os.getpid())\n"
        f"child=subprocess.Popen([sys.executable,'-c',{wrapper!r}],env=env,**{group!r})\n"
        "child.wait()\n"
    )
    parent = subprocess.Popen([sys.executable, "-B", "-c", parent_code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              **({"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}))
    worker_pid = None
    try:
        deadline = time.monotonic() + 8
        while not marker.exists() and time.monotonic() < deadline:
            assert parent.poll() is None
            time.sleep(.05)
        assert marker.exists(), "worker never reached its payload"
        worker_pid = int(marker.read_text())
        assert ds._pid_alive(worker_pid)
        parent.terminate()  # simulate app crash, deliberately only the parent
        parent.wait(timeout=5)
        deadline = time.monotonic() + 8
        while ds._pid_alive(worker_pid) and time.monotonic() < deadline:
            time.sleep(.05)
        assert not ds._pid_alive(worker_pid), "installer survived its application parent"
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait()
        if worker_pid and ds._pid_alive(worker_pid):
            os.kill(worker_pid, signal.SIGTERM)


def test_guarded_validation_supports_code_and_driver_arguments(tmp_path):
    import os
    env = dict(os.environ)
    assert ru._run_process([sys.executable, "-B", "-c", "print('import-check-ok')"], env, lambda: None).strip() == "import-check-ok"
    driver = tmp_path / "driver.py"
    driver.write_text("import sys\nfrom pathlib import Path\nprint(Path(__file__).name, sys.argv[1])\n")
    assert ru._run_process([sys.executable, str(driver), "--help"], env, lambda: None).strip() == "driver.py --help"


def test_progress_retains_pip_package_line_separately_from_phase(sandbox):
    cfg, _ = sandbox
    seen = []
    ru._event(cfg, seen.append, "installing", .3, "Resolving / downloading", line="Collecting torch~=2.8.0")
    assert ru.status(cfg)["line"] == "Collecting torch~=2.8.0"
    assert seen[0]["phase"] == "Resolving / downloading"
    assert seen[0]["line"] == "Collecting torch~=2.8.0"


@pytest.mark.parametrize("model,device,error", [
    ("htdemucs_6s", "cpu", "Choose bs_roformer_sw before Restore"),
    ("bs_roformer_sw", "cuda", "Choose CPU before Restore"),
    ("bs_roformer_sw", "cuda:1", "Choose CPU before Restore"),
])
def test_rollback_rejects_incompatible_selection_before_touching_running_server(sandbox, fake_runtime, model, device, error):
    cfg, base = sandbox
    ru.apply_update(cfg, make_plan(cfg)["plan_id"])
    previous = ru.active_root(cfg)
    ru.apply_update(cfg, make_plan(cfg)["plan_id"])
    active = ru.active_root(cfg)
    pointer_before = (base / "active.json").read_bytes()
    before_activate = mock.Mock()
    with mock.patch.object(ds, "is_running", return_value=(True, 9123)), \
            mock.patch.object(ds, "stop_server") as stop, mock.patch.object(ru, "_start_verified") as start:
        with pytest.raises(ValueError, match=error):
            ru.rollback(cfg, model=model, device=device, before_activate=before_activate)
    stop.assert_not_called()
    start.assert_not_called()
    before_activate.assert_not_called()
    assert (base / "active.json").read_bytes() == pointer_before
    assert ru.active_root(cfg) == active
    assert previous.exists()


def test_recovery_rollback_does_not_silently_substitute_recorded_model_for_user_selection(sandbox, fake_runtime):
    cfg, base = sandbox
    ru.apply_update(cfg, make_plan(cfg)["plan_id"])
    previous = ru.active_root(cfg)
    ru.apply_update(cfg, make_plan(cfg)["plan_id"])
    pointer_before = (base / "active.json").read_bytes()
    ru._persist(cfg, active=False, state="recovery_required", activation={"previous_generation": previous.name,
                "was_running": True, "device": "cpu", "model": "bs_roformer_sw", "port": 9123})
    with mock.patch.object(ds, "stop_server") as stop, pytest.raises(ValueError, match="Choose bs_roformer_sw"):
        ru.rollback(cfg, model="htdemucs_6s", device="cpu")
    stop.assert_not_called()
    assert (base / "active.json").read_bytes() == pointer_before
