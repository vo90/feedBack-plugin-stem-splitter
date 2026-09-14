"""Exercise the real HTTP update controls and the queue/cutover boundary."""
import json
import sys
import threading
import time
from pathlib import Path
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import routes
import runtime_update

PREFIX = "/api/plugins/stem_splitter/server/runtime"


@pytest.fixture
def runtime_api(tmp_path):
    settings = {"remote_model": "htdemucs_6s", "local_server_port": 19861,
                "local_server_device": "cpu", "local_server_gpu": False,
                "local_server_ref": "release-test", "local_server_cuda_tag": "",
                "local_server_autostart": False}
    (tmp_path / "stem_splitter.json").write_text(json.dumps(settings))
    managers = []

    class CapturedManager(routes.JobManager):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            managers.append(self)

    app = FastAPI()
    with mock.patch.object(routes, "JobManager", CapturedManager), \
            mock.patch.object(CapturedManager, "_worker_loop", return_value=None):
        routes.setup(app, {"config_dir": str(tmp_path)})
    return TestClient(app), managers[0], tmp_path


def test_inventory_and_status_never_check_or_install(runtime_api):
    client, _, cfg = runtime_api
    with mock.patch.object(runtime_update, "check_updates") as check, \
            mock.patch.object(runtime_update, "apply_update") as apply:
        assert client.get(PREFIX + "/inventory").status_code == 200
        assert client.get(PREFIX + "/status").status_code == 200
    check.assert_not_called()
    apply.assert_not_called()
    assert not (cfg / "demucs-server").exists()


def test_check_uses_current_settings_and_explicit_blank_default(runtime_api):
    client, _, cfg = runtime_api
    with mock.patch.object(runtime_update, "check_updates", return_value={"can_update": True}) as check:
        response = client.post(PREFIX + "/check", json={"ref": "", "gpu": False})
    assert response.json()["can_update"]
    check.assert_called_once_with(cfg, ref=None, model="htdemucs_6s", gpu=False, cuda_tag=None)


def test_unsaved_model_cannot_mismatch_the_checked_runtime(runtime_api):
    client, _, _ = runtime_api
    with mock.patch.object(runtime_update, "check_updates") as check:
        response = client.post(PREFIX + "/check", json={"model": "bs_roformer_sw"}).json()
    assert response["can_update"] is False
    assert "Save" in response["reason"]
    check.assert_not_called()


def test_update_requires_checked_plan_not_browser_package_specs(runtime_api):
    client, _, _ = runtime_api
    with mock.patch.object(runtime_update, "apply_update") as apply:
        response = client.post(PREFIX + "/update", json={"packages": ["arbitrary-package"]})
    assert response.status_code == 400
    apply.assert_not_called()


def test_update_preserves_device_port_model_and_passes_only_plan_id(runtime_api):
    client, mgr, cfg = runtime_api
    done = threading.Event()

    def apply(*args, **kwargs):
        assert args == (cfg, "checked-plan")
        assert kwargs["port"] == 19861
        assert kwargs["device"] == "cpu"
        assert kwargs["model"] == "htdemucs_6s"
        assert kwargs["before_activate"] == mgr.wait_for_runtime_activation
        assert "packages" not in kwargs
        done.set()
        return {"state": "waiting_to_activate", "phase": "Prepared", "pct": .8}

    with mock.patch.object(runtime_update, "apply_update", side_effect=apply):
        response = client.post(PREFIX + "/update", json={"plan_id": "checked-plan", "packages": ["ignored"]})
        assert response.json()["started"] == "runtime_update"
        assert done.wait(2)


def test_other_lifecycle_operation_blocks_update_and_check(runtime_api):
    client, mgr, _ = runtime_api
    mgr._server_op_active = "prepare_models"
    with mock.patch.object(runtime_update, "apply_update") as apply, \
            mock.patch.object(runtime_update, "check_updates") as check:
        assert client.post(PREFIX + "/update", json={"plan_id": "checked"}).json()["busy"] == "prepare_models"
        assert client.post(PREFIX + "/check").json()["busy"] == "prepare_models"
    apply.assert_not_called()
    check.assert_not_called()


def test_cancellation_is_available_while_lifecycle_lock_is_held(runtime_api):
    client, mgr, cfg = runtime_api
    mgr._server_op_active = "runtime_update"
    with mock.patch.object(runtime_update, "cancel_update", return_value={"cancel_requested": True}) as cancel:
        assert client.post(PREFIX + "/cancel").json()["cancel_requested"]
    cancel.assert_called_once_with(cfg)


def test_start_activates_pending_candidate_instead_of_starting_old_source(runtime_api):
    client, mgr, cfg = runtime_api
    finished = threading.Event()

    def activate(*args, **kwargs):
        assert args == (cfg,)
        assert kwargs["start_after"] is True
        assert kwargs["model"] == "htdemucs_6s"
        finished.set()
        return {"state": "active"}

    with mock.patch.object(runtime_update, "status", return_value={"pending_activation": True}), \
            mock.patch.object(runtime_update, "activate_pending", side_effect=activate), \
            mock.patch.object(routes.demucs_server, "start_server") as old_start:
        response = client.post("/api/plugins/stem_splitter/server/start")
        assert response.json()["started"] == "runtime_activate"
        assert finished.wait(2)
    old_start.assert_not_called()


def test_activation_waits_for_current_job_and_preserves_user_pause(runtime_api):
    _, mgr, _ = runtime_api
    mgr.paused.set()
    mgr.jobs["current"] = {"status": "running"}
    finished = threading.Event()
    thread = threading.Thread(target=lambda: (mgr.wait_for_runtime_activation(), finished.set()), daemon=True)
    thread.start()
    assert mgr._runtime_dispatch_paused.wait(1)
    assert not finished.wait(.05)
    with mgr.lock:
        mgr.jobs["current"]["status"] = "done"
    assert finished.wait(1)
    assert mgr.paused.is_set()
    assert mgr._runtime_dispatch_paused.is_set()


def test_failed_activation_releases_only_update_gate(runtime_api):
    _, mgr, _ = runtime_api
    mgr.paused.set()
    finished = threading.Event()

    def fail(_cb):
        mgr.wait_for_runtime_activation()
        raise RuntimeError("candidate failed")

    with mock.patch.object(mgr, "push_event", side_effect=lambda event: finished.set() if event["type"] == "server_error" else None):
        assert mgr.run_server_op("runtime_update", fail)
        assert finished.wait(2)
        deadline = time.monotonic() + 2
        while mgr._server_op_active and time.monotonic() < deadline:
            time.sleep(.01)
    assert not mgr._runtime_dispatch_paused.is_set()
    assert mgr.paused.is_set()


def test_managed_split_never_downloads_unrelated_lyrics_models(runtime_api):
    _, mgr, _ = runtime_api
    with mock.patch.object(routes.demucs_server, "missing_models", return_value=["whisperx", "whisperx aligner"]), \
            mock.patch.object(mgr, "local_server_url", return_value="http://127.0.0.1:19861"), \
            mock.patch.object(mgr, "resolve_split_engine", return_value=("remote", "")), \
            mock.patch.object(runtime_update, "inventory", return_value={"legacy": False}):
        assert mgr.needs_server_setup("split") is None


def test_status_passes_saved_device_and_selected_model_to_presentation(runtime_api):
    client, _, cfg = runtime_api
    with mock.patch.object(routes.demucs_server, "server_status", return_value={"presentation": {}}) as status:
        assert client.get("/api/plugins/stem_splitter/server_status").status_code == 200
    status.assert_called_once_with(cfg, model="htdemucs_6s", requested_device="cpu")
