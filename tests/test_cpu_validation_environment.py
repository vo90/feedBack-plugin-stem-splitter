"""CPU inference masking is scoped to validation, not GPU capability checks."""
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import demucs_server as ds
import runtime_update as ru


def _plan(gpu, device, engine="audio-separator"):
    return {"gpu": gpu, "device": device, "model": "bs_roformer_sw", "model_specs": {
        "bs_roformer_sw": {"engine": engine, "entrypoint": "model.ckpt", "stems": ["guitar", "other"]}}}


@pytest.mark.parametrize("gpu,device,engine,expected", [
    (True, "cpu", "audio-separator", "-1"),
    (False, "cpu", "audio-separator", "-1"),
    (False, "", "roformer", "-1"),
    (True, "cpu", "demucs", "-1"),
    (True, "cuda", "audio-separator", "0,1"),
    (True, "cuda:1", "audio-separator", "0,1"),
    (True, "", "audio-separator", "0,1"),
])
def test_real_inference_smoke_builds_device_specific_child_environment(tmp_path, monkeypatch, gpu, device, engine, expected):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    root = tmp_path / "candidate"
    (root / "src").mkdir(parents=True)
    calls = []
    def process(args, env, cancel, **kwargs):
        calls.append((list(args), dict(env)))
        cancel()
    # Do not mock _inference_smoke: exercise fixture creation, device selection,
    # child arguments and the output-validation launch. Only the subprocess is fake.
    with mock.patch.object(ru, "_run_process", side_effect=process):
        ru._inference_smoke(tmp_path, root, _plan(gpu, device, engine), lambda: None)
    assert len(calls) == 2
    assert calls[0][0][calls[0][0].index("-d") + 1] == (device or ("cuda" if gpu else "cpu"))
    assert all(env["CUDA_VISIBLE_DEVICES"] == expected for _, env in calls)
    assert all(env["HF_HUB_OFFLINE"] == "1" for _, env in calls)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert (root / "validation" / "fixture.wav").stat().st_size > 44100 * 11 * 4


def test_cpu_selection_does_not_mask_gpu_dependency_capability_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    root = tmp_path / "candidate"
    (root / "src").mkdir(parents=True)
    calls = []
    with mock.patch.object(ds, "write_launcher"), mock.patch.object(ds, "patch_driver_scripts"), \
            mock.patch.object(ru, "_run_process", side_effect=lambda args, env, *a, **kw: calls.append((args, dict(env)))), \
            mock.patch.object(ru, "_probe_candidate_health") as health:
        ru._validate_candidate(tmp_path, root, _plan(True, "cpu"), lambda: None)
    assert "CUDA_PROBE:" in calls[0][0][-1]
    assert "device='cuda'" in calls[0][0][-1]
    assert all(env["CUDA_VISIBLE_DEVICES"] == "0,1" for _, env in calls)
    assert health.call_args.args[3]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
