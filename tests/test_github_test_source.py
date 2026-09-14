"""Test-branch GitHub routing with real plans/files and HTTP boundary fixtures."""
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from unittest import mock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import demucs_server as ds
import runtime_update as ru
from test_runtime_update import fake_runtime, legacy, make_plan, manifest  # noqa: F401

REPO = "vo90/feedBack-demucs-server"
REF = "review/managed-runtime"
COMMIT = "b" * 40
ENV = ("FEEDBACK_STEM_TEST_REPO", "FEEDBACK_STEM_TEST_CONFIG_DIR", "STEM_SPLITTER_SERVER_REF")


@pytest.fixture(autouse=True)
def clean_source_environment(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)


def enable(monkeypatch, cfg):
    monkeypatch.setenv(ENV[0], REPO)
    monkeypatch.setenv(ENV[1], str(cfg.resolve()))
    monkeypatch.setenv(ENV[2], REF)


def archive_bytes(spec=None):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name in ds.RUNTIME_SOURCE_FILES:
            data = json.dumps(spec or manifest()) if name == ru.MANIFEST_NAME else "# server source\n"
            archive.writestr("feedBack-demucs-server-" + COMMIT + "/" + name, data)
    return stream.getvalue()


class Response:
    status_code = 200

    def __init__(self, text="", data=None, content=b""):
        self.text, self.data, self.content = text, data, content

    def json(self):
        return self.data

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def github_fixture(monkeypatch, spec=None):
    urls = []
    payload = archive_bytes(spec)

    def get(url, **kwargs):
        urls.append(url)
        if url == f"https://api.github.com/repos/{REPO}/commits/review%2Fmanaged-runtime":
            return Response(text=COMMIT)
        if url == f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{ru.MANIFEST_NAME}":
            return Response(data=manifest())
        if url == f"https://codeload.github.com/{REPO}/zip/{COMMIT}":
            return Response(content=payload)
        raise AssertionError("Unexpected network destination: " + url)

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(ru, "_candidate_versions", lambda *args: {})
    return urls, payload


def test_github_check_and_archive_use_exact_fork_commit_and_record_hash(tmp_path, monkeypatch):
    cfg = tmp_path / "profile"
    enable(monkeypatch, cfg)
    urls, payload = github_fixture(monkeypatch)
    plan = ru.check_updates(cfg, ref="main", gpu=False)
    assert plan["can_update"], plan
    assert plan["ref"] == REF
    assert plan["source"]["repo"] == REPO
    assert plan["source"]["commit"] == COMMIT
    assert plan["manifest_sha256"] == ru._digest(manifest())
    assert ru._get_plan(cfg, plan["plan_id"])["source"] == plan["source"]
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    ru._stage_source(cfg, candidate, plan, lambda: None)
    record = ru._read(candidate / "source.json")
    assert record["repo"] == REPO and record["commit"] == COMMIT
    assert record["archive_sha256"] == hashlib.sha256(payload).hexdigest()
    assert not (candidate / "source.zip").exists()
    assert len(urls) == 3


def test_scoped_source_is_visible_offline_and_other_profile_uses_official_main(tmp_path, monkeypatch):
    cfg = tmp_path / "test"
    enable(monkeypatch, cfg)
    monkeypatch.setattr(ds, "DEFAULT_SOURCE_REF", REF)  # import during the test launch
    with mock.patch.object(requests, "get", side_effect=AssertionError("status must be offline")):
        assert ru.inventory(cfg)["source"]["kind"] == "github_test"
        assert ru.status(cfg)["source"]["ref"] == REF
        other = ds.update_source(tmp_path / "normal")
        assert other["repo"] == ds.SOURCE_REPO and other["ref"] == "main"
        assert ds.update_source(tmp_path / "normal", "release-v1")["ref"] == "release-v1"


@pytest.mark.parametrize("key,value", [(ENV[0], "other/arbitrary-repo"), (ENV[1], ""),
                                      (ENV[1], "relative/profile"), (ENV[2], ""),
                                      (ENV[2], "https://example.test/bad")])
def test_invalid_opt_in_keeps_inventory_and_blocks_without_network(tmp_path, monkeypatch, key, value):
    cfg = tmp_path / "profile"
    legacy(cfg)
    enable(monkeypatch, cfg)
    monkeypatch.setenv(key, value)
    with mock.patch.object(requests, "get") as network:
        inv = ru.inventory(cfg)
        assert inv["installed"] and inv["source"]["error"]
        plan = ru.check_updates(cfg)
        assert not plan["can_update"] and plan["reason"]
        network.assert_not_called()


@pytest.mark.parametrize("change", ["repo", "ref", "scope", "disabled"])
def test_saved_plan_rejects_source_switch_before_apply(tmp_path, monkeypatch, change):
    cfg = tmp_path / "profile"
    enable(monkeypatch, cfg)
    plan = make_plan(cfg)
    assert plan["can_update"]
    if change == "repo":
        monkeypatch.setenv(ENV[0], "other/repo")
    elif change == "ref":
        monkeypatch.setenv(ENV[2], "different-ref")
    elif change == "scope":
        monkeypatch.setenv(ENV[1], str(tmp_path / "different-profile"))
    else:
        monkeypatch.delenv(ENV[0])
    with mock.patch.object(ru, "_stage_candidate") as stage:
        with pytest.raises(ValueError):
            ru.apply_update(cfg, plan["plan_id"])
        stage.assert_not_called()
    assert not (ru._base(cfg) / "active.json").exists()


def test_official_checked_plan_cannot_be_used_after_enabling_fork(tmp_path, monkeypatch):
    plan = make_plan(tmp_path)
    enable(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="source changed"):
        ru._get_plan(tmp_path, plan["plan_id"])


def test_persisted_identity_and_commit_changes_are_rejected(tmp_path, monkeypatch):
    enable(monkeypatch, tmp_path)
    plan = make_plan(tmp_path)
    path = ru._base(tmp_path) / "plans" / (plan["plan_id"] + ".json")
    for key, value in (("source_identity_sha256", "0" * 64), ("source_commit", "a" * 40)):
        ru._atomic_json(path, {**plan, key: value})
        with pytest.raises(ValueError, match="source changed"):
            ru._get_plan(tmp_path, plan["plan_id"])


@pytest.mark.usefixtures("fake_runtime")
def test_source_change_during_validation_preserves_active_pointer(tmp_path, monkeypatch):
    cfg = tmp_path / "profile"
    base = legacy(cfg)
    enable(monkeypatch, cfg)
    plan = make_plan(cfg)
    with mock.patch.object(ru, "_inference_smoke", side_effect=lambda *args: monkeypatch.setenv(ENV[2], "changed")), \
            mock.patch.object(ru, "_stage_models", return_value={}):
        with pytest.raises(ValueError, match="source changed"):
            ru.apply_update(cfg, plan["plan_id"])
    assert not (base / "active.json").exists()
    assert (base / "src" / "server.py").read_text() == "OLD_SERVER"


def test_manifest_mismatch_rejected_on_real_archive_path(tmp_path, monkeypatch):
    enable(monkeypatch, tmp_path)
    changed = manifest()
    changed["models"]["bs_roformer_sw"]["revision"] = "unexpected"
    github_fixture(monkeypatch, changed)
    plan = ru.check_updates(tmp_path, gpu=False)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    with pytest.raises(ValueError, match="planned compatibility manifest"):
        ru._stage_source(tmp_path, candidate, plan, lambda: None)
    assert not (candidate / "source.json").exists()


def test_models_only_requires_same_installed_repository(tmp_path, monkeypatch):
    legacy(tmp_path)
    enable(monkeypatch, tmp_path)
    with mock.patch.object(requests, "get") as network:
        plan = ru.check_updates(tmp_path, components=["models"])
    assert not plan["can_update"] and "include Server" in plan["reason"]
    network.assert_not_called()


def test_installed_origin_remains_separate_after_source_disabled(tmp_path, monkeypatch):
    base = legacy(tmp_path)
    ru._atomic_json(base / "source.json", {"repo": REPO, "ref": REF, "commit": COMMIT})
    enable(monkeypatch, tmp_path)
    monkeypatch.delenv(ENV[0])
    inv = ru.inventory(tmp_path)
    assert inv["source"]["kind"] == "github"
    assert inv["installed_source"]["repo"] == REPO


def test_legacy_download_routes_to_same_fork_and_records_origin(tmp_path, monkeypatch):
    enable(monkeypatch, tmp_path)
    urls, _ = github_fixture(monkeypatch)
    ds.download_source(tmp_path, ref="main")
    assert len(urls) == 2
    assert ds.source_meta(tmp_path)["repo"] == REPO
    assert ds.source_meta(tmp_path)["ref"] == REF


def test_legacy_test_download_does_not_fallback_to_unresolved_branch(tmp_path, monkeypatch):
    enable(monkeypatch, tmp_path)
    with mock.patch.object(ds, "_resolve_commit", return_value=None), mock.patch.object(requests, "get") as network:
        with pytest.raises(ValueError, match="immutable"):
            ds.download_source(tmp_path)
    network.assert_not_called()


def test_ml_child_environment_drops_test_override_and_preserves_parent(tmp_path, monkeypatch):
    enable(monkeypatch, tmp_path)
    with mock.patch.object(ds, "_migrate_torch_home"), mock.patch("audio._ffmpeg_cmd", return_value="ffmpeg"):
        env = ds._server_env(tmp_path)
    assert all(name not in env for name in ENV)
    assert os.environ[ENV[0]] == REPO and os.environ[ENV[2]] == REF


def test_models_only_keeps_installed_fork_commit_when_branch_has_moved(tmp_path, monkeypatch):
    base = legacy(tmp_path)
    enable(monkeypatch, tmp_path)
    ru._atomic_json(base / "source.json", {"repo": REPO, "ref": REF, "commit": "a" * 40})
    with mock.patch.object(ds, "_resolve_commit") as resolve, \
            mock.patch.object(ru, "_load_manifest", return_value=manifest()) as catalog, \
            mock.patch.object(ru, "_check_installed_requirements"):
        plan = ru.check_updates(tmp_path, components=["models"])
    assert plan["can_update"], plan
    assert plan["source_commit"] == "a" * 40
    resolve.assert_not_called()
    catalog.assert_called_once_with("a" * 40, REPO)


@pytest.mark.usefixtures("fake_runtime")
def test_prepared_candidate_rejects_changed_source_after_restart(tmp_path, monkeypatch):
    cfg = tmp_path / "profile"
    legacy(cfg)
    enable(monkeypatch, cfg)
    plan = make_plan(cfg)
    with mock.patch.object(ds, "is_running", return_value=(True, 7865)), \
            mock.patch.object(ru, "_managed_identity", return_value={}), \
            mock.patch.object(ru, "_stage_models", return_value={}):
        ru.apply_update(cfg, plan["plan_id"])
    assert ru.status(cfg)["pending_activation"]
    monkeypatch.setenv(ENV[2], "changed-after-restart")
    with pytest.raises(ValueError, match="source changed"):
        ru.activate_pending(cfg)
    assert not (ru._base(cfg) / "active.json").exists()


def test_fork_archive_failure_does_not_try_another_source(tmp_path, monkeypatch):
    enable(monkeypatch, tmp_path)
    plan = make_plan(tmp_path)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    with mock.patch.object(requests, "get", side_effect=requests.ConnectionError("offline")) as network:
        with pytest.raises(requests.ConnectionError):
            ru._stage_source(tmp_path, candidate, plan, lambda: None)
    assert network.call_count == 1
    assert network.call_args.args[0] == f"https://codeload.github.com/{REPO}/zip/{COMMIT}"
    assert not (candidate / "source.json").exists()


def test_official_source_urls_remain_official_without_override(tmp_path, monkeypatch):
    urls = []

    def get(url, **kwargs):
        urls.append(url)
        if "/commits/" in url:
            return Response(text=COMMIT)
        return Response(data=manifest())

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(ru, "_candidate_versions", lambda *args: {})
    plan = ru.check_updates(tmp_path, ref="main", gpu=False)
    assert plan["can_update"] and plan["source"]["kind"] == "github"
    assert urls == [f"https://api.github.com/repos/{ds.SOURCE_REPO}/commits/main",
                    f"https://raw.githubusercontent.com/{ds.SOURCE_REPO}/{COMMIT}/{ru.MANIFEST_NAME}"]


@pytest.mark.parametrize("test_launch,expected", [(True, "main"), (False, "official-pinned-ref")])
def test_imported_defaults_do_not_keep_test_ref_after_override_removed(tmp_path, test_launch, expected):
    env = dict(os.environ)
    env[ENV[2]] = REF if test_launch else "official-pinned-ref"
    if test_launch:
        env[ENV[0]], env[ENV[1]] = REPO, str(tmp_path)
    code = (
        "import os,json; from pathlib import Path; import demucs_server as ds; "
        "os.environ.pop('FEEDBACK_STEM_TEST_REPO',None); "
        "os.environ.pop('FEEDBACK_STEM_TEST_CONFIG_DIR',None); "
        "print(json.dumps(ds.update_source_status(Path('.'))))"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], env=env,
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    source = json.loads(result.stdout)
    assert source["kind"] == "github" and source["ref"] == expected
