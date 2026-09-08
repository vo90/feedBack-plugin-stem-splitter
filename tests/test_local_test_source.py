"""Explicit local-source transport uses real archives and the ordinary updater."""
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import demucs_server as ds
import runtime_update as ru
from test_runtime_update import manifest, legacy


@pytest.fixture
def local_bundle(tmp_path, monkeypatch):
    cfg = tmp_path / "test-profile"
    legacy(cfg)
    directory = tmp_path / "bundle"
    directory.mkdir()
    spec = manifest()
    raw_manifest = json.dumps(spec).encode()
    (directory / ru.MANIFEST_NAME).write_bytes(raw_manifest)
    archive = directory / "server.zip"
    commit = "b" * 40
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.comment = commit.encode()
        for name in ds.RUNTIME_SOURCE_FILES:
            zipped.writestr("server/" + name, raw_manifest if name == ru.MANIFEST_NAME else "# test source\n")
    bundle = {"schema_version": 1, "kind": "local_git_archive", "source_commit": commit,
              "ref": "feat/unpublished-runtime", "archive": archive.name,
              "archive_size": archive.stat().st_size, "archive_sha256": ru._hash_file(archive),
              "manifest": ru.MANIFEST_NAME, "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest()}
    path = directory / "source-bundle.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    monkeypatch.setenv("FEEDBACK_STEM_TEST_SOURCE", str(path))
    monkeypatch.setenv("FEEDBACK_STEM_TEST_CONFIG_DIR", str(cfg))
    return cfg, path, bundle


def checked(cfg):
    with mock.patch.object(ds, "_resolve_commit") as resolve, \
            mock.patch.object(ru, "_load_manifest") as fetch, \
            mock.patch.object(ru, "_candidate_versions", return_value={}):
        result = ru.check_updates(cfg, ref="main")
    resolve.assert_not_called()
    fetch.assert_not_called()
    assert result["can_update"], result.get("reason")
    return result


def test_scoped_source_is_visible_before_check_and_uses_fixed_bundle_revision(local_bundle):
    cfg, _, bundle = local_bundle
    for status in (ru.inventory(cfg), ru.status(cfg)):
        assert status["source"]["kind"] == "local_test"
        assert status["source"]["commit"] == bundle["source_commit"]
        assert "not published" in status["source"]["label"]
        assert str(cfg.parent) not in json.dumps(status["source"])
    plan = checked(cfg)
    assert plan["ref"] == bundle["ref"]
    assert plan["source_commit"] == bundle["source_commit"]
    assert ru._get_plan(cfg, plan["plan_id"])["source"] == plan["source"]


def test_local_archive_uses_normal_contract_extraction_without_network(local_bundle, tmp_path):
    cfg, _, bundle = local_bundle
    plan = checked(cfg)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    with mock.patch.object(ru, "_download_file") as download:
        ru._stage_source(cfg, candidate, plan, lambda: None)
    download.assert_not_called()
    assert set(p.name for p in (candidate / "src").iterdir()) == set(ds.RUNTIME_SOURCE_FILES)
    assert ru._read(candidate / "src" / ru.MANIFEST_NAME) == plan["manifest"]
    assert ru._read(candidate / "source.json")["source"] == plan["source"]
    assert not (candidate / "source.zip").exists()


def test_inherited_environment_cannot_redirect_other_profile(local_bundle, tmp_path):
    _, path, _ = local_bundle
    other = tmp_path / "normal-profile"
    # Even an unavailable bundle must not affect a different profile.
    path.unlink()
    with mock.patch.object(ds, "_resolve_commit", return_value="c" * 40) as resolve, \
            mock.patch.object(ru, "_load_manifest", return_value=manifest()), \
            mock.patch.object(ru, "_candidate_versions", return_value={}):
        plan = ru.check_updates(other, ref="main")
    resolve.assert_called_once_with("main")
    assert plan["can_update"]
    assert plan["source"]["kind"] == "github"
    assert plan["source_commit"] == "c" * 40


def test_configured_source_and_installed_origin_are_distinct(local_bundle, monkeypatch):
    cfg, _, _ = local_bundle
    plan = checked(cfg)
    ru._atomic_json(ru._base(cfg) / "source.json", {"commit": plan["source_commit"], "source": plan["source"]})
    monkeypatch.delenv("FEEDBACK_STEM_TEST_SOURCE")
    monkeypatch.delenv("FEEDBACK_STEM_TEST_CONFIG_DIR")
    inv = ru.inventory(cfg)
    assert inv["source"]["kind"] == "github"
    assert inv["installed_source"] == plan["source"]


@pytest.mark.parametrize("change", ["metadata", "archive", "manifest", "disabled", "different-profile"])
def test_persisted_plan_rejects_changed_source_before_staging(local_bundle, monkeypatch, change):
    cfg, path, bundle = local_bundle
    plan = checked(cfg)
    if change == "metadata":
        path.write_text(path.read_text() + "\n")
    elif change in {"archive", "manifest"}:
        with (path.parent / bundle[change]).open("ab") as handle:
            handle.write(b"changed after checking")
    elif change == "disabled":
        monkeypatch.delenv("FEEDBACK_STEM_TEST_SOURCE")
        monkeypatch.delenv("FEEDBACK_STEM_TEST_CONFIG_DIR")
    else:
        monkeypatch.setenv("FEEDBACK_STEM_TEST_CONFIG_DIR", str(cfg.parent / "other"))
    with mock.patch.object(ru, "_stage_candidate") as stage, pytest.raises(ValueError, match="Local test source"):
        ru.apply_update(cfg, plan["plan_id"])
    stage.assert_not_called()
    assert not (ru._base(cfg) / "active.json").exists()


def test_persisted_plan_can_be_loaded_by_a_fresh_process(local_bundle):
    cfg, _, _ = local_bundle
    plan = checked(cfg)
    code = ("from pathlib import Path;import runtime_update as ru;"
            f"p=ru._get_plan(Path({str(cfg)!r}), {plan['plan_id']!r});print(p['source']['kind'])")
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(ru.__file__).parent,
                            env=dict(os.environ), capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "local_test"


def test_archive_replaced_after_validation_is_rejected_before_extraction(local_bundle, tmp_path):
    cfg, path, bundle = local_bundle
    plan = checked(cfg)
    source = ru._validate_source_plan(cfg, plan)
    with (path.parent / bundle["archive"]).open("ab") as handle:
        handle.write(b"changed while copying")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    with mock.patch.object(ru, "_validate_source_plan", return_value=source), \
            pytest.raises(ValueError, match="changed during staging"):
        ru._stage_source(cfg, candidate, plan, lambda: None)
    assert not (candidate / "source.zip").exists()
    assert not (candidate / "src").exists()


def test_cancel_local_copy_cleans_archive_before_any_source_execution(local_bundle, tmp_path):
    cfg, _, _ = local_bundle
    plan = checked(cfg)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    def cancel():
        raise ru.UpdateCancelled("user canceled")
    with pytest.raises(ru.UpdateCancelled):
        ru._stage_source(cfg, candidate, plan, cancel)
    assert not (candidate / "source.zip").exists()
    assert not (candidate / "src").exists()


@pytest.mark.parametrize("change", ["commit", "absolute-path", "escape", "size", "bad-json", "array"])
def test_bad_bundle_fails_check_but_does_not_hide_installed_inventory(local_bundle, change):
    cfg, path, bundle = local_bundle
    if change == "commit":
        bundle["source_commit"] = "c" * 40
    elif change == "absolute-path":
        bundle["archive"] = str(path.parent / bundle["archive"])
    elif change == "escape":
        bundle["manifest"] = "../outside.json"
    elif change == "size":
        bundle["archive_size"] += 1
    path.write_text("{broken" if change == "bad-json" else "[]" if change == "array" else json.dumps(bundle))
    inv = ru.inventory(cfg)
    assert inv["installed"]
    assert inv["error"] is None
    assert inv["source"]["kind"] == "local_test"
    assert inv["source"]["error"]
    result = ru.check_updates(cfg)
    assert not result["can_update"]
    assert result["reason"]


def test_source_change_requires_server_component(local_bundle):
    cfg, _, _ = local_bundle
    result = ru.check_updates(cfg, components=["libraries", "models"])
    assert not result["can_update"]
    assert "include Server" in result["reason"]


def test_local_transport_still_rejects_archive_contract_mismatch(local_bundle, tmp_path):
    cfg, path, bundle = local_bundle
    archive = path.parent / bundle["archive"]
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.comment = bundle["source_commit"].encode()
        for name in ds.RUNTIME_SOURCE_FILES:
            zipped.writestr("server/" + name, "{}" if name == ru.MANIFEST_NAME else "# source")
    bundle.update(archive_size=archive.stat().st_size, archive_sha256=ru._hash_file(archive))
    path.write_text(json.dumps(bundle))
    plan = checked(cfg)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    with pytest.raises(ValueError, match="does not match"):
        ru._stage_source(cfg, candidate, plan, lambda: None)
    assert not (candidate / "source.zip").exists()


def test_official_checked_plan_cannot_silently_switch_to_local_transport(local_bundle, monkeypatch):
    cfg, path, _ = local_bundle
    monkeypatch.delenv("FEEDBACK_STEM_TEST_SOURCE")
    monkeypatch.delenv("FEEDBACK_STEM_TEST_CONFIG_DIR")
    with mock.patch.object(ds, "_resolve_commit", return_value="b" * 40), \
            mock.patch.object(ru, "_load_manifest", return_value=manifest()), \
            mock.patch.object(ru, "_candidate_versions", return_value={}):
        plan = ru.check_updates(cfg)
    monkeypatch.setenv("FEEDBACK_STEM_TEST_SOURCE", str(path))
    monkeypatch.setenv("FEEDBACK_STEM_TEST_CONFIG_DIR", str(cfg))
    with pytest.raises(ValueError, match="source changed"):
        ru._get_plan(cfg, plan["plan_id"])
