"""Portable FFmpeg catalog, staging and generation-containment contracts."""
import hashlib
import io
import json
import os
import stat
import sys
import tarfile
import types
import zipfile
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import demucs_server as ds
import runtime_update as ru


MODEL = b"model"
CONFIG = b"config"
FFMPEG = b"contained ffmpeg"
FFPROBE = b"contained ffprobe"


def _asset(path, payload):
    return {"path": path, "url": "https://models.example.test/" + path,
            "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _manifest(toolset, profile="test-profile"):
    return {"schema_version": 2,
            "profiles": {profile: {"python": ">=3.10,<3.14", "platforms": ["win32", "linux", "darwin"],
                "backends": ["cpu", "cuda"], "requirements": ["torch>=2.8,<2.9"], "no_deps": []}},
            "media_tools": {toolset["set_id"]: {key: value for key, value in toolset.items() if key != "set_id"}},
            "models": {"bs_roformer_sw": {"engine": "audio-separator", "revision": "model-v1",
                "entrypoint": "model.ckpt", "stems": ["vocals", "other"], "profiles": [profile],
                "assets": [_asset("model.ckpt", MODEL), _asset("model.yaml", CONFIG)]}}}


def _toolset(target=None, architecture=None, archive=None):
    target = target or sys.platform
    architecture = architecture or ru._normalized_architecture()
    suffix = ".exe" if target == "win32" else ""
    archive = archive or b"verified archive"
    return {"set_id": "portable-media-v1", "revision": "ffmpeg-test-v1", "platform": target,
            "architecture": architecture, "profiles": ["test-profile"], "license": "GPL-3.0-or-later",
            "source_url": "https://media.example.test/source", "archives": [{
                "url": "https://media.example.test/tools.zip", "size": len(archive),
                "sha256": hashlib.sha256(archive).hexdigest(), "format": "zip",
                "members": {"ffmpeg" + suffix: "ffmpeg" + suffix,
                            "ffprobe" + suffix: "ffprobe" + suffix}}]}


def _zip(entries):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries:
            archive.writestr(name, payload)
    return output.getvalue()


def _tar_xz(entries):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:xz") as archive:
        for name, payload in entries:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def _stage(tmp_path, toolset, archive_bytes):
    cfg = tmp_path / "profile"
    root = ru._generation(cfg, "g-test")
    root.mkdir(parents=True)
    plan = {"media_tool_spec": toolset}

    def download(_url, target, _cancel, **_kwargs):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive_bytes)

    with mock.patch.object(ru, "_download_file", side_effect=download):
        receipt = ru._stage_media_tools(cfg, root, plan, lambda: None, None)
    return cfg, root, receipt


def _managed_generation(tmp_path):
    target = sys.platform
    names = ru._canonical_media_names(target)
    archive = b"verified archive"
    toolset = _toolset(target, archive=archive)
    manifest = _manifest(toolset)
    cfg = tmp_path / "profile"
    root = ru._generation(cfg, "g-test")
    (root / "src").mkdir(parents=True)
    (root / "src" / ru.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    tool_dir = root / "tools" / "bin"
    tool_dir.mkdir(parents=True)
    files = []
    for name, payload in zip(names, (FFMPEG, FFPROBE)):
        path = tool_dir / name
        path.write_bytes(payload)
        path.chmod(0o755)
        files.append({"path": name, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    receipt = {"schema_version": 2, "generation_id": root.name, "profile_id": "test-profile",
               "validated": True, "media_tools": {"set_id": toolset["set_id"],
                   "revision": toolset["revision"], "platform": target,
                   "architecture": toolset["architecture"], "asset_dir": "tools/bin",
                   "archives": [{"sha256": toolset["archives"][0]["sha256"],
                                 "size": toolset["archives"][0]["size"]}], "files": files}}
    (root / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return cfg, root, tool_dir, names


@pytest.mark.parametrize("machine,expected", [("AMD64", "x86_64"), ("x86_64", "x86_64"),
                                                ("ARM64", "arm64"), ("aarch64", "arm64")])
def test_architecture_aliases_are_canonical(machine, expected):
    assert ru._normalized_architecture(machine) == expected


def test_manifest_requires_schema_v2_and_exact_canonical_pair():
    toolset = _toolset("win32", "x86_64")
    manifest = _manifest(toolset)
    ru._validate_manifest(manifest)
    manifest["schema_version"] = 1
    with pytest.raises(ValueError, match="manifest version"):
        ru._validate_manifest(manifest)
    manifest["schema_version"] = 2
    del manifest["media_tools"][toolset["set_id"]]["archives"][0]["members"]["ffprobe.exe"]
    with pytest.raises(ValueError, match="ffmpeg and ffprobe"):
        ru._validate_manifest(manifest)


def test_selection_rejects_missing_and_ambiguous_target():
    selected = _toolset("win32", "x86_64")
    manifest = _manifest(selected)
    assert ru._select_media_tools(manifest, "test-profile", target_platform="win32",
                                  architecture="AMD64")["set_id"] == selected["set_id"]
    with pytest.raises(ValueError, match="No verified"):
        ru._select_media_tools(manifest, "test-profile", target_platform="darwin", architecture="arm64")
    manifest["media_tools"]["portable-media-copy"] = dict(manifest["media_tools"][selected["set_id"]])
    with pytest.raises(ValueError, match="Ambiguous"):
        ru._select_media_tools(manifest, "test-profile", target_platform="win32", architecture="x86_64")


def test_verified_zip_extracts_only_named_members_to_canonical_paths(tmp_path):
    payload = _zip([("bundle/bin/ffmpeg.exe", FFMPEG), ("bundle/bin/ffprobe.exe", FFPROBE),
                    ("bundle/README.txt", b"ignored")])
    toolset = _toolset("win32", "x86_64", payload)
    cfg, root, receipt = _stage(tmp_path, toolset, payload)
    assert sorted(path.name for path in (root / "tools" / "bin").iterdir()) == ["ffmpeg.exe", "ffprobe.exe"]
    assert (root / "tools" / "bin" / "ffmpeg.exe").read_bytes() == FFMPEG
    assert receipt["archives"] == [{"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}]
    assert {row["path"] for row in receipt["files"]} == {"ffmpeg.exe", "ffprobe.exe"}
    assert (ru._base(cfg) / "media-assets" / hashlib.sha256(payload).hexdigest()).is_file()


def test_tar_xz_pair_is_supported_and_marked_executable(tmp_path):
    payload = _tar_xz([("bundle/ffmpeg", FFMPEG), ("bundle/ffprobe", FFPROBE)])
    toolset = _toolset("linux", "x86_64", payload)
    toolset["archives"][0].update(format="tar.xz", members={"ffmpeg": "ffmpeg", "ffprobe": "ffprobe"})
    _, root, receipt = _stage(tmp_path, toolset, payload)
    assert {row["path"] for row in receipt["files"]} == {"ffmpeg", "ffprobe"}
    if os.name != "nt":
        assert all(os.access(root / "tools" / "bin" / name, os.X_OK) for name in ("ffmpeg", "ffprobe"))


@pytest.mark.parametrize("entries,error", [
    ([("../ffmpeg.exe", FFMPEG), ("ffprobe.exe", FFPROBE)], "unsafe path"),
    ([("a/ffmpeg.exe", FFMPEG), ("b/ffmpeg.exe", FFMPEG), ("ffprobe.exe", FFPROBE)], "ambiguous"),
])
def test_archive_traversal_and_duplicate_member_are_rejected(tmp_path, entries, error):
    payload = _zip(entries)
    toolset = _toolset("win32", "x86_64", payload)
    with pytest.raises(ValueError, match=error):
        _stage(tmp_path, toolset, payload)
    assert not (tmp_path / "evil.exe").exists()


def test_zip_symlink_cannot_supply_a_required_executable(tmp_path):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        link = zipfile.ZipInfo("bundle/ffmpeg.exe")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, "elsewhere/ffmpeg.exe")
        archive.writestr("bundle/ffprobe.exe", FFPROBE)
    payload = output.getvalue()
    with pytest.raises(ValueError, match="unsafe"):
        _stage(tmp_path, _toolset("win32", "x86_64", payload), payload)


@pytest.mark.parametrize("member_type", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_tar_links_cannot_supply_a_required_executable(tmp_path, member_type):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:xz") as archive:
        link = tarfile.TarInfo("bundle/ffmpeg")
        link.type = member_type
        link.linkname = "elsewhere/ffmpeg"
        archive.addfile(link)
        probe = tarfile.TarInfo("bundle/ffprobe")
        probe.size = len(FFPROBE)
        archive.addfile(probe, io.BytesIO(FFPROBE))
    payload = output.getvalue()
    toolset = _toolset("linux", "x86_64", payload)
    toolset["archives"][0].update(format="tar.xz", members={"ffmpeg": "ffmpeg", "ffprobe": "ffprobe"})
    with pytest.raises(ValueError, match="unsafe"):
        _stage(tmp_path, toolset, payload)


def test_v2_environment_uses_rehashed_generation_tools_without_external_fallback(tmp_path, monkeypatch):
    cfg, root, tool_dir, names = _managed_generation(tmp_path)
    monkeypatch.setenv("PATH", "external-media-tools")
    resolver = mock.Mock(return_value="external-media-tools/ffmpeg.exe")
    with ds.installation_context(root), mock.patch.dict(sys.modules, {"audio": types.SimpleNamespace(_ffmpeg_cmd=resolver)}):
        env = ds._server_env(cfg)
    assert env["PATH"].split(os.pathsep)[0] == str(tool_dir)
    resolver.assert_not_called()

    damaged = tool_dir / names[0]
    damaged.write_bytes(b"X" * damaged.stat().st_size)
    with ds.installation_context(root), mock.patch.dict(sys.modules, {"audio": types.SimpleNamespace(_ffmpeg_cmd=resolver)}):
        with pytest.raises(ValueError, match="receipt verification"):
            ds._server_env(cfg)
    resolver.assert_not_called()


def test_inventory_reports_presence_not_fresh_hash_verification(tmp_path):
    cfg, root, tool_dir, names = _managed_generation(tmp_path)
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    assert ru._media_tools_inventory(cfg, root, receipt)["state"] == "present"
    damaged = tool_dir / names[0]
    damaged.write_bytes(b"X" * damaged.stat().st_size)
    result = ru._media_tools_inventory(cfg, root, receipt)
    assert result["state"] == "present"
    assert result["verification"] == "rehash_on_use"


def test_v1_receipt_keeps_legacy_bundle_resolver(tmp_path):
    cfg = tmp_path / "profile"
    root = ru._generation(cfg, "g-old")
    root.mkdir(parents=True)
    (root / "receipt.json").write_text('{"schema_version":1}', encoding="utf-8")
    bundled = tmp_path / "app" / "bin" / "ffmpeg.exe"
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(b"ffmpeg")
    resolver = mock.Mock(return_value=str(bundled))
    with ds.installation_context(root), mock.patch.dict(sys.modules, {"audio": types.SimpleNamespace(_ffmpeg_cmd=resolver)}):
        env = ds._server_env(cfg)
    assert env["PATH"].split(os.pathsep)[0] == str(bundled.parent)
    resolver.assert_called_once()


def test_early_validation_checks_bare_names_and_audio_roundtrip(tmp_path):
    root = tmp_path / "candidate"
    tool_dir = root / "tools" / "bin"
    tool_dir.mkdir(parents=True)
    names = ru._canonical_media_names(sys.platform)
    for name in names:
        (tool_dir / name).write_bytes(b"tool")
    calls = []

    def run(args, env, cancel, **kwargs):
        calls.append(list(args))
        if "-show_entries" in args:
            return json.dumps({"streams": [{"codec_name": "flac", "sample_rate": "8000", "channels": 2}]})
        return "ok"

    with mock.patch.object(ru, "_candidate_env", return_value={"PATH": str(tool_dir)}), \
            mock.patch.object(ru, "_verified_media_tools", return_value=(tool_dir, {})), \
            mock.patch.object(ru, "_run_process", side_effect=run), \
            mock.patch.object(ru, "_run_external_process", side_effect=run):
        ru._validate_media_tools(tmp_path, root, lambda: None)
    assert len(calls) == 3
    assert "subprocess.run" in calls[0][-1]
    assert names[0] in calls[0][-1] and names[1] in calls[0][-1]
    assert calls[1][0] == str(tool_dir / names[0])
    assert calls[2][0] == str(tool_dir / names[1])


def test_external_validation_runs_beneath_the_guarded_worker(tmp_path):
    output = ru._run_external_process(
        [sys.executable, "-c", "print('MEDIA_TOOL_CHILD_OK')"],
        dict(os.environ), lambda: None, timeout=30, cwd=tmp_path,
    )
    assert "MEDIA_TOOL_CHILD_OK" in output
