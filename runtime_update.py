"""Transactional updates for the plugin-owned local separation server.

Checking never imports the ML runtime or installs packages. Apply builds an
immutable generation; activation changes a small pointer only after validation.
The legacy layout remains a valid rollback target. This module never manages a
custom remote server, the in-process engine, or a song library.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import re
import secrets
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
import wave
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Callable
from urllib.parse import quote, urlparse

# Keep the on-disk operation/activation protocol stable while independently
# versioning the published server contract and immutable generation receipt.
# Existing v1 pointers and receipts remain readable; newly prepared runtimes
# consume manifest v2 and carry receipt v2 media-tool evidence.
SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 2
RECEIPT_SCHEMA_VERSION = 2
MANIFEST_NAME = "runtime-manifest.json"
_ID = re.compile(r"^[a-zA-Z0-9_-]{1,100}$")
_TOOLSET_ID = re.compile(r"^[a-zA-Z0-9_.-]{1,100}$")
_SHA = re.compile(r"^[a-fA-F0-9]{64}$")
_COMMIT = re.compile(r"^[a-fA-F0-9]{40}$")
_LOCK = threading.RLock()
_RUNNING: dict[str, threading.Event] = {}
_TERMINAL = {"idle", "active", "current", "rolled_back", "canceled", "failed", "discarded"}
# Readers and antivirus can briefly hold Windows handles without delete sharing.
# Bound the wait to 1.585 seconds; never remove the destination to work around it.
_REPLACE_RETRY_DELAYS = (.01, .025, .05, .1, .2, .4, .8)
_WINDOWS_REPLACE_ERRORS = {5, 32, 33}  # access denied, sharing violation, lock violation
_MEDIA_TOOL_NAMES = ("ffmpeg", "ffprobe")
_MEDIA_PLATFORMS = {"win32", "linux", "darwin"}
_MEDIA_ARCHES = {"x86_64", "arm64"}
_MEDIA_ARCHIVE_FORMATS = {"zip", "tar.xz"}
_MAX_MEDIA_TOOL_BYTES = 1024 ** 3


class UpdateCancelled(RuntimeError):
    """A user canceled candidate preparation, before pointer activation."""


def _base(config_dir: Path) -> Path:
    return Path(config_dir).resolve() / "demucs-server"


def owned_path(config_dir: Path, path: str | Path) -> Path:
    """Reject paths outside this managed installation, including link escapes."""
    base = _base(config_dir)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(base.resolve()):
        raise ValueError("Runtime path escapes the managed installation")
    return candidate


def _read(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {} if default is None else default


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(len(_REPLACE_RETRY_DELAYS) + 1):
            try:
                os.replace(temp, path)
                break
            except OSError as exc:
                if (os.name != "nt" or getattr(exc, "winerror", None) not in _WINDOWS_REPLACE_ERRORS
                        or attempt == len(_REPLACE_RETRY_DELAYS)):
                    raise
                time.sleep(_REPLACE_RETRY_DELAYS[attempt])
    finally:
        # A cleanup sharing violation must not hide the original write/replace
        # failure. A locked leftover is harmless; it is never an active pointer.
        with contextlib.suppress(OSError):
            temp.unlink(missing_ok=True)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _hash_file(path: Path, cancel: Callable[[], None] | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            if cancel:
                cancel()
            digest.update(block)
    return digest.hexdigest()


def _generation(config_dir: Path, generation_id: str) -> Path:
    if generation_id == "legacy":
        return _base(config_dir)
    if not _ID.fullmatch(str(generation_id)):
        raise ValueError("Invalid runtime generation ID")
    return owned_path(config_dir, Path("generations") / generation_id)


def active_root(config_dir: Path) -> Path:
    path = _base(config_dir) / "active.json"
    if not path.exists():
        return _base(config_dir)
    pointer = _read(path)
    if pointer.get("schema_version") != SCHEMA_VERSION or not pointer.get("generation_id"):
        raise RuntimeError("The active runtime pointer is damaged; use runtime recovery")
    root = _generation(config_dir, pointer["generation_id"])
    if pointer["generation_id"] != "legacy":
        receipt = _read(root / "receipt.json")
        if not receipt.get("validated") or receipt.get("generation_id") != pointer["generation_id"]:
            raise RuntimeError("The active generation is incomplete; use runtime recovery")
    return root


def _active_id(config_dir: Path) -> str:
    return _read(_base(config_dir) / "active.json").get("generation_id") or "legacy"


def _receipt(config_dir: Path, generation_id: str | None = None) -> dict:
    root = _generation(config_dir, generation_id) if generation_id else active_root(config_dir)
    return _read(root / "receipt.json")


def management_token(config_dir: Path) -> str:
    """Create only during an explicit install/start, never during status checks."""
    path = _base(config_dir) / "management-token"
    with _LOCK:
        if path.exists():
            value = path.read_text(encoding="utf-8").strip()
            if len(value) < 32:
                raise RuntimeError("Managed-server token is invalid")
            return value
        path.parent.mkdir(parents=True, exist_ok=True)
        value = secrets.token_urlsafe(48)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
        return value


def _package_versions(target: Path) -> dict[str, str]:
    """Read metadata without loading Torch or locking a Windows extension."""
    versions = {}
    if not target.is_dir():
        return versions
    for metadata in target.glob("*.dist-info/METADATA"):
        try:
            info = Parser().parsestr(metadata.read_text(encoding="utf-8"), headersonly=True)
            name, version = info.get("Name"), info.get("Version")
            if name and version:
                versions[re.sub(r"[-_.]+", "-", name).lower()] = version
        except OSError:
            continue
    return dict(sorted(versions.items()))


def _lock_file(handle, unlock: bool = False) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK if unlock else msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN if unlock else fcntl.LOCK_EX | fcntl.LOCK_NB)


def _external_operation_running(config_dir: Path) -> bool:
    """A second host process must not mistake a live update for a crash."""
    try:
        handle = (_base(config_dir) / "update.lock").open("r+b")
    except FileNotFoundError:
        return False
    with handle:
        try:
            _lock_file(handle)
        except OSError:
            return True
        _lock_file(handle, unlock=True)
        return False


def _acquire_operation_lock(config_dir: Path):
    path = _base(config_dir) / "update.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    if path.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
    try:
        _lock_file(handle)
    except OSError as exc:
        handle.close()
        raise RuntimeError("Another application process is updating this runtime") from exc
    return handle


def inventory(config_dir: Path) -> dict:
    """Local inventory. Receipt versions are authoritative for a managed generation."""
    import demucs_server as ds
    try:
        root = active_root(config_dir)
        receipt = _read(root / "receipt.json")
        versions = receipt.get("dependencies") or _package_versions(root / "pylibs")
        source = _read(root / "source.json")
        generation_id = _active_id(config_dir)
        present = model_presence(config_dir)
        media_tools = _media_tools_inventory(config_dir, root, receipt)
        models = {key: {"revision": value.get("revision"), "state": "verified" if present.get(key) else "damaged",
                        "engine": value.get("engine"), "stems": value.get("stems", [])}
                  for key, value in receipt.get("models", {}).items()}
        if generation_id == "legacy":
            # Legacy caches have no trusted revision receipt. Include recognized
            # installed stem models in planning, then hash them against catalog
            # assets during staging before permitting reuse.
            cache = _base(config_dir) / "cache"
            legacy_models = {
                "bs_roformer_sw": ("audio-separator", [cache / "_roformer-models" / "BS-Roformer-SW.ckpt"]),
                "htdemucs_6s": ("demucs", [cache / "torch" / "hub" / "checkpoints" / "5c90dfd2-34c22ccb.th",
                                           cache / "hub" / "checkpoints" / "5c90dfd2-34c22ccb.th"]),
            }
            for name, (engine, paths) in legacy_models.items():
                if any(p.is_file() for p in paths):
                    models[name] = {"revision": None, "state": "verification_pending", "engine": engine, "stems": []}
        return {"installed": (root / "src" / "server.py").is_file(),
                "generation_id": generation_id, "legacy": generation_id == "legacy",
                "source_commit": receipt.get("source_commit") or source.get("commit"),
                "ref": source.get("ref"), "profile_id": receipt.get("profile_id"),
                "dependencies": versions, "audio_separator": versions.get("audio-separator"),
                "models": models, "model_verification": "verified" if models and generation_id != "legacy" else "verification_pending",
                "media_tools": media_tools,
                "python": platform.python_version(), "platform": sys.platform,
                "install_info": _read(root / "install.json"), "error": None}
    except Exception as exc:
        return {"installed": False, "state": "damaged", "error": str(exc), "dependencies": {},
                "models": {}, "media_tools": {}}


def model_presence(config_dir: Path) -> dict[str, bool]:
    """Cheap receipt-aware presence check. Full hashing occurs before promotion."""
    result = {}
    for name, spec in _receipt(config_dir).get("models", {}).items():
        try:
            directory = owned_path(config_dir, spec["asset_dir"])
            result[name] = bool(spec.get("assets")) and all(
                (directory / _safe_relative(asset["path"])).is_file()
                and (directory / _safe_relative(asset["path"])).stat().st_size == asset["size"]
                for asset in spec["assets"])
        except (OSError, ValueError, KeyError):
            result[name] = False
    return result


def _is_link_like(path: Path) -> bool:
    try:
        return path.is_symlink() or bool(getattr(os.path, "isjunction", lambda _path: False)(path))
    except OSError:
        return True


def _verified_media_tools(config_dir: Path, root: Path, *, verify_hashes: bool = True,
                          cancel: Callable[[], None] | None = None) -> tuple[Path, dict]:
    """Validate a receipt-v2 tool pair without consulting the host PATH.

    The published archive digest establishes provenance. Per-file hashes in the
    local receipt then detect damage after extraction and across future starts.
    """
    root = Path(root).resolve()
    base = _base(config_dir).resolve()
    if root != base and not root.is_relative_to(base):
        raise ValueError("Runtime root escapes the managed installation")
    receipt = _read(root / "receipt.json")
    if type(receipt.get("schema_version")) is not int or receipt["schema_version"] != RECEIPT_SCHEMA_VERSION:
        raise ValueError("The managed runtime does not have a v2 media-tool receipt")
    recorded = receipt.get("media_tools")
    if not isinstance(recorded, dict):
        raise ValueError("The managed runtime has no verified media-tool receipt")

    manifest = _read(root / "src" / MANIFEST_NAME)
    _validate_manifest(manifest)
    selected = _select_media_tools(manifest, receipt.get("profile_id", ""))
    for receipt_key, catalog_key in (("set_id", "set_id"), ("revision", "revision"),
                                     ("platform", "platform"), ("architecture", "architecture")):
        if recorded.get(receipt_key) != selected.get(catalog_key):
            raise ValueError("The media-tool receipt does not match the runtime catalog")
    expected_archives = sorted(
        ({"sha256": archive["sha256"].lower(), "size": archive["size"]}
         for archive in selected["archives"]), key=lambda row: row["sha256"])
    actual_archives = recorded.get("archives")
    if not isinstance(actual_archives, list):
        raise ValueError("The media-tool archive receipt is missing")
    normalized_archives = []
    for archive in actual_archives:
        if (not isinstance(archive, dict) or set(archive) != {"sha256", "size"}
                or not _SHA.fullmatch(str(archive.get("sha256", "")))
                or type(archive.get("size")) is not int or archive["size"] < 1):
            raise ValueError("The media-tool archive receipt is invalid")
        normalized_archives.append({"sha256": archive["sha256"].lower(), "size": archive["size"]})
    if sorted(normalized_archives, key=lambda row: row["sha256"]) != expected_archives:
        raise ValueError("The media-tool archive receipt does not match the runtime catalog")

    if recorded.get("asset_dir") != "tools/bin":
        raise ValueError("The media-tool directory is not generation-owned")
    relative = _safe_relative(recorded["asset_dir"])
    unresolved = root / relative
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if _is_link_like(cursor):
            raise ValueError("The media-tool directory cannot contain links")
    tool_dir = unresolved.resolve()
    if tool_dir != (root / "tools" / "bin").resolve() or not tool_dir.is_relative_to(root):
        raise ValueError("The media-tool directory escapes its generation")

    files = recorded.get("files")
    expected_names = set(_canonical_media_names(recorded.get("platform")))
    if not isinstance(files, list) or len(files) != len(expected_names):
        raise ValueError("The media-tool file receipt is incomplete")
    seen: set[str] = set()
    normalized_files = []
    for entry in files:
        if (not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}
                or type(entry.get("size")) is not int or not 1 <= entry["size"] <= _MAX_MEDIA_TOOL_BYTES
                or not _SHA.fullmatch(str(entry.get("sha256", "")))):
            raise ValueError("The media-tool file receipt is invalid")
        name = _safe_basename(entry.get("path"), "media-tool output")
        if name not in expected_names or name in seen:
            raise ValueError("The media-tool file receipt is not the canonical pair")
        seen.add(name)
        path = tool_dir / name
        if _is_link_like(path) or not path.is_file() or path.stat().st_size != entry["size"]:
            raise ValueError(f"Managed media tool is missing or damaged: {name}")
        if os.name != "nt" and not os.access(path, os.X_OK):
            raise ValueError(f"Managed media tool is not executable: {name}")
        if verify_hashes and _hash_file(path, cancel).lower() != entry["sha256"].lower():
            raise ValueError(f"Managed media tool failed receipt verification: {name}")
        normalized_files.append({"path": name, "size": entry["size"], "sha256": entry["sha256"].lower()})
    if seen != expected_names:
        raise ValueError("The media-tool file receipt is not the canonical pair")
    return tool_dir, {**recorded, "archives": normalized_archives,
                      "files": sorted(normalized_files, key=lambda row: row["path"])}


def verified_media_tools_dir(config_dir: Path, root: Path | None = None) -> Path:
    """Return a fully rehashed generation tool directory or fail closed."""
    directory, _ = _verified_media_tools(config_dir, root or active_root(config_dir))
    return directory


def _media_tools_inventory(config_dir: Path, root: Path, receipt: dict) -> dict:
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        return {"state": "legacy_unverified"}
    recorded = receipt.get("media_tools") if isinstance(receipt.get("media_tools"), dict) else {}
    public = {key: recorded.get(key) for key in ("set_id", "revision", "platform", "architecture")}
    try:
        _, verified = _verified_media_tools(config_dir, root, verify_hashes=False)
        # Inventory is used by the settings poll endpoint, so keep it cheap and
        # call this what it is: catalog/receipt/size presence. Every child start,
        # candidate validation and rollback performs the full per-file rehash.
        public.update(state="present", verification="rehash_on_use",
                      files=[{"path": row["path"], "size": row["size"]}
                             for row in verified["files"]])
    except Exception as exc:
        public.update(state="damaged", error=str(exc), files=[])
    return public


def status(config_dir: Path) -> dict:
    """Recoverable transaction snapshot; no network calls or implicit recovery."""
    key = str(_base(config_dir))
    data = _read(_base(config_dir) / "update-operation.json")
    with _LOCK:
        running = key in _RUNNING
    running = running or _external_operation_running(config_dir)
    interrupted = bool(data.get("active") and not running)
    pointer = _read(_base(config_dir) / "active.json")
    data = {"state": "idle", "active": False, "phase": "Idle", "pct": 0,
            "cancel_requested": False, "error": None, **data}
    data.update(active=running, needs_recovery=interrupted or data.get("state") == "recovery_required",
                pending_activation=data.get("state") == "waiting_to_activate",
                rollback_available=bool(pointer.get("previous_generation")))
    if interrupted:
        data["phase"] = "Interrupted — recovery required"
    plan_id = data.get("plan_id")
    if plan_id and _ID.fullmatch(str(plan_id)) and data.get("generation_id"):
        checked = _read(_base(config_dir) / "plans" / (plan_id + ".json"))
        if checked.get("plan_id") == plan_id:
            # Enough persisted, public metadata to review a prepared candidate
            # after UI reload. No manifest, asset paths, credentials or ability
            # to reapply the old transaction is exposed through this snapshot.
            data["checked_plan"] = {key: checked.get(key) for key in
                ("plan_id", "model", "available", "dependency_graph", "server_state", "libraries_state")}
            data["checked_plan"]["can_update"] = False
    return data


def _persist(config_dir: Path, **changes) -> dict:
    with _LOCK:
        path = _base(config_dir) / "update-operation.json"
        current = _read(path)
        current.update(changes, updated_at=time.time())
        _atomic_json(path, current)
        return current


def _event(config_dir: Path, callback, state: str, pct: float, message: str, **changes):
    line = changes.pop("line", message)
    record = _persist(config_dir, state=state, phase=message, pct=pct, line=line, **changes)
    if callback:
        callback({"pct": pct, "phase": message, "line": line, "state": state})
    return record


def _json_get(url: str) -> dict:
    import requests
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object from update metadata")
    return value


def _load_manifest(commit: str) -> dict:
    import demucs_server as ds
    return _json_get(f"https://raw.githubusercontent.com/{ds.SOURCE_REPO}/{commit}/{MANIFEST_NAME}")


def _safe_relative(value: str) -> Path:
    posix = PurePosixPath(str(value))
    if not value or posix.is_absolute() or ".." in posix.parts or "\\" in value or ":" in value:
        raise ValueError("Invalid runtime asset path")
    return Path(*posix.parts)


def _normalized_architecture(value: str | None = None) -> str:
    raw = str(value if value is not None else platform.machine()).strip().lower()
    aliases = {
        "amd64": "x86_64", "x64": "x86_64", "x86-64": "x86_64", "x86_64": "x86_64",
        "arm64": "arm64", "aarch64": "arm64",
    }
    try:
        return aliases[raw]
    except KeyError as exc:
        raise ValueError(f"Unsupported runtime architecture: {raw or 'unknown'}") from exc


def _canonical_media_names(target_platform: str) -> tuple[str, str]:
    if target_platform not in _MEDIA_PLATFORMS:
        raise ValueError(f"Unsupported media-tool platform: {target_platform}")
    suffix = ".exe" if target_platform == "win32" else ""
    return tuple(name + suffix for name in _MEDIA_TOOL_NAMES)


def _safe_basename(value, label: str) -> str:
    text = str(value)
    if (not text or text in {".", ".."} or "/" in text or "\\" in text
            or ":" in text or "\x00" in text or Path(text).name != text):
        raise ValueError(f"Invalid {label} basename")
    return text


def _verified_https_url(value, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} requires HTTPS")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError(f"{label} requires HTTPS")
    return value


def _validate_media_tool_catalog(manifest: dict) -> None:
    catalog = manifest.get("media_tools")
    if not isinstance(catalog, dict) or not catalog:
        raise ValueError("Server did not publish a media-tool catalog")
    profiles = manifest.get("profiles", {})
    claimed_targets: set[tuple[str, str, str]] = set()
    for set_id, toolset in catalog.items():
        if not isinstance(set_id, str) or not _TOOLSET_ID.fullmatch(set_id) or not isinstance(toolset, dict):
            raise ValueError("Invalid media-tool set identity")
        target_platform = toolset.get("platform")
        architecture = toolset.get("architecture")
        if target_platform not in _MEDIA_PLATFORMS or architecture not in _MEDIA_ARCHES:
            raise ValueError(f"Media-tool set {set_id} has an unsupported target")
        for field in ("revision", "license", "source_url"):
            value = toolset.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Media-tool set {set_id} requires {field}")
        _verified_https_url(toolset["source_url"], "Media-tool source metadata")
        supported_profiles = toolset.get("profiles")
        if (not isinstance(supported_profiles, list) or not supported_profiles
                or any(not isinstance(profile_id, str) for profile_id in supported_profiles)
                or len(supported_profiles) != len(set(supported_profiles))
                or any(profile_id not in profiles for profile_id in supported_profiles)):
            raise ValueError(f"Media-tool set {set_id} references an invalid profile")
        for profile_id in supported_profiles:
            target = (profile_id, target_platform, architecture)
            if target in claimed_targets:
                raise ValueError(f"Duplicate media-tool target: {profile_id}/{target_platform}/{architecture}")
            claimed_targets.add(target)
        archives = toolset.get("archives")
        if not isinstance(archives, list) or not archives:
            raise ValueError(f"Media-tool set {set_id} has no verified archive")
        expected = set(_canonical_media_names(target_platform))
        outputs: set[str] = set()
        source_names: set[str] = set()
        archive_digests: set[str] = set()
        for archive in archives:
            if not isinstance(archive, dict):
                raise ValueError("Invalid media-tool archive entry")
            url, digest = archive.get("url"), str(archive.get("sha256", ""))
            size, archive_format = archive.get("size"), archive.get("format")
            _verified_https_url(url, "Media-tool download")
            if not _SHA.fullmatch(digest) or digest.lower() in archive_digests:
                raise ValueError("Media-tool archives require distinct full SHA-256 digests")
            if type(size) is not int or size < 1:
                raise ValueError("Media-tool archives require byte sizes")
            if archive_format not in _MEDIA_ARCHIVE_FORMATS:
                raise ValueError("Unsupported media-tool archive format")
            members = archive.get("members")
            if not isinstance(members, dict) or not members:
                raise ValueError("Media-tool archives require an explicit member map")
            archive_digests.add(digest.lower())
            for output, source in members.items():
                if not isinstance(output, str) or not isinstance(source, str):
                    raise ValueError("Media-tool member mappings require string basenames")
                output = _safe_basename(output, "media-tool output")
                source = _safe_basename(source, "media-tool source")
                if output not in expected or output in outputs or source in source_names:
                    raise ValueError("Media-tool member mappings must be canonical and unique")
                outputs.add(output)
                source_names.add(source)
        if outputs != expected:
            raise ValueError(f"Media-tool set {set_id} must provide ffmpeg and ffprobe")


def _validate_manifest(manifest: dict):
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported server runtime manifest version")
    if not isinstance(manifest.get("profiles"), dict) or not manifest["profiles"]:
        raise ValueError("Server did not publish a compatibility profile")
    if not isinstance(manifest.get("models"), dict):
        raise ValueError("Server did not publish a model catalog")
    for name, model in manifest["models"].items():
        if not _ID.fullmatch(name) or not model.get("revision"):
            raise ValueError("Invalid model identity in catalog")
        if model.get("engine") not in {"demucs", "audio-separator", "roformer"}:
            raise ValueError("Unsupported model backend")
        _safe_relative(model.get("entrypoint", ""))
        seen = set()
        for asset in model.get("assets", []):
            path = str(_safe_relative(asset.get("path", "")))
            if path in seen or not _SHA.fullmatch(str(asset.get("sha256", ""))):
                raise ValueError("Model assets require distinct paths and full SHA-256 digests")
            seen.add(path)
            if not isinstance(asset.get("size"), int) or asset["size"] < 1:
                raise ValueError("Model assets require byte sizes")
            if not str(asset.get("url", "")).startswith("https://"):
                raise ValueError("Model downloads require HTTPS")
        entry = model["entrypoint"] + (".yaml" if model["engine"] == "demucs" and not model["entrypoint"].endswith(".yaml") else "")
        if not seen or str(_safe_relative(entry)) not in seen:
            raise ValueError("The model entrypoint must be a verified catalog asset")
    _validate_media_tool_catalog(manifest)


def _select_media_tools(manifest: dict, profile_id: str, *, target_platform: str | None = None,
                        architecture: str | None = None) -> dict:
    target_platform = target_platform or sys.platform
    architecture = _normalized_architecture(architecture)
    matches = []
    for set_id, toolset in manifest["media_tools"].items():
        if (toolset["platform"] == target_platform and toolset["architecture"] == architecture
                and profile_id in toolset["profiles"]):
            matches.append({"set_id": set_id, **toolset})
    if not matches:
        raise ValueError(f"No verified FFmpeg tools for {target_platform}/{architecture} and profile {profile_id}")
    if len(matches) != 1:
        raise ValueError(f"Ambiguous FFmpeg tools for {target_platform}/{architecture} and profile {profile_id}")
    return matches[0]


def _select_profile(manifest: dict, gpu: bool, cuda_tag: str | None) -> tuple[str, dict]:
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
    backend = "cuda" if gpu else "cpu"
    for identity, profile in manifest["profiles"].items():
        if sys.platform not in profile.get("platforms", []) or backend not in profile.get("backends", []):
            continue
        if gpu and sys.platform not in (profile.get("cuda_platforms") or profile.get("cuda_torch", {}).get("platforms", ["win32", "linux"])):
            continue
        if Version(platform.python_version()) not in SpecifierSet(profile.get("python", "")):
            continue
        tags = profile.get("cuda_tags") or profile.get("cuda_torch", {}).get("tags")
        if gpu and tags and cuda_tag not in tags:
            continue
        return identity, profile
    raise ValueError(f"No supported runtime profile for Python {platform.python_version()} on {sys.platform} ({backend})")


def _candidate_versions(profile: dict, gpu: bool, cuda_tag: str) -> dict:
    """Lightweight direct-release discovery, deliberately not a pip dry-run."""
    from packaging.requirements import Requirement
    from packaging.version import InvalidVersion, Version
    from packaging.specifiers import SpecifierSet
    available = {}
    for text in profile.get("requirements", []) + profile.get("no_deps", []):
        req = Requirement(text)
        if req.marker and not req.marker.evaluate():
            continue
        name = re.sub(r"[-_.]+", "-", req.name).lower()
        data = _json_get(f"https://pypi.org/pypi/{quote(req.name)}/json")
        versions = []
        all_stable = []
        for raw, files in data.get("releases", {}).items():
            try:
                version = Version(raw)
            except InvalidVersion:
                continue
            if version.is_prerelease or version.is_devrelease or not files or all(f.get("yanked") for f in files):
                continue
            all_stable.append(version)
            python_ok = any(not f.get("yanked") and (
                not f.get("requires_python") or Version(platform.python_version()) in SpecifierSet(f["requires_python"])) for f in files)
            if version in req.specifier and python_ok:
                versions.append(version)
        if not versions:
            raise ValueError(f"No stable release satisfies {text}")
        chosen = str(max(versions))
        available[name] = {"version": chosen, "requirement": text,
                           "latest": str(max(all_stable)), "newer_unsupported": max(all_stable) > max(versions)}
    return available


def check_updates(config_dir: Path, ref: str | None = None, model: str = "bs_roformer_sw",
                  gpu: bool | None = None, cuda_tag: str | None = None,
                  components: list[str] | None = None) -> dict:
    """Check published contracts. Network errors are unknown, never up-to-date."""
    import demucs_server as ds
    installed = inventory(config_dir)
    selected = list(dict.fromkeys(components or ["server", "libraries", "models"]))
    if not selected or set(selected) - {"server", "libraries", "models"}:
        raise ValueError("Unknown runtime update component")
    result = {"schema_version": SCHEMA_VERSION, "installed": installed, "components": selected,
              "model": model, "can_update": False, "state": "unknown", "reason": None,
              "dependency_graph": "Resolved and validated during explicit staging; direct version checks do not certify transitive packages."}
    try:
        gpu = bool(installed.get("install_info", {}).get("gpu")) if gpu is None else bool(gpu)
        tag = str(cuda_tag or installed.get("install_info", {}).get("cuda_tag") or ds.DEFAULT_CUDA_TAG).strip()
        ref = ds._norm_ref(ref)
        commit = ds._resolve_commit(ref) if "server" in selected else installed.get("source_commit")
        if not commit or not _COMMIT.fullmatch(commit):
            raise ValueError("Could not resolve an immutable server revision; nothing was installed")
        manifest = _load_manifest(commit)
        _validate_manifest(manifest)
        profile_id, profile = _select_profile(manifest, gpu, tag)
        media_tools = _select_media_tools(manifest, profile_id)
        model_ids = sorted(set(installed.get("models", {})) | {model})
        missing = [name for name in model_ids if name not in manifest["models"]]
        if missing:
            raise ValueError("No verified update catalog for model(s): " + ", ".join(missing) + ". The existing runtime remains usable.")
        for name in model_ids:
            if profile_id not in manifest["models"][name].get("profiles", [profile_id]):
                raise ValueError(f"Model {name} does not support profile {profile_id}")
        releases = _candidate_versions(profile, gpu, tag) if "libraries" in selected else {}
        if "libraries" not in selected:
            _check_installed_requirements(installed["dependencies"], profile)
        models = {name: manifest["models"][name] for name in model_ids}
        if "models" not in selected:
            old = _receipt(config_dir).get("models", {})
            for name in model_ids:
                if name not in old or old[name].get("revision") != models[name]["revision"]:
                    raise ValueError("This combination needs a model update; include Models")
        model_rows = [{"id": name, "installed": installed.get("models", {}).get(name, {}).get("revision"),
                       "available": spec["revision"], "state": "current" if installed.get("models", {}).get(name, {}).get("revision") == spec["revision"] else "available",
                       "download_bytes": sum(asset["size"] for asset in spec["assets"])} for name, spec in models.items()]
        media_download = sum(archive["size"] for archive in media_tools["archives"])
        media_available = {key: media_tools[key] for key in
                           ("set_id", "revision", "platform", "architecture", "license", "source_url")}
        media_available["download_bytes"] = media_download
        plan = {**result, "plan_id": uuid.uuid4().hex, "created_at": time.time(), "state": "available",
                "can_update": True, "ref": ref, "source_commit": commit, "profile_id": profile_id,
                "profile": profile, "manifest": manifest, "manifest_sha256": _digest(manifest),
                "available": {"server": commit, "dependencies": releases, "models": model_rows,
                              "media_tools": media_available},
                "gpu": gpu, "cuda_tag": tag, "model_specs": models, "media_tool_spec": media_tools,
                "server_state": "current" if installed.get("source_commit") == commit else "available",
                "libraries_state": "refresh_available" if "libraries" in selected else "unchanged",
                "base_generation": installed.get("generation_id", "legacy")}
        _atomic_json(_base(config_dir) / "plans" / (plan["plan_id"] + ".json"), plan)
        return plan
    except Exception as exc:
        result.update(reason=str(exc), state="incompatible" if isinstance(exc, ValueError) else "unknown")
        return result


def _check_installed_requirements(versions: dict, profile: dict):
    from packaging.requirements import Requirement
    from packaging.version import Version
    for text in profile.get("requirements", []) + profile.get("no_deps", []):
        req = Requirement(text)
        if req.marker and not req.marker.evaluate():
            continue
        name = re.sub(r"[-_.]+", "-", req.name).lower()
        if name not in versions or Version(versions[name]) not in req.specifier:
            raise ValueError(f"The selected server requires {text}; include Libraries to update its dependencies")


@contextlib.contextmanager
def _operation(config_dir: Path, state: str, *, recovery: bool = False):
    key = str(_base(config_dir))
    with _LOCK:
        if key in _RUNNING:
            raise RuntimeError("Another runtime operation is already running")
        previous = status(config_dir)
        if previous.get("needs_recovery") and not recovery:
            raise RuntimeError("An interrupted update requires recovery or discard before another operation")
        handle = _acquire_operation_lock(config_dir)
        (_base(config_dir) / "update.cancel").unlink(missing_ok=True)
        event = threading.Event()
        _RUNNING[key] = event
        try:
            _persist(config_dir, active=True, state=state, operation_id=uuid.uuid4().hex,
                     cancel_requested=False, error=None, needs_recovery=False)
        except BaseException:
            _RUNNING.pop(key, None)
            _lock_file(handle, unlock=True)
            handle.close()
            raise

    def checkpoint():
        if event.is_set() or (_base(config_dir) / "update.cancel").exists() or _read(_base(config_dir) / "update-operation.json").get("cancel_requested"):
            raise UpdateCancelled("Runtime update canceled; the working installation was preserved")

    try:
        yield checkpoint
    except UpdateCancelled as exc:
        _persist(config_dir, state="canceled", phase="Canceled", error=str(exc))
        raise
    except BaseException as exc:
        current = _read(_base(config_dir) / "update-operation.json")
        if current.get("state") != "recovery_required":
            _persist(config_dir, state="failed", phase="Failed", error=str(exc))
        raise
    finally:
        with _LOCK:
            _RUNNING.pop(key, None)
            try:
                _persist(config_dir, active=False)
            finally:
                _lock_file(handle, unlock=True)
                handle.close()
                import demucs_server as ds
                ds._invalidate_disk_size(config_dir)


def cancel_update(config_dir: Path) -> dict:
    with _LOCK:
        current = status(config_dir)
        if current.get("state") == "activating":
            return {**current, "cancel_accepted": False, "reason": "Activation must finish or roll back"}
        event = _RUNNING.get(str(_base(config_dir)))
        if event:
            event.set()
        if current.get("pending_activation") and not current.get("active"):
            _persist(config_dir, state="canceled", phase="Canceled", cancel_requested=True,
                     error="Prepared update canceled; the working installation was preserved")
            return {**status(config_dir), "cancel_accepted": True}
        if current.get("active") or current.get("pending_activation"):
            # An independent marker cannot be lost to a simultaneous progress
            # journal write from another application process.
            (_base(config_dir) / "update.cancel").touch()
            _persist(config_dir, cancel_requested=True)
            return {**status(config_dir), "cancel_accepted": True}
        return {**current, "cancel_accepted": False}


def _get_plan(config_dir: Path, plan_id: str) -> dict:
    if not _ID.fullmatch(str(plan_id)):
        raise ValueError("Invalid update plan")
    plan = _read(owned_path(config_dir, Path("plans") / (plan_id + ".json")))
    if not plan.get("can_update") or plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Update plan is missing or incompatible; check for updates again")
    if plan.get("base_generation") != _active_id(config_dir):
        raise ValueError("The active runtime changed since this plan; check for updates again")
    _validate_manifest(plan["manifest"])
    if _digest(plan["manifest"]) != plan.get("manifest_sha256"):
        raise ValueError("Update manifest changed after planning")
    selected_tools = _select_media_tools(plan["manifest"], plan.get("profile_id", ""))
    if _digest(selected_tools) != _digest(plan.get("media_tool_spec")):
        raise ValueError("Update media-tool selection changed after planning")
    return plan


def _download_file(url: str, target: Path, cancel, *, size: int | None = None,
                   sha256: str | None = None) -> None:
    import requests
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    digest = hashlib.sha256()
    count = 0
    try:
        with requests.get(url, stream=True, timeout=(20, 30)) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for block in response.iter_content(1024 * 1024):
                    cancel()
                    if not block:
                        continue
                    count += len(block)
                    if size is not None and count > size:
                        raise ValueError("Downloaded runtime asset is larger than its published size")
                    digest.update(block)
                    handle.write(block)
        cancel()
        if size is not None and count != size:
            raise ValueError("Downloaded runtime asset has an incorrect size")
        if sha256 and digest.hexdigest().lower() != sha256.lower():
            raise ValueError("Downloaded runtime asset failed SHA-256 verification")
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def _safe_archive_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Media-tool archive contains an unsafe path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or any(":" in part for part in path.parts):
        raise ValueError("Media-tool archive contains an unsafe path")
    return path


def _write_media_member(source, target: Path, cancel) -> dict:
    partial = target.with_name(target.name + ".partial")
    digest = hashlib.sha256()
    count = 0
    try:
        with partial.open("wb") as output:
            while True:
                cancel()
                block = source.read(1024 * 1024)
                if not block:
                    break
                count += len(block)
                if count > _MAX_MEDIA_TOOL_BYTES:
                    raise ValueError("Extracted media tool exceeds the safety limit")
                digest.update(block)
                output.write(block)
        if count < 1:
            raise ValueError("Extracted media tool is empty")
        os.replace(partial, target)
        if os.name != "nt":
            target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return {"path": target.name, "size": count, "sha256": digest.hexdigest()}
    finally:
        partial.unlink(missing_ok=True)


def _extract_media_archive(archive_path: Path, archive: dict, target_dir: Path, cancel) -> list[dict]:
    """Copy only explicitly mapped regular files to canonical destinations."""
    requested = dict(archive["members"])
    selected = {}
    if archive["format"] == "zip":
        with zipfile.ZipFile(archive_path) as handle:
            infos = handle.infolist()
            if len(infos) > 100000:
                raise ValueError("Media-tool archive contains too many entries")
            for info in infos:
                cancel()
                path = _safe_archive_path(info.filename)
                matches = [output for output, source in requested.items() if path.name == source]
                if not matches:
                    continue
                mode = (info.external_attr >> 16) & 0xFFFF
                if (len(matches) != 1 or matches[0] in selected or info.is_dir()
                        or (mode and stat.S_ISLNK(mode)) or info.flag_bits & 0x1
                        or not 1 <= info.file_size <= _MAX_MEDIA_TOOL_BYTES):
                    raise ValueError("Media-tool archive member is ambiguous or unsafe")
                selected[matches[0]] = info
            if set(selected) != set(requested):
                raise ValueError("Media-tool archive is missing a unique required member")
            results = []
            for output in requested:
                cancel()
                with handle.open(selected[output], "r") as source:
                    results.append(_write_media_member(source, target_dir / output, cancel))
            return results

    if archive["format"] == "tar.xz":
        with tarfile.open(archive_path, mode="r:xz") as handle:
            infos = handle.getmembers()
            if len(infos) > 100000:
                raise ValueError("Media-tool archive contains too many entries")
            for info in infos:
                cancel()
                path = _safe_archive_path(info.name)
                matches = [output for output, source in requested.items() if path.name == source]
                if not matches:
                    continue
                if (len(matches) != 1 or matches[0] in selected or not info.isfile()
                        or not 1 <= info.size <= _MAX_MEDIA_TOOL_BYTES):
                    raise ValueError("Media-tool archive member is ambiguous or unsafe")
                selected[matches[0]] = info
            if set(selected) != set(requested):
                raise ValueError("Media-tool archive is missing a unique required member")
            results = []
            for output in requested:
                cancel()
                source = handle.extractfile(selected[output])
                if source is None:
                    raise ValueError("Media-tool archive member could not be read")
                with source:
                    results.append(_write_media_member(source, target_dir / output, cancel))
            return results
    raise ValueError("Unsupported media-tool archive format")


def _stage_media_tools(config_dir: Path, root: Path, plan: dict, cancel, callback) -> dict:
    spec = plan["media_tool_spec"]
    target_dir = root / "tools" / "bin"
    target_dir.mkdir(parents=True, exist_ok=False)
    archive_receipts = []
    file_receipts = []
    for archive in spec["archives"]:
        cancel()
        digest = archive["sha256"].lower()
        store = owned_path(config_dir, Path("media-assets") / digest)
        reusable = (store.is_file() and not _is_link_like(store)
                    and store.stat().st_size == archive["size"]
                    and _hash_file(store, cancel).lower() == digest)
        if not reusable:
            _event(config_dir, callback, "downloading", .07,
                   f"Downloading verified media tools for {spec['platform']}/{spec['architecture']}")
            _download_file(archive["url"], store, cancel, size=archive["size"], sha256=digest)
        else:
            _event(config_dir, callback, "validating", .07,
                   f"Reusing verified media tools for {spec['platform']}/{spec['architecture']}")
        file_receipts.extend(_extract_media_archive(store, archive, target_dir, cancel))
        archive_receipts.append({"sha256": digest, "size": archive["size"]})
    expected = set(_canonical_media_names(spec["platform"]))
    if {entry["path"] for entry in file_receipts} != expected or len(file_receipts) != len(expected):
        raise ValueError("Staged media tools are not the canonical FFmpeg/FFprobe pair")
    return {"set_id": spec["set_id"], "revision": spec["revision"],
            "platform": spec["platform"], "architecture": spec["architecture"],
            "asset_dir": "tools/bin", "archives": archive_receipts,
            "files": sorted(file_receipts, key=lambda row: row["path"])}


def _validate_media_tools(config_dir: Path, root: Path, cancel) -> None:
    """Exercise the contained pair before installing Python or model payloads."""
    validation = root / "validation"
    validation.mkdir(exist_ok=True)
    env = _candidate_env(config_dir, root, validation / "cache")
    tool_dir, _ = _verified_media_tools(config_dir, root, cancel=cancel)
    canonical = _canonical_media_names(sys.platform)
    version_probe = (
        "import subprocess\n"
        f"for command in {list(canonical)!r}:\n"
        " p=subprocess.run([command,'-version'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=30)\n"
        " assert p.returncode == 0 and p.stdout.strip(), command+' version check failed'\n"
    )
    _run_process([sys.executable, "-I", "-B", "-c", version_probe], env, cancel, timeout=60)

    fixture = validation / "media-input.wav"
    with wave.open(str(fixture), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\0\0\0\0" * 800)
    encoded = validation / "media-output.flac"
    _run_process([str(tool_dir / canonical[0]), "-nostdin", "-hide_banner", "-loglevel", "error",
                  "-y", "-i", str(fixture), "-c:a", "flac", str(encoded)], env, cancel, timeout=60)
    output = _run_process([str(tool_dir / canonical[1]), "-v", "error", "-select_streams", "a:0",
                           "-show_entries", "stream=codec_name,sample_rate,channels", "-of", "json",
                           str(encoded)], env, cancel, timeout=60)
    try:
        stream = json.loads(output)["streams"][0]
        valid = stream.get("codec_name") == "flac" and int(stream.get("sample_rate")) == 8000 and int(stream.get("channels")) == 2
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        valid = False
    if not valid:
        raise RuntimeError("Contained FFmpeg/FFprobe round-trip produced invalid audio metadata")


def _stage_source(config_dir: Path, root: Path, plan: dict, cancel) -> None:
    import demucs_server as ds
    archive = root / "source.zip"
    _download_file(f"https://codeload.github.com/{ds.SOURCE_REPO}/zip/{plan['source_commit']}", archive, cancel)
    source = root / "src"
    source.mkdir()
    try:
        found = set()
        with zipfile.ZipFile(archive) as handle:
            for info in handle.infolist():
                cancel()
                parts = PurePosixPath(info.filename).parts
                if len(parts) != 2 or parts[1] not in ds.RUNTIME_SOURCE_FILES or info.is_dir():
                    continue
                if parts[1] in found or info.file_size > 5 * 1024 * 1024:
                    raise ValueError("Unexpected runtime source archive layout")
                found.add(parts[1])
                (source / parts[1]).write_bytes(handle.read(info))
        if set(ds.RUNTIME_SOURCE_FILES) - found:
            raise ValueError("Server source archive is missing runtime contract files")
        manifest = _read(source / MANIFEST_NAME)
        if _digest(manifest) != plan["manifest_sha256"]:
            raise ValueError("Downloaded source does not match the planned compatibility manifest")
        _atomic_json(root / "source.json", {"repo": ds.SOURCE_REPO, "ref": plan["ref"],
                                           "commit": plan["source_commit"], "installed_at": time.time()})
    finally:
        archive.unlink(missing_ok=True)


def _disk_budget(config_dir: Path, plan: dict) -> None:
    # Rebuilding always needs a separate tree, even if wheels are already in pip's
    # cache. This is a conservative estimate, not a promise of exact download size.
    models = sum(a["size"] for m in plan["model_specs"].values() for a in m["assets"])
    media_archives = sum(archive["size"] for archive in plan["media_tool_spec"]["archives"])
    libraries = (12 if plan["gpu"] else 6) * 1024**3 if "libraries" in plan["components"] else 0
    # Keep the verified archive in the content-addressed store and extract a
    # separate immutable pair. Three archive sizes conservatively cover the
    # compressed payload, extracted files and update headroom.
    needed = libraries + models * 2 + media_archives * 3 + 512 * 1024**2
    path = _base(config_dir)
    path.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path).free < needed:
        raise RuntimeError(f"Not enough disk space to stage a safe replacement (estimated {needed / 1024**3:.1f} GiB needed)")


def _install_dependencies(config_dir: Path, root: Path, plan: dict, cancel, callback) -> dict:
    import demucs_server as ds
    import engine_install
    profile = plan["profile"]
    target = root / "pylibs"
    target.mkdir()
    requirements = list(profile["requirements"])
    args = ["--target", str(target), "--ignore-installed", "--upgrade",
            "--upgrade-strategy", "eager", "--report", str(root / "pip-report.json"), *requirements]
    if plan["gpu"]:
        cuda = profile.get("cuda_torch", {})
        for name in ("torch", "torchaudio"):
            if not cuda.get(name):
                raise ValueError("The profile does not define a supported CUDA dependency build")
            args.append(f"{name}=={cuda[name]}+{plan['cuda_tag']}")
        args.extend(["--extra-index-url", ds.cuda_index(plan["cuda_tag"])])
    engine_install.stream_pip(sys.executable, args, "Compatible runtime dependencies", callback,
                              0.12, 0.36, len(requirements), cancel_cb=cancel)
    # These intentional metadata exceptions cannot perturb the graph above.
    # Record separate installation reports too, including artifact hashes.
    for index, requirement in enumerate(profile.get("no_deps", [])):
        cancel()
        engine_install.stream_pip(sys.executable,
            ["--target", str(target), "--no-deps", "--upgrade", "--report", str(root / f"pip-extra-{index}.json"), requirement],
            requirement, callback, 0.48 + index * 0.035, 0.035, 1, cancel_cb=cancel)
    versions = _package_versions(target)
    _check_installed_requirements(versions, profile)
    if plan["gpu"] and not all(plan["cuda_tag"] in versions.get(name, "") for name in ("torch", "torchaudio")):
        raise RuntimeError("Requested CUDA wheels were not installed; the existing runtime was preserved")
    return versions


def _asset_candidates(config_dir: Path, asset: dict, model_id: str):
    path = _safe_relative(asset["path"])
    digest = asset["sha256"].lower()
    yield owned_path(config_dir, Path("assets") / digest)
    receipt = _receipt(config_dir)
    previous = receipt.get("models", {}).get(model_id, {})
    if previous.get("asset_dir"):
        yield owned_path(config_dir, previous["asset_dir"]) / path
    # Reuse legacy weights only after a complete digest check, never just size.
    cache = _base(config_dir) / "cache"
    for base in (cache / "_roformer-models", cache / "torch" / "hub" / "checkpoints",
                 cache / "hub" / "checkpoints"):
        yield base / path


def _stage_models(config_dir: Path, root: Path, plan: dict, cancel, callback) -> dict:
    models = {}
    for model_id, spec in plan["model_specs"].items():
        model_dir = root / "models" / model_id
        model_dir.mkdir(parents=True)
        for asset in spec["assets"]:
            cancel()
            store = owned_path(config_dir, Path("assets") / asset["sha256"].lower())
            source = None
            for candidate in _asset_candidates(config_dir, asset, model_id):
                if candidate.is_file() and candidate.stat().st_size == asset["size"] and _hash_file(candidate, cancel).lower() == asset["sha256"].lower():
                    source = candidate
                    break
            if source is None:
                _event(config_dir, callback, "downloading", 0.6, f"Downloading verified {model_id}: {asset['path']}")
                _download_file(asset["url"], store, cancel, size=asset["size"], sha256=asset["sha256"])
                source = store
            else:
                _event(config_dir, callback, "validating", 0.6, f"Reusing verified {model_id}: {asset['path']}")
            target = model_dir / _safe_relative(asset["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            # Separate inodes protect rollback even if a downstream library ever
            # rewrites a same-named config or checkpoint in the candidate.
            shutil.copy2(source, target)
            if _hash_file(target, cancel).lower() != asset["sha256"].lower():
                raise ValueError("Model verification changed while preparing the generation")
        models[model_id] = {**spec, "asset_dir": str(model_dir)}
    return models


def _terminate_process(proc: subprocess.Popen):
    if proc.poll() is not None:
        proc.wait()
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, timeout=15,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            import signal
            os.killpg(proc.pid, signal.SIGKILL)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=20)


def _popen(args, **kwargs):
    group = {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {"start_new_session": True}
    return subprocess.Popen(args, **kwargs, **group)


def _run_process(args: list[str], env: dict, cancel, *, timeout: float = 600, cwd: Path | None = None) -> str:
    import engine_install
    # Validation workers need the same crash protection as pip. Running the
    # payload in this guarded interpreter keeps model workers in its owned
    # process group and prevents an app crash from orphaning GPU work.
    if args[0] != sys.executable:
        raise ValueError("Validation must use this installation's Python")
    command = list(args[1:])
    while command and command[0] in {"-B", "-u"}:
        command.pop(0)
    if command[:1] == ["-c"]:
        payload = f"sys.argv=['-c'];exec(compile({command[1]!r},'<candidate-validation>','exec'))\n"
    elif command:
        payload = f"sys.argv={command!r};runpy.run_path({command[0]!r},run_name='__main__')\n"
    else:
        raise ValueError("Missing candidate validation command")
    args = [sys.executable, "-B", "-u", "-c", engine_install._MANAGED_PARENT_WATCH + payload]
    env = {**env, "STEM_SPLITTER_INSTALL_PARENT_PID": str(os.getpid())}
    # A file rather than an undrained pipe avoids deadlocks and permits responsive
    # cancellation even when a subprocess prints nothing for several minutes.
    with tempfile.TemporaryFile() as output:
        proc = _popen(args, env=env, cwd=str(cwd) if cwd else None, stdout=output, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + timeout
        try:
            while proc.poll() is None:
                cancel()
                if time.monotonic() >= deadline:
                    raise RuntimeError("Candidate validation timed out")
                time.sleep(0.15)
            output.seek(0)
            text = output.read().decode("utf-8", errors="replace")
            if proc.returncode:
                raise RuntimeError("Candidate command failed: " + text[-3500:])
            return text
        finally:
            _terminate_process(proc)


def _candidate_env(config_dir: Path, root: Path, cache: Path) -> dict:
    import demucs_server as ds
    with ds.installation_context(root, cache):
        env = ds._server_env(config_dir)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    return env


def _validate_candidate(config_dir: Path, root: Path, plan: dict, cancel):
    import demucs_server as ds
    validation = root / "validation"
    validation.mkdir(exist_ok=True)
    with ds.installation_context(root, validation / "cache"):
        ds.write_launcher(config_dir)
        ds.patch_driver_scripts(config_dir)
        libs = ds.pylibs_dir(config_dir)
    env = _candidate_env(config_dir, root, validation / "cache")
    source = root / "src"
    code = (
        "import sys,importlib,json,runpy\n"
        f"sys.path[:0] = {[str(source), str(libs)]!r}\n"
        "for m in ['fastapi','uvicorn','soundfile','torch','torchcrepe','whisperx','demucs.apply','audio_separator.separator']:\n"
        " importlib.import_module(m)\n"
        "import server\n"
        "import torch\n"
    )
    if plan["gpu"]:
        probe_device = plan.get("device") or "cuda"
        if probe_device == "cpu":
            probe_device = "cuda"
        code += (
            "assert torch.cuda.is_available(), 'Requested CUDA device is unavailable'\n"
            f"x=torch.ones((32,32),device={probe_device!r}); y=(x@x).cpu()\n"
            f"print('CUDA_PROBE:'+str(torch.cuda.get_device_capability({probe_device!r})))\n"
        )
    _run_process([sys.executable, "-B", "-c", code], env, cancel, cwd=source)
    for driver in ("run_demucs.py", "run_roformer.py"):
        _run_process([sys.executable, "-B", str(source / driver), "--help"], env, cancel, cwd=source)
    # Probe the actual service separately; imports alone do not establish readiness.
    _probe_candidate_health(config_dir, root, plan, env, cancel)


def _probe_candidate_health(config_dir: Path, root: Path, plan: dict, env: dict, cancel):
    import requests
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [sys.executable, str(root / "src" / "_launch.py"), "--host", "127.0.0.1", "--port", str(port), "--skip-warmup", "--model", plan["model"], "--device", "cpu"]
    with (root / "validation" / "health.log").open("w", encoding="utf-8") as output:
        proc = _popen(command, cwd=str(root / "src"), env=env, stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                cancel()
                if proc.poll() is not None:
                    raise RuntimeError("Candidate server exited before becoming healthy; see validation/health.log")
                try:
                    result = requests.get(f"http://127.0.0.1:{port}/runtime", timeout=1)
                    if result.status_code == 200 and result.json().get("generation_id") == root.name:
                        return
                except (requests.RequestException, ValueError):
                    pass
                time.sleep(0.2)
            raise RuntimeError("Candidate health check did not report the expected runtime identity")
        finally:
            _terminate_process(proc)


def _inference_smoke(config_dir: Path, root: Path, plan: dict, cancel):
    """Execution/output integrity fixture. This does not measure separation quality."""
    import math
    import struct
    source = root / "src"
    validation = root / "validation"
    validation.mkdir(exist_ok=True)
    fixture = validation / "fixture.wav"
    # A supported model-length clip exercises real overlap inference. Pure sine
    # waves are deliberately synthetic and contain no user library content.
    with wave.open(str(fixture), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        frames = bytearray()
        for index in range(44100 * 11):
            sample = int(1200 * math.sin(2 * math.pi * 220 * index / 44100))
            frames.extend(struct.pack("<hh", sample, sample))
        handle.writeframes(frames)
    output = validation / ("stems-" + uuid.uuid4().hex)
    output.mkdir()
    model = plan["model"]
    spec = plan["model_specs"][model]
    env = _candidate_env(config_dir, root, validation / "cache")
    device = plan.get("device") or ("cuda" if plan["gpu"] else "cpu")
    if device == "cpu":
        # An empty value can disagree between Windows CUDA and PyTorch/NVML
        # device probes. The explicit no-GPU value keeps CPU inference consistent.
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    if spec["engine"] in {"audio-separator", "roformer"}:
        args = [sys.executable, str(source / "run_roformer.py"), "-m", spec["entrypoint"],
                "-o", str(output), "-d", device, "--model-dir", str(root / "models" / model), str(fixture)]
    else:
        args = [sys.executable, str(source / "run_demucs.py"), "-n", model, "-d", device,
                "--shifts", "1", "-o", str(output), str(fixture)]
    _run_process(args, env, cancel, timeout=1800, cwd=source)
    import demucs_server as ds
    with ds.installation_context(root):
        libs = ds.pylibs_dir(config_dir)
    code = (
        f"import sys;sys.path.insert(0,{str(libs)!r})\n"
        "import soundfile as sf,numpy as np,re\nfrom pathlib import Path\n"
        f"files=[p for p in Path({str(output)!r}).rglob('*') if p.suffix.lower() in ('.flac','.wav')]\n"
        "labels = [(re.search(r'_\\(([^)]+)\\)_', p.stem).group(1).lower() if re.search(r'_\\(([^)]+)\\)_', p.stem) else p.stem.lower()) for p in files]\n"
        f"assert set(labels)==set({spec['stems']!r}) and len(labels)==len(set(labels)), 'Incorrect model stem labels'\n"
        "for p in files:\n"
        " x,sr=sf.read(p,always_2d=True); assert sr==44100 and abs(len(x)-sr*11)<=sr*.05 and x.shape[1]==2 and np.isfinite(x).all(), 'Invalid model output'\n"
    )
    _run_process([sys.executable, "-B", "-c", code], env, cancel)


def _stage_candidate(config_dir: Path, plan: dict, cancel, callback) -> Path:
    import demucs_server as ds
    _disk_budget(config_dir, plan)
    identity = "g-" + uuid.uuid4().hex
    root = _generation(config_dir, identity)
    root.mkdir(parents=True, exist_ok=False)
    _persist(config_dir, generation_id=identity, previous_generation=_active_id(config_dir), plan_id=plan["plan_id"])
    receipt = {"schema_version": RECEIPT_SCHEMA_VERSION, "generation_id": identity, "source_commit": plan["source_commit"],
               "profile_id": plan["profile_id"], "manifest_sha256": plan["manifest_sha256"],
               "created_at": time.time(), "owner": "stem_splitter", "validated": False,
               "prepared": False, "python": platform.python_version(), "platform": sys.platform,
               "dependencies": {}, "models": {}, "media_tools": {}}
    _atomic_json(root / "receipt.json", receipt)
    _event(config_dir, callback, "downloading", 0.04, "Downloading the exact server revision")
    _stage_source(config_dir, root, plan, cancel)
    cancel()
    receipt["media_tools"] = _stage_media_tools(config_dir, root, plan, cancel, callback)
    _atomic_json(root / "receipt.json", receipt)
    _event(config_dir, callback, "validating", .1, "Verifying contained FFmpeg and FFprobe")
    _validate_media_tools(config_dir, root, cancel)
    cancel()
    if "libraries" in plan["components"]:
        _event(config_dir, callback, "installing", 0.12, "Resolving compatible dependencies in a separate runtime")
        def pip_progress(ev):
            cancel()
            _event(config_dir, callback, "installing", ev.get("pct", .12), ev.get("phase") or "Installing",
                   line=ev.get("line") or ev.get("phase", "Installing"))
        receipt["dependencies"] = _install_dependencies(config_dir, root, plan, cancel, pip_progress)
    else:
        dependency_dir = ds.pylibs_dir(config_dir)
        receipt["dependency_dir"] = str(owned_path(config_dir, dependency_dir))
        receipt["dependencies"] = _package_versions(dependency_dir)
        _check_installed_requirements(receipt["dependencies"], plan["profile"])
    receipt["models"] = _stage_models(config_dir, root, plan, cancel, callback)
    _atomic_json(root / "receipt.json", receipt)
    _atomic_json(root / "install.json", {"gpu": plan["gpu"], "torch": receipt["dependencies"].get("torch"),
                                       "cuda_tag": plan["cuda_tag"] if plan["gpu"] else None,
                                       "installed_at": time.time(), "profile_id": plan["profile_id"]})
    _event(config_dir, callback, "validating", .7, "Checking candidate imports, drivers and isolated server health")
    _validate_candidate(config_dir, root, plan, cancel)
    receipt["prepared"] = True
    receipt["prepared_at"] = time.time()
    _atomic_json(root / "receipt.json", receipt)
    return root


def _runtime(url: str) -> dict:
    import requests
    try:
        response = requests.get(url.rstrip("/") + "/runtime", timeout=3)
        if response.status_code == 200:
            result = response.json()
            return result if isinstance(result, dict) else {}
    except (requests.RequestException, ValueError):
        pass
    return {}


def _managed_identity(config_dir: Path, port: int, generation: str) -> dict:
    import demucs_server as ds
    state = ds._read_state(config_dir)
    pid = state.get("pid")
    if not pid or not ds._pid_is_our_server(int(pid), config_dir):
        return {}
    runtime = _runtime(ds.url_for(port))
    if runtime.get("managed") and runtime.get("generation_id") == generation:
        return runtime
    return {}


def _lifecycle(config_dir: Path, port: int, generation: str, action: str, *, seal: bool = False) -> tuple[int, dict]:
    import demucs_server as ds
    import requests
    response = requests.post(ds.url_for(port) + "/lifecycle/" + action,
                             headers={"X-Management-Token": management_token(config_dir)},
                             json={"generation_id": generation, "lease_seconds": 60, "seal": seal}, timeout=10)
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if response.status_code not in (200, 409):
        raise RuntimeError(f"Managed server refused {action}: HTTP {response.status_code}")
    if response.status_code == 200 and payload.get("generation_id") != generation:
        raise RuntimeError("Managed server identity changed during activation")
    return response.status_code, payload


def _drain(config_dir: Path, port: int, generation: str, cancel, callback):
    deadline = time.monotonic() + 35 * 60
    while time.monotonic() < deadline:
        cancel()
        code, payload = _lifecycle(config_dir, port, generation, "drain")
        if code == 200 and payload.get("idle"):
            code, sealed = _lifecycle(config_dir, port, generation, "drain", seal=True)
            if code == 200 and sealed.get("idle") and sealed.get("sealed"):
                return
        _event(config_dir, callback, "draining", .8, "Waiting for current server work and downloads to finish")
        # Renew a bounded server lease while respecting cancellation.
        for _ in range(20):
            cancel()
            time.sleep(.25)
    raise RuntimeError("Current server work did not finish before the update timeout; service resumed")


def _start_verified(config_dir: Path, identity: str, port: int, device: str, model: str, callback):
    import demucs_server as ds
    ds.start_server(config_dir, port=port, device=device, model=model, warmup=False, progress_cb=callback)
    if identity != "legacy":
        runtime = _runtime(ds.url_for(port))
        if not runtime.get("managed") or runtime.get("generation_id") != identity:
            raise RuntimeError("Started server did not report the activated generation")


def _write_pointer(config_dir: Path, generation: str, previous: str | None):
    _atomic_json(_base(config_dir) / "active.json", {"schema_version": SCHEMA_VERSION,
                 "generation_id": generation, "previous_generation": previous, "activated_at": time.time()})


def _activate(config_dir: Path, root: Path, plan: dict, cancel, callback,
              port: int, device: str, model: str, before_activate, start_after: bool,
              *, rolling_back: bool = False) -> dict:
    import demucs_server as ds
    previous_id = _active_id(config_dir)
    target_id = "legacy" if root == _base(config_dir) else root.name
    old_pointer_path = _base(config_dir) / "active.json"
    old_pointer = _read(old_pointer_path) if old_pointer_path.exists() else None
    if rolling_back and target_id != "legacy":
        rollback_receipt = _read(root / "receipt.json")
        receipt_version = rollback_receipt.get("schema_version")
        if receipt_version == RECEIPT_SCHEMA_VERSION:
            _event(config_dir, callback, "validating", .82,
                   "Revalidating the previous runtime's contained media tools")
            _verified_media_tools(config_dir, root, cancel=cancel)
        elif receipt_version != 1:
            raise ValueError("The previous runtime receipt version is unsupported")
    was_running, live_port = ds.is_running(config_dir)
    port = int(live_port or port) if was_running else port
    old_state = ds._read_state(config_dir)
    # Newer launches record these; legacy launches used configured settings only.
    old_device = old_state.get("device", device)
    old_model = old_state.get("model", model)
    _persist(config_dir, activation={"previous_generation": previous_id, "target_generation": target_id,
                                    "old_pointer": old_pointer, "was_running": was_running,
                                    "port": port, "device": old_device, "model": old_model})
    if before_activate:
        before_activate()
    cancel()
    if was_running and not _managed_identity(config_dir, port, previous_id):
        _event(config_dir, callback, "waiting_to_activate", .78,
               "Candidate ready. Stop the existing server, then activate or start it to finish the update")
        return status(config_dir)

    drained = False
    stopped = False
    switched = False
    try:
        if was_running:
            drained = True  # cancellation while acquiring a lease must also resume
            _drain(config_dir, port, previous_id, cancel, callback)
            cancel()
            ds.stop_server(config_dir)
            stopped = True
            if ds.is_running(config_dir, port)[0] or ds.server_health(ds.url_for(port), timeout=1)[0]:
                raise RuntimeError("The existing server did not stop; activation was not attempted")
        cancel()
        if not rolling_back:
            _event(config_dir, callback, "validating", .84, "Running a disposable model/device validation")
            plan = {**plan, "device": device}
            _inference_smoke(config_dir, root, plan, cancel)
            receipt = _read(root / "receipt.json")
            receipt.update(validated=True, validated_at=time.time(), validated_device=device or ("cuda" if plan["gpu"] else "cpu"))
            _atomic_json(root / "receipt.json", receipt)
        cancel()
        _event(config_dir, callback, "activating", .93, "Activating the verified runtime")
        previous_exists = (_generation(config_dir, previous_id) / "src" / "server.py").is_file()
        previous = previous_id if previous_exists and previous_id != target_id else (
            old_pointer.get("previous_generation") if old_pointer else None)
        _write_pointer(config_dir, target_id, previous)
        switched = True
        if was_running or start_after:
            _start_verified(config_dir, target_id, port, device, model, None)
        _event(config_dir, callback, "rolled_back" if rolling_back else "active", 1.0,
               "Previous runtime restored" if rolling_back else "Runtime update complete",
               generation_id=target_id, previous_generation=previous_id, error=None)
        return status(config_dir)
    except BaseException as exc:
        rollback_error = None
        try:
            if switched:
                # Kill only the candidate we own before restoring the pointer.
                ds.stop_server(config_dir)
                if old_pointer is None:
                    old_pointer_path.unlink(missing_ok=True)
                else:
                    _atomic_json(old_pointer_path, old_pointer)
            if stopped and was_running:
                _start_verified(config_dir, previous_id, port, old_device, old_model, None)
        except Exception as restore_exc:
            rollback_error = restore_exc
        if rollback_error:
            _persist(config_dir, state="recovery_required", phase="Rollback needs recovery",
                     error=f"Update failed: {exc}. The previous runtime could not be restarted: {rollback_error}")
            raise RuntimeError(f"Update failed and rollback needs recovery: {rollback_error}") from exc
        raise
    finally:
        if drained and not stopped:
            try:
                _lifecycle(config_dir, port, previous_id, "resume")
            except Exception:
                # The server lease expires without us. Never replace the primary
                # update failure with a cleanup/network exception.
                pass


def apply_update(config_dir: Path, plan_id: str, *, port: int = 7865, device: str = "",
                 model: str = "bs_roformer_sw", progress_cb=None, before_activate=None,
                 start_after: bool = False) -> dict:
    if status(config_dir).get("pending_activation"):
        raise RuntimeError("Activate or discard the prepared candidate before starting another update")
    plan = _get_plan(config_dir, plan_id)
    if model != plan["model"]:
        raise ValueError("The selected model changed; check for updates again")
    _validate_device(device, plan)
    plan = {**plan, "device": device}
    with _operation(config_dir, "downloading") as cancel:
        root = _stage_candidate(config_dir, plan, cancel, progress_cb)
        _activate(config_dir, root, plan, cancel, progress_cb, port, device, model,
                  before_activate, start_after)
    return status(config_dir)


def _validate_device(device: str, plan: dict):
    if device and device != "cpu" and not re.fullmatch(r"cuda(?::\d+)?", device):
        raise ValueError("Unsupported execution device: this managed runtime supports CPU or NVIDIA CUDA")
    if device.startswith("cuda") and not plan["gpu"]:
        raise ValueError("Explicit CUDA selection requires a CUDA compatibility plan")


def activate_pending(config_dir: Path, *, port: int = 7865, device: str = "",
                     model: str = "bs_roformer_sw", progress_cb=None,
                     before_activate=None, start_after: bool = False) -> dict:
    current = status(config_dir)
    if current.get("state") != "waiting_to_activate":
        raise ValueError("There is no validated candidate waiting to activate")
    if current.get("cancel_requested"):
        raise UpdateCancelled("Pending update was canceled; discard it or prepare a new update")
    root = _generation(config_dir, current["generation_id"])
    if not _read(root / "receipt.json").get("prepared"):
        raise ValueError("Pending candidate is incomplete")
    plan = _get_plan(config_dir, current["plan_id"])
    if model != plan["model"]:
        raise ValueError("The selected model changed; prepare a matching update")
    _validate_device(device, plan)
    with _operation(config_dir, "validating") as cancel:
        _activate(config_dir, root, plan, cancel, progress_cb, port, device, model,
                  before_activate, start_after)
    return status(config_dir)


def rollback(config_dir: Path, *, port: int = 7865, device: str = "", model: str = "bs_roformer_sw",
             progress_cb=None, before_activate=None, start_after: bool = False) -> dict:
    current = status(config_dir)
    pointer = _read(_base(config_dir) / "active.json")
    target = pointer.get("previous_generation")
    if current.get("needs_recovery"):
        activation = current.get("activation", {})
        target = activation.get("previous_generation") or current.get("previous_generation")
        start_after = start_after or bool(activation.get("was_running"))
        port = int(activation.get("port") or port)
    if not target:
        raise ValueError("No previous runtime generation is available")
    root = _generation(config_dir, target)
    if target != "legacy":
        receipt = _read(root / "receipt.json")
        if not receipt.get("validated"):
            raise ValueError("The previous generation is incomplete; recovery material was preserved")
        if model not in receipt.get("models", {}):
            supported = ", ".join(sorted(receipt.get("models", {}))) or "a model included in that generation"
            raise ValueError(f"The previous runtime does not include {model}. Choose {supported} before Restore.")
        gpu = bool(_read(root / "install.json").get("gpu"))
        if device.startswith("cuda") and not gpu:
            raise ValueError("The previous runtime has CPU libraries. Choose CPU before Restore.")
        _validate_device(device, {"gpu": gpu})
    if not (root / "src" / "server.py").is_file():
        raise ValueError("The previous server source is missing")
    with _operation(config_dir, "validating", recovery=True) as cancel:
        _activate(config_dir, root, {}, cancel, progress_cb, port, device, model,
                  before_activate, start_after, rolling_back=True)
    return status(config_dir)


def discard_candidate(config_dir: Path) -> dict:
    with _LOCK:
        if str(_base(config_dir)) in _RUNNING or _external_operation_running(config_dir):
            raise RuntimeError("Cancel the current operation before discarding its candidate")
        current = status(config_dir)
        generation = current.get("generation_id")
        if not generation or generation == "legacy":
            raise ValueError("No owned candidate is available to discard")
        pointer = _read(_base(config_dir) / "active.json")
        if generation in {pointer.get("generation_id"), pointer.get("previous_generation")}:
            raise ValueError("Active and rollback generations cannot be discarded")
        root = _generation(config_dir, generation)
        receipt = _read(root / "receipt.json")
        if receipt.get("owner") != "stem_splitter" or receipt.get("generation_id") != generation:
            raise ValueError("Candidate ownership is not established; recovery material was preserved")
        # Deletion is confined to this exact recorded candidate. Shared model
        # assets and every previous generation are intentionally retained.
        handle = _acquire_operation_lock(config_dir)
        try:
            # Recheck after taking the OS lock; another app may have activated
            # this candidate while we were inspecting the ownership receipt.
            pointer = _read(_base(config_dir) / "active.json")
            if generation in {pointer.get("generation_id"), pointer.get("previous_generation")}:
                raise ValueError("Active and rollback generations cannot be discarded")
            shutil.rmtree(root)
            _persist(config_dir, state="discarded", phase="Candidate discarded", active=False,
                     error=None, cancel_requested=False, generation_id=None, needs_recovery=False)
        finally:
            _lock_file(handle, unlock=True)
            handle.close()
            import demucs_server as ds
            ds._invalidate_disk_size(config_dir)
        return status(config_dir)
