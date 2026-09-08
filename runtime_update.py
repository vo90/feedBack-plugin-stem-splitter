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
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Callable
from urllib.parse import quote

SCHEMA_VERSION = 1
MANIFEST_NAME = "runtime-manifest.json"
_ID = re.compile(r"^[a-zA-Z0-9_-]{1,100}$")
_SHA = re.compile(r"^[a-fA-F0-9]{64}$")
_COMMIT = re.compile(r"^[a-fA-F0-9]{40}$")
_LOCK = threading.RLock()
_RUNNING: dict[str, threading.Event] = {}
_TERMINAL = {"idle", "active", "current", "rolled_back", "canceled", "failed", "discarded"}


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
        os.replace(temp, path)
    finally:
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


def _local_test_scope(config_dir: Path) -> Path | None:
    """Process-only opt-in, bound to one profile even if its environment leaks."""
    bundle = os.environ.get("FEEDBACK_STEM_TEST_SOURCE", "").strip()
    scope = os.environ.get("FEEDBACK_STEM_TEST_CONFIG_DIR", "").strip()
    if not bundle and not scope:
        return None
    if not scope or not Path(scope).is_absolute():
        raise ValueError("Local test source needs an absolute FEEDBACK_STEM_TEST_CONFIG_DIR")
    if Path(scope).resolve() != Path(config_dir).resolve():
        return None
    if not bundle or not Path(bundle).is_absolute():
        raise ValueError("Local test source needs an absolute FEEDBACK_STEM_TEST_SOURCE bundle path")
    return Path(bundle).resolve()


def _local_test_source(config_dir: Path) -> dict | None:
    """Read a hash-bound Git archive; never consult a working tree or network."""
    path = _local_test_scope(config_dir)
    if path is None:
        return None
    try:
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("Local test source metadata is too large")
        raw = path.read_bytes()
        bundle = json.loads(raw)
        if not isinstance(bundle, dict) or bundle.get("schema_version") != 1 or bundle.get("kind") != "local_git_archive":
            raise ValueError("Unsupported local test source bundle")
        commit = str(bundle.get("source_commit", "")).lower()
        if not _COMMIT.fullmatch(commit):
            raise ValueError("Local test source requires the full Git commit")
        ref = bundle.get("ref")
        if not isinstance(ref, str) or not ref.strip() or len(ref) > 200:
            raise ValueError("Local test source requires a revision label")
        files = {}
        for name in ("archive", "manifest"):
            digest = str(bundle.get(name + "_sha256", "")).lower()
            if not _SHA.fullmatch(digest):
                raise ValueError("Local test source requires full file SHA-256 digests")
            relative = bundle.get(name)
            if not isinstance(relative, str):
                raise ValueError("Local test source requires relative file paths")
            asset = (path.parent / _safe_relative(relative)).resolve()
            if not asset.is_relative_to(path.parent) or not asset.is_file():
                raise ValueError("Local test source files must stay inside the bundle directory")
            limit = 64 * 1024 * 1024 if name == "archive" else 1024 * 1024
            if asset.stat().st_size > limit or _hash_file(asset) != digest:
                raise ValueError(f"Local test source {name} failed SHA-256 or size verification")
            files[name] = asset
        size = bundle.get("archive_size")
        if type(size) is not int or size < 1 or files["archive"].stat().st_size != size:
            raise ValueError("Local test source archive has an incorrect size")
        with zipfile.ZipFile(files["archive"]) as archive:
            if archive.comment != commit.encode("ascii"):
                raise ValueError("Local test source archive does not identify the expected Git commit")
        manifest = json.loads(files["manifest"].read_bytes())
        if not isinstance(manifest, dict):
            raise ValueError("Local test source manifest must be an object")
        _validate_manifest(manifest)
        identity = {"kind": "local_test", "label": "Local test source (not published)",
                    "commit": commit, "ref": ref,
                    "bundle_sha256": hashlib.sha256(raw).hexdigest(),
                    "archive_sha256": bundle["archive_sha256"].lower()}
        return {"source": identity, "manifest": manifest, "archive": files["archive"], "archive_size": size}
    except (OSError, KeyError, TypeError, AttributeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Local test source is unavailable: {exc}. Restart using the test launcher") from exc


def _source_description(config_dir: Path) -> dict:
    try:
        local = _local_test_source(config_dir)
        return local["source"] if local else {"kind": "github", "label": "Official GitHub source"}
    except ValueError as exc:
        return {"kind": "local_test", "label": "Local test source (unavailable)", "error": str(exc)}


def _validate_source_plan(config_dir: Path, plan: dict) -> dict | None:
    local = _local_test_source(config_dir)
    source = plan.get("source", {})
    if source.get("kind") == "local_test":
        if not local or local["source"] != source or local["source"]["commit"] != plan["source_commit"]:
            raise ValueError("Local test source changed or is no longer enabled; restart the test launcher and check again")
        if _digest(local["manifest"]) != plan["manifest_sha256"]:
            raise ValueError("Local test source manifest changed; check for updates again")
    elif local:
        raise ValueError("Server source changed since this plan; check for updates again")
    return local


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
                "source": _source_description(config_dir),
                "installed_source": receipt.get("source") or source.get("source"),
                "ref": source.get("ref"), "profile_id": receipt.get("profile_id"),
                "dependencies": versions, "audio_separator": versions.get("audio-separator"),
                "models": models, "model_verification": "verified" if models and generation_id != "legacy" else "verification_pending",
                "python": platform.python_version(), "platform": sys.platform,
                "install_info": _read(root / "install.json"), "error": None}
    except Exception as exc:
        return {"installed": False, "state": "damaged", "error": str(exc), "dependencies": {}, "models": {}}


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
                rollback_available=bool(pointer.get("previous_generation")), source=_source_description(config_dir))
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
                ("plan_id", "model", "available", "dependency_graph", "server_state", "libraries_state", "source")}
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
        raise ValueError("Invalid model asset path")
    return Path(*posix.parts)


def _validate_manifest(manifest: dict):
    if manifest.get("schema_version") != SCHEMA_VERSION:
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
              "source": _source_description(config_dir),
              "dependency_graph": "Resolved and validated during explicit staging; direct version checks do not certify transitive packages."}
    try:
        gpu = bool(installed.get("install_info", {}).get("gpu")) if gpu is None else bool(gpu)
        tag = str(cuda_tag or installed.get("install_info", {}).get("cuda_tag") or ds.DEFAULT_CUDA_TAG).strip()
        local = _local_test_source(config_dir)
        ref = local["source"]["ref"] if local else ds._norm_ref(ref)
        if local:
            commit = local["source"]["commit"]
            if "server" not in selected and commit != installed.get("source_commit"):
                raise ValueError("This local test bundle changes the server; include Server")
        else:
            commit = ds._resolve_commit(ref) if "server" in selected else installed.get("source_commit")
        if not commit or not _COMMIT.fullmatch(commit):
            raise ValueError("Could not resolve an immutable server revision; nothing was installed")
        manifest = local["manifest"] if local else _load_manifest(commit)
        source = local["source"] if local else {"kind": "github", "label": "Official GitHub source", "commit": commit, "ref": ref}
        _validate_manifest(manifest)
        profile_id, profile = _select_profile(manifest, gpu, tag)
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
        plan = {**result, "plan_id": uuid.uuid4().hex, "created_at": time.time(), "state": "available",
                "can_update": True, "ref": ref, "source_commit": commit, "source": source, "profile_id": profile_id,
                "profile": profile, "manifest": manifest, "manifest_sha256": _digest(manifest),
                "available": {"server": commit, "dependencies": releases, "models": model_rows},
                "gpu": gpu, "cuda_tag": tag, "model_specs": models,
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
    _validate_source_plan(config_dir, plan)
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
                        raise ValueError("Downloaded model is larger than its published size")
                    digest.update(block)
                    handle.write(block)
        cancel()
        if size is not None and count != size:
            raise ValueError("Downloaded model has an incorrect size")
        if sha256 and digest.hexdigest().lower() != sha256.lower():
            raise ValueError("Downloaded model failed SHA-256 verification")
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def _stage_source(config_dir: Path, root: Path, plan: dict, cancel) -> None:
    import demucs_server as ds
    archive = root / "source.zip"
    local = _validate_source_plan(config_dir, plan)
    source = root / "src"
    try:
        if local:
            # Copy into the owned generation and verify the copy, closing the
            # check-to-copy race without executing anything from the bundle.
            with local["archive"].open("rb") as src, archive.open("xb") as dst:
                copied = 0
                for block in iter(lambda: src.read(1024 * 1024), b""):
                    cancel()
                    copied += len(block)
                    if copied > local["archive_size"]:
                        raise ValueError("Local test source archive changed during staging")
                    dst.write(block)
            if archive.stat().st_size != local["archive_size"] or _hash_file(archive, cancel) != plan["source"]["archive_sha256"]:
                raise ValueError("Local test source archive changed during staging")
        else:
            _download_file(f"https://codeload.github.com/{ds.SOURCE_REPO}/zip/{plan['source_commit']}", archive, cancel)
        source.mkdir()
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
                                           "commit": plan["source_commit"], "source": plan.get("source"), "installed_at": time.time()})
    finally:
        archive.unlink(missing_ok=True)


def _disk_budget(config_dir: Path, plan: dict) -> None:
    # Rebuilding always needs a separate tree, even if wheels are already in pip's
    # cache. This is a conservative estimate, not a promise of exact download size.
    models = sum(a["size"] for m in plan["model_specs"].values() for a in m["assets"])
    libraries = (12 if plan["gpu"] else 6) * 1024**3 if "libraries" in plan["components"] else 0
    needed = libraries + models * 2 + 512 * 1024**2
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
        env["CUDA_VISIBLE_DEVICES"] = ""
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
    receipt = {"schema_version": SCHEMA_VERSION, "generation_id": identity, "source_commit": plan["source_commit"],
               "source": plan.get("source"),
               "profile_id": plan["profile_id"], "manifest_sha256": plan["manifest_sha256"],
               "created_at": time.time(), "owner": "stem_splitter", "validated": False,
               "prepared": False, "python": platform.python_version(), "platform": sys.platform,
               "dependencies": {}, "models": {}}
    _atomic_json(root / "receipt.json", receipt)
    _event(config_dir, callback, "downloading", 0.04, "Downloading the exact server revision")
    _stage_source(config_dir, root, plan, cancel)
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
        return status(config_dir)
