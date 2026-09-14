"""Runtime journals stay atomic when Windows readers briefly deny replacement."""
import concurrent.futures
import json
import os
import sys
import threading
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime_update as ru


def win_error(code):
    error = OSError("simulated Windows file error")
    error.winerror = code
    return error


@pytest.mark.parametrize("code", [5, 32, 33])
def test_transient_replace_preserves_old_json_until_retry_succeeds(tmp_path, code):
    target = tmp_path / "update-operation.json"
    target.write_text('{"state":"working"}')
    real_replace = os.replace
    attempts = []
    def replace(source, destination):
        attempts.append(Path(source))
        assert json.loads(target.read_text()) == {"state": "working"}
        assert json.loads(Path(source).read_text()) == {"state": "updated"}
        if len(attempts) < 3:
            raise win_error(code)
        real_replace(source, destination)
    with mock.patch.object(ru, "os", wraps=os) as filesystem, \
            mock.patch.object(ru.time, "sleep") as sleep:
        filesystem.name = "nt"
        filesystem.replace.side_effect = replace
        ru._atomic_json(target, {"state": "updated"})
    assert len(set(attempts)) == 1  # retry the same fully written, closed file
    assert len(attempts) == 3
    assert sleep.call_count == 2
    assert json.loads(target.read_text()) == {"state": "updated"}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("platform,code", [("nt", 3), ("nt", 112), ("posix", 5)])
def test_permanent_or_non_windows_failure_is_not_retried(tmp_path, platform, code):
    target = tmp_path / "active.json"
    target.write_text('{"generation_id":"working"}')
    error = win_error(code)
    with mock.patch.object(ru, "os", wraps=os) as filesystem, \
            mock.patch.object(ru.time, "sleep") as sleep:
        filesystem.name = platform
        filesystem.replace.side_effect = error
        with pytest.raises(OSError) as caught:
            ru._atomic_json(target, {"generation_id": "candidate"})
        assert filesystem.replace.call_count == 1
    assert caught.value is error
    sleep.assert_not_called()
    assert json.loads(target.read_text()) == {"generation_id": "working"}
    assert not list(tmp_path.glob("*.tmp"))


def test_retry_exhaustion_preserves_destination_and_original_error(tmp_path):
    target = tmp_path / "active.json"
    before = b'{"generation_id":"working"}'
    target.write_bytes(before)
    error = win_error(5)
    with mock.patch.object(ru, "os", wraps=os) as filesystem, \
            mock.patch.object(ru.time, "sleep") as sleep:
        filesystem.name = "nt"
        filesystem.replace.side_effect = error
        with pytest.raises(OSError) as caught:
            ru._atomic_json(target, {"generation_id": "candidate"})
        assert filesystem.replace.call_count == len(ru._REPLACE_RETRY_DELAYS) + 1
    assert caught.value is error
    assert [call.args[0] for call in sleep.call_args_list] == list(ru._REPLACE_RETRY_DELAYS)
    assert sum(call.args[0] for call in sleep.call_args_list) < 2
    assert target.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_temp_cleanup_does_not_replace_original_failure(tmp_path):
    target = tmp_path / "active.json"
    target.write_text('{"generation_id":"working"}')
    error = win_error(112)
    with mock.patch.object(ru, "os", wraps=os) as filesystem, \
            mock.patch.object(Path, "unlink", side_effect=PermissionError("temp held by scanner")):
        filesystem.name = "nt"
        filesystem.replace.side_effect = error
        with pytest.raises(OSError) as caught:
            ru._atomic_json(target, {"generation_id": "candidate"})
    assert caught.value is error
    assert json.loads(target.read_text()) == {"generation_id": "working"}
    assert len(list(tmp_path.glob("*.tmp"))) == 1


@pytest.mark.skipif(os.name != "nt", reason="requires Windows file sharing semantics")
def test_real_windows_reader_handle_is_retried_without_losing_existing_json(tmp_path):
    import ctypes
    from ctypes import wintypes
    target = tmp_path / "update-operation.json"
    target.write_text('{"state":"working"}')
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                  wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    # Allow reads and writes, deliberately omit FILE_SHARE_DELETE like the
    # briefly overlapping reader/antivirus handle that broke the real updater.
    handle = kernel.CreateFileW(str(target), 0x80000000, 0x1 | 0x2, None, 3, 0x80, None)
    assert handle != ctypes.c_void_p(-1).value, ctypes.WinError(ctypes.get_last_error())
    blocked = threading.Event()
    real_replace = os.replace
    def replace(source, destination):
        try:
            return real_replace(source, destination)
        except OSError as exc:
            assert exc.winerror in {5, 32, 33}
            blocked.set()
            raise
    pool = concurrent.futures.ThreadPoolExecutor(1)
    try:
        with mock.patch.object(ru.os, "replace", side_effect=replace):
            future = pool.submit(ru._atomic_json, target, {"state": "updated"})
            assert blocked.wait(1), "Windows did not deny deletion while reader held the file"
            assert json.loads(target.read_text()) == {"state": "working"}
            kernel.CloseHandle(handle)
            handle = None
            future.result(timeout=3)
    finally:
        if handle is not None:
            kernel.CloseHandle(handle)
        pool.shutdown(wait=True)
    assert json.loads(target.read_text()) == {"state": "updated"}
    assert not list(tmp_path.glob("*.tmp"))


def test_concurrent_readers_only_observe_complete_json(tmp_path):
    target = tmp_path / "update-operation.json"
    payload = "generation-state-" * 512
    ru._atomic_json(target, {"counter": 0, "payload": payload})
    ready = [threading.Event() for _ in range(3)]
    stop = threading.Event()
    def read(reader_ready):
        observed = 0
        while not stop.is_set():
            try:
                value = json.loads(target.read_text())
            except PermissionError:
                # io.open can report CRT errno13 without a winerror while a
                # Windows replacement is completing. Only successful reads
                # participate in the complete-JSON invariant.
                if os.name == "nt":
                    stop.wait(.002)
                    continue
                raise
            assert isinstance(value["counter"], int)
            assert value["payload"] == payload
            observed += 1
            reader_ready.set()
            # Polling releases the file between reads. An uninterrupted set of
            # no-delete-sharing handles must exhaust the bounded retry budget.
            stop.wait(.002)
        return observed
    pool = concurrent.futures.ThreadPoolExecutor(3)
    readers = [pool.submit(read, event) for event in ready]
    try:
        assert all(event.wait(1) for event in ready)
        for counter in range(1, 31):
            ru._atomic_json(target, {"counter": counter, "payload": payload})
    finally:
        stop.set()
        pool.shutdown(wait=True)
    assert all(reader.result() > 0 for reader in readers)
    assert json.loads(target.read_text())["counter"] == 30
    assert not list(tmp_path.glob("*.tmp"))
