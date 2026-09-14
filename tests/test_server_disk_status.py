"""A slow filesystem scan must not block status, start or stop responses."""
import concurrent.futures
import sys
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import demucs_server as ds
import runtime_update as ru


def wait_for_refresh(cfg):
    deadline = time.perf_counter() + 3
    while time.perf_counter() < deadline:
        with ds._disk_lock:
            if str(cfg) not in ds._disk_refreshing:
                return
        threading.Event().wait(.005)
    pytest.fail("disk refresh did not finish")


def test_slow_scan_does_not_block_real_status_response(tmp_path):
    entered, release = threading.Event(), threading.Event()
    def scan(_root):
        entered.set()
        assert release.wait(3)
        return 12345
    with mock.patch.object(ds, "_dir_size", side_effect=scan), \
            mock.patch.object(ds, "is_running", return_value=(False, None)), \
            mock.patch.object(ds, "can_manage", return_value=(True, "")), \
            mock.patch.object(ds, "detect_nvidia_gpu", return_value=None):
        pool = concurrent.futures.ThreadPoolExecutor(1)
        try:
            response = pool.submit(ds.server_status, tmp_path)
            assert entered.wait(1)
            status = response.result(timeout=.5)  # scan remains deliberately blocked
            assert status["installed"] is False
            assert status["running"] is False
            assert status["disk_bytes"] == 0
        finally:
            release.set()
            pool.shutdown(wait=True)
            wait_for_refresh(tmp_path)
        assert ds.server_status(tmp_path)["disk_bytes"] == 12345


def test_concurrent_polls_share_one_refresh_and_serve_previous_total(tmp_path):
    entered, release = threading.Event(), threading.Event()
    ds._disk_memo[str(tmp_path)] = (0, 17)
    def scan(_root):
        entered.set()
        assert release.wait(3)
        return 29
    with mock.patch.object(ds, "_dir_size", side_effect=scan) as scanner:
        pool = concurrent.futures.ThreadPoolExecutor(12)
        try:
            results = [pool.submit(ds._server_disk_bytes, tmp_path) for _ in range(24)]
            assert entered.wait(1)
            assert [result.result(timeout=.5) for result in results] == [17] * 24
            assert scanner.call_count == 1
        finally:
            release.set()
            pool.shutdown(wait=True)
            wait_for_refresh(tmp_path)
        assert ds._server_disk_bytes(tmp_path) == 29
        assert scanner.call_count == 1


def test_scan_ttl_starts_when_it_finishes_not_when_it_begins(tmp_path):
    entered, release = threading.Event(), threading.Event()
    clock = [100.0]
    def scan(_root):
        entered.set()
        assert release.wait(3)
        return 55
    with mock.patch.object(ds, "_dir_size", side_effect=scan) as scanner, \
            mock.patch.object(ds.time, "monotonic", side_effect=lambda: clock[0]):
        try:
            assert ds._server_disk_bytes(tmp_path) == 0
            assert entered.wait(1)
            clock[0] += 120  # actual Windows scan exceeded the old 30-second TTL
        finally:
            release.set()
            wait_for_refresh(tmp_path)
        assert ds._server_disk_bytes(tmp_path) == 55
        assert scanner.call_count == 1


def test_invalidation_rejects_old_scan_without_starting_parallel_scan(tmp_path):
    entered, release = threading.Event(), threading.Event()
    def scan(_root):
        entered.set()
        assert release.wait(3)
        return 99
    with mock.patch.object(ds, "_dir_size", side_effect=scan) as scanner:
        try:
            assert ds._server_disk_bytes(tmp_path) == 0
            assert entered.wait(1)
            ds._invalidate_disk_size(tmp_path)
            assert ds._server_disk_bytes(tmp_path) == 0
            assert scanner.call_count == 1
        finally:
            release.set()
            wait_for_refresh(tmp_path)
        assert str(tmp_path) not in ds._disk_memo
    with mock.patch.object(ds, "_dir_size", return_value=4):
        assert ds._server_disk_bytes(tmp_path) == 0
        wait_for_refresh(tmp_path)
        assert ds._server_disk_bytes(tmp_path) == 4


def test_scan_failure_keeps_previous_size_and_does_not_retry_each_poll(tmp_path):
    ds._disk_memo[str(tmp_path)] = (0, 17)
    with mock.patch.object(ds, "_dir_size", side_effect=OSError("directory removed during walk")) as scanner:
        assert ds._server_disk_bytes(tmp_path) == 17
        wait_for_refresh(tmp_path)
        assert ds._server_disk_bytes(tmp_path) == 17
        assert scanner.call_count == 1


@pytest.mark.parametrize("fail", [False, True])
def test_managed_operation_invalidates_size_after_success_or_failure(tmp_path, fail):
    ds._disk_memo[str(tmp_path)] = (time.monotonic(), 17)
    try:
        with ru._operation(tmp_path, "installing"):
            if fail:
                raise RuntimeError("candidate failed")
    except RuntimeError:
        assert fail
    assert str(tmp_path) not in ds._disk_memo
