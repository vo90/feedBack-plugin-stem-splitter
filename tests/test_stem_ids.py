"""Unit tests for the pure stem-id helpers in ``split_stems``.

These import ``split_stems`` directly (it defers the heavy ``pak_io``/``sloppak``
import), so they run without the feedBack host:  ``python -m unittest -v`` from
the repo root, or ``python -m pytest tests``.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import split_stems as ss  # noqa: E402


class NormalizeStemId(unittest.TestCase):
    def test_demucs_bare_names(self):
        for name in ("vocals", "drums", "bass", "guitar", "piano", "other"):
            self.assertEqual(ss._normalize_stem_id(name), name)

    def test_audio_separator_paren_labels(self):
        v = "mix_(Vocals)_model_bs_roformer_ep_317_sdr_12.9755"
        i = "mix_(Instrumental)_model_bs_roformer_ep_317_sdr_12.9755"
        self.assertEqual(ss._normalize_stem_id(v), "vocals")
        self.assertEqual(ss._normalize_stem_id(i), "other")

    def test_bs_roformer_sw_6_stem_labels(self):
        # Real server naming: "<base>_(<Label>)_BS-Roformer-SW.flac"
        for label, expect in [("Vocals", "vocals"), ("Drums", "drums"),
                              ("Bass", "bass"), ("Guitar", "guitar"),
                              ("Piano", "piano"), ("Other", "other")]:
            name = f"mix_({label})_BS-Roformer-SW"
            self.assertEqual(ss._normalize_stem_id(name), expect,
                             f"{name!r} -> {expect}")

    def test_model_token_does_not_shadow_paren_label(self):
        # "BS-Roformer-SW" contains no stem word, but ensure the paren label wins.
        self.assertEqual(
            ss._normalize_stem_id("mix_(Bass)_BS-Roformer-SW"), "bass")

    def test_no_vocals_maps_to_other_not_vocals(self):
        # "no_vocals" (the instrumental companion) must not match bare "vocals".
        self.assertEqual(ss._normalize_stem_id("mix_no_vocals_htdemucs"), "other")
        self.assertEqual(ss._normalize_stem_id("mix_vocals_htdemucs"), "vocals")

    def test_word_boundary_avoids_false_match(self):
        # "brother" contains "other" but not as a token — must not map.
        self.assertIsNone(ss._normalize_stem_id("brother"))

    def test_unknown_returns_none(self):
        self.assertIsNone(ss._normalize_stem_id("mix_12345_checkpoint"))

    def test_keys_alias(self):
        self.assertEqual(ss._normalize_stem_id("mix_(Keys)_model"), "piano")


class Sanitize(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(ss._sanitize("(Vocals) 2!"), "vocals_2")

    def test_empty_falls_back(self):
        self.assertEqual(ss._sanitize("!!!"), "stem")


class SeparateAudioServiceBoundary(unittest.TestCase):
    def test_returns_canonical_paths_below_caller_workspace_and_forwards_requested_stems(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mix = root / "mix.ogg"
            mix.write_bytes(b"fake audio")
            captured = {}

            def fake_remote(mix_arg, out_dir, model, server_url, api_key,
                            stems, progress_cb, cancel_cb, cleanup_cache=False):
                captured["stems"] = stems
                captured["cleanup_cache"] = cleanup_cache
                result = out_dir / "remote_stems"
                result.mkdir()
                (result / "mix_(Guitar)_BS-Roformer-SW.flac").write_bytes(b"guitar")
                (result / "mix_(Drums)_BS-Roformer-SW.flac").write_bytes(b"drums")
                return result

            with mock.patch.object(ss, "_run_remote", side_effect=fake_remote):
                result = ss.separate_audio(
                    mix, root / "ephemeral", engine="remote",
                    server_url="http://127.0.0.1:7865", stems=("guitar",),
                    ephemeral=True,
                )

            self.assertEqual(captured["stems"], ("guitar",))
            self.assertTrue(captured["cleanup_cache"])
            self.assertEqual(set(result), {"guitar", "drums"})
            for path in result.values():
                path.resolve().relative_to((root / "ephemeral").resolve())

    def test_ephemeral_cache_cleanup_retries_after_download_handle_closes(self):
        responses = [mock.Mock(status_code=200), mock.Mock(status_code=200)]
        requests = mock.Mock()
        requests.delete.side_effect = responses

        with mock.patch.object(ss.time, "sleep") as sleep:
            ss._cleanup_remote_cache(
                requests, "http://127.0.0.1:7865", "safe-job-id",
                {"Authorization": "Bearer secret"},
            )

        self.assertEqual(requests.delete.call_count, 2)
        sleep.assert_called_once_with(0.25)
        for response in responses:
            response.close.assert_called_once_with()

    def test_ephemeral_remote_download_streams_only_requested_standard_stem(self):
        class InitialResponse:
            status_code = 200

            def __init__(self):
                self.closed = False

            def json(self):
                return {
                    "job_id": "job-1",
                    "stems": {
                        "guitar": "/download/job-1/guitar.flac",
                        "drums": "/download/job-1/drums.flac",
                    },
                }

            def close(self):
                self.closed = True

        class StreamResponse:
            status_code = 200

            def __init__(self):
                self.closed = False
                self.chunk_sizes = []

            @property
            def content(self):
                raise AssertionError("streaming code must not buffer response.content")

            def iter_content(self, chunk_size):
                self.chunk_sizes.append(chunk_size)
                yield b"guitar-"
                yield b"audio"

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mix = root / "mix.ogg"
            mix.write_bytes(b"source")
            initial = InitialResponse()
            streamed = StreamResponse()
            with mock.patch("requests.post", return_value=initial), mock.patch.object(
                ss, "_get_authed",
                return_value=(streamed, "http://127.0.0.1:7865/download/job-1/guitar.flac"),
            ) as get_authed:
                result_dir = ss._run_remote(
                    mix, root, "bs_roformer_sw", "http://127.0.0.1:7865",
                    None, ("guitar",), None,
                )

            self.assertTrue(initial.closed)
            self.assertTrue(streamed.closed)
            self.assertEqual(get_authed.call_count, 1)
            self.assertTrue(get_authed.call_args.kwargs["stream"])
            self.assertEqual(streamed.chunk_sizes, [1024 * 1024])
            self.assertEqual((result_dir / "guitar.flac").read_bytes(), b"guitar-audio")


if __name__ == "__main__":
    unittest.main()
