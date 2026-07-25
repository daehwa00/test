import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artifacts import (
    BufferedHistoryRecorder,
    append_history_event,
    initialize_run,
    read_json,
    read_status,
    run_history_path,
    run_status_history_path,
    write_matrix_manifest,
    write_status,
)


class ArtifactTest(unittest.TestCase):
    def test_run_artifacts_have_canonical_json_lifecycle(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {"run_id": "ant__time__seed-0", "seed": 0}
            initialize_run(root, config)
            append_history_event(root, config["run_id"], {"total_steps": 1, "episode_return": 3.0})
            write_status(root, config["run_id"], "running")
            write_status(root, config["run_id"], "completed")
            self.assertEqual(read_status(root, config["run_id"])["state"], "completed")
            self.assertEqual(read_json(run_history_path(root, config["run_id"]))["events"][0]["episode_return"], 3.0)

    def test_resume_preserves_history_and_rejects_identity_changes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {"run_id": "ant__time__seed-0", "seed": 0}
            initialize_run(root, config)
            append_history_event(root, config["run_id"], {"total_steps": 1})
            initialize_run(root, config)
            self.assertEqual(
                len(read_json(run_history_path(root, config["run_id"]))["events"]), 1
            )
            with self.assertRaises(RuntimeError):
                initialize_run(root, {**config, "seed": 1})
            with self.assertRaises(RuntimeError):
                initialize_run(root, {**config, "seed": True})

    def test_buffered_history_flushes_in_batches_and_at_evaluations(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_id = "ant__time__seed-0"
            initialize_run(root, {"run_id": run_id, "seed": 0})
            recorder = BufferedHistoryRecorder(root, run_id, flush_every=2)

            recorder({"event": "training", "total_steps": 1})
            self.assertEqual(
                read_json(run_history_path(root, run_id))["events"], []
            )

            recorder({"event": "training", "total_steps": 2})
            self.assertEqual(
                len(read_json(run_history_path(root, run_id))["events"]), 2
            )

            recorder({"event": "evaluation", "total_steps": 2})
            events = read_json(run_history_path(root, run_id))["events"]
            self.assertEqual(len(events), 3)
            self.assertEqual(events[-1]["event"], "evaluation")

    def test_buffered_history_rejects_invalid_flush_interval(self):
        with TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                BufferedHistoryRecorder(Path(temporary), "run", flush_every=0)
    def test_status_transitions_preserve_attempt_history(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_id = "ant__time__seed-0"
            initialize_run(root, {"run_id": run_id, "seed": 0})
            with self.assertRaises(ValueError):
                write_status(root, run_id, "completed")
            write_status(root, run_id, "running")
            write_status(root, run_id, "failed", error="boom")
            write_status(root, run_id, "running")
            history = read_json(run_status_history_path(root, run_id))["events"]
            self.assertEqual(
                [event["state"] for event in history],
                ["pending", "running", "failed", "running"],
            )
            self.assertEqual(history[2]["error"], "boom")

    def test_manifest_is_create_or_verify(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_matrix_manifest(root, {"protocol": "time-brax-corrected-v2"})
            write_matrix_manifest(root, {"protocol": "time-brax-corrected-v2"})
            with self.assertRaises(RuntimeError):
                write_matrix_manifest(root, {"protocol": "legacy"})
            write_matrix_manifest(root / "typed", {"flag": True})
            with self.assertRaises(RuntimeError):
                write_matrix_manifest(root / "typed", {"flag": 1})


if __name__ == "__main__":
    unittest.main()
