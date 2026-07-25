import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artifacts import (
    initialize_run,
    read_json,
    read_status,
    run_history_path,
    run_status_history_path,
    write_status,
)
from experiment_config import RunConfig, VARIANT_BY_NAME
from run_brax_matrix import _run_one


class RunnerResumeTest(unittest.TestCase):
    def test_running_run_is_recovered_into_a_new_attempt(self):
        run = RunConfig("swimmer", VARIANT_BY_NAME["time"], 0)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            initialize_run(root, run.to_dict())
            write_status(root, run.run_id, "running", attempt_id="attempt-0001")

            def fake_train(config, env_name):
                config["history_recorder"](
                    {
                        "event": "evaluation",
                        "total_steps": 0,
                        "evaluation_mean_return": 1.0,
                    }
                )
                return {"total_steps": 0}

            with (
                patch.dict(
                    sys.modules, {"trainer": types.SimpleNamespace(train=fake_train)}
                ),
                patch("run_brax_matrix._validate_attempt_completion"),
            ):
                failed = _run_one(root, run, "cuda", False, True)
            self.assertFalse(failed)
            status = read_status(root, run.run_id)
            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["attempt_id"], "attempt-0002")
            transitions = read_json(run_status_history_path(root, run.run_id))["events"]
            self.assertEqual(
                [event["state"] for event in transitions],
                ["pending", "running", "failed", "running", "completed"],
            )
            events = read_json(run_history_path(root, run.run_id))["events"]
            self.assertTrue(
                any(event.get("attempt_id") == "attempt-0002" for event in events)
            )


if __name__ == "__main__":
    unittest.main()
