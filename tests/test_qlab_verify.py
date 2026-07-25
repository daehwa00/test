import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "qlab_verify.py"
SPEC = importlib.util.spec_from_file_location("qlab_verify", MODULE_PATH)
qlab_verify = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(qlab_verify)


class QlabVerifierTest(unittest.TestCase):
    def test_blocks_script_and_module_campaign_launches(self):
        blocked = (
            ["python", "run_brax_matrix.py", "--execute"],
            ["python", "-m", "run_brax_matrix", "--execute"],
            ["python", "safe.py", "--authorize-full-execution"],
            ["python", "-c", "exec(open('renamed.py').read())"],
            ["bash", "-lc", "python renamed.py"],
            ["python", "renamed.py"],
            ["python-wrapper", "-m", "unittest"],
            ["python", "-m", "unittest", "discover", "-s", "/tmp/tests"],
        )
        for command in blocked:
            with self.subTest(command=command), self.assertRaises(ValueError):
                qlab_verify._validate_command(command, "qlab")

    def test_rejects_non_qlab_hosts(self):
        command = [
            "python",
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
            "-v",
        ]
        for host in ("localhost", "localhost.", "127.0.0.1", "other-gpu"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                qlab_verify._validate_command(command, host)
    def test_allows_bounded_unit_verification(self):
        qlab_verify._validate_command(
            [
                "python",
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "test_*.py",
                "-v",
            ],
            "qlab",
        )


if __name__ == "__main__":
    unittest.main()
