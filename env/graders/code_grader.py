import subprocess
import tempfile
import os
from env.tasks.code_task import TEST_CODE


def grade(task_state: dict) -> float:
    """
    Score code fix task. Returns 0.0 – 1.0.

    Runs the hidden test suite against submitted code.
    Score = tests_passed / total_tests.
    Returns 0.0 on syntax error, timeout, or runtime crash.
    """
    code = task_state.get("code", "")

    if not code.strip():
        return 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        sol_path = os.path.join(tmpdir, "solution.py")
        test_path = os.path.join(tmpdir, "test_runner.py")

        with open(sol_path, "w") as f:
            f.write(code)
        with open(test_path, "w") as f:
            f.write(TEST_CODE)

        try:
            result = subprocess.run(
                ["python3", test_path],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=tmpdir,
            )
            output = result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return 0.0
        except Exception:
            return 0.0

    passed, total = _parse_score(output)

    if total == 0:
        return 0.0

    return round(passed / total, 4)


def _parse_score(output: str) -> tuple[int, int]:
    for line in output.splitlines():
        if line.startswith("SCORE:"):
            try:
                parts = line.split(":")[1].strip().split("/")
                return int(parts[0]), int(parts[1])
            except Exception:
                pass
    return 0, 0