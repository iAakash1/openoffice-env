import subprocess
import tempfile
import os
from env.tasks.code_task import TEST_CODE


def grade(task_state: dict) -> float:
    code = task_state.get("code", "")

    if not code.strip():
        return 0.02

    with tempfile.TemporaryDirectory() as tmpdir:
        sol_path  = os.path.join(tmpdir, "solution.py")
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
            return 0.02
        except Exception:
            return 0.02

    passed, total = _parse_score(output)

    if total == 0:
        return 0.02

    score = passed / total
    score = max(0.02, min(0.98, score))
    return round(score, 4)


def _parse_score(output: str) -> tuple[int, int]:
    for line in output.splitlines():
        if line.startswith("SCORE:"):
            try:
                parts = line.split(":")[1].strip().split("/")
                return int(parts[0]), int(parts[1])
            except Exception:
                pass
    return 0, 0