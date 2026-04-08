import subprocess
import tempfile
import os

BUGGY_CODE = '''def find_max(arr):
    max_val = arr[0]
    for i in range(len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
    return max_val

def average(arr):
    return sum(arr) / len(arr)

def remove_duplicates(lst):
    result = []
    for item in lst:
        if item not in result:
            result.append(item)
    return lst  # BUG: should return result

def is_palindrome(s):
    return s == s[::-1]
'''

TEST_CODE = '''
import sys
sys.path.insert(0, '.')
from solution import find_max, average, remove_duplicates, is_palindrome

passed = 0
total = 0

def test(name, got, expected):
    global passed, total
    total += 1
    if got == expected:
        passed += 1
        print(f"PASS: {name}")
    else:
        print(f"FAIL: {name} — got {got!r}, expected {expected!r}")

test("find_max basic",      find_max([3, 1, 4, 1, 5, 9]), 9)
test("find_max negatives",  find_max([-1, -5, -2]),       -1)
test("average",             average([1, 2, 3, 4, 5]),     3.0)
test("remove_duplicates",   remove_duplicates([1,2,2,3,3,3]), [1,2,3])
test("is_palindrome true",  is_palindrome("racecar"),     True)
test("is_palindrome false", is_palindrome("hello"),       False)

print(f"\\nSCORE: {passed}/{total}")
'''


class CodeTask:
    def __init__(self):
        self.reset()

    def reset(self):
        self.current_code = BUGGY_CODE
        self.test_results: dict = {}
        self.action_count = 0
        self.last_run_output = ""

    def get_observation(self) -> dict:
        return {
            "code": self.current_code,
            "last_run_output": self.last_run_output,
            "test_results": self.test_results,
            "instructions": (
                "Fix the buggy Python code. "
                "Use edit_code(new_code) to update, run_tests() to check."
            ),
        }

    def apply_action(self, action_type: str, params: dict) -> tuple[float, str, bool]:
        self.action_count += 1

        if action_type == "edit_code":
            new_code = params.get("code", "")
            if not new_code.strip():
                return -0.1, "Empty code submitted", False
            self.current_code = new_code
            return 0.05, "Code updated.", False

        elif action_type == "run_tests":
            return self._run_tests()

        elif action_type == "finish":
            passed, total = self._get_score()
            if total == 0:
                return -0.1, "Run tests before finishing.", True
            ratio = passed / total
            reward = 0.5 if ratio == 1.0 else (0.2 if ratio >= 0.5 else -0.1)
            return reward, f"Finished. {passed}/{total} tests passed.", True

        else:
            return -0.1, f"Unknown action: {action_type}", False

    def _run_tests(self) -> tuple[float, str, bool]:
        with tempfile.TemporaryDirectory() as tmpdir:
            sol_path = os.path.join(tmpdir, "solution.py")
            test_path = os.path.join(tmpdir, "test_runner.py")
            with open(sol_path, "w") as f:
                f.write(self.current_code)
            with open(test_path, "w") as f:
                f.write(TEST_CODE)

            try:
                result = subprocess.run(
                    ["python3", test_path],
                    capture_output=True, text=True, timeout=10, cwd=tmpdir
                )
                output = result.stdout + result.stderr
            except subprocess.TimeoutExpired:
                return -0.3, "Tests timed out — possible infinite loop", False
            except Exception as e:
                return -0.2, f"Runtime error: {e}", False

        self.last_run_output = output
        passed, total = self._parse_score(output)
        self.test_results = {"passed": passed, "total": total, "output": output}

        if total == 0:
            return -0.1, "Tests failed to run (syntax error?)", False

        ratio = passed / total
        reward = 0.3 * ratio
        feedback = f"{passed}/{total} tests passed"
        return reward, feedback, ratio == 1.0

    def _parse_score(self, output: str) -> tuple[int, int]:
        for line in output.splitlines():
            if line.startswith("SCORE:"):
                try:
                    parts = line.split(":")[1].strip().split("/")
                    return int(parts[0]), int(parts[1])
                except Exception:
                    pass
        return 0, 0

    def _get_score(self) -> tuple[int, int]:
        return self._parse_score(self.last_run_output)

    def is_done(self) -> bool:
        p, t = self._get_score()
        return t > 0 and p == t