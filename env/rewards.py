"""
rewards.py — centralized dense reward shaping

compute_reward() is called OPTIONALLY from inference.py or evaluation
scripts to get a state-based reward signal independent of task actions.

This satisfies the OpenEnv requirement:
  "Reward function must provide feedback throughout the task trajectory"
"""


def compute_reward(
    task_name: str,
    current_state: dict,
    previous_state: dict,
) -> tuple[float, str]:
    """
    Compute a dense reward by comparing current vs previous task state.
    Returns (reward_value, feedback_string).

    Works across all three tasks without knowing task internals —
    delegates to the right scorer based on task_name.
    """
    if task_name == "email":
        return _reward_email(current_state, previous_state)
    elif task_name == "data":
        return _reward_data(current_state, previous_state)
    elif task_name == "code":
        return _reward_code(current_state, previous_state)
    else:
        return 0.0, f"Unknown task: {task_name}"


# ------------------------------------------------------------------ #
# Email                                                                #
# ------------------------------------------------------------------ #

def _reward_email(curr: dict, prev: dict) -> tuple[float, str]:
    from env.graders.email_grader import grade

    curr_score = grade(curr)
    prev_score = grade(prev)
    delta = round(curr_score - prev_score, 4)

    if delta > 0:
        return delta, f"Progress: email score +{delta:.4f}"
    elif delta < 0:
        return delta - 0.05, f"Regression: email score {delta:.4f}"
    else:
        return -0.02, "No change in email state (possible no-op)"


# ------------------------------------------------------------------ #
# Data                                                                 #
# ------------------------------------------------------------------ #

def _reward_data(curr: dict, prev: dict) -> tuple[float, str]:
    from env.graders.data_grader import grade

    curr_score = grade(curr)
    prev_score = grade(prev)
    delta = round(curr_score - prev_score, 4)

    if delta > 0:
        return delta, f"Progress: data quality +{delta:.4f}"
    elif delta < 0:
        return delta - 0.05, f"Regression: data quality {delta:.4f}"
    else:
        return -0.02, "No change in data state (possible no-op)"


# ------------------------------------------------------------------ #
# Code                                                                 #
# ------------------------------------------------------------------ #

def _reward_code(curr: dict, prev: dict) -> tuple[float, str]:
    curr_tests = curr.get("test_results", {})
    prev_tests = prev.get("test_results", {})

    curr_passed = curr_tests.get("passed", 0)
    prev_passed = prev_tests.get("passed", 0)
    curr_total  = curr_tests.get("total",  0)

    # Before any test run
    if curr_total == 0 and prev.get("test_results", {}).get("total", 0) == 0:
        curr_code = curr.get("code", "")
        prev_code = prev.get("code", "")
        if curr_code != prev_code:
            return 0.05, "Code edited — run tests to evaluate"
        return -0.02, "No change in code state (possible no-op)"

    delta = curr_passed - prev_passed

    if delta > 0:
        gain = round(delta / max(curr_total, 1), 4)
        return gain, f"Tests improved: +{delta} passing"
    elif delta < 0:
        loss = round(delta / max(curr_total, 1), 4)
        return loss - 0.05, f"Regression: {abs(delta)} tests broke"
    else:
        return -0.02, "No change in test results (possible no-op)"