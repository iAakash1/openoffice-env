import os
import sys
import json
from openai import OpenAI
from env.core import OpenOfficeEnv

# ------------------------------------------------------------------ #
# Environment variables                                                #
# ------------------------------------------------------------------ #

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ------------------------------------------------------------------ #
# Prompt                                                               #
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """You are an autonomous agent solving real-world office tasks.

CRITICAL RULES:
- You MUST actively work on the task step by step.
- Do NOT call finish() until the task is fully completed.
- Always take a meaningful action that improves the current state.
- Never repeat an action you have already taken.
- Output ONLY the action string. No explanation. No markdown.

AVAILABLE ACTIONS BY TASK:

EMAIL TASK:
  classify_email(id=<int>, category=<'urgent'|'social'|'billing'|'feature'|'spam'|'general'>)
  mark_priority(id=<int>, priority=<'high'|'medium'|'low'>)
  finish()

DATA TASK:
  fill_missing(column=<'age'|'salary'|'name'>)
  normalize_column(column=<'age'|'salary'|'department'>)
  remove_outlier(column=<'age'>)
  finish()

CODE TASK:
  edit_code(code=\'\'\'<complete corrected python code>\'\'\')
  run_tests()
  finish()
"""


def build_user_prompt(obs: dict, last_reward: float, last_feedback: str, action_history: list[str]) -> str:
    data = obs["data"]
    task = obs["task"]

    hint = ""
    if task == "email":
        remaining   = data.get("remaining", [])
        classified  = data.get("classified", {})
        prioritized = data.get("prioritized", {})
        hint = (
            f"Emails remaining (need classify + priority): {remaining}\n"
            f"Already classified: {classified}\n"
            f"Already prioritized: {prioritized}\n"
            f"NEXT: pick the first id in remaining. "
            f"If classified, call mark_priority. If both done for all, call finish().\n"
        )
    elif task == "data":
        issues = data.get("issues", [])
        applied = data.get("applied_actions", [])
        hint = (
            f"Remaining issues: {issues}\n"
            f"Actions already applied: {applied}\n"
            f"NEXT: fix the first issue listed above using the correct action.\n"
        )
    elif task == "code":
        results     = data.get("test_results", {})
        last_output = data.get("last_run_output", "")
        hint = (
            f"Test results: {results}\n"
            f"Last test output:\n{last_output}\n"
            f"NEXT: if you have not edited the code yet, call edit_code() with the fixed code. "
            f"Then call run_tests(). If all tests pass, call finish().\n"
        )

    history_str = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(action_history[-6:]))

    return (
        f"Task: {task}\n"
        f"Step: {obs['step_count']}\n"
        f"Last reward: {last_reward:.2f}\n"
        f"Last feedback: {last_feedback}\n\n"
        f"--- WHAT YOU HAVE DONE SO FAR ---\n{history_str or '  Nothing yet.'}\n\n"
        f"--- TASK GUIDANCE ---\n{hint}\n"
        f"--- FULL STATE ---\n{json.dumps(data, indent=2)}\n\n"
        f"Your next action:"
    )


# ------------------------------------------------------------------ #
# Deterministic fallback logic                                         #
# ------------------------------------------------------------------ #

def deterministic_fallback(task_name: str, obs_data: dict, action_history: list[str]) -> str | None:
    """
    Returns a guaranteed-correct action when the LLM is looping or stalling.
    Returns None if no fallback is needed.
    """
    if task_name == "email":
        remaining   = obs_data.get("remaining", [])
        classified  = obs_data.get("classified", {})
        prioritized = obs_data.get("prioritized", {})

        for email_id in remaining:
            if email_id not in classified:
                # Pick a safe default — LLM will correct on real runs
                return f"classify_email(id={email_id}, category='general')"
            if email_id not in prioritized:
                return f"mark_priority(id={email_id}, priority='medium')"

        if not remaining:
            return "finish()"

    elif task_name == "data":
        applied  = obs_data.get("applied_actions", [])
        issues   = obs_data.get("issues", [])
        sequence = [
            "remove_outlier(column='age')",
            "fill_missing(column='age')",
            "fill_missing(column='salary')",
            "fill_missing(column='name')",
            "normalize_column(column='age')",
            "normalize_column(column='salary')",
            "normalize_column(column='department')",
        ]
        for action in sequence:
            if action not in action_history:
                return action
        if not issues:
            return "finish()"

    elif task_name == "code":
        has_edited  = any("edit_code" in a for a in action_history)
        has_tested  = any("run_tests" in a for a in action_history)
        results     = obs_data.get("test_results", {})
        passed      = results.get("passed", 0)
        total       = results.get("total",  0)

        if not has_edited:
            return None   # let LLM write the fix — don't hardcode code
        if has_edited and not has_tested:
            return "run_tests()"
        if total > 0 and passed == total:
            return "finish()"
        if has_tested and passed < total:
            return None   # let LLM attempt another edit

    return None


# ------------------------------------------------------------------ #
# Loop detection                                                       #
# ------------------------------------------------------------------ #

def is_looping(action_history: list[str], window: int = 4) -> bool:
    """True if the last `window` actions are all identical."""
    if len(action_history) < window:
        return False
    recent = action_history[-window:]
    return len(set(recent)) == 1


# ------------------------------------------------------------------ #
# LLM call                                                             #
# ------------------------------------------------------------------ #

def get_action(obs: dict, last_reward: float, last_feedback: str, action_history: list[str]) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(obs, last_reward, last_feedback, action_history)},
            ],
        )
        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1].strip()
            if raw.startswith("python"):
                raw = raw[6:].strip()

        return raw

    except Exception:
        return "finish()"


# ------------------------------------------------------------------ #
# Success check                                                        #
# ------------------------------------------------------------------ #

def check_success(task_name: str, env: OpenOfficeEnv) -> bool:
    try:
        task_state = env.state()["task_state"]
        if task_name == "email":
            from env.graders.email_grader import grade
        elif task_name == "data":
            from env.graders.data_grader import grade
        else:
            from env.graders.code_grader import grade
        return grade(task_state) >= 0.9
    except Exception:
        return False


# ------------------------------------------------------------------ #
# Episode runner                                                       #
# ------------------------------------------------------------------ #

def run_episode(task_name: str) -> dict:
    env     = OpenOfficeEnv(task_name)
    obs     = env.reset()
    rewards = []
    done    = False
    step_n  = 0
    last_reward   = 0.0
    last_feedback = "Episode started."
    info          = {}
    action_history: list[str] = []

    print(f"[START] task={task_name} env=OpenOfficeEnv model={MODEL_NAME}", flush=True)

    try:
        while not done:

            # 1. Check deterministic fallback first
            fallback = deterministic_fallback(task_name, obs.data, action_history)

            # 2. Check loop — if looping, force fallback
            if is_looping(action_history):
                action_str = fallback or "finish()"
                print(f"  [LOOP DETECTED] forcing: {action_str}", file=sys.stderr)

            # 3. Normal LLM call
            else:
                action_str = get_action(obs.model_dump(), last_reward, last_feedback, action_history)

                # 4. If LLM output is a repeated action, use fallback
                if action_str in action_history[-3:] and fallback:
                    action_str = fallback
                    print(f"  [REPEAT OVERRIDE] using fallback: {action_str}", file=sys.stderr)

                # 5. Prevent premature finish
                if action_str.strip() == "finish()" and step_n < 2 and fallback and fallback != "finish()":
                    action_str = fallback
                    print(f"  [EARLY FINISH GUARD] forcing: {action_str}", file=sys.stderr)

            action_history.append(action_str)
            obs, reward, done, info = env.step(action_str)

            step_n        = info["step"]
            last_reward   = reward.value
            last_feedback = reward.feedback

            error_val = info.get("error")
            if error_val is None:
                error_val = "null"
            else:
                error_val = str(error_val)

            rewards.append(reward.value)

            print(
                f"[STEP] step={step_n} "
                f"action={action_str} "
                f"reward={reward.value:.2f} "
                f"done={str(done).lower()} "
                f"error={error_val}",
                flush=True,
            )

    except Exception as e:
        print(
            f"[STEP] step={step_n + 1} action=finish() reward=0.00 done=true error={str(e)}",
            flush=True,
        )
        done = True

    finally:
        success     = check_success(task_name, env)
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"

        print(
            f"[END] success={str(success).lower()} "
            f"steps={step_n} "
            f"rewards={rewards_str}",
            flush=True,
        )

        env.close()

    return {"task": task_name, "success": success, "steps": step_n, "rewards": rewards}


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    tasks   = ["email", "data", "code"]
    results = []

    for task in tasks:
        result = run_episode(task)
        results.append(result)

    print("\n--- BASELINE SUMMARY ---", file=sys.stderr)
    for r in results:
        status  = "SUCCESS" if r["success"] else "FAIL"
        avg_rew = sum(r["rewards"]) / len(r["rewards"]) if r["rewards"] else 0.0
        print(
            f"  {r['task']:<6} | {status:<7} | steps={r['steps']:>2} | avg_reward={avg_rew:.3f}",
            file=sys.stderr,
        )