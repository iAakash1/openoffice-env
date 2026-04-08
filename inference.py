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
- Never stop early. If unsure, take the most reasonable next step.
- Output ONLY the action string. No explanation. No markdown.

AVAILABLE ACTIONS BY TASK:

EMAIL TASK — you must classify AND prioritize every email before finishing:
  classify_email(id=<int>, category=<'urgent'|'social'|'billing'|'feature'|'spam'|'general'>)
  mark_priority(id=<int>, priority=<'high'|'medium'|'low'>)
  finish()

DATA TASK — you must fix missing values, outliers, and formatting before finishing:
  fill_missing(column=<'age'|'salary'|'name'>)
  normalize_column(column=<'age'|'salary'|'department'>)
  remove_outlier(column=<'age'>)
  finish()

CODE TASK — you must edit the code, run tests, verify all pass, then finish:
  edit_code(code=\'\'\'<complete corrected python code>\'\'\')
  run_tests()
  finish()

STRATEGY PER TASK:

EMAIL: Look at 'remaining' list in state. For each email id in remaining,
call classify_email() then mark_priority(). Only call finish() when remaining is empty.

DATA: Look at 'issues' list in state. Fix each issue using the appropriate action.
Call finish() only when issues list is empty or data looks clean.

CODE: First call edit_code() with the fully corrected code.
Then call run_tests() to verify. Only call finish() when all tests pass.
"""


def build_user_prompt(obs: dict, last_reward: float, last_feedback: str) -> str:
    data = obs["data"]

    # Surface the most important signal per task
    hint = ""
    if obs["task"] == "email":
        remaining = data.get("remaining", [])
        hint = f"Emails still needing action: {remaining}\n"
    elif obs["task"] == "data":
        issues = data.get("issues", [])
        hint = f"Remaining issues to fix: {issues}\n"
    elif obs["task"] == "code":
        results = data.get("test_results", {})
        hint = f"Test results so far: {results}\n"

    return (
        f"Task: {obs['task']}\n"
        f"Step: {obs['step_count']}\n"
        f"Last reward: {last_reward:.2f}\n"
        f"Last feedback: {last_feedback}\n"
        f"{hint}\n"
        f"Current state:\n{json.dumps(data, indent=2)}\n\n"
        f"What is your next action?"
    )

# ------------------------------------------------------------------ #
# LLM call                                                             #
# ------------------------------------------------------------------ #

def get_action(obs: dict, last_reward: float, last_feedback: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(obs, last_reward, last_feedback)},
            ],
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences
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

    print(f"[START] task={task_name} env=OpenOfficeEnv model={MODEL_NAME}", flush=True)

    try:
        while not done:
            action_str = get_action(obs.model_dump(), last_reward, last_feedback)

            # Anti-finish guard — prevent model quitting before doing any work
            if action_str.strip() == "finish()" and step_n < 2:
                if task_name == "email":
                    remaining = obs.data.get("remaining", [])
                    if remaining:
                        action_str = f"classify_email(id={remaining[0]}, category='general')"
                elif task_name == "data":
                    action_str = "fill_missing(column='age')"
                elif task_name == "code":
                    action_str = "run_tests()"

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