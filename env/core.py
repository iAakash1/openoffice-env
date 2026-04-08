import re
import ast
from typing import Any
from env.models import Observation, Action, Reward
from env.tasks.email_task import EmailTask
from env.tasks.data_task import DataTask
from env.tasks.code_task import CodeTask

TASK_MAP = {
    "email": EmailTask,
    "data": DataTask,
    "code": CodeTask,
}


def parse_action(raw: str) -> Action:
    """
    Parse LLM output like:
        classify_email(id=1, category='urgent')
        edit_code(code='''def foo(): pass''')
        finish()
    """
    raw = raw.strip()

    match = re.match(r"^(\w+)\((.*)\)$", raw, re.DOTALL)
    if not match:
        return Action(raw=raw, action_type="unknown", params={})

    action_type = match.group(1)
    args_str = match.group(2).strip()
    params: dict[str, Any] = {}

    if not args_str:
        return Action(raw=raw, action_type=action_type, params=params)

    kv_pattern = re.compile(
        r"(\w+)\s*=\s*("
        r"'{3}[\s\S]*?'{3}"
        r'|"{3}[\s\S]*?"{3}'
        r"|'(?:[^'\\]|\\.)*'"
        r'|"(?:[^"\\]|\\.)*"'
        r"|-?\d+\.\d+"
        r"|-?\d+"
        r")"
    )

    for m in kv_pattern.finditer(args_str):
        key = m.group(1)
        val_str = m.group(2)
        try:
            params[key] = ast.literal_eval(val_str)
        except Exception:
            params[key] = val_str

    return Action(raw=raw, action_type=action_type, params=params)


class OpenOfficeEnv:
    MAX_STEPS = 30

    def __init__(self, task_name: str = "email"):
        if task_name not in TASK_MAP:
            raise ValueError(f"Unknown task '{task_name}'. Choose from: {list(TASK_MAP)}")
        self.task_name = task_name
        self._task = None
        self._step_count = 0
        self._done = False
        self._action_history: list[str] = []

    # ------------------------------------------------------------------ #
    # OpenEnv interface                                                    #
    # ------------------------------------------------------------------ #

    def reset(self) -> Observation:
        self._task = TASK_MAP[self.task_name]()
        self._step_count = 0
        self._done = False
        self._action_history = []
        return self._make_observation()

    def step(self, raw_action: str) -> tuple[Observation, Reward, bool, dict]:
        if self._done:
            obs = self._make_observation()
            reward = Reward(value=0.0, feedback="Episode already done.")
            return obs, reward, True, {"error": "already_done"}

        # Repeated action guard
        if self._action_history.count(raw_action) >= 2:
            self._step_count += 1
            self._action_history.append(raw_action)
            obs = self._make_observation()
            reward = Reward(value=-0.1, feedback="Repeated action detected.")
            return obs, reward, False, {
                "step": self._step_count,
                "task": self.task_name,
                "action_type": "repeated",
                "params": {},
                "error": "repeated_action",
            }

        action = parse_action(raw_action)
        self._action_history.append(raw_action)
        self._step_count += 1

        # Parse failure
        if action.action_type == "unknown":
            obs = self._make_observation()
            reward = Reward(value=-0.1, feedback=f"Could not parse action: '{raw_action}'")
            return obs, reward, False, {
                "step": self._step_count,
                "task": self.task_name,
                "action_type": "unknown",
                "params": {},
                "error": "parse_failed",
            }

        # Delegate to task
        reward_value, feedback, task_done = self._task.apply_action(
            action.action_type, action.params
        )

        error_msg = None

        # Efficiency penalty: encourage fewer steps
        reward_value -= 0.01

        # Step limit
        if self._step_count >= self.MAX_STEPS and not task_done:
            feedback += " | Step limit reached."
            reward_value -= 0.2
            task_done = True

        self._done = task_done
        reward = Reward(value=round(reward_value, 4), feedback=feedback)
        obs = self._make_observation()

        info = {
            "step": self._step_count,
            "task": self.task_name,
            "action_type": action.action_type,
            "params": action.params,
            "error": error_msg,
        }

        return obs, reward, self._done, info

    def state(self) -> dict:
        return {
            "task": self.task_name,
            "step_count": self._step_count,
            "done": self._done,
            "task_state": self._task.get_observation() if self._task else {},
            "action_history": self._action_history,
        }

    def close(self):
        pass

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _make_observation(self) -> Observation:
        task_data = self._task.get_observation() if self._task else {}
        return Observation(
            task=self.task_name,
            data=task_data,
            step_count=self._step_count,
        )