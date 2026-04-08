from pydantic import BaseModel
from typing import Any, Dict


class Observation(BaseModel):
    task: str
    data: Dict[str, Any]
    step_count: int


class Action(BaseModel):
    raw: str  # raw string from LLM
    action_type: str
    params: Dict[str, Any]


class Reward(BaseModel):
    value: float
    feedback: str