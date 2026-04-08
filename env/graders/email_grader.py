from env.tasks.email_task import GROUND_TRUTH


def grade(task_state: dict) -> float:
    """
    Score email triage task. Returns 0.0 – 1.0.

    Breakdown:
      - 0.6 → classification accuracy (0.12 per email × 5)
      - 0.4 → priority accuracy     (0.08 per email × 5)
    """
    classified = task_state.get("classified", {})
    prioritized = task_state.get("prioritized", {})

    total_emails = len(GROUND_TRUTH)
    correct_categories = 0
    correct_priorities = 0

    for email_id, truth in GROUND_TRUTH.items():
        if classified.get(email_id) == truth["category"]:
            correct_categories += 1
        if prioritized.get(email_id) == truth["priority"]:
            correct_priorities += 1

    category_score = (correct_categories / total_emails) * 0.6
    priority_score = (correct_priorities / total_emails) * 0.4

    return round(category_score + priority_score, 4)