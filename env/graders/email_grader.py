from env.tasks.email_task import GROUND_TRUTH


def grade(task_state: dict) -> float:
    classified  = task_state.get("classified", {})
    prioritized = task_state.get("prioritized", {})

    total_emails       = len(GROUND_TRUTH)
    correct_categories = 0
    correct_priorities = 0

    for email_id, truth in GROUND_TRUTH.items():
        if classified.get(email_id) == truth["category"]:
            correct_categories += 1
        if prioritized.get(email_id) == truth["priority"]:
            correct_priorities += 1

    category_score = (correct_categories / total_emails) * 0.6
    priority_score = (correct_priorities / total_emails) * 0.4
    score          = category_score + priority_score

    score = max(0.02, min(0.98, score))
    return round(score, 4)