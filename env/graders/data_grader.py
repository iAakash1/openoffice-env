from env.tasks.data_task import EXPECTED


def grade(task_state: dict) -> float:
    current_data = task_state.get("data", [])

    if not current_data:
        return 0.02

    fields  = ["age", "salary", "department", "name"]
    total   = len(EXPECTED) * len(fields)
    correct = 0

    for exp, got in zip(EXPECTED, current_data):
        for field in fields:
            exp_val = exp.get(field)
            got_val = got.get(field)
            try:
                if field == "age":
                    match = int(got_val) == int(exp_val)
                elif field == "salary":
                    match = abs(float(got_val) - float(exp_val)) < 0.01
                else:
                    match = str(got_val).strip() == str(exp_val).strip()
            except (TypeError, ValueError):
                match = False
            if match:
                correct += 1

    if total == 0:
        return 0.5

    score = correct / total
    score = max(0.01, min(0.99, score))
    if score >= 0.99:
        score = 0.98
    if score <= 0.01:
        score = 0.02
    return round(score, 4)