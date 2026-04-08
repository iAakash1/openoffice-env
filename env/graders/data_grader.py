from env.tasks.data_task import EXPECTED


def grade(task_state: dict) -> float:
    """
    Score data cleaning task. Returns 0.0 – 1.0.

    Checks 4 fields per row: age, salary, department, name.
    Each correct field = 1 point. Max = len(EXPECTED) × 4.
    """
    current_data = task_state.get("data", [])

    if not current_data:
        return 0.0

    fields = ["age", "salary", "department", "name"]
    total = len(EXPECTED) * len(fields)
    correct = 0

    for exp, got in zip(EXPECTED, current_data):
        for field in fields:
            exp_val = exp.get(field)
            got_val = got.get(field)

            # Normalize for type-safe comparison
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

    return round(correct / total, 4)