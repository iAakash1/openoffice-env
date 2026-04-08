import copy

RAW_DATA = [
    {"id": 1, "age": "25",   "salary": "50000", "department": "Engineering",  "name": "Alice"},
    {"id": 2, "age": None,   "salary": "75000", "department": "marketing",    "name": "Bob"},
    {"id": 3, "age": "999",  "salary": None,    "department": "Engineering",  "name": "Carol"},
    {"id": 4, "age": "30",   "salary": "60000", "department": "HR",           "name": "Dave"},
    {"id": 5, "age": "28",   "salary": "55000", "department": "Engineering",  "name": ""},
]

EXPECTED = [
    {"id": 1, "age": 25,  "salary": 50000.0, "department": "Engineering", "name": "Alice"},
    {"id": 2, "age": 27,  "salary": 75000.0, "department": "Marketing",   "name": "Bob"},
    {"id": 3, "age": 27,  "salary": 60000.0, "department": "Engineering", "name": "Carol"},
    {"id": 4, "age": 30,  "salary": 60000.0, "department": "HR",          "name": "Dave"},
    {"id": 5, "age": 28,  "salary": 55000.0, "department": "Engineering", "name": "Unknown"},
]

VALID_ACTIONS = {"fill_missing", "normalize_column", "remove_outlier", "finish"}


class DataTask:
    def __init__(self):
        self.reset()

    def reset(self):
        self.data = copy.deepcopy(RAW_DATA)
        self.action_count = 0
        self.applied: list[str] = []

    def get_observation(self) -> dict:
        return {
            "data": self.data,
            "issues": self._detect_issues(),
            "applied_actions": self.applied,
        }

    def _detect_issues(self) -> list[str]:
        issues = []
        for row in self.data:
            if row.get("age") is None or row.get("age") == "999":
                issues.append(f"Row {row['id']}: age is missing or outlier")
            if row.get("salary") is None:
                issues.append(f"Row {row['id']}: salary is missing")
            if row.get("name") == "":
                issues.append(f"Row {row['id']}: name is empty")
            if isinstance(row.get("department"), str) and row["department"] != row["department"].title():
                issues.append(f"Row {row['id']}: department not title-cased")
        return issues

    def apply_action(self, action_type: str, params: dict) -> tuple[float, str, bool]:
        self.action_count += 1

        if action_type == "fill_missing":
            col = params.get("column")
            return self._fill_missing(col)

        elif action_type == "normalize_column":
            col = params.get("column")
            return self._normalize(col)

        elif action_type == "remove_outlier":
            col = params.get("column")
            return self._remove_outlier(col)

        elif action_type == "finish":
            score = self._score()
            reward = 0.4 if score >= 0.9 else -0.1
            return reward, f"Finished. Data quality score: {score:.2f}", True

        else:
            return -0.1, f"Unknown action: {action_type}", False

    def _fill_missing(self, col: str) -> tuple[float, str, bool]:
        if col == "age":
            valid_ages = [int(r["age"]) for r in self.data if r["age"] not in (None, "999")]
            mean = int(sum(valid_ages) / len(valid_ages))
            changed = 0
            for row in self.data:
                if row["age"] is None:
                    row["age"] = str(mean)
                    changed += 1
            self.applied.append(f"fill_missing:{col}")
            return (0.15, f"Filled {changed} missing age(s) with mean {mean}", False)

        elif col == "salary":
            valid = [float(r["salary"]) for r in self.data if r["salary"] is not None]
            mean = sum(valid) / len(valid)
            changed = 0
            for row in self.data:
                if row["salary"] is None:
                    row["salary"] = str(mean)
                    changed += 1
            self.applied.append(f"fill_missing:{col}")
            return (0.15, f"Filled {changed} missing salary(ies) with mean {mean:.0f}", False)

        elif col == "name":
            changed = 0
            for row in self.data:
                if row["name"] == "":
                    row["name"] = "Unknown"
                    changed += 1
            self.applied.append(f"fill_missing:{col}")
            return (0.1, f"Replaced {changed} empty name(s) with 'Unknown'", False)

        return -0.05, f"fill_missing: unsupported column '{col}'", False

    def _normalize(self, col: str) -> tuple[float, str, bool]:
        if col == "department":
            changed = 0
            for row in self.data:
                orig = row["department"]
                row["department"] = orig.title()
                if row["department"] != orig:
                    changed += 1
            self.applied.append(f"normalize:{col}")
            return (0.1, f"Title-cased {changed} department(s)", False)

        elif col == "age":
            changed = 0
            for row in self.data:
                if isinstance(row["age"], str):
                    row["age"] = int(row["age"])
                    changed += 1
            self.applied.append(f"normalize:{col}")
            return (0.1, f"Converted {changed} age values to int", False)

        elif col == "salary":
            changed = 0
            for row in self.data:
                if isinstance(row["salary"], str):
                    row["salary"] = float(row["salary"])
                    changed += 1
            self.applied.append(f"normalize:{col}")
            return (0.1, f"Converted {changed} salary values to float", False)

        return -0.05, f"normalize: unsupported column '{col}'", False

    def _remove_outlier(self, col: str) -> tuple[float, str, bool]:
        if col == "age":
            changed = 0
            for row in self.data:
                val = row["age"]
                if val not in (None, "") and int(val) > 120:
                    row["age"] = None
                    changed += 1
            self.applied.append(f"remove_outlier:{col}")
            return (0.1, f"Removed {changed} age outlier(s)", False)
        return -0.05, f"remove_outlier: unsupported column '{col}'", False

    def _score(self) -> float:
        correct = 0
        total = len(EXPECTED) * 4  # age, salary, department, name
        for exp, got in zip(EXPECTED, self.data):
            if str(got.get("age")) == str(exp["age"]):
                correct += 1
            if str(got.get("salary")) == str(exp["salary"]):
                correct += 1
            if got.get("department") == exp["department"]:
                correct += 1
            if got.get("name") == exp["name"]:
                correct += 1
        return correct / total

    def is_done(self) -> bool:
        return self._score() >= 0.95