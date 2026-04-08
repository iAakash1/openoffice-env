from typing import Optional

EMAILS = [
    {"id": 1, "subject": "Server down in prod", "body": "Production API is returning 500 errors. Users affected."},
    {"id": 2, "subject": "Team lunch Friday", "body": "Hey, are you joining us for lunch this Friday at noon?"},
    {"id": 3, "subject": "Invoice #4421 due", "body": "Payment of $2,400 is due in 3 days for your subscription."},
    {"id": 4, "subject": "New feature request", "body": "Client wants dark mode added before end of quarter."},
    {"id": 5, "subject": "Happy birthday!", "body": "Wishing you a great birthday from the whole team!"},
]

GROUND_TRUTH = {
    1: {"category": "urgent",   "priority": "high"},
    2: {"category": "social",   "priority": "low"},
    3: {"category": "billing",  "priority": "high"},
    4: {"category": "feature",  "priority": "medium"},
    5: {"category": "social",   "priority": "low"},
}

VALID_CATEGORIES = {"urgent", "social", "billing", "feature", "spam", "general"}
VALID_PRIORITIES = {"high", "medium", "low"}


class EmailTask:
    def __init__(self):
        self.reset()

    def reset(self):
        self.classifications: dict[int, str] = {}
        self.priorities: dict[int, str] = {}
        self.action_count = 0

    def get_observation(self) -> dict:
        return {
            "emails": EMAILS,
            "classified": self.classifications,
            "prioritized": self.priorities,
            "remaining": [
                e["id"] for e in EMAILS
                if e["id"] not in self.classifications
                or e["id"] not in self.priorities
            ],
        }

    def apply_action(self, action_type: str, params: dict) -> tuple[float, str, bool]:
        self.action_count += 1
        reward = 0.0
        feedback = ""

        if action_type == "classify_email":
            email_id = params.get("id")
            category = params.get("category", "").lower()
            if email_id not in GROUND_TRUTH:
                return -0.1, f"Invalid email id {email_id}", False
            if category not in VALID_CATEGORIES:
                return -0.05, f"Invalid category '{category}'", False
            self.classifications[email_id] = category
            if GROUND_TRUTH[email_id]["category"] == category:
                reward = 0.2
                feedback = f"Correct category for email {email_id}"
            else:
                reward = -0.05
                feedback = f"Wrong category for email {email_id}"

        elif action_type == "mark_priority":
            email_id = params.get("id")
            priority = params.get("priority", "").lower()
            if email_id not in GROUND_TRUTH:
                return -0.1, f"Invalid email id {email_id}", False
            if priority not in VALID_PRIORITIES:
                return -0.05, f"Invalid priority '{priority}'", False
            self.priorities[email_id] = priority
            if GROUND_TRUTH[email_id]["priority"] == priority:
                reward = 0.1
                feedback = f"Correct priority for email {email_id}"
            else:
                reward = -0.05
                feedback = f"Wrong priority for email {email_id}"

        elif action_type == "finish":
            done = self.is_done()
            if done:
                reward = 0.3
                feedback = "All emails triaged correctly!"
            else:
                reward = -0.1
                feedback = "Finished early — not all emails triaged."
            return reward, feedback, True

        else:
            return -0.1, f"Unknown action: {action_type}", False

        return reward, feedback, self.is_done()

    def is_done(self) -> bool:
        all_classified = all(
            self.classifications.get(e["id"]) == GROUND_TRUTH[e["id"]]["category"]
            for e in EMAILS
        )
        all_prioritized = all(
            self.priorities.get(e["id"]) == GROUND_TRUTH[e["id"]]["priority"]
            for e in EMAILS
        )
        return all_classified and all_prioritized