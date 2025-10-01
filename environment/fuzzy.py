from typing import Dict


class Fuzzy:
    """
    Fuzzy Logic module for autoscaling decisions.
    """

    def __init__(self):
        self.memberships = {
            "cpu_usage": {
                "low": lambda x: max(0, min(1, (50 - x) / 50)) if x <= 50 else 0,
                "medium": lambda x: max(0, 1 - abs(x - 50) / 25) if 25 <= x <= 75 else 0,
                "high": lambda x: max(0, min(1, (x - 50) / 50)) if x >= 50 else 0,
            },
            "memory_usage": {
                "low": lambda x: max(0, min(1, (50 - x) / 50)) if x <= 50 else 0,
                "medium": lambda x: max(0, 1 - abs(x - 50) / 25) if 25 <= x <= 75 else 0,
                "high": lambda x: max(0, min(1, (x - 50) / 50)) if x >= 50 else 0,
            },
            "response_time": {
                "fast": lambda x: max(0, min(1, (300 - x) / 300)) if x <= 300 else 0,
                "normal": lambda x: max(0, 1 - abs(x - 500) / 200) if 300 <= x <= 700 else 0,
                "slow": lambda x: max(0, min(1, (x - 500) / 500)) if x >= 500 else 0,
            },
        }

        self.action_map = {
            "scale_down": -1,
            "no_change": 0,
            "scale_up": 1,
        }

    def fuzzify(self, observation: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        fuzzy_state = {}
        for metric, value in observation.items():
            if metric in self.memberships:
                fuzzy_state[metric] = {
                    label: fn(value) for label, fn in self.memberships[metric].items()
                }
        return fuzzy_state

    def apply_rules(self, fuzzy_state: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        cpu = fuzzy_state.get("cpu_usage", {})
        mem = fuzzy_state.get("memory_usage", {})
        resp = fuzzy_state.get("response_time", {})

        return {
            "scale_up": max(cpu.get("high", 0), mem.get("high", 0), resp.get("slow", 0)),
            "scale_down": max(cpu.get("low", 0), mem.get("low", 0), resp.get("fast", 0)),
            "no_change": max(cpu.get("medium", 0), mem.get("medium", 0), resp.get("normal", 0)),
        }

    def defuzzify(self, action_memberships: Dict[str, float]) -> int:
        num = sum(self.action_map[a] * w for a, w in action_memberships.items())
        den = sum(action_memberships.values())
        if den == 0:
            return 0
        return round(num / den)

    def decide(self, observation: Dict[str, float]) -> int:
        fuzzy_state = self.fuzzify(observation)
        action_memberships = self.apply_rules(fuzzy_state)
        return self.defuzzify(action_memberships)
