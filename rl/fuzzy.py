# fuzzy.py
from typing import Dict

class Fuzzy:
    """
    Fuzzy Logic module for autoscaling decisions.

    - fuzzify(observation) -> fuzzy state
    - apply_rules(fuzzy_state) -> action memberships {'scale_up':..., 'no_change':..., 'scale_down':...}
    - influence(action_memberships) -> continuous float in [-1.0, 1.0]:
        -1.0 means strong scale_down preference, +1.0 strong scale_up preference, 0.0 neutral.
    """

    def __init__(self):
        self.memberships = {
            "cpu_usage": {
                "low": lambda x: max(0.0, min(1.0, (50.0 - x) / 50.0)) if x <= 50 else 0.0,
                "medium": lambda x: max(0.0, 1.0 - abs(x - 50.0) / 25.0) if 25 <= x <= 75 else 0.0,
                "high": lambda x: max(0.0, min(1.0, (x - 50.0) / 50.0)) if x >= 50 else 0.0,
            },
            "memory_usage": {
                "low": lambda x: max(0.0, min(1.0, (50.0 - x) / 50.0)) if x <= 50 else 0.0,
                "medium": lambda x: max(0.0, 1.0 - abs(x - 50.0) / 25.0) if 25 <= x <= 75 else 0.0,
                "high": lambda x: max(0.0, min(1.0, (x - 50.0) / 50.0)) if x >= 50 else 0.0,
            },
            "response_time": {
                "fast": lambda x: max(0.0, min(1.0, (300.0 - x) / 300.0)) if x <= 300 else 0.0,
                "normal": lambda x: max(0.0, 1.0 - abs(x - 500.0) / 200.0) if 300 <= x <= 700 else 0.0,
                "slow": lambda x: max(0.0, min(1.0, (x - 500.0) / 500.0)) if x >= 500 else 0.0,
            },
        }

    def fuzzify(self, observation: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        fuzzy_state = {}
        for metric, value in observation.items():
            if metric in self.memberships:
                fuzzy_state[metric] = {
                    label: float(fn(value)) for label, fn in self.memberships[metric].items()
                }
        return fuzzy_state

    def apply_rules(self, fuzzy_state: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        cpu = fuzzy_state.get("cpu_usage", {})
        mem = fuzzy_state.get("memory_usage", {})
        resp = fuzzy_state.get("response_time", {})

        return {
            "scale_up": max(cpu.get("high", 0.0), mem.get("high", 0.0), resp.get("slow", 0.0)),
            "scale_down": max(cpu.get("low", 0.0), mem.get("low", 0.0), resp.get("fast", 0.0)),
            "no_change": max(cpu.get("medium", 0.0), mem.get("medium", 0.0), resp.get("normal", 0.0)),
        }

    def influence(self, action_memberships: Dict[str, float]) -> float:
        """
        Convert memberships to continuous influence in [-1, 1]:
        scale_down -> negative, scale_up -> positive, no_change -> 0 bias.
        Interpretation: -1 strong down, +1 strong up.
        """
        up = action_memberships.get("scale_up", 0.0)
        down = action_memberships.get("scale_down", 0.0)
        no_change = action_memberships.get("no_change", 0.0)

        # Simple signed combination: (up - down) normalized by total mass (plus small epsilon)
        denom = up + down + no_change + 1e-8
        score = (up - down) / denom

        # Optionally dampen extreme values (smooth)
        return max(-1.0, min(1.0, score))

    def decide(self, observation: Dict[str, float]) -> Dict[str, float]:
        """
        Returns both memberships and a continuous influence for diagnostics:
        {'memberships': {...}, 'influence': -0.3}
        """
        fuzzy_state = self.fuzzify(observation)
        memberships = self.apply_rules(fuzzy_state)
        inf = self.influence(memberships)
        return {"memberships": memberships, "influence": inf}
