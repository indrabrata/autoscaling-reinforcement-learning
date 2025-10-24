from logging import Logger
from typing import Dict, Optional

class Fuzzy:
    """
    Fuzzy Logic module for autoscaling decisions using continuous memberships.

    - fuzzify(observation) -> fuzzy state (continuous membership values)
    - apply_rules(fuzzy_state) -> action memberships {'scale_up':..., 'no_change':..., 'scale_down':...}
    - influence(action_memberships) -> continuous float in [-1.0, 1.0]:
        -1.0 means strong scale_down preference, +1.0 strong scale_up preference, 0.0 neutral.
    """

    def __init__(self, logger: Optional[Logger] = None):
        # Define membership functions for each metric
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
        self.logger = logger or Logger(__name__)
        

    def fuzzify(self, observation: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Convert raw observations to fuzzy membership values."""
        fuzzy_state = {}
        for metric, value in observation.items():
            if metric in self.memberships:
                fuzzy_state[metric] = {
                    label: float(fn(value)) for label, fn in self.memberships[metric].items()
                }
        self.logger.info(f"fuzzy state: {fuzzy_state}")
        return fuzzy_state

    def apply_rules(self, fuzzy_state: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Apply fuzzy rules to compute scale_up, scale_down, no_change memberships."""
        cpu = fuzzy_state.get("cpu_usage", {})
        mem = fuzzy_state.get("memory_usage", {})
        resp = fuzzy_state.get("response_time", {})

        applied_rules = {
            "scale_up": max(cpu.get("high", 0.0), mem.get("high", 0.0), resp.get("slow", 0.0)),
            "scale_down": max(cpu.get("low", 0.0), mem.get("low", 0.0), resp.get("fast", 0.0)),
            "no_change": max(cpu.get("medium", 0.0), mem.get("medium", 0.0), resp.get("normal", 0.0)),
        }
        self.logger.info(f"applied rules: {applied_rules}")
        return applied_rules

    def influence(self, action_memberships: Dict[str, float]) -> float:
        """
        Convert fuzzy action memberships into a continuous influence in [-1, 1].
        - Negative = bias toward scale down
        - Positive = bias toward scale up
        - 0 = neutral
        """
        up = action_memberships.get("scale_up", 0.0)
        down = action_memberships.get("scale_down", 0.0)
        no_change = action_memberships.get("no_change", 0.0)

        denom = up + down + no_change + 1e-8  # normalize
        score = (up - down) / denom

        return max(-1.0, min(1.0, score))

    def decide(self, observation: Dict[str, float]) -> Dict[str, float]:
        """
        Compute both the fuzzy memberships and the continuous influence.
        Returns:
            {
                'memberships': {...},
                'influence': float in [-1, 1]
            }
        """
        fuzzy_state = self.fuzzify(observation)
        memberships = self.apply_rules(fuzzy_state)
        inf = self.influence(memberships)
        result = {"memberships": memberships, "influence": inf}
        self.logger.info(f"result decide: {result}")
        return result
