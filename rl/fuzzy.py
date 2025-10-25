from logging import Logger
import math
from typing import Dict, Optional


class Fuzzy:
    """
    Enhanced Fuzzy Logic module for autoscaling decisions.

    Improvements:
    - Smoother membership overlap with trapezoids
    - Priority-based rule system (Response > CPU/Mem)
    - Weighted influence computation (stability + hysteresis)
    - Adaptive confidence for better maintain decisions
    """

    def __init__(self, logger: Optional[Logger] = None):
        def _trapezoidal(x, a, b, c, d):
            if x <= a or x >= d:
                return 0.0
            elif b <= x <= c:
                return 1.0
            elif a < x < b:
                return (x - a) / (b - a) if (b - a) != 0 else 0.0
            else:  # c < x < d
                return (d - x) / (d - c) if (d - c) != 0 else 0.0

        self.memberships = {
            "cpu_usage": {
                "very_low": lambda x: _trapezoidal(x, 0, 0, 10, 25),
                "low": lambda x: _trapezoidal(x, 15, 25, 35, 45),
                "medium": lambda x: _trapezoidal(x, 40, 50, 60, 70),
                "high": lambda x: _trapezoidal(x, 65, 75, 85, 90),
                "very_high": lambda x: _trapezoidal(x, 85, 95, 100, 100),
            },
            "memory_usage": {
                "very_low": lambda x: _trapezoidal(x, 0, 0, 10, 25),
                "low": lambda x: _trapezoidal(x, 15, 25, 35, 45),
                "medium": lambda x: _trapezoidal(x, 40, 50, 60, 70),
                "high": lambda x: _trapezoidal(x, 65, 75, 85, 90),
                "very_high": lambda x: _trapezoidal(x, 85, 95, 100, 100),
            },
            "response_time": {
                "very_low": lambda x: _trapezoidal(x, 0, 0, 10, 25),
                "low": lambda x: _trapezoidal(x, 20, 30, 45, 55),
                "medium": lambda x: _trapezoidal(x, 50, 60, 70, 80),
                "high": lambda x: _trapezoidal(x, 75, 85, 90, 95),
                "very_high": lambda x: _trapezoidal(x, 90, 95, 100, 100),
            },
        }

        self.logger = logger or Logger(__name__)

    def fuzzify(self, obs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        fuzzy_state = {}
        for metric, value in obs.items():
            if metric in self.memberships:
                fuzzy_state[metric] = {
                    label: fn(value) for label, fn in self.memberships[metric].items()
                }
        self.logger.info(f"Fuzzified: {fuzzy_state}")
        return fuzzy_state

    def apply_rules(self, fz: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        cpu = fz.get("cpu_usage", {})
        mem = fz.get("memory_usage", {})
        resp = fz.get("response_time", {})

        # Scale up: triggered by bad response time or high utilization
        scale_up = max(
            resp.get("high", 0.0) * 1.0,
            resp.get("very_high", 0.0) * 1.0,
            min(resp.get("medium", 0.0), max(cpu.get("high", 0.0), mem.get("high", 0.0))) * 0.9,
            max(cpu.get("very_high", 0.0), mem.get("very_high", 0.0)) * 0.8,
        )

        # Scale down: when utilization low & response acceptable
        scale_down = max(
            min(cpu.get("low", 0.0), mem.get("low", 0.0), resp.get("low", 0.0)) * 1.0,
            min(cpu.get("low", 0.0), mem.get("low", 0.0), resp.get("medium", 0.0)) * 0.9,
            min(cpu.get("very_low", 0.0), mem.get("very_low", 0.0)) * 0.95,
        )

        # No change: balanced or conflicting situation
        no_change = max(
            min(cpu.get("medium", 0.0), mem.get("medium", 0.0), resp.get("low", 0.0)) * 1.0,
            min(cpu.get("medium", 0.0), mem.get("medium", 0.0), resp.get("medium", 0.0)) * 0.9,
            min(max(cpu.get("high", 0.0), mem.get("high", 0.0)), resp.get("low", 0.0)) * 0.8,
            min(cpu.get("low", 0.0), mem.get("low", 0.0), resp.get("high", 0.0)) * 0.7,
        )

        # Normalize
        total = scale_up + scale_down + no_change + 1e-6
        res = {
            "scale_up": scale_up / total,
            "scale_down": scale_down / total,
            "no_change": no_change / total,
        }

        self.logger.debug(f"Rule result: {res}")
        return res

    def influence(self, act: Dict[str, float]) -> float:
        """
        Convert fuzzy rule output into scalar influence [-1, 1].
        Includes hysteresis and confidence weighting.
        """
        up, down, stay = act["scale_up"], act["scale_down"], act["no_change"]
        total = up + down + stay + 1e-6

        # Base direction
        direction = (up - down) / total

        # Confidence: how dominant one decision is
        confidence = max(up, down, stay)
        neutrality = stay / total

        # Hysteresis: reduce influence near neutral
        hysteresis = 1.0 - 0.6 * neutrality

        # Weighted final influence
        influence = direction * hysteresis * confidence

        # Clip
        influence = max(-1.0, min(1.0, influence))

        self.logger.debug(
            f"Influence={influence:.3f} (up={up:.2f}, down={down:.2f}, stay={stay:.2f}, conf={confidence:.2f})"
        )
        return influence

    def decide(self, obs: Dict[str, float]) -> Dict[str, float]:
        fz = self.fuzzify(obs)
        acts = self.apply_rules(fz)
        infl = self.influence(acts)

        # Adaptive decision threshold
        if infl > 0.4:
            rec = "scale_up"
        elif infl < -0.4:
            rec = "scale_down"
        else:
            rec = "maintain"

        result = {"memberships": acts, "influence": infl, "recommendation": rec}
        self.logger.info(f"Decision: {result}")
        return result
