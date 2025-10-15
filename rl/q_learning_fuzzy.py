import numpy as np
from rl import QLearning
from .fuzzy import Fuzzy

class QFuzzyHybrid(QLearning):
    def __init__(self, *args, fuzzy_weight: float = 0.3, **kwargs):
        """
        Hybrid agent combining Q-learning and Fuzzy Logic.
        fuzzy_weight = 0.0 → pure Q-learning
        fuzzy_weight = 1.0 → pure fuzzy
        """
        super().__init__(*args, **kwargs)
        self.agent_type = "Q-Fuzzy"
        self.fuzzy = Fuzzy()
        self.fuzzy_weight = fuzzy_weight

    def get_action(self, observation: dict) -> int:
        q_action = super().get_action(observation)
        fuzzy_action = self.fuzzy.decide(observation)

        if fuzzy_action == -1:
            fuzzy_action_idx = 0
        elif fuzzy_action == 0:
            fuzzy_action_idx = 50
        else:
            fuzzy_action_idx = 99

        if np.random.rand() < self.fuzzy_weight:
            return fuzzy_action_idx
        return q_action
