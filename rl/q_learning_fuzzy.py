# inside your rl.py where QFuzzyHybrid is defined
import numpy as np
from rl import QLearning
from .fuzzy import Fuzzy

class QFuzzyHybrid(QLearning):
    def __init__(self, *args, fuzzy_weight: float = 0.3, fuzziness_bandwidth: float = 8.0, **kwargs):
        """
        fuzzy_weight: how strongly fuzzy influences action selection (0..1).
            0.0 -> pure Q, 1.0 -> pure fuzzy (but we still use fuzzy to construct a distribution)
        fuzziness_bandwidth: controls spread of fuzzy preference across actions (higher -> wider)
        """
        super().__init__(*args, **kwargs)
        self.agent_type = "Q-Fuzzy"
        self.fuzzy = Fuzzy()
        self.fuzzy_weight = float(fuzzy_weight)
        self.fuzziness_bandwidth = float(fuzziness_bandwidth)

    def _state_key_for_table(self, observation: dict):
        # reuse parent's conversion to state key function
        return self.get_state_key(observation)

    def get_action(self, observation: dict) -> int:
        """
        Blend Q-values with a fuzzy-derived preference distribution.

        Steps:
        1. get state key and ensure q_table entry exists
        2. get Q-values vector (shape n_actions)
        3. create fuzzy influence in [-1,1], map to preferred action center (index)
           - influence < 0 => bias toward lower indices (scale down)
           - influence > 0 => bias toward higher indices (scale up)
           - influence == 0 => center = middle
        4. build a gaussian-like fuzzy score around that center and normalize
        5. final_score = (1 - fuzzy_weight) * q_norm + fuzzy_weight * fuzzy_pref
        6. choose argmax(final_score) (or sample proportional to softmax if you prefer exploration)
        """

        state_key = self._state_key_for_table(observation)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)

        q_values = np.array(self.q_table[state_key], dtype=float)

        # normalize Q-values to 0..1 (min-max). If all equal, fall back to uniform.
        q_min, q_max = q_values.min(), q_values.max()
        if q_max - q_min < 1e-8:
            q_norm = np.ones_like(q_values) / len(q_values)
        else:
            q_norm = (q_values - q_min) / (q_max - q_min)
            q_norm = q_norm / (q_norm.sum() + 1e-12)

        # Get fuzzy influence in [-1,1]
        fuzzy_out = self.fuzzy.decide(observation)
        if isinstance(fuzzy_out, dict):
            influence = float(fuzzy_out.get("influence", 0.0))
            memberships = fuzzy_out.get("memberships", {})
        else:
            # backward compatibility if decide returned float
            influence = float(fuzzy_out)
            memberships = {}

        # Map influence to an action-center index across 0..n_actions-1.
        center = (influence + 1.0) / 2.0 * (self.n_actions - 1)  # -1 -> 0, 0 -> mid, +1 -> n_actions-1

        # Create Gaussian-like preference distribution around center
        indices = np.arange(self.n_actions)
        # bandwidth controls how sharp / wide the bias is; lower -> sharper
        denom = 2 * (self.fuzziness_bandwidth ** 2)
        fuzzy_pref = np.exp(-((indices - center) ** 2) / denom)
        fuzzy_pref = fuzzy_pref / (fuzzy_pref.sum() + 1e-12)

        # Combine distributions
        w = float(self.fuzzy_weight)
        combined = (1.0 - w) * q_norm + w * fuzzy_pref
        combined = combined / (combined.sum() + 1e-12)

        # Epsilon-greedy: maintain QLearning's exploration but sample from combined distribution when exploring
        if np.random.rand() < self.epsilon:
            # sample action according to combined distribution
            action = int(np.random.choice(self.n_actions, p=combined))
        else:
            action = int(np.argmax(combined))

        # Do NOT change epsilon decay here (keep in update function)
        return action
