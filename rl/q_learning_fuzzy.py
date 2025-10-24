from logging import Logger
from pathlib import Path
import pickle
from typing import Dict, Optional
import numpy as np
from .fuzzy import Fuzzy  


class QFuzzyHybrid:
    """Q-learning agent using fuzzy categorical states with fuzzy action bias."""

    def __init__(
        self,
        n_actions=10,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_start=0.1,
        epsilon_decay=0.99,
        epsilon_min=0.01,
        fuzzy_weight=0.3,
        fuzziness_bandwidth=1.0,
        created_at: int = 0,
        logger: Optional[Logger] = None,
    ):
        self.agent_type = "QFuzzyHybrid"
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.fuzzy_weight = fuzzy_weight
        self.fuzziness_bandwidth = fuzziness_bandwidth
        self.created_at = created_at
        self.episodes_trained = 0
        self.q_table = {}
        self.fuzzy = Fuzzy(logger)
        self.logger = logger


    def get_state_key(self, observation: dict) -> tuple:
        """Wrapper for Trainer compatibility."""
        return self._state_key(observation)        

    def add_episode_count(self):
        """Increment the number of episodes trained."""
        self.episodes_trained += 1


    def _state_key(self, observation: Dict[str, float]):
        """Convert observation to fuzzy categorical state key (no last_action)."""
        fuzzy_state = self.fuzzy.fuzzify(observation)

        cpu_label = max(fuzzy_state["cpu_usage"], key=fuzzy_state["cpu_usage"].get)
        mem_label = max(fuzzy_state["memory_usage"], key=fuzzy_state["memory_usage"].get)
        resp_label = max(fuzzy_state["response_time"], key=fuzzy_state["response_time"].get)
 
        return (cpu_label, mem_label, resp_label)


    def get_action(self, observation: Dict[str, float]) -> int:
        """Choose an action using Q-values + fuzzy bias (epsilon-greedy)."""
        state_key = self._state_key(observation)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)

        q_values = np.array(self.q_table[state_key], dtype=float)
        q_min, q_max = q_values.min(), q_values.max()
        if q_max - q_min < 1e-8:
            q_norm = np.ones_like(q_values) / len(q_values)
        else:
            q_norm = (q_values - q_min) / (q_max - q_min)
            q_norm /= (q_norm.sum() + 1e-12)

        # Fuzzy bias
        fuzzy_out = self.fuzzy.influence(self.fuzzy.apply_rules(self.fuzzy.fuzzify(observation)))
        center = (fuzzy_out + 1.0) / 2.0 * (self.n_actions - 1)
        indices = np.arange(self.n_actions)
        denom = 2 * (self.fuzziness_bandwidth ** 2)
        fuzzy_pref = np.exp(-((indices - center) ** 2) / denom)
        fuzzy_pref /= (fuzzy_pref.sum() + 1e-12)

        # Combine Q + fuzzy
        combined = (1.0 - self.fuzzy_weight) * q_norm + self.fuzzy_weight * fuzzy_pref
        combined /= (combined.sum() + 1e-12)

        # Epsilon-greedy
        if np.random.rand() < self.epsilon:
            action = int(np.random.choice(self.n_actions, p=combined))
        else:
            action = int(np.argmax(combined))

        return action


    def update_q_table(
        self,
        observation: Dict[str, float],
        action: int,
        reward: float,
        next_observation: Dict[str, float],
    ):
        """Update Q-table using Q-learning."""
        state_key = self._state_key(observation)
        next_key = self._state_key(next_observation)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.n_actions)

        target = reward + self.gamma * np.max(self.q_table[next_key])
        self.q_table[state_key][action] += self.lr * (target - self.q_table[state_key][action])

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


    def save_model(self, filepath: str, episode_count: int = 0):
        """Save Q-table and parameters."""
        try:
            model_data = {
                "q_table": self.q_table,
                "learning_rate": self.lr,
                "discount_factor": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "n_actions": self.n_actions,
                "created_at": self.created_at,
                "episodes_trained": episode_count,
                "fuzzy_weight": self.fuzzy_weight,
                "fuzziness_bandwidth": self.fuzziness_bandwidth,
            }
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "wb") as f:
                pickle.dump(model_data, f)
            if self.logger:
                self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save model: {e}")
            raise


    def load_model(self, filepath: str):
        """Load Q-table and parameters."""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.q_table = model_data.get("q_table", {})
        self.lr = model_data.get("learning_rate", self.lr)
        self.gamma = model_data.get("discount_factor", self.gamma)
        self.epsilon = model_data.get("epsilon", self.epsilon)
        self.epsilon_min = model_data.get("epsilon_min", self.epsilon_min)
        self.epsilon_decay = model_data.get("epsilon_decay", self.epsilon_decay)
        self.n_actions = model_data.get("n_actions", self.n_actions)
        self.created_at = model_data.get("created_at", self.created_at)
        self.episodes_trained = model_data.get("episodes_trained", 0)
        self.fuzzy_weight = model_data.get("fuzzy_weight", self.fuzzy_weight)
        self.fuzziness_bandwidth = model_data.get("fuzziness_bandwidth", self.fuzziness_bandwidth)

        if self.logger:
            self.logger.info(f"Model loaded from {filepath}")
            self.logger.info(f"Q-table size: {len(self.q_table)} states")
