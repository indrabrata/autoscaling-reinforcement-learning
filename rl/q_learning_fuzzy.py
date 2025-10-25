import pickle
from pathlib import Path
from typing import Optional, Dict, Tuple
from logging import Logger
import numpy as np
import urllib3

from .fuzzy import Fuzzy 

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class QFuzzyHybrid:
    """
    Q-Fuzzy Hybrid Agent for intelligent auto-scaling.

    Integrates:
    - Fuzzy logic for state representation and influence bias
    - Q-learning for value-based decision optimization
    - Replica scaling represented as percentage (0–100)
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 0.1,
        epsilon_decay: float = 0.0,
        epsilon_min: float = 0.01,
        created_at: int = 0,
        n_actions: int = 100,
        logger: Optional[Logger] = None,
    ):
        self.agent_type = "QFuzzyHybrid"
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.created_at = created_at
        self.episodes_trained = 0
        self.n_actions = n_actions
        self.q_table: Dict[Tuple[str, str, str], np.ndarray] = {}
        self.fuzzy = Fuzzy(logger=logger)
        self.logger = logger or Logger(__name__)

        self.logger.info("Initialized QFuzzyHybrid agent")
        self.logger.info(f"Agent parameters: {self.__dict__}")

    def get_state_key(self, observation: dict) -> Tuple[str, str, str]:
        """Convert observation to fuzzy state key (discrete labels)."""
        response_time_raw = observation["response_time"]
        if np.isnan(response_time_raw) or response_time_raw is None:
            response_time = 0
        else:
            response_time = int(response_time_raw)
        
        observation["response_time"] = response_time
        fuzzy_state = self.fuzzy.fuzzify(observation)
        cpu_label = max(fuzzy_state["cpu_usage"], key=fuzzy_state["cpu_usage"].get)
        mem_label = max(fuzzy_state["memory_usage"], key=fuzzy_state["memory_usage"].get)
        resp_label = max(fuzzy_state["response_time"], key=fuzzy_state["response_time"].get)
        return (cpu_label, mem_label, resp_label)

    def get_action(self, observation: dict) -> int:
        """
        Choose action using epsilon-greedy + fuzzy influence bias.
        Influence ∈ [-1, 1] adjusts scaling direction and intensity.
        """
        fuzzy_result = self.fuzzy.decide(observation)
        influence = fuzzy_result["influence"]
        state_key = self.get_state_key(observation)

        # Initialize state if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)

        # ε-greedy
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
            self.logger.debug(f"[EXPLORE] Random action {action}%")
        else:
            q_values = self.q_table[state_key]
            action = int(np.argmax(q_values))
            self.logger.debug(f"[EXPLOIT] Best Q action {action}%")

        # Influence biasing
        bias = int((influence + 1) / 2 * (self.n_actions - 1))  # map to 0–99
        final_action = int((action + bias) / 2)
        final_action = np.clip(final_action, 0, self.n_actions - 1)

        return final_action

    def update_q_table(
        self,
        observation: dict,
        action: int,
        reward: float,
        next_observation: dict,
    ):
        """Update Q-table using off-policy Q-learning."""
        state_key = self.get_state_key(observation)
        next_state_key = self.get_state_key(next_observation)

        # Ensure both states exist
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.n_actions)

        # Q-learning update
        best_next_action = np.max(self.q_table[next_state_key])
        old_value = self.q_table[state_key][action]

        self.q_table[state_key][action] += self.learning_rate * (
            reward + self.discount_factor * best_next_action - old_value
        )

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.logger.debug(
            f"Q-update | S={state_key} | A={action} | R={reward:.3f} | "
            f"NewQ={self.q_table[state_key][action]:.3f}"
        )

    def add_episode_count(self, count: int = 1):
        self.episodes_trained += count

    def save_model(self, filepath: str, episode_count: int = 0):
        """Save Q-table and parameters to file."""
        try:
            model_data = {
                "q_table": self.q_table,
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "n_actions": self.n_actions,
                "created_at": self.created_at,
                "episodes_trained": episode_count,
            }
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with Path(filepath).open("wb") as f:
                pickle.dump(model_data, f)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise

    def load_model(self, filepath: str):
        """Load Q-table and parameters from file."""
        try:
            if not Path(filepath).exists():
                raise FileNotFoundError(f"Model file not found: {filepath}")

            with Path(filepath).open("rb") as f:
                model_data = pickle.load(f)  # noqa: S301

            self.q_table = model_data["q_table"]
            self.learning_rate = model_data["learning_rate"]
            self.discount_factor = model_data["discount_factor"]
            self.epsilon = model_data["epsilon"]
            self.epsilon_min = model_data["epsilon_min"]
            self.epsilon_decay = model_data["epsilon_decay"]
            self.n_actions = model_data["n_actions"]
            self.created_at = model_data.get("created_at", None)
            self.episodes_trained = model_data.get("episodes_trained", 0)

            self.logger.info(f"Model loaded from {filepath}")
            self.logger.info(f"Q-table size: {len(self.q_table)} states")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
