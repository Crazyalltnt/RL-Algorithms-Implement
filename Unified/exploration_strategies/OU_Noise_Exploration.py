from utilities.OU_Noise import OU_Noise
from exploration_strategies.Base_Exploration_Strategy import Base_Exploration_Strategy

class OU_Noise_Exploration(Base_Exploration_Strategy):
    """Ornstein-Uhlenbeck 噪声过程探索策略"""
    def __init__(self, config):
        super().__init__(config)
        self.noise = OU_Noise(self.config.action_size, self.config.seed, self.config.hyperparameters["mu"],
                              self.config.hyperparameters["theta"], self.config.hyperparameters["sigma"])

    def perturb_action_for_exploration_purposes(self, action_info):
        """干扰智能体的行为以鼓励探索"""
        action = action_info["action"]
        action += self.noise.sample()
        return action

    def add_exploration_rewards(self, reward_info):
        """探索的动作奖励"""
        raise ValueError("Must be implemented")

    def reset(self):
        """重置噪声过程"""
        self.noise.reset()