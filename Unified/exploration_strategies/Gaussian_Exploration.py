from exploration_strategies.Base_Exploration_Strategy import Base_Exploration_Strategy
import torch
from torch.distributions.normal import Normal

class Gaussian_Exploration(Base_Exploration_Strategy):

    """高斯噪声探索策略"""
    def __init__(self, config):
        super().__init__(config)
        self.action_noise_std = self.config.hyperparameters["action_noise_std"]
        self.action_noise_distribution = Normal(torch.Tensor([0.0]), torch.Tensor([self.action_noise_std]))
        self.action_noise_clipping_range = self.config.hyperparameters["action_noise_clipping_range"]


    def perturb_action_for_exploration_purposes(self, action_info):
        """干扰智能体的行为以鼓励探索"""
        action = action_info["action"]
        action_noise = self.action_noise_distribution.sample(sample_shape=action.shape)
        action_noise = action_noise.squeeze(-1)
        clipped_action_noise = torch.clamp(action_noise, min=-self.action_noise_clipping_range,
                                           max=self.action_noise_clipping_range)
        action += clipped_action_noise
        return action

    def add_exploration_rewards(self, reward_info):
        """探索奖励"""
        raise ValueError("Must be implemented")

    def reset(self):
        """重置高斯噪声"""
        pass

