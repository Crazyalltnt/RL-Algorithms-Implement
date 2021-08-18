from exploration_strategies.Base_Exploration_Strategy import Base_Exploration_Strategy
import numpy as np
import random
import torch

class Epsilon_Greedy_Exploration(Base_Exploration_Strategy):
    """实现epsilon-greedy探索策略"""

    def __init__(self, config):
        super().__init__(config)
        self.notified_that_exploration_turned_off = False  # 是否已通知探索关闭
        # 是否周期性探索
        if "exploration_cycle_episodes_length" in self.config.hyperparameters.keys():
            print("Using a cyclical exploration strategy")
            self.exploration_cycle_episodes_length = self.config.hyperparameters["exploration_cycle_episodes_length"]
        else:
            self.exploration_cycle_episodes_length = None

        # 设置初始随机探索的episodes数
        if "random_episodes_to_run" in self.config.hyperparameters.keys():
            self.random_episodes_to_run = self.config.hyperparameters["random_episodes_to_run"]
            print("Running {} random episodes".format(self.random_episodes_to_run))
        else:
            self.random_episodes_to_run = 0

    def perturb_action_for_exploration_purposes(self, action_info):
        """干扰智能体的行为以鼓励探索"""
        action_values = action_info["action_values"]
        turn_off_exploration = action_info["turn_off_exploration"]
        episode_number = action_info["episode_number"]
        # 通知探索关闭
        if turn_off_exploration and not self.notified_that_exploration_turned_off:
            print(" ")
            print("Exploration has been turned OFF")
            print(" ")
            self.notified_that_exploration_turned_off = True
        epsilon = self.get_updated_epsilon_exploration(action_info)

        # 贪婪
        if (random.random() > epsilon or turn_off_exploration) and (episode_number >= self.random_episodes_to_run):
            return torch.argmax(action_values).item()
        # 探索
        return np.random.randint(0, action_values.shape[1])

    def get_updated_epsilon_exploration(self, action_info, epsilon=1.0):
        """获取随机探索概率，随着episode增加概率衰减"""
        episode_number = action_info["episode_number"]
        epsilon_decay_denominator = self.config.hyperparameters["epsilon_decay_rate_denominator"]  # 概率衰减速度控制因子

        if self.exploration_cycle_episodes_length is None:
            epsilon = epsilon / (1.0 + (episode_number / epsilon_decay_denominator))
        else:
            epsilon = self.calculate_epsilon_with_cyclical_strategy(episode_number)
        return epsilon

    def calculate_epsilon_with_cyclical_strategy(self, episode_number):
        """根据周期性策略计算epsilon"""
        max_epsilon = 0.5
        min_epsilon = 0.001
        increment = (max_epsilon - min_epsilon) / float(self.exploration_cycle_episodes_length / 2)
        cycle = [ix for ix in range(int(self.exploration_cycle_episodes_length / 2))] + [ix for ix in range(
            int(self.exploration_cycle_episodes_length / 2), 0, -1)]
        cycle_ix = episode_number % self.exploration_cycle_episodes_length
        epsilon = max_epsilon - cycle[cycle_ix] * increment
        return epsilon

    def add_exploration_rewards(self, reward_info):
        """鼓励探索的动作奖励"""
        return reward_info["reward"]

    def reset(self):
        """重置噪声过程"""
        pass
