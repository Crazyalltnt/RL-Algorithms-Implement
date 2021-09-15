import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
from agents.Base_Agent import Base_Agent

class REINFORCE(Base_Agent):
    """REINFORCE algorithm"""

    agent_name = "REINFORCE"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.policy = self.create_NN(input_dim=self.state_size, output_dim=self.action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.hyperparameters["learning_rate"])
        self.episode_rewards = []
        self.episode_log_probabilities = []

    def reset_game(self):
        """重置游戏信息，开始新的episode"""
        self.state = self.environment.reset()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        self.episode_rewards = []
        self.episode_log_probabilities = []
        self.episode_step_number = 0

    def step(self):
        """在游戏中运行一个episode，如果需要，进行迭代学习"""
        while not self.done:
            self.pick_and_conduct_action_and_save_log_probabilities()
            self.update_next_state_reward_done_and_score()
            self.store_reward()
            if self.time_to_learn():
                self.actor_learn()
            self.state = self.next_state  # 这是为下一次迭代设置状态
            self.episode_step_number += 1
        self.episode_number += 1

    def pick_and_conduct_action_and_save_log_probabilities(self):
        """选择然后执行动作。 然后保存它执行的动作对数概率分布以供以后学习使用"""
        action, log_probabilities = self.pick_action_and_get_log_probabilities()
        self.store_log_probabilities(log_probabilities)
        self.store_action(action)
        self.conduct_action()

    def pick_action_and_get_log_probabilities(self):
        """选择动作，然后计算给定策略的对数概率分布"""
        state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        action_probabilities = self.policy.forward(state).cpu()
        action_distribution = Categorical(action_probabilities)  # 这将创建一个分布以从中采样
        action = action_distribution.sample()
        return action.item(), action_distribution.log_prob(action)

    def store_log_probabilities(self, log_probabilities):
        """存储选择动作的对数概率以供以后学习"""
        self.episode_log_probabilities.append(log_probabilities)

    def store_action(self, action):
        """存储选择的动作"""
        self.action = action

    def store_reward(self):
        """存储选择的奖励"""
        self.episode_rewards.append(self.reward)

    def actor_learn(self):
        """运行策略的学习迭代"""
        total_discounted_reward = self.calculate_episode_discounted_reward()
        policy_loss = self.calculate_policy_loss_on_episode(total_discounted_reward)
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def calculate_episode_discounted_reward(self):
        """计算episode的累积折扣回报"""
        discounts = self.hyperparameters["discount_rate"] ** np.arange(len(self.episode_rewards))
        total_discounted_reward = np.dot(discounts, self.episode_rewards)
        return total_discounted_reward

    def calculate_policy_loss_on_episode(self, total_discounted_reward):
        """计算一个episode的loss"""
        policy_loss = []
        for log_prob in self.episode_log_probabilities:
            policy_loss.append(-log_prob * total_discounted_reward)
        policy_loss = torch.cat(policy_loss).sum()  # 我们需要将整个小批量的损失相加以获得 1 个整体损失
        return policy_loss

    def time_to_learn(self):
        """告诉我们是否是算法学习的时间。 使用 REINFORCE，我们只在每一个episode结束时学习，所以这只会返回该集是否结束"""
        return self.done
