import random
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from collections import Counter
from agents.Base_Agent import Base_Agent
from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
from utilities.data_structures.Replay_Buffer import Replay_Buffer

class DQN(Base_Agent):
    """Deep Q-Learning"""

    agent_name = "DQN"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.agent_round = config.agent_round
        self.memory = Replay_Buffer(self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"], config.seed, self.device)
        self.q_network_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size)
        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(),
                                              lr=self.hyperparameters["learning_rate"], eps=1e-4)
        self.exploration_strategy = Epsilon_Greedy_Exploration(config)

    def reset_game(self):
        """重置游戏信息，开始新的episode"""
        super().reset_game()
        self.update_learning_rate(self.hyperparameters["learning_rate"], self.q_network_optimizer)

    def step(self):
        """在游戏中运行一个episode，每n步进行一次迭代学习"""
        while not self.done:
            if self.config.train_render:
                self.config.environment.render()  # 可视化
            self.action = self.pick_action()  # 选择动作
            self.conduct_action(self.action)  # 执行一步动作
            if self.time_for_q_network_to_learn():  # 每经过n步并且经验池样本足够才会学习 dqn=1
                for _ in range(self.hyperparameters["learning_iterations"]):  # 学习迭代次数，dqn=1
                    self.learn()
            self.save_experience()  # 保存经验到经验池
            self.state = self.next_state
            self.global_step_number += 1
        self.episode_number += 1

    def pick_action(self, state=None):
        """使用原始Q网络和epsilon贪婪策略选择动作"""
        # PyTorch 只接受mini-batch而不接受单个观测，所以我们必须使用 unsqueeze 来添加
        # 一个“假”维度，使它成为一个min-batch而不是一个一维的观测
        if state is None: state = self.state
        if isinstance(state, np.int64) or isinstance(state, int): state = np.array([state])
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if len(state.shape) < 2: state = state.unsqueeze(0)
        self.q_network_local.eval()  # 将网络设为评估模式
        with torch.no_grad():  # 不跟踪梯度
            action_values = self.q_network_local(state)  # 动作价值列表
        self.q_network_local.train()   # 将网络设为训练模式
        # 选择动作
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
                                                                                    "turn_off_exploration": self.turn_off_exploration,
                                                                                    "episode_number": self.episode_number})
        self.logger.info("Q values {} -- Action chosen {}".format(action_values, action))
        return action

    def learn(self, experiences=None):
        """运行Q网络的学习迭代"""
        if experiences is None: states, actions, rewards, next_states, dones = self.sample_experiences()  # 采样经验
        else: states, actions, rewards, next_states, dones = experiences
        loss = self.compute_loss(states, next_states, rewards, actions, dones)

        actions_list = [action_X.item() for action_X in actions ]

        self.logger.info("Action counts {}".format(Counter(actions_list)))
        # 反向传播优化更新参数
        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, loss, self.hyperparameters["gradient_clipping_norm"])

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """计算loss"""
        with torch.no_grad():
            Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions)
        # loss = r + γmaxQ(s',a') - Q(s,a)
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss

    def compute_q_targets(self, next_states, rewards, dones):
        """计算目标Q值用于计算loss"""
        Q_targets_next = self.compute_q_values_for_next_states(next_states)
        Q_targets = self.compute_q_values_for_current_states(rewards, Q_targets_next, dones)
        return Q_targets

    def compute_q_values_for_next_states(self, next_states):
        """计算下一状态Q目标值用于计算loss"""
        # 计算 q-learning 更新时下一状态选择最大价值动作 maxQ(s',a')
        Q_targets_next = self.q_network_local(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        """计算当前状态Q目标值用于计算loss"""
        # 计算Q目标值 r + γmaxQ(s',a')
        Q_targets_current = rewards + (self.hyperparameters["discount_rate"] * Q_targets_next * (1 - dones))
        return Q_targets_current

    def compute_expected_q_values(self, states, actions):
        """计算Q期望值用于计算loss"""
        # 计算 Q(s,a)，必须将动作转换为long才能用作gather索引
        Q_expected = self.q_network_local(states).gather(1, actions.long())
        return Q_expected

    def locally_save_policy(self):
        """保存策略模型"""
        model_save_path = self.cur_run_data_dir + "/models"
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        torch.save(self.q_network_local.state_dict(), model_save_path + "/{}_{}_local_network.pt".format(self.agent_name, self.agent_round))

    def time_for_q_network_to_learn(self):
        """返回布尔值，指示是否已采取足够的步骤来学习，并且重放缓冲区中是否有足够的经验可供学习"""
        return self.right_amount_of_steps_taken() and self.enough_experiences_to_learn_from()

    def right_amount_of_steps_taken(self):
        """返回布尔值，指示是否已采取足够的步骤来开始学习"""
        return self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def sample_experiences(self):
        """从记忆缓冲区中随机抽取经验样本"""
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        return states, actions, rewards, next_states, dones