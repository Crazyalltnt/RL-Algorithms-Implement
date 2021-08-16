import torch
import torch.nn as nn
import torch.optim as optimizer
import random
import numpy as np

from networks.critic import Critic

import os
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from common.buffer import Replay_buffer as buffer

def get_trajectory_property():
    return ["action"]

class DQN(object):
    def __init__(self, args):
        self.state_dim = args.obs_space
        self.action_dim = args.action_space

        self.hidden_size = args.hidden_size
        self.lr = args.c_lr  # 学习率
        self.buffer_size = args.buffer_capacity  # 经验池容量
        self.batch_size = args.gamma  # 奖励折扣因子

        self.critic_eval = Critic(self.state_dim, self.action_dim, self.hidden_size)  # eval-动作价值网络
        self.critic_target = Critic(self.state_dim, self.action_dim, self.hidden_size)  # target-动作价值网络
        self.optimizer = optimizer.Adam(self.critic_eval.parameters(), lr=self.lr)

        # 探索
        self.eps = args.epsilon  # 探索概率
        self.eps_end = args.epsilon_end  # 最低探索概率
        self.eps_delay = 1 / (args.max_episodes * 100)  # 探索概率衰减因子

        # 更新target网络
        self.learn_step_counter = 0  # 学习步数计数器
        self.target_replace_iter = args.target_replace  # target网络更新步长

        trajectory_property = get_trajectory_property()
        self.memory = buffer(self.buffer_size, trajectory_property)
        self.memory.init_item_buffers()
    
    def choose_action(self, observation, train=True):
        """
        动作选择
        """
        inference_output = self.inference(observation, train)
        if train:
            self.add_experience(inference_output)
        return inference_output

    def inference(self, observation, train):
        """
        运行模式：训练 / 在线推断
        """
        if train:
            self.eps = max(self.eps_end, self.eps - self.eps_delay)
            if random.random() < self.eps:
                action = random.randrange(self.action_dim)
            else:
                observation = torch.tensor(observation, dtype=torch.float).view(1, -1)
                action = torch.argmax(self.critic_eval(observation)).item()
        else:
            observation = torch.tensor(observation, dtype=torch.float).view(1, -1)
            action = torch.argmax(self.critic_eval(observation)).item()
        
        return {"action": action}
    
    def add_experience(self, output):
        agent_id = 0
        for k, v in output.items():
            self.memory.insert(k, agent_id, v)
    
    def learn(self):
        """
        训练，学习参数
        """
        data_length = len(self.memory.item_buffers["rewards"].data)
        if data_length < self.buffer_size:
            return
        
        data = self.memory.sample(self.batch_size)

        transitions = {
            "o_0": np.array(data["states"]),
            "o_next_0": np.array(data["states_next"]),
            "r_0": np.array(data["rewards"]).reshape(-1, 1),
            "a_0": np.array(data["action"]),
            "d_0": np.array(data["dones"]).reshape(-1, 1),
        }

        obs = torch.tensor(transitions["o_0"], dtype=torch.float)
        obs_ = torch.tensor(transitions["o_next_0"], dtype=torch.float)
        action = torch.tensor(transitions["a_0"], dtype=torch.long).view(self.batch_size, -1)
        reward = torch.tensor(transitions["r_0"], dtype=torch.float).squeeze()
        done = torch.tensor(transitions["d_0"], dtype=torch.float).squeeze()

        q_eval = self.critic_eval(obs).gather(1, action)
        q_next = self.critic_target(obs_).detach()
        q_target = (reward * self.gamma * q_next.max(1)[0] * (1 - done)).view(self.batch_size, 1)
        loss_fn = nn.MSELoss()
        loss = loss_fn(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新target网络参数
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.critic_target.load_state_dict(self.critic_eval.state_dict())
        
        self.learn_step_counter += 1

        return loss
    
    def save(self, save_path, episode):
        """
        保存模型
        """
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        torch.save(self.critic_eval.state_dict(), model_critic_path)
    
    def load(self, file):
        self.critic_eval.load_state_dict(torch.load(file))