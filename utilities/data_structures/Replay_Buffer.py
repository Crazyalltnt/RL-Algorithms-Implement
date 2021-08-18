from collections import namedtuple, deque
import random
import torch
import numpy as np

class Replay_Buffer(object):
    """经验重放缓冲区，存储过去的经验，然后智能体可以将其作为训练数据"""
    
    def __init__(self, buffer_size, batch_size, seed, device=None):
        self.memory = deque(maxlen=buffer_size)  # 经验池
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])  # 设置经验元组结构
        self.seed = random.seed(seed)
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add_experience(self, states, actions, rewards, next_states, dones):
        """将经验添加到重放缓冲区中"""
        if type(dones) == list:  # 批量经验
            assert type(dones[0]) != list, "A done shouldn't be a list"  # 断言，done不能是列表
            experiences = [self.experience(state, action, reward, next_state, done)
                           for state, action, reward, next_state, done in
                           zip(states, actions, rewards, next_states, dones)]
            self.memory.extend(experiences)
        else:  # 单个经验
            experience = self.experience(states, actions, rewards, next_states, dones)
            self.memory.append(experience)
   
    def sample(self, num_experiences=None, separate_out_data_types=True):
        """从重放缓冲区中随机抽取经验样本"""
        experiences = self.pick_experiences(num_experiences)
        if separate_out_data_types:  # 是否分离数据类型
            states, actions, rewards, next_states, dones = self.separate_out_data_types(experiences)
            return states, actions, rewards, next_states, dones
        else:
            return experiences
            
    def separate_out_data_types(self, experiences):
        """将采样经验转化为 PyTorch 神经网络的正确格式"""
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float().to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def pick_experiences(self, num_experiences=None):
        """选择经验"""
        if num_experiences is not None: batch_size = num_experiences
        else: batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)
