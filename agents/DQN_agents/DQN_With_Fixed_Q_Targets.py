import os
import torch
from agents.Base_Agent import Base_Agent
from agents.DQN_agents.DQN import DQN

class DQN_With_Fixed_Q_Targets(DQN):
    """A DQN agent that uses an older version of the q_network as the target network"""

    agent_name = "DQN with Fixed Q Targets"
    
    def __init__(self, config):
        DQN.__init__(self, config)
        self.q_network_target = self.create_NN(input_dim=self.state_size, output_dim=self.action_size)
        Base_Agent.copy_model_over(from_model=self.q_network_local, to_model=self.q_network_target)

    def learn(self, experiences=None):
        """运行Q网络的学习迭代"""
        super().learn(experiences=experiences)
        self.soft_update_of_target_network(self.q_network_local, self.q_network_target,
                                           self.hyperparameters["tau"])  # 更新目标网络

    def compute_q_values_for_next_states(self, next_states):
        """计算下一状态Q目标值用于计算loss"""
        Q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next