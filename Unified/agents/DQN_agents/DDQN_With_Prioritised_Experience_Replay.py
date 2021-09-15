import torch
import torch.nn.functional as F
from agents.DQN_agents.DDQN import DDQN
from utilities.data_structures.Prioritised_Replay_Buffer import Prioritised_Replay_Buffer

class DDQN_With_Prioritised_Experience_Replay(DDQN):
    """A DQN agent with prioritised experience replay"""
    
    agent_name = "DDQN with Prioritised Replay"

    def __init__(self, config):
        DDQN.__init__(self, config)
        self.memory = Prioritised_Replay_Buffer(self.hyperparameters, config.seed, self.device)

    def learn(self):
        """在以优先方式从重放缓冲区采样后，为 Q 网络运行学习迭代"""
        sampled_experiences, importance_sampling_weights = self.memory.sample()
        states, actions, rewards, next_states, dones = sampled_experiences
        loss, td_errors = self.compute_loss_and_td_errors(states, next_states, rewards, actions, dones, importance_sampling_weights)
        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, loss, self.hyperparameters["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.q_network_local, self.q_network_target, self.hyperparameters["tau"])
        self.memory.update_td_errors(td_errors.squeeze(1))

    def save_experience(self):
        """保存包括 td_error 在内的最新经验"""
        max_td_error_in_experiences = self.memory.give_max_td_error() + 1e-9
        self.memory.add_experience(max_td_error_in_experiences, self.state, self.action, self.reward, self.next_state, self.done)

    def compute_loss_and_td_errors(self, states, next_states, rewards, actions, dones, importance_sampling_weights):
        """计算本地 Q 网络的损失。 它根据来自优先重放缓冲区的重要性采样权重来衡量每个观察损失"""
        Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        loss = loss * importance_sampling_weights
        loss = torch.mean(loss)
        td_errors = Q_targets.data.cpu().numpy() - Q_expected.data.cpu().numpy()
        return loss, td_errors