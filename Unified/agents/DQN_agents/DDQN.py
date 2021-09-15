from agents.DQN_agents.DQN_With_Fixed_Q_Targets import DQN_With_Fixed_Q_Targets

class DDQN(DQN_With_Fixed_Q_Targets):
    """A double DQN agent"""

    agent_name = "DDQN"

    def __init__(self, config):
        DQN_With_Fixed_Q_Targets.__init__(self, config)

    def compute_q_values_for_next_states(self, next_states):
        """计算下一状态Q目标值用于计算loss。
        Double DQN 使用原始 Q 网络选择最大 Q 值动作，然后用目标网络估计算 Q 目标值，这有助于阻止网络高估 Q 值"""
        max_action_indexes = self.q_network_local(next_states).detach().argmax(1)
        Q_targets_next = self.q_network_target(next_states).gather(1, max_action_indexes.unsqueeze(1))
        return Q_targets_next 
            

