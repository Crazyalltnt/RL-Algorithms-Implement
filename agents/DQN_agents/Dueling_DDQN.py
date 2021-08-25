import numpy as np
import time
import torch
from torch import optim
from agents.Base_Agent import Base_Agent
from agents.DQN_agents.DDQN import DDQN

class Dueling_DDQN(DDQN):
    """A dueling double DQN agent as described in the paper http://proceedings.mlr.press/v48/wangf16.pdf"""
    
    agent_name = "Dueling DDQN"

    def __init__(self, config):
        DDQN.__init__(self, config)
        print("duel init")
        self.q_network_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size + 1)  # 输出增加一个状态价值函数
        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(), lr=self.hyperparameters["learning_rate"], eps=1e-4)
        self.q_network_target = self.create_NN(input_dim=self.state_size, output_dim=self.action_size + 1)
        Base_Agent.copy_model_over(from_model=self.q_network_local, to_model=self.q_network_target)

    def pick_action(self, state=None):
        """使用原始 Q 网络和 epsilon 贪婪策略来选择一个动作
        
        PyTorch 只接受mini-batch而不接受单个观测，所以我们必须使用 unsqueeze 来添加
        一个“假”维度，使它成为一个min-batch而不是一个一维的观测
        """
        if state is None: state = self.state
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if len(state.shape) < 2: state = state.unsqueeze(0)
        self.q_network_local.eval()
        with torch.no_grad():
            action_values = self.q_network_local(state)
            action_values = action_values[:, :-1]  # 我们将最后一个输出元素视为状态值并将其余元素视为优势
        self.q_network_local.train()
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
                                                                                    "turn_off_exploration": self.turn_off_exploration,
                                                                                    "episode_number": self.episode_number})
        return action

    def compute_q_values_for_next_states(self, next_states):
        """计算下一个状态的 q_values，我们将使用它来创建 loss 以训练 Q 网络。 
        双 DQN 使用原始 Q 网络选择最大 q_value 动作，然后使用目标 Q 网络计算 q_value，这将有助于阻止网络高估 q 值
        """
        max_action_indexes = self.q_network_local(next_states)[:, :-1].detach().argmax(1)
        dueling_network_output = self.q_network_target(next_states)
        q_values = self.calculate_dueling_q_values(dueling_network_output)
        Q_targets_next = q_values.gather(1, max_action_indexes.unsqueeze(1))
        return Q_targets_next

    def compute_expected_q_values(self, states, actions):
        """计算我们将用于创建 loss 以训练 Q 网络的期望 q_values"""
        dueling_network_output = self.q_network_local(states)
        q_values = self.calculate_dueling_q_values(dueling_network_output)
        Q_expected = q_values.gather(1, actions.long())
        return Q_expected

    def calculate_dueling_q_values(self, duelling_q_network_output):
        """使用dueling网络架构计算 q_values"""
        state_value = duelling_q_network_output[:, -1]
        avg_advantage = torch.mean(duelling_q_network_output[:, :-1], dim=1)
        q_values = state_value.unsqueeze(1) + (duelling_q_network_output[:, :-1] - avg_advantage.unsqueeze(1))
        return q_values

    # def locally_load_policy(self):
        """加载已有模型"""
        print("dueling")
        model = torch.load(self.config.model_load_path)
        self.q_network_local.load_state_dict(model['model'])
        self.q_network_optimizer.load_state_dict(model['optimizer'])
        self.exploration_strategy = model['exploration_strategy']
        print("The model was loaded successfully")
        self.terminal_logger.info("The model was loaded successfully")

    def eval_agent(self):
        """评估智能体"""
        def do():
            self.config.environment.render()  # 可视化
            state = self.state
            if isinstance(state, np.int64) or isinstance(state, int): state = np.array([state])
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            if len(state.shape) < 2: state = state.unsqueeze(0)
            self.q_network_local.eval()  # 将网络设为评估模式
            with torch.no_grad():  # 不跟踪梯度
                action_values = self.q_network_local(state)  # 动作价值列表
                action_values = action_values[:, :-1]  # 我们将最后一个输出元素视为状态值并将其余元素视为优势
            # self.q_network_local.train()   # 将网络设为训练模式
            self.turn_off_any_epsilon_greedy_exploration()
            # 选择动作
            self.action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
                                                                                        "turn_off_exploration": self.turn_off_exploration,
                                                                                        "episode_number": self.episode_number})
            self.conduct_action(self.action)  # 执行一步动作
            self.state = self.next_state
            time.sleep(0.01)

        rounds = 10
        steps = 300
        for round in range(rounds):
            print("round " + str(round))
            super().reset_game()
            do_step = 0
            while not self.done:
                do()
                do_step += 1
                print("do_step: " + str(do_step))
            for step in range(steps):
                do()
                print("more step: " + str(step))
            
        self.environment.env.close()