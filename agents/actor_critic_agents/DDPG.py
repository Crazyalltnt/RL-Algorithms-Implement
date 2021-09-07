import os
import time
import numpy as np
import torch
import torch.nn.functional as functional
from torch import optim
from agents.Base_Agent import Base_Agent
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from exploration_strategies.OU_Noise_Exploration import OU_Noise_Exploration

class DDPG(Base_Agent):
    """A DDPG Agent"""

    agent_name = "DDPG"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.agent_round = config.agent_round
        self.hyperparameters = config.hyperparameters
        self.critic_local = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic")
        self.critic_target = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)

        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed, self.device)
        self.actor_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Actor")
        self.actor_target = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Actor")
        Base_Agent.copy_model_over(self.actor_local, self.actor_target)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        self.exploration_strategy = OU_Noise_Exploration(self.config)

    def step(self):
        """运行一个episode"""
        while not self.done:
            # print("State ", self.state.shape)
            self.action = self.pick_action()
            self.conduct_action(self.action)
            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    states, actions, rewards, next_states, dones = self.sample_experiences()
                    self.critic_learn(states, actions, rewards, next_states, dones)
                    self.actor_learn(states)
            self.save_experience()
            self.state = self.next_state
            self.global_step_number += 1
        self.episode_number += 1

    def sample_experiences(self):
        """采样经验"""
        return self.memory.sample()

    def pick_action(self, state=None):
        """使用原始 actor 网络选择一个动作，然后添加一些噪声确保探索"""
        if state is None: state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action": action})
        return action.squeeze(0)

    def critic_learn(self, states, actions, rewards, next_states, dones):
        """对 critic 网络运行一次学习迭代"""
        loss = self.compute_loss(states, next_states, rewards, actions, dones)
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, loss, self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target, self.hyperparameters["Critic"]["tau"])

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """计算 critic 的 loss"""
        with torch.no_grad():
            critic_targets = self.compute_critic_targets(next_states, rewards, dones)
        critic_expected = self.compute_expected_critic_values(states, actions)
        loss = functional.mse_loss(critic_expected, critic_targets)
        return loss

    def compute_critic_targets(self, next_states, rewards, dones):
        """计算 critic 的 target value"""
        critic_targets_next = self.compute_critic_values_for_next_states(next_states)
        critic_targets = self.compute_critic_values_for_current_states(rewards, critic_targets_next, dones)
        return critic_targets

    def compute_critic_values_for_next_states(self, next_states):
        """计算下一个状态的 critic value"""
        # Q'(s',a',θ^(Q‘))
        with torch.no_grad():
            actions_next = self.actor_target(next_states)
            critic_targets_next = self.critic_target(torch.cat((next_states, actions_next), 1))
        return critic_targets_next

    def compute_critic_values_for_current_states(self, rewards, critic_targets_next, dones):
        """计算当前状态的 critic value"""
        # r + γQ'(s',a',θ^(Q‘))
        critic_targets_current = rewards + (self.hyperparameters["discount_rate"] * critic_targets_next * (1.0 - dones))
        return critic_targets_current

    def compute_expected_critic_values(self, states, actions):
        """计算 critic 期望值"""
        # Q'(s,a,θ^(Q))
        critic_expected = self.critic_local(torch.cat((states, actions), 1))
        return critic_expected

    def time_for_critic_and_actor_to_learn(self):
        """返回布尔值，指示是否有足够的经验可供学习，并且是时候为 actor 和 critic 学习了"""
        return self.enough_experiences_to_learn_from() and self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def actor_learn(self, states):
        """对 critic 网络运行一次学习迭代"""
        if self.done: #we only update the learning rate at end of each episode
            self.update_learning_rate(self.hyperparameters["Actor"]["learning_rate"], self.actor_optimizer)
        actor_loss = self.calculate_actor_loss(states)
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.actor_local, self.actor_target, self.hyperparameters["Actor"]["tau"])

    def calculate_actor_loss(self, states):
        """计算 actor 网络的 loss"""
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(torch.cat((states, actions_pred), 1)).mean()
        return actor_loss
    
    def locally_save_policy(self):
        """保存策略模型"""
        model_save_path = self.cur_run_data_dir + "/models"
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        
        state = {'critic_local': self.critic_local.state_dict(),
                 'critic_target': self.critic_target.state_dict(),
                 'critic_optimizer': self.critic_optimizer.state_dict(),
                 'actor_local': self.actor_local.state_dict(),
                 'actor_target': self.actor_target.state_dict(),
                 'actor_optimizer': self.actor_optimizer.state_dict(),
                 'exploration_strategy': self.exploration_strategy}
        torch.save(state, model_save_path + "/{}_{}_model.pt".format(self.agent_name, self.agent_round))
        print("The model was saved successfully")
        self.terminal_logger.info("The model was saved successfully")

    def locally_load_policy(self):
        """加载已有模型"""
        model = torch.load(self.config.model_load_path)
        self.critic_local.load_state_dict(model['critic_local'])
        self.critic_target.load_state_dict(model['critic_target'])
        self.critic_optimizer.load_state_dict(model['critic_optimizer'])
        self.actor_local.load_state_dict(model['actor_local'])
        self.actor_target.load_state_dict(model['actor_target'])
        self.actor_optimizer.load_state_dict(model['actor_optimizer'])
        self.exploration_strategy = model['exploration_strategy']
        print("The model was loaded successfully")
        self.terminal_logger.info("The model was loaded successfully")

    def eval_agent(self):
        """评估智能体"""
        def do():
            self.config.environment.render()  # 可视化
            state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
            self.actor_local.eval()
            with torch.no_grad():
                action = self.actor_local(state).cpu().data.numpy()
            self.action = action.squeeze(0)
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
