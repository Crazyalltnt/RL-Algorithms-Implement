import copy
import random
import logging
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import wrappers


class Evaluator(object):
    """使用的智能体运行游戏。 可选择将结果可视化并保存"""
    
    def __init__(self, config, agents):
        self.config = config
        self.agents = agents
        self.config.terminal_logger = logging.getLogger(__name__)
        self.terminal_logger = self.config.terminal_logger  # 终端日志管理器
        self.agent_to_agent_group = self.create_agent_to_agent_group_dictionary()
        self.config.load_model = True
    
    def create_agent_to_agent_group_dictionary(self):
        """创建一个将智能体映射到更广泛的智能体组的字典"""
        agent_to_agent_group_dictionary = {
            "DQN": "DQN_Agents",
            "DQN-HER": "DQN_Agents",
            "DDQN": "DQN_Agents",
            "DDQN with Prioritised Replay": "DQN_Agents",
            "DQN with Fixed Q Targets": "DQN_Agents",
            "Duelling DQN": "DQN_Agents",
            "PPO": "Policy_Gradient_Agents",
            "REINFORCE": "Policy_Gradient_Agents",
            "Genetic_Agent": "Stochastic_Policy_Search_Agents",
            "Hill Climbing": "Stochastic_Policy_Search_Agents",
            "DDPG": "Actor_Critic_Agents",
            "DDPG-HER": "Actor_Critic_Agents",
            "TD3": "Actor_Critic_Agents",
            "A2C": "Actor_Critic_Agents",
            "A3C": "Actor_Critic_Agents",
            "h-DQN": "h_DQN",
            "SNN-HRL": "SNN_HRL",
            "HIRO": "HIRO",
            "SAC": "Actor_Critic_Agents",
            "HRL": "HRL",
            "Model_HRL": "HRL",
            "DIAYN": "DIAYN",
            "Dueling DDQN": "DQN_Agents"
        }
        return agent_to_agent_group_dictionary

    def evaluate_agents(self):
        for agent_number, agent_class in enumerate(self.agents):
            self.evaluate_agent(agent_number + 1, agent_class)  # 评估每个智能体
            print("-----------------------------------------------------------------------------------")
            print("-----------------------------------------------------------------------------------")
            
    
    def evaluate_agent(self, agent_number, agent_class):
        agent_config = copy.deepcopy(self.config)
        agent_name = agent_class.agent_name
        agent_group = self.agent_to_agent_group[agent_name]  # 智能体分类：dqn/pg/ac/...
        agent_config.agent_round = 1
        agent_config.hyperparameters = agent_config.hyperparameters[agent_group]  # 智能体参数
        agent = agent_class(agent_config)
        self.environment_name = agent.environment_title  # 环境名
        print("ENVIRONMENT NAME: {}".format(self.environment_name))
        
        print("AGENT NAME: {}".format(agent_name))  # 智能体类名
        print("\033[1m" + "{}: {}".format(agent_number, agent_name) + "\033[0m", flush=True)
        agent.eval_agent()