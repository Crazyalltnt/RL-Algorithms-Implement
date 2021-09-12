import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

import gym
from utilities.data_structures.Config import Config
from utilities.Utility_Functions import create_cur_run_data_dir
from agents.Trainer import Trainer
from agents.Evaluator import Evaluator

from agents.actor_critic_agents.DDPG import DDPG
from agents.actor_critic_agents.TD3 import TD3
from agents.policy_gradient_agents.PPO import PPO
# from agents.actor_critic_agents.SAC import SAC

config = Config()
config.seed = 1
config.environment = gym.make("MountainCarContinuous-v0")
config.num_episodes_to_run = 450
# config.num_episodes_to_run = 1000

config.show_solution_score = False
config.visualise_individual_results = True
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 10
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = True

config.save_model = False
config.load_model = False

config.hyperparameters = {
    "Policy_Gradient_Agents": {
            "learning_rate": 0.05,
            "linear_hidden_units": [30, 15],
            "final_layer_activation": "TANH",
            "learning_iterations_per_round": 10,
            "discount_rate": 0.9,
            "batch_norm": False,
            "clip_epsilon": 0.2,
            "episodes_per_learning_round": 10,
            "normalise_rewards": True,
            "gradient_clipping_norm": 5,
            "mu": 0.0,
            "theta": 0.15,
            "sigma": 0.2,
            "epsilon_decay_rate_denominator": 1,
            "clip_rewards": False
        },

    "Actor_Critic_Agents": {
            "Actor": {
                "learning_rate": 0.003,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": None,
                "batch_norm": False,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "initialiser": "Xavier"
            },

            "Critic": {
                "learning_rate": 0.02,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": None,
                "batch_norm": False,
                "buffer_size": 1000000,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "initialiser": "Xavier"
            },

        "min_steps_before_learning": 1000, #for SAC only
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0,  # for O-H noise
        "theta": 0.15,  # for O-H noise
        "sigma": 0.25,  # for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 20,
        "learning_updates_per_learning_session": 10,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": True,
        "do_evaluation_iterations": True,
        "clip_rewards": False

    }

}

if __name__ == "__main__":
    # AGENTS = [TD3, DDPG, PPO, SAC]
    AGENTS = [PPO]
    num = 1  # 执行次数
    # config.eval_render = True  # 评估模式
    for i in range(num):
        if config.eval_render:
            config.load_model = True
            config.cur_run_data_dir = r"E:\RL\RLAlogorithmsImplement\results\data\20210825-192621"
            config.model_load_path = r"E:\RL\RLAlogorithmsImplement\results\data\20210825-192621\models\DDPG_3_model.pt"
            evaluator = Evaluator(config, AGENTS)
            evaluator.evaluate_agents()
        else:
            config.cur_run_data_dir = create_cur_run_data_dir()
            config.file_to_save_data_results = config.cur_run_data_dir + "/Mountain_Car_Results_Data.pkl"
            config.file_to_save_results_graph = config.cur_run_data_dir + "/Mountain_Car_Results_Graph.png"
            trainer = Trainer(config, AGENTS)
            trainer.run_games_for_agents()




