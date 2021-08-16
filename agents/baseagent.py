class Baseagent(object):
    """
    智能体基类
    """
    def __init__(self, args):
        self.args = args
        self.agent = list()
    
    def choose_action_to_env(self, observation):
        """
        接口
        """
        observation_copy = observation.copy()
        obs = observation_copy["obs"]
        agent_id = observation_copy["controlled_player_index"]
        action_from_algo = self.agent[agent_id].choose_action(obs, train=True)
        action_to_env = self.action_from_algo_to_env(action_from_algo)
        return action_to_env
    
    def learn(self):
        """
        更新算法
        """
        for agent in self.agent:
            agent.learn()
    
    def save(self, save_path, episode):
        """
        保存模型
        """
        for agent in self.agent:
            agent.save(save_path, episode)
    
    def load(self, file):
        """
        加载模型
        """
        for agent in self.agent:
            agent.load(file)
