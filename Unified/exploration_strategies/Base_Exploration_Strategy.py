class Base_Exploration_Strategy(object):
    """智能体探索的基础抽象类，每个探索策略都必须继承和实现以下函数
    """

    def __init__(self, config):
        self.config = config

    def perturb_action_for_exploration_purposes(self, action_info):
        """干扰智能体的行为以鼓励探索"""
        raise ValueError("Must be implemented")

    def add_exploration_rewards(self, reward_info):
        """鼓励探索的动作奖励"""
        raise ValueError("Must be implemented")

    def reset(self):
        """重置噪声过程"""
        raise ValueError("Must be implemented")