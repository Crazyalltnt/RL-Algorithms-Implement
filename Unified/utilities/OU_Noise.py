import numpy as np
import random
import copy

class OU_Noise(object):
    """Ornstein-Uhlenbeck process / 奥恩斯坦-乌伦贝克过程
    
    连续空间：dx_t = θ * (μ - x_t) * dt + σ * dW_t
    离散空间 x_t - x_(t-1) = θ * (μ - x_(t - 1)) + σ * W_t
    其中，​x_t 就表示我们需要刻画的量，θ 表示均值回归的速率（大于0），μ 表示它的均值，​σ 是随机噪声的权重，
    W_t 表示维纳过程（也叫做布朗运动），是一种外界的随机噪声，
    """

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """将内部状态（= 噪声）重置为平均值 (mu)"""
        self.state = copy.copy(self.mu)

    def sample(self):
        """更新内部状态并将其作为噪声样本返回"""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.array([np.random.normal() for _ in range(len(self.state))])
        self.state += dx
        return self.state