from agents.actor_critic_agents.A3C import A3C

class A2C(A3C):
    """来自 deepmind 论文 https://arxiv.org/pdf/1602.01783.pdf 的 A2C 算法的同步版本。
    这与 A3C 之间的唯一区别是梯度更新是在批处理中完成的，而不是在梯度进来时 1 x 1"""
    agent_name = "A2C"
    def __init__(self, config):
        super(A2C, self).__init__(config)

    def update_shared_model(self, gradient_updates_queue):
        """将共享模型放入队列时，使用梯度更新共享模型的 worker"""
        while True:
            gradients_seen = 0
            while gradients_seen < self.worker_processes:
                if gradients_seen == 0:
                    gradients = gradient_updates_queue.get()
                else:
                    new_grads = gradient_updates_queue.get()
                    gradients = [grad + new_grad for grad, new_grad in zip(gradients, new_grads)]
                gradients_seen += 1
            self.actor_critic_optimizer.zero_grad()
            for grads, params in zip(gradients, self.actor_critic.parameters()):
                params._grad = grads
            self.actor_critic_optimizer.step()