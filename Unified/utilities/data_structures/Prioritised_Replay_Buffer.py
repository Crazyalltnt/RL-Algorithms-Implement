import numpy as np
import torch
from utilities.data_structures.Deque import Deque
from utilities.data_structures.Max_Heap import Max_Heap

class Prioritised_Replay_Buffer(Max_Heap, Deque):
    """维护一个 deque、max_heap 和 array 的数据结构。 
    
    deque 会跟踪哪些体验是最旧的，因此告诉我们一旦缓冲区开始变满要删除哪些体验。 
    max_heap 让我们可以使用最大 td_value 快速检索经验。 
    array 让我们可以快速随机抽样，其概率等比例于 td 误差。
    我们还使用一个简单的变量来跟踪 td 值的总和。
    """

    def __init__(self, hyperparameters, seed=0, device=None):
        Max_Heap.__init__(self, hyperparameters["buffer_size"], dimension_of_value_attribute=5, default_key_to_use=0)
        Deque.__init__(self, hyperparameters["buffer_size"], dimension_of_value_attribute=5)
        np.random.seed(seed)

        self.deques_td_errors = self.initialise_td_errors_array()  # 使用 td_error 数组初始化双端队列

        self.heap_index_to_overwrite_next = 1  # 要重写的堆索引
        self.number_experiences_in_deque = 0  # 队列中的经验数
        self.adapted_overall_sum_of_td_errors = 0  # 调整后的 td_errors 的总和

        self.alpha = hyperparameters["alpha_prioritised_replay"]
        self.beta = hyperparameters["beta_prioritised_replay"]
        self.incremental_td_error = hyperparameters["incremental_td_error"]  # 增量 td_error
        self.batch_size = hyperparameters["batch_size"]

        self.heap_indexes_to_update_td_error_for = None

        self.indexes_in_node_value_tuple = {
            "state": 0,
            "action": 1,
            "reward": 2,
            "next_state": 3,
            "done": 4
        }

        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def initialise_td_errors_array(self):
        """初始化一个长度为 self.max_size 的节点的双端队列"""
        return np.zeros(self.max_size)

    def add_experience(self, raw_td_error, state, action, reward, next_state, done):
        """保存经验"""
        td_error = (abs(raw_td_error) + self.incremental_td_error) ** self.alpha
        self.update_overall_sum(td_error, self.deque[self.deque_index_to_overwrite_next].key)
        self.update_deque_and_deque_td_errors(td_error, state, action, reward, next_state, done)
        self.update_heap_and_heap_index_to_overwrite()
        self.update_number_experiences_in_deque()
        self.update_deque_index_to_overwrite_next()

    def update_overall_sum(self, new_td_error, old_td_error):
        """更新缓冲区中存在的 td_error 的总和"""
        self.adapted_overall_sum_of_td_errors += new_td_error  - old_td_error

    def update_deque_and_deque_td_errors(self, td_error, state, action, reward, next_state, done):
        """通过使用提供的经验覆盖最旧的经验来更新双端队列"""
        self.deques_td_errors[self.deque_index_to_overwrite_next] = td_error
        self.add_element_to_deque(td_error, (state, action, reward, next_state, done))

    def add_element_to_deque(self, new_key, new_value):
        """添加元素到双端队列"""
        self.update_deque_node_key_and_value(self.deque_index_to_overwrite_next, new_key, new_value)

    def update_heap_and_heap_index_to_overwrite(self):
        """根据刚刚合并的新经验，通过重新排列堆来更新堆。 如果我们还没有达到最大容量，那么新经验将直接添加到堆中，
        否则堆上的指针更改以反映新经验，因此无需添加。
        """
        if not self.reached_max_capacity:
            self.update_heap_element(self.heap_index_to_overwrite_next, self.deque[self.deque_index_to_overwrite_next])
            self.deque[self.deque_index_to_overwrite_next].heap_index = self.heap_index_to_overwrite_next
            self.update_heap_index_to_overwrite_next()

        heap_index_change = self.deque[self.deque_index_to_overwrite_next].heap_index
        self.reorganise_heap(heap_index_change)

    def update_heap_index_to_overwrite_next(self):
        """这将更新堆索引以进行下一次写入。 一旦缓冲区满了，我们就停止调用这个函数，因为堆指向的节点开始直接改变，而不是堆上的指针改变"""
        self.heap_index_to_overwrite_next += 1

    def swap_heap_elements(self, index1, index2):
        """交换两个堆元素的位置，然后更新存储在两个节点中的 heap_index。 我们必须从 Max_Heap 覆盖这个方法，以便它也更新 heap_index 变量"""
        self.heap[index1], self.heap[index2] = self.heap[index2], self.heap[index1]
        self.heap[index1].heap_index = index1
        self.heap[index2].heap_index = index2

    def sample(self, rank_based=True):
        """从经验中随机抽样一批，给具有较高 td 误差的经验提供更高的可能性。 然后它计算每个采样经验的重要性采样权重，您可以在论文中了解这一点：
        https://arxiv.org/pdf/1511.05952.pdf
        """
        experiences, deque_sample_indexes = self.pick_experiences_based_on_proportional_td_error()
        states, actions, rewards, next_states, dones = self.separate_out_data_types(experiences)
        self.deque_sample_indexes_to_update_td_error_for = deque_sample_indexes
        importance_sampling_weights = self.calculate_importance_sampling_weights(experiences)
        return (states, actions, rewards, next_states, dones), importance_sampling_weights

    def pick_experiences_based_on_proportional_td_error(self):
        """随机选择一批经验，概率等比例于 td_errors"""
        probabilities = self.deques_td_errors / self.give_adapted_sum_of_td_errors()
        deque_sample_indexes = np.random.choice(range(len(self.deques_td_errors)), size=self.batch_size, replace=False, p=probabilities)
        experiences = self.deque[deque_sample_indexes]
        return experiences, deque_sample_indexes

    def separate_out_data_types(self, experiences):
        """将经验分成不同的部分，并使它们准备好在 pytorch 模型中使用的张量"""
        states = torch.from_numpy(np.vstack([e.value[self.indexes_in_node_value_tuple["state"]] for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.value[self.indexes_in_node_value_tuple["action"]] for e in experiences])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.value[self.indexes_in_node_value_tuple["reward"]] for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.value[self.indexes_in_node_value_tuple["next_state"]] for e in experiences])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([int(e.value[self.indexes_in_node_value_tuple["done"]]) for e in experiences])).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def calculate_importance_sampling_weights(self, experiences):
        """计算样本中每个观测值的重要性抽样权重。 权重与 observation 的 td_error 成正比，
        请参阅此处的论文了解更多详情：https://arxiv.org/pdf/1511.05952.pdf
        """
        td_errors = [experience.key for experience in experiences]
        importance_sampling_weights = [((1.0 / self.number_experiences_in_deque) * (self.give_adapted_sum_of_td_errors() / td_error)) ** self.beta for td_error in td_errors]
        sample_max_importance_weight = max(importance_sampling_weights)
        importance_sampling_weights = [is_weight / sample_max_importance_weight for is_weight in importance_sampling_weights]
        importance_sampling_weights = torch.tensor(importance_sampling_weights).float().to(self.device)
        return importance_sampling_weights

    def update_td_errors(self, td_errors):
        """更新提供的堆索引的 td_errors。 索引应该是 give_sample 方法最近提供的 observation 结果"""
        for raw_td_error, deque_index in zip(td_errors, self.deque_sample_indexes_to_update_td_error_for):
            td_error =  (abs(raw_td_error) + self.incremental_td_error) ** self.alpha
            corresponding_heap_index = self.deque[deque_index].heap_index
            self.update_overall_sum(td_error, self.heap[corresponding_heap_index].key)
            self.heap[corresponding_heap_index].key = td_error
            self.reorganise_heap(corresponding_heap_index)
            self.deques_td_errors[deque_index] = td_error

    def give_max_td_error(self):
        """返回当前堆中的最大 td_error，即最大堆的顶部元素"""
        return self.give_max_key()

    def give_adapted_sum_of_td_errors(self):
        """返回堆中当前经验的 td_error 总和"""
        return self.adapted_overall_sum_of_td_errors

    def __len__(self):
        """重放缓冲区的经验数"""
        return self.number_experiences_in_deque

