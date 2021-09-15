import numpy as np
from utilities.data_structures.Node import Node

class Deque(object):
    """通用双端队列对象"""

    def __init__(self, max_size, dimension_of_value_attribute):
        self.max_size = max_size  # 队列容量
        self.dimension_of_value_attribute = dimension_of_value_attribute  # 值属性的维度
        self.deque = self.initialise_deque()  # 初始化队列
        self.deque_index_to_overwrite_next = 0  # 要重写的经验索引
        self.reached_max_capacity = False  # 是否达到最大容量
        self.number_experiences_in_deque = 0  # 队列中的经验数

    def initialise_deque(self):
        """初始化一个长度为 self.max_size 的节点队列"""
        deque = np.array([Node(0, tuple([None for _ in range(self.dimension_of_value_attribute)])) for _ in range(self.max_size)])
        return deque

    def add_element_to_deque(self, new_key, new_value):
        """向双端队列添加一个元素，然后更新下一个要覆盖的元素的索引以及双端队列中的元素数量 """
        self.update_deque_node_key_and_value(self.deque_index_to_overwrite_next, new_key, new_value)
        self.update_number_experiences_in_deque()
        self.update_deque_index_to_overwrite_next()

    def update_deque_node_key_and_value(self, index, new_key, new_value):
        """更新队列中节点的键和值"""
        self.update_deque_node_key(index, new_key)
        self.update_deque_node_value(index, new_value)

    def update_deque_node_key(self, index, new_key):
        """更新队列中节点的键"""
        self.deque[index].update_key(new_key)

    def update_deque_node_value(self, index, new_value):
        """更新队列中节点的值"""
        self.deque[index].update_value(new_value)

    def update_deque_index_to_overwrite_next(self):
        """更新我们接下来应该覆盖的双端队列索引，当缓冲区已满时，我们开始覆盖旧经验"""
        if self.deque_index_to_overwrite_next < self.max_size - 1:
            self.deque_index_to_overwrite_next += 1
        else:
            self.reached_max_capacity = True
            self.deque_index_to_overwrite_next = 0

    def update_number_experiences_in_deque(self):
        """跟踪缓冲区中有多少经验"""
        if not self.reached_max_capacity:
            self.number_experiences_in_deque += 1