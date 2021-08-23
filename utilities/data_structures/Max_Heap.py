import numpy as np
from utilities.data_structures.Node import Node

class Max_Heap(object):
    """通用最大堆对象"""

    def __init__(self, max_size, dimension_of_value_attribute, default_key_to_use):
        self.max_size = max_size  # 堆容量
        self.dimension_of_value_attribute = dimension_of_value_attribute  # 值属性的维度
        self.default_key_to_use = default_key_to_use  # 默认键
        self.heap = self.initialise_heap()  # 初始化堆

    def initialise_heap(self):
        """初始化堆，长度为 self.max_size * 4 + 1"""
        heap = np.array([Node(self.default_key_to_use, tuple([None for _ in range(self.dimension_of_value_attribute)])) for _ in range(self.max_size * 4 + 1)])

        # 我们不使用堆中的第 0 个元素，因此我们希望它具有无限大值，因此它永远不会与较低的节点交换
        heap[0] = Node(float("inf"), (None, None, None, None, None))
        return heap

    def update_element_and_reorganise_heap(self, heap_index_for_change, new_element):
        """更新元素并重新组织堆"""
        self.update_heap_element(heap_index_for_change, new_element)
        self.reorganise_heap(heap_index_for_change)

    def update_heap_element(self, heap_index, new_element):
        """更新堆元素"""
        self.heap[heap_index] = new_element

    def reorganise_heap(self, heap_index_changed):
        """添加新值后重新组织堆，以便将最大值保持在堆的顶部，即数组 self.heap 中的索引位置 1"""

        node_key = self.heap[heap_index_changed].key
        parent_index = int(heap_index_changed / 2)

        if node_key > self.heap[parent_index].key:
            self.swap_heap_elements(heap_index_changed, parent_index)
            self.reorganise_heap(parent_index)

        else:
            biggest_child_index = self.calculate_index_of_biggest_child(heap_index_changed)
            if node_key < self.heap[biggest_child_index].key:
                self.swap_heap_elements(heap_index_changed, biggest_child_index)
                self.reorganise_heap(biggest_child_index)

    def swap_heap_elements(self, index1, index2):
        """交换堆中的两个元素"""
        self.heap[index1], self.heap[index2] = self.heap[index2], self.heap[index1]

    def calculate_index_of_biggest_child(self, heap_index_changed):
        """计算 td_error 值最大的节点子节点的堆索引"""
        left_child = self.heap[int(heap_index_changed * 2)]
        right_child = self.heap[int(heap_index_changed * 2) + 1]

        if left_child.key > right_child.key:
            biggest_child_index = heap_index_changed * 2
        else:
            biggest_child_index = heap_index_changed * 2 + 1

        return biggest_child_index

    def give_max_key(self):
        """返回当前堆中的最大 td_error，即最大堆的顶部元素"""
        return self.heap[1].key
