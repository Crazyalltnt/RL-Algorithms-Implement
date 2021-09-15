class Node(object):
    """通用节点类，用于优先级经验重放缓冲区的实现"""

    def __init__(self, key, value):
        self.key = key
        self.value = value

    def update_key_and_value(self, new_key, new_value):
        self.update_key(new_key)
        self.update_value(new_value)

    def update_key(self, new_key):
        self.key = new_key

    def update_value(self, new_value):
        self.value = new_value

    def __eq__(self, other):
        """比较Node内容是否相等"""
        return self.key == other.key and self.value == other.value