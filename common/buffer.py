import numpy as np

class Replay_buffer(object):
    """
    经验池
    """

    def __init__(self, max_size, trajectory_property):
        self.storage = []
        self.max_size = max_size  # 经验池容量

        self.property_list = ['states', 'states_next', 'rewards', 'dones']
        self.property_additional = trajectory_property
        self.properties_all = self.property_list + self.property_additional
        self.item_buffers = dict()  # 经验池
        self.step_index_by_env = 0

        self.buffer_dict = dict()  # 字典批量存储经验序列
        self.buffer_dict_clear()
        self.ptr = 0
    
    def buffer_dict_clear(self):
        """
        经验序列重置
        """
        for item in self.properties_all:
            self.buffer_dict[item] = list()
    
    def init_item_buffers(self):
        """
        初始化经验池
        """
        for property in self.properties_all:
            self.item_buffers[property] = ItemBuffer(self.max_size, property)
    
    def insert(self, item_name:str, agent_id:int, data:np.ndarray, step=None):
        """
        插入经验属性序列
        """
        if item_name == 'dones':
            agent_id = 0
        self.item_buffers[item_name].insert(agent_id, step, data)
    
    def sample(self, batch_size):
        """
        批量采样经验序列
        """
        self.buffer_dict_clear()
        data_length = len(self.item_buffers["action"].data)
        ind = np.random.randint(0, data_length, size=batch_size)
        for name, item_buffer in self.item_buffers.items():
            for i in ind:
                self.buffer_dict[name].append(np.array(item_buffer.data[i], copy=False))
        return self.buffer_dict
    
    def item_buffer_clear(self):
        for p in self.properties_all:
            self.item_buffers[p].clear()


class ItemBuffer(object):
    """
    经验属性列表
    """
    def __init__(self, max_size, name):
        self.name = name  # 属性名：states/states_next/rewards/dones
        self.max_size = max_size  # 容量
        self.A = 1
        self.data = list()  # 属性经验列表
        self.ptr = 0
    
    def insert(self, agent_id:int, step:int, data:np.ndarray):
        """
        插入经验属性
        """
        if len(self.data) == self.max_size:
            self.data.pop(0)
        self.data.append(data)
    
    def clear(self):
        """
        清空经验属性列表
        """
        del self.data[:]