class Config(object):
    """设定智能体/游戏环境的配置要求的对象"""
    def __init__(self):
        self.train_render = False  # 训练模式渲染
        self.eval_render = False  # 评估模式渲染
        self.cur_run_data_dir = None  # 本次运行数据存放目录
        self.seed = None  # 随机种子
        self.environment = None  # 环境
        self.requirements_to_solve_game = None  # 是否要求解决问题
        self.num_episodes_to_run = None  # 训练的episode数
        self.file_to_save_data_results = None  # 训练结果数据保存路径
        self.file_to_save_results_graph = None  # 训练结果可视化图像保存路径
        self.runs_per_agent = None  # 每个智能体训练轮次次数
        # self.visualise_overall_results = None
        self.visualise_overall_agent_results = None  # 可视化所有轮次结果
        self.visualise_individual_results = None  # 可视化每一轮结果
        self.hyperparameters = None  # 智能体超参数
        self.use_GPU = None  # GPU开关
        self.overwrite_existing_results_file = None  # 是否覆盖已存在结果文件
        self.save_model = False  # 是否保存模型
        self.load_model = False  # 是否加载已保存模型
        self.standard_deviation_results = 1.0  # 标准差放大系数
        self.randomise_random_seed = True  # 是否随机化随机种子
        self.show_solution_score = False  # 是否显示解决问题的成绩参考线s
        self.debug_mode = False  # 是否调试模式