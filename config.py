import os
#路径配置
DATASET_PATH = "ml-1m/ratings.dat"
OUTPUT_PATH = "output/"  # 保存训练日志、损失曲线的路径
#超参数配置（论文默认）
K = 50  # 潜在因子维度（论文默认）
LAMBDA_REG = 0.001  # 正则化参数（论文默认）
LEARNING_RATE = 0.01  # 学习率（客户端+服务器共用）
ADAM_BETA1 = 0.9  # Adam优化器参数（论文默认）
ADAM_BETA2 = 0.999  # Adam优化器参数（论文默认）
NUM_EPOCHS = 20  # 联邦训练迭代次数（论文默认）
TOP_K = 10  # 推荐Top-K（评估指标用）
SAMPLE_CLIENT_NUM = 200  # 抽样客户端数量（全量6040个太慢）
#实验配置
TRAIN_TEST_SPLIT_RATIO = 0.9  # 90%训练集，10%测试集（按用户交互时间戳拆分）
RUN_CENTRALIZED_CF = True  # 是否运行集中式CF（对比实验，验证论文结论）
EARLY_STOP_PATIENCE = 3  # 早停耐心值（连续3轮损失不变则停止）