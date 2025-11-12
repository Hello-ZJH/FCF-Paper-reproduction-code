import pandas as pd
import numpy as np
from config import *
import os

def create_output_dir():
    """创建输出文件夹（保存损失曲线、日志）"""
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)


def load_and_preprocess_data():
    """加载MovieLens 1M数据，转为隐式反馈格式"""
    # 检查数据集路径是否存在
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"数据集文件未找到！请检查路径：{DATASET_PATH}\n"
        )
    # 读取数据
    data = pd.read_csv(
        DATASET_PATH,
        sep='::',
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        engine='python',
        encoding='latin-1'
    )

    # 1. 重新编号用户ID和电影ID（从0开始，方便矩阵索引）
    user_ids = sorted(data['user_id'].unique())
    movie_ids = sorted(data['movie_id'].unique())
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(user_ids)}
    movie_id_map = {old_id: new_id for new_id, old_id in enumerate(movie_ids)}
    data['user_id_new'] = data['user_id'].map(user_id_map)
    data['movie_id_new'] = data['movie_id'].map(movie_id_map)

    # 2. 转为隐式反馈：评分≥3为正样本（1），否则为负样本（0）
    data['is_like'] = (data['rating'] >= 3).astype(int)

    # 3. 按用户时间戳拆分训练集/测试集（后10%为测试集）
    data['train_mask'] = data.groupby('user_id_new')['timestamp'].transform(
        lambda x: x <= x.quantile(TRAIN_TEST_SPLIT_RATIO)
    )
    train_data = data[data['train_mask']].copy()
    test_data = data[~data['train_mask']].copy()

    # 4. 按用户分组，构建客户端本地交互数据（训练集）
    train_user_groups = train_data.groupby('user_id_new')[
        ['movie_id_new', 'is_like']
    ].apply(lambda x: list(x.itertuples(index=False, name=None))).to_dict()

    # 输出数据基本信息
    print(f"数据预处理完成！")
    print(f"用户数：{len(user_ids)}，电影数：{len(movie_ids)}")
    print(f"训练集交互数：{len(train_data)}，测试集交互数：{len(test_data)}")
    return (
        train_data, test_data, train_user_groups,
        len(user_ids), len(movie_ids), user_id_map, movie_id_map
    )


# 初始化输出文件夹
create_output_dir()
