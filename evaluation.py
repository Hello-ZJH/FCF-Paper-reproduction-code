import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score


def evaluate_fcf(server, clients, test_data, num_items):
    """评估FCF模型（仅评估抽样的客户端用户）"""
    server.eval()
    for client in clients:
        client.eval()

    all_true = []
    all_pred = []
    all_pred_scores = []
    all_true_scores = []

    # 仅处理FCF抽样客户端的测试数据
    sample_user_ids = [client.user_id for client in clients]
    test_data_fcf = test_data[test_data['user_id_new'].isin(sample_user_ids)]

    # 按用户分组评估（避免不同用户的正负样本混淆）
    for user_id, group in test_data_fcf.groupby('user_id_new'):
        # 找到对应客户端的用户向量X
        client = next(c for c in clients if c.user_id == user_id)
        user_vec = client.X.detach()
        # 获取服务器全局物品向量Y
        item_vecs = server.global_Y.detach()
        # 计算该用户对所有物品的推荐分数
        all_scores = torch.matmul(user_vec, item_vecs).squeeze().numpy()
        # 测试集中的物品ID和真实标签
        test_item_ids = group['movie_id_new'].values
        test_labels = group['is_like'].values
        # 记录真实标签和预测分数（用于RMSE计算）
        all_true_scores.extend(test_labels)
        all_pred_scores.extend(all_scores[test_item_ids])
        # 按Top-10筛选预测结果（推荐分数前10的物品视为正例）
        top10_item_indices = np.argsort(all_scores)[-10:]
        pred_labels = np.zeros(num_items)
        pred_labels[top10_item_indices] = 1
        # 提取测试集物品的预测标签，用于精确率/召回率计算
        test_pred_labels = pred_labels[test_item_ids]
        all_true.extend(test_labels)
        all_pred.extend(test_pred_labels)

    # 计算评估指标
    precision = precision_score(all_true, all_pred, zero_division=0)
    recall = recall_score(all_true, all_pred, zero_division=0)
    rmse = np.sqrt(np.mean((np.array(all_pred_scores) - np.array(all_true_scores)) ** 2))
    return {
        'precision': precision,
        'recall': recall,
        'rmse': rmse
    }


def evaluate_centralized_cf(centralized_cf, test_data, sample_user_ids):
    """
    评估集中式CF模型（与FCF评估样本完全对齐）
    :param sample_user_ids: FCF抽样的用户ID列表（从train.py传递，确保样本一致）
    """
    centralized_cf.eval()
    all_true = []
    all_pred = []
    all_pred_scores = []
    all_true_scores = []
    # 核心：仅评估FCF抽样的用户（样本完全对齐）
    test_data_centralized = test_data[test_data['user_id_new'].isin(sample_user_ids)]
    # 按用户分组评估（与FCF评估逻辑一致）
    for user_id, group in test_data_centralized.groupby('user_id_new'):
        # 获取该用户的向量X和所有物品向量Y
        user_vec = centralized_cf.X[user_id].detach().unsqueeze(0)
        item_vecs = centralized_cf.Y.detach()
        # 计算该用户对所有物品的推荐分数（与FCF计算逻辑一致）
        all_scores = torch.matmul(user_vec, item_vecs).squeeze().numpy()
        # 测试集中的物品ID和真实标签
        test_item_ids = group['movie_id_new'].values
        test_labels = group['is_like'].values
        # 记录真实标签和预测分数（用于RMSE计算）
        all_true_scores.extend(test_labels)
        all_pred_scores.extend(all_scores[test_item_ids])
        # 按Top-10筛选预测结果（与FCF评估标准一致）
        top10_item_indices = np.argsort(all_scores)[-10:]
        pred_labels = np.zeros(len(all_scores))
        pred_labels[top10_item_indices] = 1
        # 提取测试集物品的预测标签，用于精确率/召回率计算
        test_pred_labels = pred_labels[test_item_ids]
        all_true.extend(test_labels)
        all_pred.extend(test_pred_labels)

    # 计算评估指标（与FCF指标计算方式完全一致）
    precision = precision_score(all_true, all_pred, zero_division=0)
    recall = recall_score(all_true, all_pred, zero_division=0)
    rmse = np.sqrt(np.mean((np.array(all_pred_scores) - np.array(all_true_scores)) ** 2))
    return {
        'precision': precision,
        'recall': recall,
        'rmse': rmse
    }
