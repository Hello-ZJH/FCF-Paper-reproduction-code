import torch
import torch.nn as nn
import torch.optim as optim
from config import *


class FCFClient(nn.Module):
    """联邦协同过滤客户端（对应用户设备）"""

    def __init__(self, user_id, local_interactions, K, I):
        super().__init__()
        self.user_id = user_id
        self.local_interactions = local_interactions  # 本地交互数据（(movie_id, is_like)列表）
        self.K = K
        self.I = I
        # 本地用户因子向量X（1×K）
        self.X = nn.Parameter(torch.randn(1, K) * 0.01)  # 小方差初始化，避免梯度爆炸
        # Adam优化器（对齐论文）
        self.optimizer = optim.Adam(
            [self.X],
            lr=LEARNING_RATE,
            betas=(ADAM_BETA1, ADAM_BETA2),
            weight_decay=LAMBDA_REG  # 正则化（L2）
        )

    def local_update(self, global_Y):
        """本地更新用户因子X"""
        self.train()
        self.optimizer.zero_grad()
        # 计算本地损失（隐式反馈对数似然损失简化版）
        total_loss = 0.0
        for (movie_id, is_like) in self.local_interactions:
            # 推荐分数：x_u × y_i（1×K @ K×1 = 1×1）
            pred_score = torch.matmul(self.X, global_Y[:, movie_id:movie_id + 1])
            # 损失：MSE（贴合论文优化目标）
            loss = (pred_score - is_like) ** 2
            total_loss += loss
        # 平均损失（避免用户交互数差异影响）
        avg_loss = total_loss / len(self.local_interactions)
        avg_loss.backward()
        self.optimizer.step()
        return avg_loss.item()

    def compute_y_gradient(self, global_Y):
        """计算全局Y的梯度（上传给服务器）"""
        self.eval()
        y_grad = torch.zeros_like(global_Y)  # K×I
        with torch.no_grad():
            for (movie_id, is_like) in self.local_interactions:
                pred_score = torch.matmul(self.X, global_Y[:, movie_id:movie_id + 1])
                # 梯度公式：dLoss/dy_i = 2*(pred - is_like)*x_u^T + 2*lambda*y_i
                grad = 2 * (pred_score - is_like) * self.X.T + 2 * LAMBDA_REG * global_Y[:, movie_id:movie_id + 1]
                y_grad[:, movie_id:movie_id + 1] = grad
        # 平均梯度（按本地交互数归一化）
        return y_grad / len(self.local_interactions)


class FCFServer(nn.Module):
    """联邦协同过滤服务器（对应平台总部）"""

    def __init__(self, K, I):
        super().__init__()
        self.K = K
        self.I = I
        # 全局项目因子矩阵Y（K×I）
        self.global_Y = nn.Parameter(torch.randn(K, I) * 0.01)
        # Adam优化器（对齐论文）
        self.optimizer = optim.Adam(
            [self.global_Y],
            lr=LEARNING_RATE,
            betas=(ADAM_BETA1, ADAM_BETA2),
            weight_decay=LAMBDA_REG
        )

    def distribute_global_Y(self):
        """分发当前全局Y给所有客户端（detach避免客户端修改原始数据）"""
        return self.global_Y.detach().clone()

    def aggregate_gradients(self, client_gradients):
        """聚合所有客户端的Y梯度（平均聚合）"""
        if not client_gradients:
            return torch.zeros_like(self.global_Y)
        # 求和后平均
        total_grad = sum(client_gradients)
        avg_grad = total_grad / len(client_gradients)
        return avg_grad

    def update_global_Y(self, avg_grad):
        """更新全局Y（用聚合后的梯度）"""
        self.train()
        self.optimizer.zero_grad()
        # 手动赋值梯度（因为梯度来自客户端聚合，而非服务器本地反向传播）
        self.global_Y.grad = avg_grad
        self.optimizer.step()


class CentralizedCF(nn.Module):
    """传统集中式协同过滤（适配隐式反馈+BPR损失，修复维度不匹配）"""

    def __init__(self, num_users, num_items, K):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.K = K

        # 核心修复：Y维度保持与FCF一致（K×I），避免评估时矩阵乘法错误
        self.X = nn.Parameter(torch.randn(num_users, K) * 0.01)  # 用户向量：U×K
        self.Y = nn.Parameter(torch.randn(K, num_items) * 0.01)  # 物品向量：K×I（与FCFServer的Y维度一致）

        # 优化器适配BPR损失
        self.optimizer = optim.Adam(
            [self.X, self.Y],
            lr=0.05,
            betas=(ADAM_BETA1, ADAM_BETA2),
            weight_decay=0.0001
        )

    def forward(self, user_ids, pos_item_ids, neg_item_ids=None):
        """
        修复维度计算：Y为K×I，提取物品向量后转置为batch_size×K
        """
        user_vecs = self.X[user_ids]  # (batch_size, K)

        # 提取正样本物品向量：Y[:, pos_item_ids] → (K, batch_size) → 转置为 (batch_size, K)
        pos_item_vecs = self.Y[:, pos_item_ids].T  # 关键修复：增加转置

        if neg_item_ids is not None:
            # 提取负样本物品向量：同样转置
            neg_item_vecs = self.Y[:, neg_item_ids].T  # 关键修复：增加转置
            pos_scores = torch.sum(user_vecs * pos_item_vecs, dim=1)  # (batch_size,)
            neg_scores = torch.sum(user_vecs * neg_item_vecs, dim=1)  # (batch_size,)
            return pos_scores - neg_scores
        else:
            # 评估时：返回单个物品得分（与FCF评估逻辑一致）
            return torch.sum(user_vecs * pos_item_vecs, dim=1)

    def train_step(self, user_ids, pos_item_ids, neg_item_ids):
        """BPR损失训练步骤（维度修复后正常计算）"""
        self.train()
        self.optimizer.zero_grad()

        score_diff = self(user_ids, pos_item_ids, neg_item_ids)
        loss = -torch.mean(torch.sigmoid(score_diff))

        loss.backward()
        self.optimizer.step()
        return loss.item()
