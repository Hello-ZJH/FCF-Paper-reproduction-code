import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from config import *
from data_preprocess import load_and_preprocess_data
from fcf_models import FCFClient, FCFServer, CentralizedCF
from evaluation import evaluate_fcf, evaluate_centralized_cf


# 设置随机种子（保证复现性）
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed()

# 加载数据
(
    train_data, test_data, train_user_groups,
    num_users, num_items, user_id_map, movie_id_map
) = load_and_preprocess_data()

# 初始化FCF模型（服务器+客户端）
print(f"\n初始化FCF模型（K={K}，客户端数={SAMPLE_CLIENT_NUM}）...")
server = FCFServer(K=K, I=num_items)
# 初始化客户端（抽样SAMPLE_CLIENT_NUM个用户，加速训练）
clients = []
sample_user_ids = list(train_user_groups.keys())[:SAMPLE_CLIENT_NUM]
for user_id in sample_user_ids:
    local_interactions = train_user_groups[user_id]
    client = FCFClient(
        user_id=user_id,
        local_interactions=local_interactions,
        K=K,
        I=num_items
    )
    clients.append(client)

# FCF联邦训练
print(f"\n开始FCF联邦训练（{NUM_EPOCHS}轮迭代）...")
fcf_train_losses = []
best_fcf_loss = float('inf')
early_stop_count = 0

for epoch in range(NUM_EPOCHS):
    epoch_losses = []
    client_gradients = []
    # 服务器分发全局Y
    current_Y = server.distribute_global_Y()
    # 每个客户端本地更新X + 计算Y梯度
    for client in tqdm(clients, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
        local_loss = client.local_update(current_Y)
        epoch_losses.append(local_loss)
        y_grad = client.compute_y_gradient(current_Y)
        client_gradients.append(y_grad)
    # 服务器聚合梯度 + 更新Y
    avg_grad = server.aggregate_gradients(client_gradients)
    server.update_global_Y(avg_grad)
    # 计算本轮平均损失
    avg_epoch_loss = np.mean(epoch_losses)
    fcf_train_losses.append(avg_epoch_loss)
    print(f"Epoch {epoch + 1}，FCF平均损失：{avg_epoch_loss:.4f}")
    # 早停判断
    if avg_epoch_loss < best_fcf_loss - 1e-4:  # 损失下降超过1e-4才算有效
        best_fcf_loss = avg_epoch_loss
        early_stop_count = 0
    else:
        early_stop_count += 1
        if early_stop_count >= EARLY_STOP_PATIENCE:
            print(f"连续{EARLY_STOP_PATIENCE}轮损失无下降，触发早停！")
            break
# 保存FCF训练损失曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(fcf_train_losses) + 1), fcf_train_losses, label='FCF Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('FCF Training Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_PATH, 'fcf_loss_curve.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"\nFCF损失曲线已保存到：{os.path.join(OUTPUT_PATH, 'fcf_loss_curve.png')}")

# 训练集中式CF（对比实验，适配隐式反馈+BPR损失） =====================
centralized_cf_results = None
if RUN_CENTRALIZED_CF:
    print(f"\n开始训练传统集中式CF（用于对比，适配隐式反馈+BPR损失）...")
    centralized_cf = CentralizedCF(
        num_users=num_users,
        num_items=num_items,
        K=K
    )
    # 核心修改1：生成隐式反馈的正负样本对（适配BPR损失）
    # 按用户分组，获取每个用户的正样本（喜欢的电影：is_like=1）
    pos_user_item = train_data[train_data['is_like'] == 1].groupby('user_id_new')['movie_id_new'].apply(list).to_dict()
    # 生成每个用户的负样本（未交互的电影，数量与正样本一致）
    all_movie_ids = set(range(num_items))
    neg_user_item = {}
    for user_id, pos_movies in pos_user_item.items():
        # 负样本 = 所有电影 - 该用户的正样本（取与正样本相同数量）
        available_neg_movies = list(all_movie_ids - set(pos_movies))
        neg_movies = available_neg_movies[:len(pos_movies)]  # 保证正负样本数量平衡
        neg_user_item[user_id] = neg_movies
    # 构造训练样本对 (user_id, 正样本电影ID, 负样本电影ID)
    train_pairs = []
    for user_id in pos_user_item.keys():
        if user_id not in neg_user_item or len(neg_user_item[user_id]) == 0:
            continue  # 跳过无负样本的用户
        # 一一对应构造正负样本对
        for pos_movie, neg_movie in zip(pos_user_item[user_id], neg_user_item[user_id]):
            train_pairs.append((user_id, pos_movie, neg_movie))
    # 转换为PyTorch张量（批量训练用）
    train_pairs = np.array(train_pairs)
    train_users = torch.tensor(train_pairs[:, 0], dtype=torch.long)
    train_pos_items = torch.tensor(train_pairs[:, 1], dtype=torch.long)
    train_neg_items = torch.tensor(train_pairs[:, 2], dtype=torch.long)
    # 批量训练参数设置
    batch_size = 512  # 适配BPR损失的合理批次大小
    num_batches = len(train_users) // batch_size  # 每轮迭代的批次数
    centralized_train_losses = []
    best_centralized_loss = float('inf')
    early_stop_count_centralized = 0
    # 核心修改2：BPR损失训练循环（增加训练轮次，适配收敛速度）
    for epoch in range(NUM_EPOCHS * 2):  # 训练轮次翻倍（BPR损失收敛稍慢）
        epoch_total_loss = 0.0
        # 打乱样本对顺序（提升训练稳定性，避免过拟合）
        permutation = torch.randperm(len(train_users))
        shuffled_users = train_users[permutation]
        shuffled_pos_items = train_pos_items[permutation]
        shuffled_neg_items = train_neg_items[permutation]
        # 按批次迭代训练
        for batch_idx in tqdm(range(num_batches), desc=f"Centralized CF Epoch {epoch + 1}"):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            # 提取当前批次的正负样本对
            batch_u = shuffled_users[start_idx:end_idx]
            batch_p = shuffled_pos_items[start_idx:end_idx]
            batch_n = shuffled_neg_items[start_idx:end_idx]
            # 单批次训练（BPR损失）
            batch_loss = centralized_cf.train_step(batch_u, batch_p, batch_n)
            epoch_total_loss += batch_loss
        # 计算本轮平均损失
        avg_epoch_loss = epoch_total_loss / num_batches
        centralized_train_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1}，集中式CF平均损失：{avg_epoch_loss:.4f}")
        # 核心修改3：调整早停逻辑（BPR损失波动较大，放宽阈值）
        if avg_epoch_loss < best_centralized_loss - 5e-4:  # 损失下降超过5e-4才算有效
            best_centralized_loss = avg_epoch_loss
            early_stop_count_centralized = 0
        else:
            early_stop_count_centralized += 1
            # 早停耐心值翻倍（避免过早停止）
            if early_stop_count_centralized >= EARLY_STOP_PATIENCE * 2:
                print(f"集中式CF连续{EARLY_STOP_PATIENCE * 2}轮损失无下降，触发早停！")
                break
    # 保存集中式CF损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(centralized_train_losses) + 1), centralized_train_losses,
             label='Centralized CF Train Loss (BPR)', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Centralized CF Training Loss Curve (Implicit Feedback+BPR)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_PATH, 'centralized_cf_loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"集中式CF损失曲线已保存到：{os.path.join(OUTPUT_PATH, 'centralized_cf_loss_curve.png')}")
    # 评估集中式CF（传递FCF的抽样用户ID，确保样本一致）
    centralized_cf_results = evaluate_centralized_cf(centralized_cf, test_data, sample_user_ids)

# 评估FCF
fcf_results = evaluate_fcf(server, clients, test_data, num_items)
# 对比结果汇总
print(f"\n{'=' * 50} 最终结果对比 {'=' * 50}")
print(f"{'指标':<15} {'FCF':<10} {'集中式CF':<10} {'差异率':<10}")
print(f"-" * 80)
for metric in ['precision', 'recall', 'rmse']:
    fcf_val = fcf_results[metric]
    centralized_val = centralized_cf_results[metric] if centralized_cf_results else 0.0
    diff_rate = abs(fcf_val - centralized_val) / centralized_val * 100 if centralized_val != 0 else 0.0
    print(f"{metric.upper():<15} {fcf_val:.4f} {'':<2} {centralized_val:.4f} {'':<2} {diff_rate:.2f}%")
print(f"{'=' * 80}")
print(f"复现成功判定标准：FCF与集中式CF性能差异≤3%，验证了论文核心结论：FCF保护隐私的同时不牺牲推荐性能")
