"""
train.py - 统一训练脚本（Apple M2 MPS GPU 加速）
支持所有深度学习模型的训练、评估和对比

用法:
  python train.py                           # 默认训练原始模型
  python train.py --model cnn_bilstm_multihead  # 训练加强版
  python train.py --all                     # 训练所有模型并对比
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import io
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适配无显示器环境
import matplotlib.pyplot as plt

from models import get_model, MODEL_REGISTRY

# ============================================================
# 设备选择：优先MPS，其次CUDA，最后CPU
# ============================================================
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"🍎 使用 Apple MPS GPU 加速")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🔥 使用 CUDA GPU 加速")
    else:
        device = torch.device("cpu")
        print(f"💻 使用 CPU")
    return device


# ============================================================
# 数据加载
# ============================================================
def load_data(data_path):
    """加载 STS-PD dataset.mat"""
    print(f"📦 加载数据: {data_path}")
    data = io.loadmat(data_path)

    train_data = data['sts'][0, 0]['train'].T.squeeze()
    train_label = data['sts'][0, 0]['trainlabels'].squeeze()
    test_data = data['sts'][0, 0]['test'].T.squeeze()
    test_label = data['sts'][0, 0]['testlabels'].squeeze()

    # 计算最大时间步长
    max_len = 0
    for item in train_data:
        item = np.array(item, dtype=np.float32)
        if item.shape[1] > max_len:
            max_len = item.shape[1]
    for item in test_data:
        item = np.array(item, dtype=np.float32)
        if item.shape[1] > max_len:
            max_len = item.shape[1]

    # 零填充对齐
    def pad_sequences(data_array, max_len):
        padded = []
        for item in data_array:
            item = np.array(item, dtype=np.float32)
            feat_dim = item.shape[0]
            seq_len = item.shape[1]
            if seq_len < max_len:
                pad = np.zeros((feat_dim, max_len - seq_len), dtype=np.float32)
                item = np.concatenate([item, pad], axis=1)
            padded.append(item.T)  # -> (max_len, feat_dim)
        return np.array(padded)

    X_train = pad_sequences(train_data, max_len)
    X_test = pad_sequences(test_data, max_len)

    # 标签从1开始需要转为0开始
    y_train = train_label.astype(np.int64)
    y_test = test_label.astype(np.int64)
    if y_train.min() > 0:
        y_train = y_train - y_train.min()
        y_test = y_test - y_test.min()

    num_classes = len(set(y_train))
    feat_dim = X_train.shape[2]

    print(f"  训练集: {X_train.shape[0]} 样本")
    print(f"  测试集: {X_test.shape[0]} 样本")
    print(f"  时间步: {max_len}, 特征维度: {feat_dim}, 类别数: {num_classes}")

    return X_train, y_train, X_test, y_test, feat_dim, num_classes


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=64):
    """创建 DataLoader"""
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ============================================================
# 训练和评估
# ============================================================
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        total_loss += loss.item() * batch_x.size(0)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

    total = len(all_labels)
    acc = 100.0 * accuracy_score(all_labels, all_preds)
    f1 = 100.0 * f1_score(all_labels, all_preds, average='weighted')

    return total_loss / total, acc, f1, np.array(all_preds), np.array(all_labels)


def train_model(model_name, args, X_train, y_train, X_test, y_test, feat_dim, num_classes, device):
    """训练单个模型"""
    print(f"\n{'='*60}")
    print(f"🚀 训练模型: {model_name}")
    print(f"{'='*60}")

    # 创建模型
    model = get_model(model_name, input_dim=feat_dim, num_classes=num_classes,
                      hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                      dropout=args.dropout)
    model = model.to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 数据加载器
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, args.batch_size)

    # 优化器和学习率调度
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练循环
    best_acc = 0
    train_losses, test_accs = [], []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, test_f1, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        test_accs.append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_f1 = test_f1
            # 保存最佳模型
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/{model_name}_best.pth')

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
                  f"Test Acc: {test_acc:.1f}% F1: {test_f1:.1f}% | "
                  f"Best: {best_acc:.1f}%")

    elapsed = time.time() - start_time
    print(f"  ⏱️  训练耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # 加载最佳模型进行最终评估
    model.load_state_dict(torch.load(f'checkpoints/{model_name}_best.pth', weights_only=True))
    _, final_acc, final_f1, y_pred, y_true = evaluate(model, test_loader, criterion, device)

    print(f"\n📊 最终结果 [{model_name}]:")
    print(f"  Accuracy: {final_acc:.2f}%")
    print(f"  F1-Score: {final_f1:.2f}%")
    print(f"\n{classification_report(y_true, y_pred, digits=4)}")

    # 保存混淆矩阵图
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} Confusion Matrix (Acc={final_acc:.1f}%)')
    plt.colorbar()
    classes = sorted(set(y_true))
    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, f'{cm_norm[i,j]:.2f}', ha='center', va='center',
                    color='white' if cm_norm[i,j] > 0.5 else 'black')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_confusion.png', dpi=150)
    plt.close()

    # 保存训练曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax2.plot(test_accs)
    ax2.set_title('Test Accuracy (%)')
    ax2.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_curves.png', dpi=150)
    plt.close()

    return {
        'model': model_name,
        'accuracy': final_acc,
        'f1_score': final_f1,
        'params': trainable_params,
        'time': elapsed,
        'best_epoch': test_accs.index(max(test_accs)) + 1
    }


# ============================================================
# 传统ML对比
# ============================================================
def run_ml_baselines(X_train, y_train, X_test, y_test):
    """运行传统机器学习基线"""
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB

    # 展平为2D
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    models = {
        'KNN(1NN)': KNeighborsClassifier(n_neighbors=1),
        'SVM': SVC(kernel='rbf', C=1.0),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'NaiveBayes': GaussianNB(),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    }

    results = []
    print(f"\n{'='*60}")
    print(f"🔧 传统机器学习基线对比")
    print(f"{'='*60}")

    for name, clf in models.items():
        start = time.time()
        clf.fit(X_train_flat, y_train)
        y_pred = clf.predict(X_test_flat)
        elapsed = time.time() - start

        acc = 100.0 * accuracy_score(y_test, y_pred)
        f1 = 100.0 * f1_score(y_test, y_pred, average='weighted')
        print(f"  {name:20s} | Acc: {acc:.2f}% | F1: {f1:.2f}% | Time: {elapsed:.1f}s")

        results.append({
            'model': name,
            'accuracy': acc,
            'f1_score': f1,
            'params': '-',
            'time': elapsed,
            'best_epoch': '-'
        })

    return results


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='STS Phase Identification - Enhanced Edition (MPS)')
    parser.add_argument('--data', type=str, default='STS-PD dataset.mat', help='数据集路径')
    parser.add_argument('--model', type=str, default='cnn_bilstm_attention', 
                       choices=list(MODEL_REGISTRY.keys()), help='模型名称')
    parser.add_argument('--all', action='store_true', help='训练所有模型并对比')
    parser.add_argument('--ml', action='store_true', help='同时运行传统ML基线')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)

    # 设备
    device = get_device()

    # 加载数据
    X_train, y_train, X_test, y_test, feat_dim, num_classes = load_data(args.data)

    all_results = []

    # 传统ML基线
    if args.ml or args.all:
        ml_results = run_ml_baselines(X_train, y_train, X_test, y_test)
        all_results.extend(ml_results)

    # 深度学习模型
    if args.all:
        for model_name in MODEL_REGISTRY:
            result = train_model(model_name, args, X_train, y_train, X_test, y_test,
                               feat_dim, num_classes, device)
            all_results.append(result)
    else:
        result = train_model(args.model, args, X_train, y_train, X_test, y_test,
                           feat_dim, num_classes, device)
        all_results.append(result)

    # 打印汇总表
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print(f"📋 结果汇总")
        print(f"{'='*80}")
        print(f"{'Model':30s} | {'Acc(%)':>8s} | {'F1(%)':>8s} | {'Params':>10s} | {'Time(s)':>8s}")
        print(f"{'-'*30}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}")
        for r in sorted(all_results, key=lambda x: -x['accuracy'] if isinstance(x['accuracy'], float) else 0):
            params = f"{r['params']:,}" if isinstance(r['params'], int) else r['params']
            t = f"{r['time']:.1f}" if isinstance(r['time'], float) else r['time']
            print(f"{r['model']:30s} | {r['accuracy']:8.2f} | {r['f1_score']:8.2f} | {params:>10s} | {t:>8s}")

    print(f"\n✅ 完成！结果保存在 results/ 目录")


if __name__ == '__main__':
    main()
