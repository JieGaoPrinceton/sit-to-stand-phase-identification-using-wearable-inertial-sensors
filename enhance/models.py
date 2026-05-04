"""
models.py - 加强版模型定义
适配 Apple M2 MPS GPU 加速
包含原始论文模型 + 加强改进版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 原始论文模型：CNN-BiLSTM-Attention（基线复现）
# ============================================================
class CNN_BiLSTM_Attention(nn.Module):
    """论文原始模型 - 忠实复现"""
    def __init__(self, input_dim=6, hidden_dim=256, num_layers=2, num_classes=5, dropout=0.5):
        super(CNN_BiLSTM_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # CNN部分：2层Conv1d + BN + MaxPool
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=2, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)

        # BiLSTM部分
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        # Attention部分
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # 分类头
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # -> (batch, features, seq_len)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = x.permute(0, 2, 1)  # -> (batch, seq_len', hidden_dim)
        x, _ = self.lstm(x)  # -> (batch, seq_len', hidden_dim*2)
        x = self.dropout(x)

        # Attention
        attn_weights = F.softmax(self.attention(x), dim=1)  # (batch, seq_len', 1)
        attn_out = torch.bmm(attn_weights.permute(0, 2, 1), x).squeeze(1)  # (batch, hidden_dim*2)

        out = self.fc(attn_out)
        return out


# ============================================================
# 加强版1：CNN-BiLSTM-MultiHeadAttention
# 改进：多头注意力替代单头，更强的特征交互能力
# ============================================================
class CNN_BiLSTM_MultiHeadAttention(nn.Module):
    """加强版 - 多头注意力机制"""
    def __init__(self, input_dim=6, hidden_dim=256, num_layers=2, num_classes=5, 
                 dropout=0.5, num_heads=4):
        super(CNN_BiLSTM_MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # CNN部分：3层Conv1d + BN + ReLU + MaxPool（比原始多一层）
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # BiLSTM部分
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, 
                           bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)

        # Multi-Head Attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

        # 分类头
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.permute(0, 2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.dropout(x)

        # Multi-Head Self-Attention
        attn_out, _ = self.multihead_attn(x, x, x)
        x = self.layer_norm(x + attn_out)  # 残差连接

        # 全局平均池化
        out = x.mean(dim=1)
        out = self.fc(out)
        return out


# ============================================================
# 加强版2：CNN-BiLSTM-Transformer
# 改进：LSTM后接Transformer Encoder层，增强全局建模能力
# ============================================================
class CNN_BiLSTM_Transformer(nn.Module):
    """加强版 - LSTM + Transformer混合架构"""
    def __init__(self, input_dim=6, hidden_dim=256, num_layers=2, num_classes=5, 
                 dropout=0.5, nhead=4, num_transformer_layers=2):
        super(CNN_BiLSTM_Transformer, self).__init__()
        self.hidden_dim = hidden_dim

        # CNN部分
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # BiLSTM部分
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, 
                           bidirectional=True, dropout=dropout if num_layers > 1 else 0)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2, nhead=nhead, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        x = self.conv_block(x)
        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # CLS-style：取平均作为序列表示
        out = x.mean(dim=1)
        out = self.classifier(out)
        return out


# ============================================================
# 加强版3：ResNet1D-BiLSTM-Attention
# 改进：用残差卷积块替代普通CNN，防止梯度消失
# ============================================================
class ResBlock1D(nn.Module):
    """1D残差块"""
    def __init__(self, channels):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class ResNet1D_BiLSTM_Attention(nn.Module):
    """加强版 - 残差CNN + BiLSTM + Attention"""
    def __init__(self, input_dim=6, hidden_dim=256, num_layers=2, num_classes=5, 
                 dropout=0.5, num_res_blocks=3):
        super(ResNet1D_BiLSTM_Attention, self).__init__()

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 残差块
        self.res_blocks = nn.Sequential(*[ResBlock1D(hidden_dim) for _ in range(num_res_blocks)])
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # BiLSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, 
                           bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # 分类头
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.res_blocks(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)
        x = self.dropout(x)

        # Attention
        attn_weights = F.softmax(self.attention(x), dim=1)
        attn_out = torch.bmm(attn_weights.permute(0, 2, 1), x).squeeze(1)

        out = self.fc(attn_out)
        return out


# ============================================================
# 模型注册表
# ============================================================
MODEL_REGISTRY = {
    'cnn_bilstm_attention': CNN_BiLSTM_Attention,
    'cnn_bilstm_multihead': CNN_BiLSTM_MultiHeadAttention,
    'cnn_bilstm_transformer': CNN_BiLSTM_Transformer,
    'resnet1d_bilstm_attention': ResNet1D_BiLSTM_Attention,
}


def get_model(name, **kwargs):
    """获取模型实例"""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
