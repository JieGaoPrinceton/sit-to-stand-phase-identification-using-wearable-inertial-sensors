"""
polt.py - 混淆矩阵可视化工具模块

该模块为项目中所有机器学习和深度学习算法提供统一的混淆矩阵热力图绘制功能。
所有Python脚本通过 `import polt` 和 `polt.plot_matrix(y_true, y_pred, title)` 调用。

来源：该文件为项目原始代码中缺失的自定义工具模块，根据调用方式重建。
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


def plot_matrix(y_true, y_pred, title='Confusion Matrix'):
    """
    绘制混淆矩阵热力图

    Parameters
    ----------
    y_true : array-like
        真实标签
    y_pred : array-like
        预测标签
    title : str
        图表标题
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    classes = sorted(set(y_true) | set(y_pred))
    num_classes = len(classes)

    # 归一化
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # 在每个格子中添加数值
    fmt = '.2f'
    thresh = cm_normalized.max() / 2.0
    for i, j in itertools.product(range(cm_normalized.shape[0]), range(cm_normalized.shape[1])):
        plt.text(j, i, format(cm_normalized[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
