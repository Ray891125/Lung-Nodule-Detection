import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def focal_loss(logits, labels, alpha=0.25, gamma=2.0):
    """
    计算二分类的Focal Loss.
    logits: 预测的logits (bbox num, 1)
    labels: 实际标签 (bbox num,)
    """
    labels = labels.unsqueeze(1).float()  # 将 labels 转为 (bbox num, 1)

    # 计算预测概率
    probs = torch.sigmoid(logits)

    # Focal Loss 的各个项
    pt = probs * labels + (1 - probs) * (1 - labels)
    alpha_factor = labels * alpha + (1 - labels) * (1 - alpha)
    focal_weight = alpha_factor * (1 - pt).pow(gamma)

    # 二分类交叉熵损失
    bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    focal_loss = focal_weight * bce_loss

    return focal_loss.mean()

def rcnn_loss(logits, deltas, labels, targets, deltas_sigma=1.0, alpha=0.25, gamma=2.0):
    # batch_size = logits.size(0) 不需要这个变量，因为bbox num已经是整个batch的大小

    # 分类损失使用 Focal Loss
    rcnn_cls_loss = focal_loss(logits, labels, alpha=alpha, gamma=gamma)

    # 仅对正样本 (label为1) 进行回归损失计算
    positive_indices = labels > 0
    num_pos = positive_indices.sum().item()

    if num_pos > 0:
        # 选出正样本的 deltas 和 targets
        deltas_pos = deltas[positive_indices]

        # 计算回归损失 (Smooth L1 loss)
        rcnn_reg_loss = F.smooth_l1_loss(deltas_pos, targets)
    else:
        rcnn_reg_loss = torch.tensor(0.0, device=deltas.device)

    # 返回分类损失，回归损失，以及回归损失的各维度的平均值
    return rcnn_cls_loss, rcnn_reg_loss, [1,1,1,1,1,1]


if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


 