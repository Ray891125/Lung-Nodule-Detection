import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast as autocast
import math
from net.layer.ops.util import box_transform, box_transform_inv, clip_boxes

def weighted_focal_loss_for_cross_entropy(logits, labels, weights, gamma=2.):
    log_probs = F.log_softmax(logits, dim=1).gather(1, labels)
    probs     = F.softmax(logits, dim=1).gather(1, labels)
    probs     = F.softmax(logits, dim=1).gather(1, labels)

    loss = - log_probs * (1 - probs) ** gamma
    loss = (weights * loss).sum()/(weights.sum()+1e-12)

    return loss.sum()

def binary_cross_entropy_with_hard_negative_mining(logits, labels, weights, batch_size, num_hard=2):
    # with autocast(enabled=False):
    # classify_loss = nn.BCELoss()
    classify_loss = nn.BCEWithLogitsLoss()
    probs = torch.sigmoid(logits)[:, 0].view(-1, 1)
    pos_idcs = labels[:, 0] == 1

    pos_prob = probs[pos_idcs, 0]
    pos_logits = logits[pos_idcs, 0]
    pos_labels = labels[pos_idcs, 0]

    # For those weights are zero, there are 2 cases,
    # 1. Because we first random sample num_neg negative boxes for OHEM
    # 2. Because those anchor boxes have some overlap with ground truth box,
    #    we want to maintain high sensitivity, so we do not count those as
    #    negative. It will not contribute to the loss
    neg_idcs = (labels[:, 0] == 0) & (weights[:, 0] != 0)
    neg_prob = probs[neg_idcs, 0]
    neg_logits = logits[neg_idcs, 0]
    neg_labels = labels[neg_idcs, 0]
    if num_hard > 0:
        neg_prob, neg_logits, neg_labels = OHEM(neg_prob, neg_logits, neg_labels, num_hard * len(pos_prob))

    pos_correct = 0
    pos_total = 0
    if len(pos_prob) > 0:
        cls_loss = 0.5 * classify_loss(
            pos_logits, pos_labels.float()) + 0.5 * classify_loss(
            neg_logits, neg_labels.float())
        pos_correct = (pos_prob >= 0.5).sum()
        pos_total = len(pos_prob)
    else:
        cls_loss = 0.5 * classify_loss(
            neg_logits, neg_labels.float())


    neg_correct = (neg_prob < 0.5).sum()
    neg_total = len(neg_prob)
    return cls_loss, pos_correct, pos_total, neg_correct, neg_total
def binary_cross_entropy_with_hard_negative_mining_concat(logits, labels, weights, batch_size, num_hard=2):
    # with autocast(enabled=False):
    # classify_loss = nn.BCELoss()
    classify_loss = nn.BCEWithLogitsLoss()
    
    # Concatenate logits and labels
    logits_labels_concat = torch.cat((logits, labels), dim=1)
    
    probs = torch.sigmoid(logits_labels_concat[:, 0]).view(-1, 1)
    pos_idcs = labels[:, 0] == 1

    pos_prob = probs[pos_idcs, 0]
    pos_logits = logits[pos_idcs, 0]
    pos_labels = labels[pos_idcs, 0]

    neg_idcs = (labels[:, 0] == 0) & (weights[:, 0] != 0)
    neg_prob = probs[neg_idcs, 0]
    neg_logits = logits[neg_idcs, 0]
    neg_labels = labels[neg_idcs, 0]
    if num_hard > 0:
        neg_prob, neg_logits, neg_labels = OHEM(neg_prob, neg_logits, neg_labels, num_hard * len(pos_prob))

    # Concatenate negative and positive logits and labels
    logits_concat = torch.cat((neg_logits, pos_logits), dim=0)
    labels_concat = torch.cat((neg_labels, pos_labels), dim=0)

    pos_correct = 0
    pos_total = 0
    if len(pos_prob) > 0:
        pos_correct = (pos_prob >= 0.5).sum()
        cls_loss = classify_loss(logits_concat, labels_concat.float())/ (pos_correct+1)
        
        pos_total = len(pos_prob)
    else:
        cls_loss = classify_loss(logits_concat, labels_concat.float())

    neg_correct = (neg_prob < 0.5).sum()
    neg_total = len(neg_prob)
    return cls_loss, pos_correct, pos_total, neg_correct, neg_total

def OHEM(neg_output, neg_logits, neg_labels, num_hard):
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_logits = torch.index_select(neg_logits, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_logits, neg_labels


def focal_OHEM(neg_output,  neg_weights,neg_logprobs, num_hard):
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_weights = torch.index_select( neg_weights, 0, idcs)
    neg_logprobs = torch.index_select(neg_logprobs, 0, idcs)
    return neg_output, neg_weights,neg_logprobs

def weighted_focal_loss_with_logits_OHEM(logits, labels, weights, gamma=2., alpha=0.25,num_hard=3):
    log_probs = F.logsigmoid(logits)
    probs     = torch.sigmoid(logits)
    probs = torch.clamp(probs, 1e-4, 1.0 - 1e-4)
    
    pos_logprobs = log_probs[labels == 1]
    neg_logprobs = torch.log(1 - probs[labels == 0])
    pos_probs = probs[labels == 1]

    neg_idcs = (labels == 0) & (weights != 0)
    neg_probs = probs[neg_idcs]
    pos_weights = weights[labels == 1]
    neg_weights = weights[labels == 0]
#     print(len(neg_probs[neg_probs > 0.9]))
    if num_hard > 0:
        neg_probs, neg_weights,neg_logprobs = focal_OHEM(neg_probs, neg_weights,neg_logprobs, num_hard * len(pos_probs))    
    neg_probs = 1-neg_probs
    pos_probs = pos_probs.detach()
    neg_probs = neg_probs.detach()

    pos_loss = - (alpha) * pos_logprobs * (1 - pos_probs) ** gamma
    neg_loss = - (1-alpha) * neg_logprobs * (1 - neg_probs) ** gamma
    total_p_loss = (pos_loss * pos_weights).sum()/ (pos_weights.sum() + 1)
    total_n_loss = neg_loss.sum()/ (pos_weights.sum() + 1)
    loss = total_p_loss + total_n_loss
#     print(pos_weights.sum())
#     print(len(neg_loss))

    pos_correct = (probs[labels != 0] > 0.5).sum()
    pos_total = (labels != 0).sum()
    neg_correct = (neg_probs > 0.5).sum()
    neg_total = len(neg_probs)

    return loss, pos_correct, pos_total, neg_correct, neg_total,total_p_loss,total_n_loss

def weighted_focal_loss_with_logits(logits, labels, weights,alpha=0.25, gamma=2.):
    log_probs = F.logsigmoid(logits)
    probs     = torch.sigmoid(logits)
    probs = torch.clamp(probs, 1e-4, 1.0 - 1e-4)
    
    pos_logprobs = log_probs[labels == 1]
    neg_logprobs = torch.log(1 - probs[labels == 0])
    pos_probs = probs[labels == 1]
    
    neg_probs = 1 - probs[labels == 0]
    pos_weights = weights[labels == 1]
    neg_weights = weights[labels == 0]

    pos_probs = pos_probs.detach()
    neg_probs = neg_probs.detach()

    pos_loss = - (alpha)*pos_logprobs * (1 - pos_probs) ** gamma
    neg_loss = -(1-alpha)*neg_logprobs * (1 - neg_probs) ** gamma
    loss = ((pos_loss * pos_weights).sum() + (neg_loss * neg_weights).sum()) / (pos_weights.sum() + 1e-4)
    
#     print(pos_weights.sum())
#     print(neg_weights.sum())
#     print(len(neg_weights))

    pos_correct = (probs[labels != 0] > 0.5).sum()
    pos_total = (labels != 0).sum()
    neg_correct = (probs[labels == 0] < 0.5).sum()
    neg_total = (labels == 0).sum()
    return loss, pos_correct, pos_total, neg_correct, neg_total  
def box_ciou(b1, b2):
    """
    输入为：
    ----------
    b1: (x, y, z, d, h, w)
    b2: (x, y, z, d, h, w)

    返回为：
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    # 求出预测框左上角右下角
    b1_xyz = b1[..., :3]
    b1_dhw = abs(b1[..., 3:])
    b1_dhw_half = b1_dhw / 2.
    b1_mins = b1_xyz - b1_dhw_half
    b1_maxes = b1_xyz + b1_dhw_half

    # 求出真实框左上角右下角
    b2_xyz = b2[..., :3]
    b2_dhw = abs(b2[..., 3:])
    b2_dhw_half = b2_dhw / 2.
    b2_mins = b2_xyz - b2_dhw_half
    b2_maxes = b2_xyz + b2_dhw_half

    # 求真实框和预测框所有的iou
    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_dhw = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area = intersect_dhw[..., 0] * intersect_dhw[..., 1] * intersect_dhw[..., 2]
    b1_area = b1_dhw[..., 0] * b1_dhw[..., 1] * b1_dhw[..., 2]
    b2_area = b2_dhw[..., 0] * b2_dhw[..., 1] * b2_dhw[..., 2]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / torch.clamp(union_area, min=1e-6)

    # 计算中心的差距
    center_distance = torch.sum(torch.pow((b1_xyz - b2_xyz), 2), axis=-1)

    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins = torch.min(b1_mins, b2_mins)
    enclose_maxes = torch.max(b1_maxes, b2_maxes)
    enclose_dhw = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))

    # 计算对角线距离
    enclose_diagonal = torch.sum(torch.pow(enclose_dhw, 2), axis=-1)
    ciou = iou - 1.0 * center_distance / torch.clamp(enclose_diagonal, min=1e-6)

    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(
        b1_dhw[..., 1] / torch.clamp(b1_dhw[..., 2], min=1e-6)) - torch.atan(
        b2_dhw[..., 1] / torch.clamp(b2_dhw[..., 2], min=1e-6))), 2)
    alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
    ciou = ciou - alpha * v
    return 1 - ciou

def rpn_loss(logits, deltas, labels, label_weights, targets, target_weights, cfg, mode='train', delta_sigma=3.0):
    batch_size, num_windows, num_classes = logits.size()
    batch_size_k = batch_size
    labels = labels.long()

    # Calculate classification score
    pos_correct, pos_total, neg_correct, neg_total = 0, 0, 0, 0
    batch_size = batch_size*num_windows
    logits = logits.view(batch_size, num_classes)
    labels = labels.view(batch_size, 1)
    label_weights = label_weights.view(batch_size, 1)

    # Make sure OHEM is performed only in training mode
    if mode in ['train']:
        num_hard = cfg['num_hard']
    else:
        num_hard = 10000000

    rpn_cls_loss, pos_correct, pos_total, neg_correct, neg_total,p_loss,n_loss = \
       weighted_focal_loss_with_logits_OHEM(logits, labels, label_weights,gamma=2., alpha=0.75,num_hard=num_hard)
#-----------------------------------------------------
#     Calculate regression
#-----------------------------------------------------
    deltas = deltas.view(batch_size, 6)
    targets = targets.view(batch_size, 6)
    index = (labels != 0).nonzero()[:,0]
    deltas  = deltas[index]
    targets = targets[index]

    
    rpn_reg_loss = 0
    reg_losses = []
    for i in range(6):
        l = F.smooth_l1_loss(deltas[:, i], targets[:, i])
        rpn_reg_loss += l
        reg_losses.append(l.data.item())
#-----------------------------------------------------
#     Ciou loss
#-----------------------------------------------------
    # index = index % num_windows
    # np_index = index.cpu().numpy()
    # windows =windows[np_index]
    # deltas = deltas.data.cpu().numpy()
    # targets = targets.data.cpu().numpy()
    # rpn_box = rpn_decode(windows, deltas, cfg['box_reg_weight'])
    # gt_box = rpn_decode(windows, targets, cfg['box_reg_weight'])
    # iou_loss = ciou_loss(rpn_box, gt_box)
    # print(iou_loss)
    return 4*rpn_cls_loss, rpn_reg_loss, [pos_correct, pos_total, neg_correct, neg_total,p_loss,n_loss]


def ciou_loss(rpn_proposals, truth_box):
    iou_loss = 0
    
    for slice_idx in range(len(truth_box)):
        
        slice_rpn = rpn_proposals[rpn_proposals[:,0] == slice_idx, 2:-1]
        for nodule_rpn in slice_rpn: 
               
            for nodule_box in truth_box[slice_idx]:
                if box_ciou(nodule_rpn, torch.from_numpy(nodule_box).cuda()) < 0:
                    print("Unavailable box.")
                iou_score = box_ciou(nodule_rpn, torch.from_numpy(nodule_box).cuda())
                iou_loss += iou_score
                
    return iou_loss

def rpn_encode(window, truth_box, weight):
    return box_transform(window, truth_box, weight)

def rpn_decode(window, delta, weight):
    return box_transform_inv(window, delta, weight)