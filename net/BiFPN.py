
import sys

from net.Encoder import resnet
from net.Module import cgnl
from net.layer import *
# from net.layer.ops.Matcher import *
# from net.layer.ops.NMS import *
# from net.layer.ops import*
from single_config import net_config as config
import copy
from torch.nn.parallel.data_parallel import data_parallel
import time

from utils.util import center_box_to_coord_box, ext2factor, clip_boxes
from torch.nn.parallel import data_parallel
import random
from scipy.stats import norm

from .Module.transformer import build_transformer
from .Module.position_encoding import build_position_encoding
import torch.nn.functional as F
import torch.nn as nn
import torch
bn_momentum = 0.1
affine = True

class DepthwiseConvBlock3D(nn.Module):
    """
    Depthwise separable convolution for 3D inputs.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(DepthwiseConvBlock3D, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size, stride, 
                               padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False)
        
        self.bn = nn.BatchNorm3d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()
        
    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)
    
class ConvBlock3D(nn.Module):
    """
    Convolution block with Batch Normalization and ReLU activation for 3D inputs.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.act(x)

class BiFPNBlock3D(nn.Module):
    """
    Bi-directional Feature Pyramid Network for 3D inputs, using Upsample and MaxPooling for downsampling.
    """
    def __init__(self, feature_size=128, epsilon=0.0001):
        super(BiFPNBlock3D, self).__init__()
        self.epsilon = epsilon
        
        # Define depthwise convolutions
        self.p2_td = DepthwiseConvBlock3D(feature_size, feature_size)
        self.p3_td = DepthwiseConvBlock3D(feature_size, feature_size)
        self.p4_td = DepthwiseConvBlock3D(feature_size, feature_size)
        self.p5_td = DepthwiseConvBlock3D(feature_size, feature_size)
        
        self.p3_out = DepthwiseConvBlock3D(feature_size, feature_size)
        self.p4_out = DepthwiseConvBlock3D(feature_size, feature_size)
        self.p5_out = DepthwiseConvBlock3D(feature_size, feature_size)
        self.p2_out = DepthwiseConvBlock3D(feature_size, feature_size)
        
        # Upsample layers for top-down pathway
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # MaxPool layers for bottom-up pathway (downsampling)
        self.p4_down_sample = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.p5_down_sample = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.p6_down_sample = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.p7_down_sample = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Init weights
        self.w1 = nn.Parameter(torch.Tensor(2, 3))
        self.w2 = nn.Parameter(torch.Tensor(3, 3))
        nn.init.constant_(self.w1, 1.0)
        nn.init.constant_(self.w2, 1.0)
        
        self.w1_relu = nn.ReLU()
        self.w2_relu = nn.ReLU()

    def forward(self, inputs):
        p2_x, p3_x, p4_x, p5_x = inputs

        # Normalize weights
        w1 = torch.softmax(self.w1, dim=0)
        w2 = torch.softmax(self.w2, dim=0)
        
        # Top-Down Pathway using Upsample
        p5_td = p5_x
        p4_td = self.p4_td(w1[0, 0] * p4_x + w1[1, 0] * self.p5_upsample(p5_td))
        p3_td = self.p3_td(w1[0, 1] * p3_x + w1[1, 1] * self.p4_upsample(p4_td))
        p2_td = self.p2_td(w1[0, 2] * p2_x + w1[1, 2] * self.p3_upsample(p3_td))
        
        # Bottom-Up Pathway using MaxPool for downsampling
        p2_out = p2_td
        p3_out = self.p3_out(w2[0, 0] * p3_x + w2[1, 0] * p3_td + w2[2, 0] * self.p4_down_sample(p2_out))
        p4_out = self.p4_out(w2[0, 1] * p4_x + w2[1, 1] * p4_td + w2[2, 1] * self.p5_down_sample(p3_out))
        p5_out = self.p5_out(w2[0, 2] * p5_x + w2[1, 2] * p5_td + w2[2, 2] * self.p6_down_sample(p4_out))

        return [p2_out,p3_out,p4_out,p5_out]
    


class FeatureNet(nn.Module):
    def __init__(self, config,size = [64,64,64,64], feature_size=128, num_layers=1, epsilon=0.0001):
        super(FeatureNet, self).__init__()
        self.resnet50 = resnet.resnet50()
        
        self.p2 = nn.Conv3d(size[0], feature_size, kernel_size=1, stride=1, padding=0)
        self.p3 = nn.Conv3d(size[1], feature_size, kernel_size=1, stride=1, padding=0)
        self.p4 = nn.Conv3d(size[2], feature_size, kernel_size=1, stride=1, padding=0)
        self.p5 = nn.Conv3d(size[3], feature_size, kernel_size=1, stride=1, padding=0)

        self.position_embedding = build_position_encoding(config)
        self.transformer = build_transformer(config)

        bifpns = []
        for _ in range(num_layers):
            bifpns.append(BiFPNBlock3D(feature_size))
        self.bifpn = nn.Sequential(*bifpns)
    
    def forward(self,x):
        x1, c2, c3, c4, c5 = self.resnet50(x)
    
        p2_x = self.p2(c2)
        p3_x = self.p3(c3)
        p4_x = self.p4(c4)
        p5_x = self.p5(c5)
        
        features = [p2_x, p3_x, p4_x, p5_x]
        out2, out3, out4, out5 = self.bifpn(features)
        return [x1, c2, out3], c2
class RpnHead(nn.Module):
    def __init__(self, config, in_channels=128):
        super(RpnHead, self).__init__()
        self.drop = nn.Dropout3d(p=0.5, inplace=False)
        self.conv = nn.Sequential(nn.Conv3d(in_channels, 64, kernel_size=1),
                                    nn.ReLU())
        self.logits = nn.Conv3d(64, 1 * len(config['anchors']), kernel_size=1)
        self.deltas = nn.Conv3d(64, 6 * len(config['anchors']), kernel_size=1)

    def forward(self, f):
        # out = self.drop(f)
        out = self.conv(f)

        logits = self.logits(out)
        deltas = self.deltas(out)
        size = logits.size()
        logits = logits.view(logits.size(0), logits.size(1), -1)
        logits = logits.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 1)
        
        size = deltas.size()
        deltas = deltas.view(deltas.size(0), deltas.size(1), -1)
        deltas = deltas.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 6)
        

        return logits, deltas

class RcnnHead(nn.Module):
    def __init__(self, cfg, in_channels=128):
        super(RcnnHead, self).__init__()
        self.num_class = cfg['num_class']
        self.crop_size = cfg['rcnn_crop_size']

        self.fc1 = nn.Linear(in_channels * self.crop_size[0] * self.crop_size[1] * self.crop_size[2], 512)
        self.fc2 = nn.Linear(512, 256)
        self.logit = nn.Linear(256, self.num_class)
        self.delta = nn.Linear(256, self.num_class * 6)

    def forward(self, crops):
        x = crops.view(crops.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        # x = F.dropout(x, 0.5, training=self.training)
        logits = self.logit(x)
        deltas = self.delta(x)

        return logits, deltas

class MaskHead(nn.Module):
    def __init__(self, cfg, in_channels=128):
        super(MaskHead, self).__init__()
        self.num_class = cfg['num_class']

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))
        
        self.back1 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))
        
        self.back2 = nn.Sequential(
            nn.Conv3d(96, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))
        
        self.back3 = nn.Sequential(
            nn.Conv3d(65, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))

        for i in range(self.num_class):
            setattr(self, 'logits' + str(i + 1), nn.Conv3d(64, 1, kernel_size=1))

    def forward(self, detections, features):
        img, f_2, f_4 = features  

        # Squeeze the first dimension to recover from protection on avoiding split by dataparallel      
        img = img.squeeze(0)
        f_2 = f_2.squeeze(0)
        f_4 = f_4.squeeze(0)

        _, _, D, H, W = img.shape
        out = []

        for detection in detections:
            b, z_start, y_start, x_start, z_end, y_end, x_end, cat = detection

            up1 = f_4[b, :, z_start / 4:z_end / 4, y_start / 4:y_end / 4, x_start / 4:x_end / 4].unsqueeze(0)
            up2 = self.up2(up1)
            up2 = self.back2(torch.cat((up2, f_2[b, :, z_start / 2:z_end / 2, y_start / 2:y_end / 2, x_start / 2:x_end / 2].unsqueeze(0)), 1))
            up3 = self.up3(up2)
            im = img[b, :, z_start:z_end, y_start:y_end, x_start:x_end].unsqueeze(0)
            up3 = self.back3(torch.cat((up3, im), 1))

            logits = getattr(self, 'logits' + str(int(cat)))(up3)
            logits = logits.squeeze()
 
            mask = Variable(torch.zeros((D, H, W))).cuda()
            mask[z_start:z_end, y_start:y_end, x_start:x_end] = logits
            mask = mask.unsqueeze(0)
            out.append(mask)

        out = torch.cat(out, 0)

        return out


def crop_mask_regions(masks, crop_boxes):
    out = []
    for i in range(len(crop_boxes)):
        b, z_start, y_start, x_start, z_end, y_end, x_end, cat = crop_boxes[i]
        m = masks[i][z_start:z_end, y_start:y_end, x_start:x_end].contiguous()
        out.append(m)
    
    return out


def top1pred(boxes):
    res = []
    pred_cats = np.unique(boxes[:, -1])
    for cat in pred_cats:
        preds = boxes[boxes[:, -1] == cat]
        res.append(preds[0])
        
    res = np.array(res)
    return res

def random1pred(boxes):
    res = []
    pred_cats = np.unique(boxes[:, -1])
    for cat in pred_cats:
        preds = boxes[boxes[:, -1] == cat]
        idx = random.sample(range(len(preds)), 1)[0]
        res.append(preds[idx])
        
    res = np.array(res)
    return res


class CropRoi(nn.Module):
    def __init__(self, cfg, rcnn_crop_size, in_channels = 128):
        super(CropRoi, self).__init__()
        self.cfg = cfg
        self.rcnn_crop_size  = rcnn_crop_size
        self.scale = cfg['stride']
        self.DEPTH, self.HEIGHT, self.WIDTH = cfg['crop_size']

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels, 64, kernel_size=1, padding=0),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear',align_corners=True),
            nn.Conv3d(64, 64, kernel_size=1, padding=0),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace=True))
        self.back2 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=1, padding=0),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace=True))
        self.back3 = nn.Sequential(
            nn.Conv3d(65, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace=True))

    def forward(self, f, inputs, proposals):
        img, out1, comb2 = f
        self.DEPTH, self.HEIGHT, self.WIDTH = inputs.shape[2:]

        img = img.squeeze(0)
        out1 = out1.squeeze(0)
        comb2 = comb2.squeeze(0)

        crops = []
        for p in proposals:
            b, z_start, y_start, x_start, z_end, y_end, x_end = p

            # Slice 0 dim, should never happen
            c0 = np.array(torch.Tensor([z_start, y_start, x_start]))
            c1 = np.array(torch.Tensor([z_end, y_end, x_end]))
            if np.any((c1 - c0) < 1): #np.any((c1 - c0).cpu().data.numpy() < 1):
                # c0=c0+1
                # c1=c1+1
                for i in range(3):
                    if c1[i] == 0:
                        c1[i] = c1[i] + 4
                    if c1[i] - c0[i] == 0:
                        c1[i] = c1[i] + 4
                print(p)
                print('c0:', c0, ', c1:', c1)
            z_end, y_end, x_end = c1

            fe1 = comb2[int(b), :, int(z_start / 4):int(z_end / 4), int(y_start / 4):int(y_end / 4), int(x_start / 4):int(x_end / 4)].unsqueeze(0)
            fe1_up = self.up2(fe1)

            fe2 = self.back2(torch.cat((fe1_up, out1[int(b), :, int(z_start / 2):int(z_end / 2), int(y_start / 2):int(y_end / 2), int(x_start / 2):int(x_end / 2)].unsqueeze(0)), 1))
            # fe2_up = self.up3(fe2)
            # im = img[int(b), :, int(z_start / 2):int(z_end / 2), int(y_start / 2):int(y_end / 2), int(x_start / 2):int(x_end / 2)].unsqueeze(0)
            # up3 = self.back3(torch.cat((fe2_up, im), 1))
            # crop = up3.squeeze()

            crop = fe2.squeeze()
            # crop = f[b, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]]
            crop = F.adaptive_max_pool3d(crop, self.rcnn_crop_size)
            crops.append(crop)

        crops = torch.stack(crops)

        return crops

class TransBiFPN(nn.Module):
    def __init__(self, cfg, mode='train'):
        super(TransBiFPN, self).__init__()

        self.cfg = cfg
        self.mode = mode
        self.feature_net = FeatureNet(config)
        self.rpn = RpnHead(config, in_channels=128)
        self.rcnn_head = RcnnHead(config, in_channels=64)
        self.rcnn_crop = CropRoi(self.cfg, cfg['rcnn_crop_size'])
        self.use_rcnn = False
        # self.rpn_loss = Loss(cfg['num_hard'])
        

    def forward(self, inputs, truth_boxes, truth_labels, split_combiner=None, nzhw=None):
        # features, feat_4 = self.feature_net(inputs)
        #features, feat_4 = data_parallel(self.feature_net, (inputs))
        #fs = features[-1]
        features, feat_4= self.feature_net(inputs)
        fs = features[-1]
        self.rpn_logits_flat, self.rpn_deltas_flat = data_parallel(self.rpn, fs)

        b,D,H,W,_,num_class = self.rpn_logits_flat.shape

        self.rpn_logits_flat = self.rpn_logits_flat.view(b, -1, 1)
        self.rpn_deltas_flat = self.rpn_deltas_flat.view(b, -1, 6)


        self.rpn_window = make_rpn_windows(fs, self.cfg)
        self.rpn_proposals = []
        if self.use_rcnn or self.mode in ['eval', 'test']:
            self.rpn_proposals = rpn_nms(self.cfg, self.mode, inputs, self.rpn_window,
                  self.rpn_logits_flat, self.rpn_deltas_flat)

        if self.mode in ['train', 'valid']:
            # self.rpn_proposals = torch.zeros((0, 8)).cuda()
            self.rpn_labels, self.rpn_label_assigns, self.rpn_label_weights, self.rpn_targets, self.rpn_target_weights = \
                make_rpn_target(self.cfg, self.mode, inputs, self.rpn_window, truth_boxes, truth_labels)

            if self.use_rcnn:
                # self.rpn_proposals = torch.zeros((0, 8)).cuda()
                self.rpn_proposals, self.rcnn_labels, self.rcnn_assigns, self.rcnn_targets = \
                    make_rcnn_target(self.cfg, self.mode, inputs, self.rpn_proposals,
                        truth_boxes, truth_labels)

        #rcnn proposals
        self.detections = copy.deepcopy(self.rpn_proposals)
        self.ensemble_proposals = copy.deepcopy(self.rpn_proposals)

        self.mask_probs = []
        if self.use_rcnn:
            if len(self.rpn_proposals) > 0:
                proposal = self.rpn_proposals[:, [0, 2, 3, 4, 5, 6, 7]].cpu().numpy().copy()
                proposal[:, 1:] = center_box_to_coord_box(proposal[:, 1:])
                proposal = proposal.astype(np.int64)
                proposal[:, 1:] = ext2factor(proposal[:, 1:], 4)
                proposal[:, 1:] = clip_boxes(proposal[:, 1:], inputs.shape[2:])
                # rcnn_crops = self.rcnn_crop(features, inputs, torch.from_numpy(proposal).cuda())
                features = [t.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1, -1, -1, -1) for t in features]
                rcnn_crops = data_parallel(self.rcnn_crop, (features, inputs, torch.from_numpy(proposal).cuda()))
                # rcnn_crops = self.rcnn_crop(fs, inputs, self.rpn_proposals)
                # self.rcnn_logits, self.rcnn_deltas = self.rcnn_head(rcnn_crops)
                self.rcnn_logits, self.rcnn_deltas = data_parallel(self.rcnn_head, rcnn_crops)
                self.detections, self.keeps = rcnn_nms(self.cfg, self.mode, inputs, self.rpn_proposals, 
                                                                        self.rcnn_logits, self.rcnn_deltas)

                if self.mode in ['eval']:
                    #     Ensemble
                    fpr_res = get_probability(self.cfg, self.mode, inputs, self.rpn_proposals, self.rcnn_logits,
                                              self.rcnn_deltas)
                    if self.ensemble_proposals.shape[0] == fpr_res.shape[0]:
                        self.ensemble_proposals[:, 1] = (self.ensemble_proposals[:, 1] + fpr_res[:, 0]) / 2

    def loss(self, targets=None):
        cfg  = self.cfg
    
        self.rcnn_cls_loss, self.rcnn_reg_loss = torch.zeros(1).cuda(), torch.zeros(1).cuda()
        rcnn_stats = None
    
        self.rpn_cls_loss, self.rpn_reg_loss, rpn_stats = \
           rpn_loss( self.rpn_logits_flat, self.rpn_deltas_flat, self.rpn_labels,
            self.rpn_label_weights, self.rpn_targets, self.rpn_target_weights, self.cfg, mode=self.mode)
    
        if self.use_rcnn:
            self.rcnn_cls_loss, self.rcnn_reg_loss, rcnn_stats = \
                rcnn_loss(self.rcnn_logits, self.rcnn_deltas, self.rcnn_labels, self.rcnn_targets)
    
        self.total_loss = self.rpn_cls_loss + self.rpn_reg_loss \
                          + self.rcnn_cls_loss +  self.rcnn_reg_loss

    
        return self.total_loss, rpn_stats, rcnn_stats

    def set_mode(self, mode):
        assert mode in ['train', 'valid', 'eval', 'test']
        self.mode = mode
        if mode in ['train']:
            self.train()
        else:
            self.eval()

    def set_anchor_params(self, anchor_ids, anchor_params):
        self.anchor_ids = anchor_ids
        self.anchor_params = anchor_params

    def crf(self, detections):
        """
        detections: numpy array of detection results [b, z, y, x, d, h, w, p]
        """
        res = []
        config = self.cfg
        anchor_ids = self.anchor_ids
        anchor_params = self.anchor_params
        anchor_centers = []

        for a in anchor_ids:
            # category starts from 1 with 0 denoting background
            # id starts from 0
            cat = a + 1
            dets = detections[detections[:, -1] == cat]
            if len(dets):
                b, p, z, y, x, d, h, w, _ = dets[0]
                anchor_centers.append([z, y, x])
                res.append(dets[0])
            else:
                # Does not have anchor box
                return detections
        
        pred_cats = np.unique(detections[:, -1]).astype(np.uint8)
        for cat in pred_cats:
            if cat - 1 not in anchor_ids:
                cat = int(cat)
                preds = detections[detections[:, -1] == cat]
                score = np.zeros((len(preds),))
                roi_name = config['roi_names'][cat - 1]

                for k, params in enumerate(anchor_params):
                    param = params[roi_name]
                    for i, det in enumerate(preds):
                        b, p, z, y, x, d, h, w, _ = det
                        d = np.array([z, y, x]) - np.array(anchor_centers[k])
                        prob = norm.pdf(d, param[0], param[1])
                        prob = np.log(prob)
                        prob = np.sum(prob)
                        score[i] += prob

                res.append(preds[score == score.max()][0])
            
        res = np.array(res)
        return res

if __name__ == '__main__':
   import torchsummary
   net = FeatureNet()

   # Create the input tensor with specific dimensions
   input_tensor = torch.randn(1, 128, 128, 128)

   # Move to GPU if needed
   if torch.cuda.is_available():
       net = net.to('cuda')
       input_tensor = input_tensor.to('cuda')

   torchsummary.summary(net, (1, 128, 128, 128), device='cuda')
   print("summary")

