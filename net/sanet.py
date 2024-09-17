
import sys

import torch.nn.functional as F
import torch.nn as nn
import torch

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

from net.Head.rcnn_head import RcnnHead,CropRoi
from net.Head.rpn_head import RpnHead
from net.Decoder.FeatureNet import FeatureNet
from net.Encoder import resnet
bn_momentum = 0.1
affine = True




class SANet(nn.Module):
    def __init__(self, cfg, mode='train'):
        super(SANet, self).__init__()

        self.cfg = cfg
        self.mode = mode
        self.encoder = resnet.resnet50()
        self.decoder = FeatureNet(config)
        self.rpn = RpnHead(config, in_channels=128)
        self.rcnn_head = RcnnHead(config, in_channels=64)
        self.rcnn_crop = CropRoi(self.cfg, cfg['rcnn_crop_size'])
        self.use_rcnn = False
        

    def forward(self, inputs, truth_boxes, truth_labels, split_combiner=None, nzhw=None):
        #-------------------------------------
        # 1. get features 
        #-------------------------------------
        encode_feature = self.encoder(inputs)
        features, feat_4 = self.decoder(encode_feature)
        fs = features[-1]
        #-------------------------------------
        # 2. rpn detector
        #-------------------------------------
        self.rpn_logits_flat, self.rpn_deltas_flat = data_parallel(self.rpn, fs)
        b,D,H,W,_,num_class = self.rpn_logits_flat.shape
        self.rpn_logits_flat = self.rpn_logits_flat.view(b, -1, 1)
        self.rpn_deltas_flat = self.rpn_deltas_flat.view(b, -1, 6)

        #-------------------------------------
        # 3. anchor generator(ok)
        #-------------------------------------
        self.rpn_window = make_rpn_windows(fs, self.cfg)


        #-------------------------------------
        # 3. rpn matcher(ok)
        #-------------------------------------
        if self.mode in ['train', 'valid']:
            # self.rpn_proposals = torch.zeros((0, 8)).cuda()
            self.rpn_labels, self.rpn_label_assigns, self.rpn_label_weights, self.rpn_targets, self.rpn_target_weights = \
                make_rpn_target(self.cfg, self.mode, inputs, self.rpn_window, truth_boxes, truth_labels)
        #-------------------------------------
        # 4. nms(ok)
        #-------------------------------------
        self.rpn_proposals = []
        if self.use_rcnn or self.mode in ['eval', 'test']:
            self.rpn_proposals = rpn_nms(self.cfg, self.mode, inputs, self.rpn_window,
                  self.rpn_logits_flat, self.rpn_deltas_flat)


        #-------------------------------------
        # 5. rcnn matcher
        #-------------------------------------
            if self.use_rcnn:
                # self.rpn_proposals = torch.zeros((0, 8)).cuda()
                self.rpn_proposals, self.rcnn_labels, self.rcnn_assigns, self.rcnn_targets = \
                    make_rcnn_target(self.cfg, self.mode, inputs, self.rpn_proposals,
                        truth_boxes, truth_labels)

        #-------------------------------------
        # 6. rcnn detector
        #-------------------------------------
        self.detections = copy.deepcopy(self.rpn_proposals)
        self.ensemble_proposals = copy.deepcopy(self.rpn_proposals)

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
    net = SANet(config)

    input = torch.rand([2,1,304,384,384])
    input = Variable(input)
    net(input, None, None)

    import torchsummary
    torchsummary.summary(net, device='cuda')
    print("summary")

