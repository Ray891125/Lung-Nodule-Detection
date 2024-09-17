import sys
from net.layer import *
from single_config import net_config as config
import copy
from torch.nn.parallel.data_parallel import data_parallel
import time
import torch.nn.functional as F
from utils.util import center_box_to_coord_box, ext2factor, clip_boxes
from torch.nn.parallel import data_parallel
import random
from scipy.stats import norm
from net.Module.SCconv import SCConv3D
from net.Module.eca_module import ECABlock
from net.Module.DSAM import DsamModule
import math
bn_momentum = 0.1
affine = True

class ResBlock3d1(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm3d(n_out, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(n_out, momentum=bn_momentum)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.BatchNorm3d(n_out, momentum=bn_momentum))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out

class ResBlock3d(nn.Module):
    pooling_r = 4 # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, inplanes, planes, stride = 1,
                 cardinality=1, bottleneck_width=32,
                 avd=False, dilation=1, is_first=False,
                 norm_layer=nn.BatchNorm3d):
        # parameter for resnet
        super(ResBlock3d, self).__init__()

        if stride != 1 or planes != inplanes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(inplanes, planes, kernel_size = 1, stride = stride),
                nn.BatchNorm3d(planes, momentum=bn_momentum))
        else:
            self.shortcut = None
        # parameter for SCconv
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality

        # SCconv3D won't change the channel of input here
        self.SCconv1 = SCConv3D(inplanes, planes, stride = stride,
                 cardinality=cardinality, bottleneck_width=bottleneck_width,
                 avd=avd, dilation=dilation, is_first=is_first,
                 norm_layer=norm_layer)
        # self.SCconv2 = SCConv3D(planes, planes, stride = stride,
        #          cardinality=cardinality, bottleneck_width=bottleneck_width,
        #          avd=avd, dilation=dilation, is_first=is_first,
        #          norm_layer=norm_layer)
        self.eca1 = ECABlock(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dilation = dilation


    def forward(self, x):
        residual = x
        # for scconv
        out = self.SCconv1(x)
        # out = self.SCconv2(out)
        out = self.eca1(out)
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out += residual
        out = self.relu(out)
        return out

class FeatureNet(nn.Module):
    def __init__(self, config, in_channels, out_channels):
        super(FeatureNet, self).__init__()
        self.preBlock = nn.Sequential(
            nn.Conv3d(in_channels, 24, kernel_size = 3, padding = 1, stride=2),
            nn.BatchNorm3d(24, momentum=bn_momentum),
            nn.ReLU(inplace = True),
            nn.Conv3d(24, 24, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(24, momentum=bn_momentum),
            nn.ReLU(inplace = True))

        self.forw1 = nn.Sequential(
            ResBlock3d(24, 48),
            ResBlock3d(48, 48))

        self.forw2 = nn.Sequential(
            ResBlock3d(48, 72),
            ResBlock3d(72, 72))

        self.forw3 = nn.Sequential(
            ResBlock3d(72, 96),
            ResBlock3d(96, 96),
            ResBlock3d(96, 96))

        self.forw4 = nn.Sequential(
            ResBlock3d(96, 120),
            ResBlock3d(120, 120),
            ResBlock3d(120, 120))

        # skip connection in U-net
        self.back2 = nn.Sequential(
            # 注意back2、back3順序
            # 128 + 72 = 200
            ResBlock3d(200, 128),
            ResBlock3d(128, 128),
            ResBlock3d(128, 128))

        # skip connection in U-net
        self.back3 = nn.Sequential(
            # 注意back2、back3順序
            #120+96=216
            ResBlock3d(216, 128),
            ResBlock3d(128, 128),
            ResBlock3d(128, 128))
        self.dsam1 = DsamModule(96)
        self.dsam2 = DsamModule(72)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        
        # upsampling in U-net
        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(120, 120, kernel_size=2, stride=2),
            nn.BatchNorm3d(120),
            nn.ReLU(inplace=True))

        # upsampling in U-net
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True))
        
        # multi-scale aggregation
#         self.MSA1 = MultiScalAggregation3fs([48,72,96])
#         self.MSA2 = MultiScalAggregation3fs([72,96,120])
#         self.MSA3 = MultiScalAggregation2fs([96,120])


    def forward(self, x):
        out = self.preBlock(x)#24
        out_pool = out
        out1 = self.forw1(out_pool)#48
        out1_pool, _ = self.maxpool2(out1)
        out2 = self.forw2(out1_pool)#72
        #out2 = self.drop(out2)
        out2_pool, _ = self.maxpool3(out2)
        out3 = self.forw3(out2_pool)#96
        out3_pool, _ = self.maxpool4(out3)
        out4 = self.forw4(out3_pool)#120
        #out4 = self.drop(out4)
#         mid3 = self.MSA3([out3,out4])       # output = 6 * 6 * 6 * 120
#         mid2 = self.MSA2([out2,out3,out4])  # 12* 12* 12* 96
#         mid1 = self.MSA1([out1,out2,out3])  # 24* 24* 24* 72
        # mid3 = self.dsam1(out3)
        # mid2 = self.dsam2(out2)
        rev3 = self.path1(out4)             # upsample,12*12*12*120
        comb3 = self.back3(torch.cat((rev3, out3), 1))#120+96 ->128
        rev2 = self.path2(comb3)            # 24*24*24*128
        comb2 = self.back2(torch.cat((rev2, out2), 1))#128+72->128
        return [x, out1, comb2], out2

class RpnHead(nn.Module):
    def __init__(self, config, in_channels=128):
        super(RpnHead, self).__init__()
        self.drop = nn.Dropout3d(p=0.5, inplace=False)
        self.conv = nn.Sequential(nn.Conv3d(in_channels, 64, kernel_size=1),
                                    nn.ReLU())
        self.logits = nn.Conv3d(64, 1 * len(config['anchors']), kernel_size=1)
        self.deltas = nn.Conv3d(64, 6 * len(config['anchors']), kernel_size=1)

    def forward(self, f):
        out = self.drop(f)
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
        x = F.dropout(x, 0.5, training=self.training)
        logits = self.logit(x)
        deltas = self.delta(x)

        return logits, deltas



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
            nn.Conv3d(in_channels, 48, kernel_size=1, padding=0),
            nn.InstanceNorm3d(48, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear',align_corners=True),
            nn.Conv3d(64, 64, kernel_size=1, padding=0),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace=True))
        self.back2 = nn.Sequential(
            nn.Conv3d(96, 64, kernel_size=1, padding=0),
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
            #
            # im = img[int(b), :, int(z_start / 2):int(z_end / 2), int(y_start / 2):int(y_end / 2), int(x_start / 2):int(x_end / 2)].unsqueeze(0)
            # up3 = self.back3(torch.cat((fe2_up, im), 1))
            # crop = up3.squeeze()

            crop = fe2.squeeze()
            # crop = f[b, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]]
            crop = F.adaptive_max_pool3d(crop, self.rcnn_crop_size)
            crops.append(crop)

        crops = torch.stack(crops)

        return crops

class MsaNet(nn.Module):
    def __init__(self, cfg, mode='train'):
        super(MsaNet, self).__init__()

        self.cfg = cfg
        self.mode = mode
        self.feature_net = FeatureNet(config,1  ,128)
        self.rpn = RpnHead(config, in_channels=128)
        self.rcnn_head = RcnnHead(config, in_channels=64)
        self.rcnn_crop = CropRoi(self.cfg, cfg['rcnn_crop_size'])
        self.use_rcnn = False
        # self.rpn_loss = Loss(cfg['num_hard'])
        

    def forward(self, inputs, truth_boxes, truth_labels, split_combiner=None, nzhw=None):
        # features, feat_4 = self.feature_net(inputs)
        if self.mode in ['train', 'valid']:

            features, feat_4 = data_parallel(self.feature_net, (inputs))
            fs = features[-1]
            fs_shape = fs.shape
            # self.rpn_logits_flat, self.rpn_deltas_flat = self.rpn(fs)
            self.rpn_logits_flat, self.rpn_deltas_flat = data_parallel(self.rpn, fs)

            b,D,H,W,_,num_class = self.rpn_logits_flat.shape

            self.rpn_logits_flat = self.rpn_logits_flat.view(b, -1, 1)
            self.rpn_deltas_flat = self.rpn_deltas_flat.view(b, -1, 6)
            self.rpn_window = make_rpn_windows(fs_shape, self.cfg)

        elif self.mode == 'eval':
            input_shape = inputs.shape
            B,C,D,H,W = input_shape
            # breakpoint()
            self.rpn_logits_flat = torch.tensor([]).cuda()
            self.rpn_deltas_flat = torch.tensor([]).cuda()
            self.rpn_window = np.empty((0,6))
            
            last_crop = 0

            rpn_windows = make_rpn_windows([1, 128, 32, 128, 128], self.cfg)

            for i in range(math.ceil(D/64)-1):

                if i*64+128 >= D:
                    crop_input = inputs[:,:,-128:]
                    overlap_slice = (D-(i*64))//4
                    last_crop = 1
                    start_slice = D-128
                else:
                    crop_input = inputs[:,:,i*64: i*64+128]
                    overlap_slice = 8
                    start_slice = i*64

                with torch.no_grad():
                    features, _ = data_parallel(self.feature_net,(crop_input))
                    self.fs = features[-1]
                    fs_shape = self.fs.shape
                    crop_rpn_logits_flat, crop_rpn_deltas_flat = data_parallel(self.rpn, self.fs)
                
                b,d,_,_,_,_ = crop_rpn_logits_flat.shape

                crop_rpn_window = rpn_windows.copy() + [start_slice, 0, 0, 0, 0, 0]
                if i == 0:
                    crop_rpn_logits_flat = crop_rpn_logits_flat[:,:24]
                    crop_rpn_deltas_flat = crop_rpn_deltas_flat[:,:24]
                    crop_rpn_window = crop_rpn_window.reshape(d, -1, 6)[:24]
                elif last_crop == 1:
                    crop_rpn_logits_flat = crop_rpn_logits_flat[:,overlap_slice//2:]
                    crop_rpn_deltas_flat = crop_rpn_deltas_flat[:,overlap_slice//2:]
                    crop_rpn_window = crop_rpn_window.reshape(d, -1, 6)[overlap_slice//2:]
                else:
                    crop_rpn_logits_flat = crop_rpn_logits_flat[:,8:24]
                    crop_rpn_deltas_flat = crop_rpn_deltas_flat[:,8:24]
                    crop_rpn_window = crop_rpn_window.reshape(d, -1, 6)[8:24]
                crop_rpn_window = crop_rpn_window.reshape(-1, 6)

                crop_rpn_logits_flat = crop_rpn_logits_flat.view(b, -1, 1)
                crop_rpn_deltas_flat = crop_rpn_deltas_flat.view(b, -1, 6)

                self.rpn_logits_flat = torch.cat((self.rpn_logits_flat, crop_rpn_logits_flat), dim=1)
                self.rpn_deltas_flat = torch.cat((self.rpn_deltas_flat, crop_rpn_deltas_flat), dim=1)
                self.rpn_window = np.concatenate((self.rpn_window, crop_rpn_window), axis=0)


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
   net = FeatureNet(config,1,1)

   # Create the input tensor with specific dimensions
   input_tensor = torch.randn(1, 128, 128, 128)

   # Move to GPU if needed
   if torch.cuda.is_available():
       net = net.to('cuda')
       input_tensor = input_tensor.to('cuda')

   torchsummary.summary(net, (1, 128, 128, 128), device='cuda')
   print("summary")
