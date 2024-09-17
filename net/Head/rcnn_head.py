from net.layer import *
from single_config import net_config as config
bn_momentum = 0.1
affine = True

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
            # pe = self.position_embedding(fe1)
            # fe1 = self.transformer(fe1,pe) # 64
            fe1_up = self.up2(fe1)

            fe2 = self.back2(torch.cat((fe1_up, out1[int(b), :, int(z_start / 2):int(z_end / 2), int(y_start / 2):int(y_end / 2), int(x_start / 2):int(x_end / 2)].unsqueeze(0)), 1))
            # pe = self.position_embedding(fe2)
            # fe2 = self.transformer(fe2,pe) # 64
            
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