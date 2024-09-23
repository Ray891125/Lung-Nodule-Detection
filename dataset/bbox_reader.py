import os
import time
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import rotate
#import SimpleITK as sitk
import nrrd
import warnings
import json
import matplotlib.pyplot as plt
#from dataset.split_combine import SplitComb
import torch.nn.functional as F
from dataset.split_combine import SplitComb
class BboxReader(Dataset):
    def __init__(self, data_dir, set_name, augtype, cfg, mode='train'):
        #self.split_com ber = SplitComb(64,cfg['max_stride'],cfg['stride'],32,cfg['pad_value'])
        self.split_comber = SplitComb(crop_size=cfg['crop_size'], overlap_size=[32,32,32], pad_value=cfg['pad_value'], do_padding=True)
        self.mode = mode
        self.cfg = cfg
        self.r_rand = cfg['r_rand_crop']
        self.augtype = augtype
        self.pad_value = cfg['pad_value']
        self.data_dir = data_dir
        self.stride = cfg['stride']
        self.annotation_dir = cfg['annotation_dir']
        self.blacklist = cfg['blacklist']
        self.set_name = set_name
        self.clip_min = cfg['clip_min']
        self.clip_max = cfg['clip_max']
        labels = []
        if set_name.endswith('.csv'):
            self.filenames = np.genfromtxt(set_name, dtype=str)
        elif set_name.endswith('.npy'):
            self.filenames = np.load(set_name)
        elif set_name.endswith('.txt'):
            with open(self.set_name, "r") as f:
                self.filenames = f.read().splitlines()
            #self.filenames = ['%05d' % int(i) for i in self.filenames]

        self.filenames = [f for f in self.filenames if (f not in self.blacklist)]

        for fn in self.filenames:
            with open(os.path.join(self.annotation_dir, '%s_nodule_count_crop.json' % (fn)), 'r') as f:
                annota= json.load(f)
                bboxes = annota['bboxes']
                l = []
                if len(bboxes) > 0:
                    for nodule in bboxes:  # 遍历输入列表
                        top_left = nodule[0]  # 左上角坐标
                        bottom_right = nodule[1]  # 右下角坐标
                        z = (bottom_right[2] + top_left[2]) / 2
                        y = (bottom_right[0] + top_left[0]) / 2
                        x = (bottom_right[1] + top_left[1]) / 2
                        d = (bottom_right[2] - top_left[2]) + 1
                        h = (bottom_right[0] - top_left[0]) + 1
                        w = (bottom_right[1] - top_left[1]) + 1
                        if (d*h*w) <=27:
                            continue
                        l.append([z, y, x, d, h, w]) 
                else:
                    print("No bboxes for %s" % fn)
                l = np.array(l)
                labels.append(l)        

        self.sample_bboxes = labels
        if self.mode in ['train', 'val', 'eval']:
            self.bboxes = []
            for i, l in enumerate(labels):
                if len(l) > 0:
                    for t in l:
                        self.bboxes.append([np.concatenate([[i], t])])
            if self.bboxes == []:
                print()
            self.bboxes = np.concatenate(self.bboxes, axis=0).astype(np.float16)
        self.crop = Crop(cfg)

    def __getitem__(self, idx):
        """
        idx: gt box base
        
        bboxes: all gt box

        bbox: idxth box = [filename index, z,y,x,d,h,w]
        """
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time

        if self.mode in ['train', 'val']:
            #-----------------
            # select sample
            #-----------------
            bbox = self.bboxes[idx]
            filename = self.filenames[int(bbox[0])]
            imgs = self.load_img(filename)
            bboxes = self.sample_bboxes[int(bbox[0])]
            isScale = self.augtype['scale'] and (self.mode=='train')
            #---------------------
            # random crop, scale(ok)
            #---------------------
            
            #draw_image_bbox(imgs,bboxes)
            sample, target, bboxes = self.crop(imgs, bbox[1:], bboxes, isScale)
            #draw_image_bbox(sample,[target])
            #-----------------
            # augmentation
            #-----------------
            if self.mode == 'train':                   
                sample, target, bboxes = augment(sample, target, bboxes, do_flip=self.augtype['flip'],
                                                    do_rotate=self.augtype['rotate'], do_swap=self.augtype['swap'])      
            #draw_image_bbox(sample,[target])
            check_sample_shape(sample, filename, self.cfg)
            # ------------
            # Normalize
            #-------------
            sample = self.hu_normalize(sample)
            bboxes = fillter_box(bboxes, self.cfg['crop_size'])
            label = np.ones(len(bboxes), dtype=np.int32)
            if len(bboxes.shape) != 1:
                for i in range(3):
                    bboxes[:, i+3] = bboxes[:, i+3] + self.cfg['bbox_border']
            return [torch.from_numpy(sample), bboxes, label]

        if self.mode in ['eval']:

        # 取得當前索引處的檔名
            filename = self.filenames[idx]
            
            # 加載圖像 (通常是 3D CT Scan 圖像)
            imgs = self.load_img(filename)
            # draw_image(imgs)
            # 從 sample_bboxes 取得對應的 bounding boxes
            bboxes = self.sample_bboxes[idx]
            
            

            # 使用 split_comber 分割圖像，這個步驟會將圖像分成多個小塊
            imgs, nzhw, dwh = self.split_comber.split(imgs)
        
            
            # 對圖像進行 HU 值的標準化處理，這是對 CT 圖像的特定處理方法
            imgs = self.hu_normalize(imgs)
            # for img in imgs :
            #     draw_image(img)
            # 最後將圖像和座標轉換為 PyTorch 張量並返回，同時返回 bounding boxes 和分割資訊 nzhw
            return [torch.from_numpy(imgs), bboxes, np.array(nzhw),np.array(dwh)]


    def __len__(self):
        """
        return:
            train,valid: bbox number
            eval: filename
        """
        if self.mode == 'train1':
            print(int(len(self.bboxes) / (1-self.r_rand)))
            return int(len(self.bboxes) / (1-self.r_rand))
        elif self.mode =='val1':
            return len(self.bboxes)
        else:
            return len(self.filenames)
    
    def load_img(self,filename):
#         img = np.load(os.path.join(self.data_dir, '%s.npy' % (filename)))
        img = np.load(os.path.join(self.data_dir, '%s_crop.npy' % (filename)))
        # (y,x,z)->(z,y,x)
        img = img.transpose(2,0,1)
        img = img[np.newaxis,...]
        img = np.clip(img,self.clip_min,self.clip_max)
        return img
    
    def hu_normalize(self,images):
        images = np.array(images, float)
        images = images - self.clip_min
        images = images / (self.clip_max - self.clip_min)
        return images.astype(np.float32)

def pad2factor(image, factor=32, pad_value=5):
    depth, height, width = image.shape
    d = int(math.ceil(depth / float(factor))) * factor
    h = int(math.ceil(height / float(factor))) * factor
    w = int(math.ceil(width / float(factor))) * factor

    pad = []
    pad.append([0, d - depth])
    pad.append([0, h - height])
    pad.append([0, w - width])

    image = np.pad(image, pad, 'constant', constant_values=pad_value)

    return image



def fillter_box(bboxes, size):
    res = []
    for box in bboxes:
        if box[0] - box[0+3] / 2 > 0 and box[0] + box[0+3] / 2 < size[0] and \
           box[1] - box[1+3] / 2 > 0 and box[1] + box[1+3] / 2 < size[1] and \
           box[2] - box[2+3] / 2 > 0 and box[2] + box[2+3] / 2 < size[2]:
            res.append(box)
    return np.array(res)

def augment(sample, target, bboxes, do_flip = True, do_rotate=True, do_swap = True):
    #                     angle1 = np.random.rand()*180
    if do_rotate:
        validrot = False
        counter = 0
        while not validrot:
            newtarget = np.copy(target)
            angle1 = np.random.rand()*180
            size = np.array(sample.shape[2:4]).astype('float')
            rotmat = np.array([[np.cos(angle1/180*np.pi),-np.sin(angle1/180*np.pi)],[np.sin(angle1/180*np.pi),np.cos(angle1/180*np.pi)]])
            newtarget[1:3] = np.dot(rotmat,target[1:3]-size/2)+size/2
            if np.all(newtarget[:3]>target[3]) and np.all(newtarget[:3]< np.array(sample.shape[1:4])-newtarget[3]):
                validrot = True
                target = newtarget
                sample = rotate(sample,angle1,axes=(2,3),reshape=False)
                for box in bboxes:
                    box[1:3] = np.dot(rotmat,box[1:3]-size/2)+size/2
            else:
                counter += 1
                if counter ==3:
                    break
    if do_swap:
        if sample.shape[1]==sample.shape[2] and sample.shape[1]==sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample,np.concatenate([[0],axisorder+1]))
            target[:3] = target[:3][axisorder]
            bboxes[:,:3] = bboxes[:,:3][:,axisorder]

    if do_flip:
        # Optimized flipid generation
        flipid = np.array([1, np.random.choice([1, -1]), np.random.choice([1, -1])])
        
        # Use efficient slicing for flipping
        sample = sample[:, ::flipid[0], ::flipid[1], ::flipid[2]]

        # Update target and bounding boxes after flipping
        shape_dims = sample.shape[1:4]
        for ax in range(3):
            if flipid[ax] == -1:
                # No need for redundant np.array conversions
                target[ax] = shape_dims[ax] - target[ax]
                if bboxes.ndim == 2:  # Added check for multidimensional bboxes
                    bboxes[:, ax] = shape_dims[ax] - bboxes[:, ax]

    return sample, target, bboxes


class Crop(object):
    def __init__(self, config):
        self.crop_size = config['crop_size']
        self.bound_size = config['bound_size']
        self.stride = config['stride']
        self.pad_value = config['pad_value']

    def __call__(self, imgs, target, bboxes, isScale=False):
        """
        imgs: the CT imgae

        target: the select bbox for center crop

        bboxes: all bboxes of imgs
        """
        if isScale:
            # if scale,calculate the crop size and scale range
            radiusLim = [7.,15.]
            scaleLim = [1,2]
            min_r = np.min(target[3:6])
            max_r = np.max(target[3:6])
            scaleRange = [np.min([np.max([(radiusLim[0]/min_r),scaleLim[0]]),1])
                         ,np.max([np.min([(radiusLim[1]/max_r),scaleLim[1]]),1])]              
            scale = np.random.rand()*(scaleRange[1]-scaleRange[0])+scaleRange[0]
            crop_size = (np.array(self.crop_size).astype('float')/scale).astype('int')
            if scaleRange[0] == 1 and scaleRange[1] == 1:
                isScale=False
                crop_size = self.crop_size
        else:
            crop_size = self.crop_size

        bound_size = self.bound_size
        target = np.copy(target)
        bboxes = np.copy(bboxes)
        
        # get start and end point
        start = []
        for i in range(3):
            r = target[i+3]/2
            s = np.floor(target[i] - r) + 1 - bound_size
            e = np.ceil(target[i] + r) + 1 + bound_size - crop_size[i]
            e = max(e,0)
            s = min(s,imgs.shape[i+1]-crop_size[i])
            if s > e:                 
                start.append(np.random.randint(e,s))
            else:
                start.append(int(e)+np.random.randint(0,bound_size/2))

        # padding 
        pad = []
        pad.append([0,0])
        for i in range(3):
            leftpad = max(0,-start[i])
            rightpad = max(0,start[i]+crop_size[i]-imgs.shape[i+1])
            pad.append([leftpad,rightpad])
        # get cube by cropping from imgs
        crop = imgs[:,
            int(max(start[0],0)):int(min(start[0] + int(crop_size[0]),imgs.shape[1])),
            int(max(start[1],0)):int(min(start[1] + int(crop_size[1]),imgs.shape[2])),
            int(max(start[2],0)):int(min(start[2] + int(crop_size[2]),imgs.shape[3]))]
        crop = np.pad(crop, pad, 'constant', constant_values=self.pad_value)
        # adjust target and bboxes coordinate of new cube 
        for i in range(3):
            target[i] = target[i] - start[i]
        for i in range(len(bboxes)):
            for j in range(3):
                bboxes[i][j] = bboxes[i][j] - start[j]
        # if scale, interpolate the cropped cube

        if isScale:
            # 使用 torch.nn.functional.interpolate 取代room操作，加速至少10倍 
            crop_tensor = torch.from_numpy(crop).float().unsqueeze(0)
            new_size = [int(scale * s) for s in crop_tensor.shape[2:]]
            crop_tensor = F.interpolate(crop_tensor, size=new_size, mode='trilinear', align_corners=False)
            crop = crop_tensor.squeeze(0).numpy()

            newpad = self.crop_size[0] - crop.shape[1:][0]
            if newpad < 0:
                crop = crop[:, :-newpad, :-newpad, :-newpad]
            elif newpad > 0:
                pad2 = [[0, 0], [0, newpad], [0, newpad], [0, newpad]]
                crop = np.pad(crop, pad2, 'constant', constant_values=self.pad_value)
            for i in range(6):
                target[i] = target[i] * scale
            for i in range(len(bboxes)):
                for j in range(6):
                    bboxes[i][j] = bboxes[i][j] * scale
        return crop, target, bboxes

def check_sample_shape(sample, filename, cfg):
    if (sample.shape[1] != cfg['crop_size'][0] or 
        sample.shape[2] != cfg['crop_size'][1] or 
        sample.shape[3] != cfg['crop_size'][2]):
        
        warning_message = f"File '{filename}' has shape {sample.shape}, " \
                          f"which does not match expected crop size {cfg['crop_size']}."
        warnings.warn(warning_message, UserWarning)

def draw_image_bbox(image, bboxes):
    """
    draw center image of CT with box
    
    image: the ct scan

    bbox : a single bbox
    """
    import matplotlib
    matplotlib.use('TkAgg') 
    for bbox in bboxes : 
        z,y,x,d,h,w = bbox 

        x = x - w / 2   
        y = y - h / 2

        img = image[0][int(z),:,:]
        plt.figure(figsize=(5,5))
        plt.imshow(img, cmap='gray', interpolation='nearest')  
        plt.axis('off')  # 不顯示座標軸
        # 繪製 bounding box
        plt.gca().add_patch(plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none'))  # 綠色邊框
        plt.show() 
def draw_image(image):
    import matplotlib
    matplotlib.use('TkAgg')
    img = image[0][int(20),:,:]
    plt.figure(figsize=(5,5))
    plt.imshow(img, cmap='gray', interpolation='nearest')  
    plt.axis('off')  # 不顯示座標軸
    plt.show()
    plt.pause(1)
    plt.close() 


