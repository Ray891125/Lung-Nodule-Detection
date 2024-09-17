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
from dataset.split_combine import SplitComb
class BboxReader(Dataset):
    def __init__(self, data_dir, set_name, augtype, cfg, mode='train'):
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
                #l = fillter_box(l, [512, 512, 512])
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
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time
        is_random_img = False
        if self.mode in ['train', 'val']:
            if idx >= len(self.bboxes):
                is_random_crop = True
                idx = idx % len(self.bboxes)
                is_random_img = np.random.randint(2)
            else:
                is_random_crop = False
        else:
            is_random_crop = False

        if self.mode in ['train', 'val']:
            if not is_random_img:
                bbox = self.bboxes[idx]
                filename = self.filenames[int(bbox[0])]
                imgs = self.load_img(filename)
                bboxes = self.sample_bboxes[int(bbox[0])]
                # bboxes = fillter_box(bboxes, imgs.shape[1:])
                isScale = self.augtype['scale'] and (self.mode=='train')
                sample, target, bboxes, coord = self.crop(imgs, bbox[1:], bboxes, isScale, is_random_crop)
                draw_all_bbox(sample,target,filename)
                if self.mode == 'train' and not is_random_crop:                   
                    sample, target, bboxes = augment(sample, target, bboxes, do_flip=self.augtype['flip'],
                                                     do_rotate=self.augtype['rotate'], do_swap=self.augtype['swap'])
                    draw_all_bbox(sample,target,filename)
            else:
                randimid = np.random.randint(len(self.filenames))
                filename = self.filenames[randimid]
                imgs = self.load_img(filename)
                bboxes = self.sample_bboxes[randimid]
                bboxes = fillter_box(bboxes, imgs.shape[1:])
                isScale = self.augtype['scale'] and (self.mode=='train')
                sample, target, bboxes, coord = self.crop(imgs, [], bboxes, isScale=False, isRand=True)

            if sample.shape[1] != self.cfg['crop_size'][0] or sample.shape[2] != \
                self.cfg['crop_size'][1] or sample.shape[3] != self.cfg['crop_size'][2]:
                print(filename, sample.shape)

            # Normalize
            sample = self.hu_normalize(sample)
            #draw_all_bbox(sample,target,filename)
            bboxes = fillter_box(bboxes, self.cfg['crop_size'])
            label = np.ones(len(bboxes), dtype=np.int32)
            # print(bboxes.shape)
            if len(bboxes.shape) != 1:
                for i in range(3):
                    bboxes[:, i+3] = bboxes[:, i+3] + self.cfg['bbox_border']

            return [torch.from_numpy(sample), bboxes, label]

        if self.mode in ['eval']:
            filename = self.filenames[idx]
            imgs = self.load_img(filename)
            image = pad2factor(imgs[0])
            image = np.expand_dims(image, 0)

            bboxes = self.sample_bboxes[idx]
            bboxes = fillter_box(bboxes, imgs.shape[1:])
            for i in range(3):
                bboxes[:, i + 3] = bboxes[:, i + 3] + self.cfg['bbox_border']
            bboxes = np.array(bboxes)
            label = np.ones(len(bboxes), dtype=np.int32)

            input = self.hu_normalize(image)
#             input = (image.astype(np.float32) - 128.) / 128.
                
            return [torch.from_numpy(input).float(), bboxes, label]
            # imgs = self.load_img(self.filenames[idx])
            # bboxes = self.sample_bboxes[idx]
            # nz, nh, nw = imgs.shape[1:]
            # pz = int(np.ceil(float(nz) / self.stride)) * self.stride
            # ph = int(np.ceil(float(nh) / self.stride)) * self.stride
            # pw = int(np.ceil(float(nw) / self.stride)) * self.stride
            # imgs = np.pad(imgs, [[0,0],[0, pz - nz], [0, ph - nh], [0, pw - nw]], 'constant',constant_values = self.pad_value)
            
            # xx,yy,zz = np.meshgrid(np.linspace(-0.5,0.5,imgs.shape[1]/self.stride),
            #                        np.linspace(-0.5,0.5,imgs.shape[2]/self.stride),
            #                        np.linspace(-0.5,0.5,imgs.shape[3]/self.stride),indexing ='ij')
            # coord = np.concatenate([xx[np.newaxis,...], yy[np.newaxis,...],zz[np.newaxis,:]],0).astype('float32')
            # imgs, nzhw = self.split_comber.split(imgs)
            # coord2, nzhw2 = self.split_comber.split(coord,
            #                                        side_len = self.split_comber.side_len/self.stride,
            #                                        max_stride = self.split_comber.max_stride/self.stride,
            #                                        margin = self.split_comber.margin/self.stride)
            # assert np.all(nzhw==nzhw2)
            # imgs = (imgs.astype(np.float32)-128)/128
            # return torch.from_numpy(imgs), bboxes, torch.from_numpy(coord2), np.array(nzhw)



    def __len__(self):
        if self.mode == 'train':
            print(int(len(self.bboxes) / (1-self.r_rand)))
            return int(len(self.bboxes) / (1-self.r_rand))
        elif self.mode =='val':
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
        # flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        flipid = np.array([1,np.random.randint(2),np.random.randint(2)])*2-1
        sample = np.ascontiguousarray(sample[:,::flipid[0],::flipid[1],::flipid[2]])

        for ax in range(3):
            if flipid[ax] == -1:
                target[ax] = np.array(sample.shape[ax + 1]) - target[ax]
                if len(bboxes.shape) == 1:
                    print()
                bboxes[:, ax] = np.array(sample.shape[ax + 1]) - bboxes[:, ax]
    return sample, target, bboxes

class Crop(object):
    def __init__(self, config):
        self.crop_size = config['crop_size']
        self.bound_size = config['bound_size']
        self.stride = config['stride']
        self.pad_value = config['pad_value']

    def __call__(self, imgs, target, bboxes, isScale=False, isRand=False):
        if isScale:
            radiusLim = [7.,15.]
            scaleLim = [0.85,1.15]
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

        start = []
        for i in range(3):
            # start.append(int(target[i] - crop_size[i] / 2))
            if not isRand:
                r = target[i+3]/2
                s = np.floor(target[i] - r) + 1 - bound_size
                e = np.ceil(target[i] + r) + 1 + bound_size - crop_size[i]
                e = max(e,0)
                s = min(s,imgs.shape[i+1]-crop_size[i])
            else:
                s = np.max([imgs.shape[i+1]-crop_size[i]/2, imgs.shape[i+1]/2+bound_size])
                e = np.min([crop_size[i]/2, imgs.shape[i+1]/2-bound_size])
                target = np.array([np.nan, np.nan, np.nan, np.nan])
            if s > e:                 
                start.append(np.random.randint(e,s))
            else:
                start.append(int(e)+np.random.randint(0,bound_size/2))
            # if s > e:
            #     start.append(np.random.randint(e, s))#!
            # else:
            #     start.append(int(target[i])-crop_size[i]/2+np.random.randint(-bound_size/2,bound_size/2))
        coord=[]

        pad = []
        pad.append([0,0])
        for i in range(3):
            leftpad = max(0,-start[i])
            rightpad = max(0,start[i]+crop_size[i]-imgs.shape[i+1])
            pad.append([leftpad,rightpad])
        crop = imgs[:,
            int(max(start[0],0)):int(min(start[0] + int(crop_size[0]),imgs.shape[1])),
            int(max(start[1],0)):int(min(start[1] + int(crop_size[1]),imgs.shape[2])),
            int(max(start[2],0)):int(min(start[2] + int(crop_size[2]),imgs.shape[3]))]
        crop = np.pad(crop, pad, 'constant', constant_values=self.pad_value)
        for i in range(3):
            target[i] = target[i] - start[i]
        for i in range(len(bboxes)):
            for j in range(3):
                bboxes[i][j] = bboxes[i][j] - start[j]

        if isScale:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                crop = zoom(crop, [1, scale, scale, scale], order=1)
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

        return crop, target, bboxes, coord


import cv2
def draw_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


def draw_bbox(image, bbox,filename):
    return
    plt.clf()
    plt.title(filename)
    plt.figure(figsize=(5,5))
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # 不顯示座標軸
    # 解析 bounding box 的坐標
    z,y,x,d ,h,w = bbox 
    x = x - w / 2
    y = y - h / 2
    # 繪製 bounding box
    plt.gca().add_patch(plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none'))  # 綠色邊框
    plt.show()

def draw_all_bbox(image, bbox,filename):
    return
    import matplotlib
    matplotlib.use('TkAgg') 
    plt.clf()

    z,y,x,d,h,w = bbox 
    z = int(z - d / 2)
    x = x - w / 2   
    y = y - h / 2
    for i in range(1):
        img = image[0][z+i,:,:]
        plt.figure(figsize=(5,5))
        plt.imshow(img, cmap='gray')
        plt.axis('off')  # 不顯示座標軸
        
        # 繪製 bounding box
        plt.gca().add_patch(plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none'))  # 綠色邊框
        plt.show() 