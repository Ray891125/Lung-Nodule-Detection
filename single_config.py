import os
import numpy as np
import torch
import random


# Set seed
SEED = 35202
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Preprocessing using preserved HU in dilated part of mask
# BASE: Dataset path
BASE = r'E:/desktop/training_data/My_SAnet/' # make sure you have the ending '/'
SAVE_BASE = r'E:/desktop/training_data/My_SAnet/Save/'
SAVE_DIR = "TEST"
test_config = { 
    'dataset_train': 'Dicom_crop',  
    'dataset_test': 'Dicom_crop',
    'load_epoch': 'epoch_150_loss_0.788940',
}
dataset = 'Dicom_crop' # dataset_test
datasets_info = {}
if dataset == 'ME_dataset':
    datasets_info['dataset'] = 'ME_dataset'
    datasets_info['train_list'] = [BASE+'train.txt']
    datasets_info['val_list'] = [BASE+'val.txt']
    datasets_info['test_name'] = BASE+'test.txt' # test
    datasets_info['data_dir'] = BASE+dataset+'/'
    datasets_info['annotation_dir'] = BASE+'Bbox_Label'
    datasets_info['test_annotation_dir'] = BASE+'annotation_validate.csv'
    datasets_info['BATCH_SIZE'] = 16
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 2
    datasets_info['pad_value'] = -1200
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}
elif dataset == 'Dicom_crop':
    datasets_info['dataset'] = 'Dicom_crop'
    datasets_info['train_list'] = [BASE+'Data\\train_300.txt']
    datasets_info['val_list'] = [BASE+'Data\\crop_test_60.txt']
    datasets_info['test_name'] = BASE+'Data\\crop_test_60.txt' # test
    datasets_info['data_dir'] = BASE+'Data\\'+dataset+'\\'
    datasets_info['annotation_dir'] = BASE+'Data\\Bbox_crop_anno\\'
    datasets_info['test_annotation_dir'] = BASE+'Data\\crop_test_27_combine.csv'
    datasets_info['series_uids_path'] = BASE+'Data\\crop_seriesuid_val.csv'
    datasets_info['BATCH_SIZE'] = 8
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128,128]
    datasets_info['bbox_border'] = 0
    datasets_info['pad_value'] = 5
    datasets_info['augtype'] = {'flip': True, 'rotate': False, 'scale': False, 'swap': False}
elif dataset == 'Dicom_NPY_Order3_Temp':
    datasets_info['dataset'] = 'Dicom_NPY_Order3_Temp'
    datasets_info['train_list'] = [BASE+'BME/train.txt']
    datasets_info['val_list'] = [BASE+'BME/val.txt']
    datasets_info['test_name'] = BASE+'BME/test.txt' # test
    datasets_info['data_dir'] = BASE+dataset+'/'
    datasets_info['annotation_dir'] = BASE+'Bbox_Label'
    datasets_info['test_annotation_dir'] = BASE+'BME/annotation_validate.csv'
    datasets_info['BATCH_SIZE'] = 16
    datasets_info['label_types'] = ['bbox']
    datasets_info['roi_names'] = ['nodule']
    datasets_info['crop_size'] = [128, 128, 128]
    datasets_info['bbox_border'] = 2
    datasets_info['pad_value'] = -1200
    datasets_info['augtype'] = {'flip': True, 'rotate': True, 'scale': True, 'swap': False}

def get_anchors(bases, aspect_ratios):
    anchors = []
    for b in bases:
        for asp in aspect_ratios:
            d, h, w = b * asp[0], b * asp[1], b * asp[2]
            anchors.append([d, h, w])

    return anchors

bases = [5]
aspect_ratios = [[1, 1, 1]]
# bases = [5]
# aspect_ratios = [[1, 1, 1]]
net_config = {
    # Net configuration
    'anchors': get_anchors(bases, aspect_ratios),
    'channel': 1,
    'crop_size': datasets_info['crop_size'],
    'croplen':128,
    'margin':0,
    'stride': 4,
    'max_stride': 16,
    'num_neg': 80000,
    'num_hard': 100,
    'bound_size': 12,
    'blacklist': ['CHEST1849', 'CHESTCT1190','CHESTCT1068','CHESTCT1520','CHESTCT1682','CHESTCT1425','CHESTCT1788','CHEST1649'],

    'r_rand_crop': 0.,
    'pad_value': 5,
    'clip_min': -1000,
    'clip_max': 400,
    # region proposal network configuration
    'rpn_train_bg_thresh_high': 0.1,
    'rpn_train_fg_thresh_low': 0.4,
    
    #'rpn_train_nms_num': 300,
    'rpn_train_nms_pre_score_threshold': 0.5,
    'rpn_train_nms_overlap_threshold': 0.1,
    'rpn_test_nms_pre_score_threshold': 0.5,
    'rpn_test_nms_overlap_threshold': 0.1,

    # false positive reduction network configuration
    # 'num_class': len(datasets_info['roi_names']) + 1,
    'num_class': len(datasets_info['roi_names']),
    'rcnn_crop_size': (7,7,7), # can be set smaller, should not affect much
    'rcnn_train_fg_thresh_low': 0.5,
    'rcnn_train_bg_thresh_high': 0.1,
    'rcnn_train_batch_size': 64,
    'rcnn_train_fg_fraction': 0.5,
    'rcnn_train_nms_pre_score_threshold': 0.5,
    'rcnn_train_nms_overlap_threshold': 0.1,
    'rcnn_test_nms_pre_score_threshold': 0.0,
    'rcnn_test_nms_overlap_threshold': 0.1,

    'box_reg_weight': [1., 1., 1., 1., 1., 1.],

        # nodule-detr config
    'hidden_dim': 64,
    'dropout': 0.1,
    'nheads': 8,
    'dim_feedforward': 256,
    'enc_layers': 6,
    'dec_layers': 6,
    'pre_norm': '',
    'return_intermediate_dec': True,
    'position_embedding': 'sine',
    'num_queries': 512,
    # Atten conv
    'bn_momentum': 0.1,
}



train_config = {
 
    'net': 'SANet',
    'net_name': 'SANet',

    'num_groups': 4,
    'batch_size': datasets_info['BATCH_SIZE'],

    'optimizer': 'AdamW',
    'momentum': 0.9,
    'weight_decay': 1e-4,
    

    'epochs': 170, #200 #400
    'epoch_save': 5,
    'epoch_rcnn': 180, #20 #47  
    'num_workers':2, #30

}

if train_config['optimizer'] == 'SGD':
    train_config['init_lr'] = 0.01
elif train_config['optimizer'] == 'Adam'or train_config['optimizer'] == 'AdamW':
    train_config['init_lr'] = 0.001
elif train_config['optimizer'] == 'RMSprop':
    train_config['init_lr'] = 2e-3

train_config['RESULTS_DIR'] = os.path.join(SAVE_BASE+SAVE_DIR+'/{}_results'.format(train_config['net_name']),
                                  datasets_info['dataset'])
train_config['out_dir'] = os.path.join(train_config['RESULTS_DIR'], 'cross_val_test_{}'.format(train_config['epoch_rcnn']))
train_config['initial_checkpoint'] = r"E:\desktop\training_data\My_SAnet\Save\test\SANet_results\Dicom_crop\cross_val_test_180\model\epoch_040_loss_1.830506.ckpt"
test_config['checkpoint'] = SAVE_BASE+SAVE_DIR+'/{}_results/{}/cross_val_test_{}/model/{}.ckpt'.format(
    train_config['net_name'], test_config['dataset_train'], train_config['epoch_rcnn'], test_config['load_epoch'])
test_config['out_dir'] = SAVE_BASE+SAVE_DIR+'/{}/cross_val_test_{}/{}/'.format(
    train_config['net_name'], test_config['dataset_train'], train_config['epoch_rcnn'], test_config['dataset_test'])

config = dict(datasets_info, **net_config)
config = dict(config, **train_config)
config = dict(config, **test_config)
    