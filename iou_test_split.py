import numpy as np
import torch
import os
import traceback
import time
import nrrd
import sys
import matplotlib.pyplot as plt
import logging
import argparse
import torch.nn.functional as F
from scipy.stats import norm
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parallel.data_parallel import data_parallel
from scipy.ndimage import label
from scipy.ndimage import center_of_mass
from net.sanet import SANet
from net.MSANet import MsaNet
from net.TransUnet import TransUnet
from net.TicNet import TicNet
#from net.MSANet_PC import MsaNet
from dataset.collate import train_collate, test_collate, eval_collate,split_eval_collate
from dataset.bbox_reader import BboxReader
from single_config import datasets_info, train_config, test_config, net_config, config
from utils.util import dice_score_seperate, get_contours_from_masks, merge_contours, hausdorff_distance
from utils.util import onehot2multi_mask, normalize, pad2factor, load_dicom_image, crop_boxes2mask_single, \
    npy2submission
import pandas as pd
from evaluationScript.uni_noduleCADEvaluation import noduleCADEvaluation
from evaluationScript.eval import nodule_evaluation
from evaluationScript.logs import setup_logging
import logging
from dataset.split_combine import SplitComb
from utils.pybox import *
logger = logging.getLogger(__name__)


plt.rcParams['figure.figsize'] = (24, 16)
plt.switch_backend('agg')
this_module = sys.modules[__name__] 
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

parser = argparse.ArgumentParser(description='PyTorch Detector')
parser.add_argument('--net', '-m', metavar='NET', default=train_config['net'],
                    help='neural net')
parser.add_argument("--mode", type=str, default='eval',
                    help="you want to test or val")
parser.add_argument('--ckpt', default=test_config['checkpoint'], type=str, metavar='CKPT',
                    help='checkpoint to use')
# parser.add_argument("--dicom-path", type=str, default=None,
#                     help="path to dicom files of patient")
parser.add_argument('--out_dir', default=test_config['out_dir'], type=str, metavar='OUT',
                    help='path to save the results')
parser.add_argument('--test_name', default=datasets_info['test_name'], type=str,
                    help='test set name')
parser.add_argument('--data_dir', default=datasets_info['data_dir'], type=str, metavar='OUT',
                    help='path to load data')
parser.add_argument('--test_annotation_dir', default=datasets_info['test_annotation_dir'], type=str, metavar='OUT',
                    help='path to load annotation')
parser.add_argument('--series_uids_path', default=datasets_info['series_uids_path'], type=str, metavar='OUT',
                    help='path to load annotation')
parser.add_argument('--augtype', default=datasets_info['augtype'], type=str, metavar='OUT',
                    help='augment type')


def main():
    logging.basicConfig(format='[%(levelname)s][%(asctime)s] %(message)s', level=logging.INFO)
    args = parser.parse_args()

    if args.mode == 'eval':
        net = args.net
        initial_checkpoint = args.ckpt
        out_dir = args.out_dir
        test_name = args.test_name
        data_dir = args.data_dir
        test_annotation_dir = args.test_annotation_dir
        series_uids_path = args.series_uids_path
        label_types = config['label_types'][0]
        augtype = args.augtype
        num_workers = config['num_workers']
        net = getattr(this_module, net)(config)
        net = net.cuda()
        # print(net)
        
        if initial_checkpoint:
            print('[Loading model from %s]' % initial_checkpoint)
            checkpoint = torch.load(initial_checkpoint)
            # out_dir = checkpoint['out_dir']   
            epoch = checkpoint['epoch']

            net.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            print('No model weight file specified')
            return
        
        print('out_dir', out_dir)
        save_dir = os.path.join(out_dir, 'res', str(epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(os.path.join(save_dir, 'FROC')):
            os.makedirs(os.path.join(save_dir, 'FROC'))
    
        if label_types == 'bbox':
            dataset = BboxReader(data_dir, test_name, augtype, config, mode='eval')
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                              num_workers=num_workers, pin_memory=False, collate_fn=split_eval_collate)
        eval_rpn(net, test_loader, test_annotation_dir, data_dir, save_dir,series_uids_path,predict=True)

    else:
        logging.error('Mode %s is not supported' % (args.mode))
def eval_rpn(net, dataset, test_annotation_dir, data_dir, save_dir=None, series_uids_path = None,predict=True):
    """
    predict the result of the dataset and save the result as csv file
    """
    net.set_mode('eval')
    net.use_rcnn = False
    print('Total # of eval data %d' % (len(dataset)))
    split_comber = SplitComb(crop_size=config['crop_size'], overlap_size=[32,32,32], pad_value=config['pad_value'], do_padding=True)
    #------------------------------
    # predict and save results as npy files
    #------------------------------
    if predict == True:
        # start predict
        for i, inputs in tqdm(enumerate(dataset), total=len(dataset), desc='eval'):
            rpns = []
            rpn_outputs_list = []
            pid = dataset.dataset.filenames[i]
            inputs = inputs[0]

            datas, bboxes, nzhw, dwh = inputs
            print('[%d] Predicting %s' % (i, pid))

            # 將 datas 分成批次
            for data_batch in torch.split(datas,8):
                # 將每個 batch 的資料堆疊成 Tensor
                try:
                    data_batch = Variable(data_batch).cuda()               
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            net.forward(data_batch)
                    # 取得網絡輸出，並將其儲存在 rpns 列表中
                    rpn_output = net.rpn_proposals  
                    for out in rpn_output :
                        rpn_outputs_list.append(out)
                    #rpn_outputs = np.concatenate(rpn_outputs_list, 0)  
                except Exception as e:
                    del inputs,datas
                    torch.cuda.empty_cache()
                    traceback.print_exc()
                    print
                    return
            rpns = split_comber.combine(rpn_outputs_list, nzhw, dwh)
            rpns = np.concatenate(rpns, 0)
            rpns,_ = torch_nms(rpns, 0.5)
            print('rpn', len(rpns))
            # save network output
            if len(rpns) > 0:
                rpns = np.array(rpns)
                np.save(os.path.join(save_dir, '%s_rpns.npy' % (pid)), rpns)
            # Clear gpu memory
            del data_batch
            torch.cuda.empty_cache()
    #------------------------------
    # Generate prediction csv for the use of performing FROC analysis
    #------------------------------

    # read rpn and rcnn npy results
    rpn_res = []
    for pid in dataset.dataset.filenames:
        if os.path.exists(os.path.join(save_dir, '%s_rpns.npy' % (pid))):
            rpns = np.load(os.path.join(save_dir, '%s_rpns.npy' % (pid)))
            rpns = rpns[:, [3, 2, 1, 6, 5, 4, 0]]
            names = np.array([[pid]] * len(rpns))
            rpn_res.append(np.concatenate([names, rpns], axis=1))

    rpn_res = np.concatenate(rpn_res, axis=0)
    col_names = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'w', 'h', 'd', 'probability']
    eval_dir = os.path.join(save_dir, 'FROC')
    rpn_submission_path = os.path.join(eval_dir, 'submission_rpn.csv')

    df = pd.DataFrame(rpn_res, columns=col_names)
    df.to_csv(rpn_submission_path, index=False)

    # ------------------------------
    # Start evaluating
    # ------------------------------
    if not os.path.exists(os.path.join(eval_dir, 'rpn')):
        os.makedirs(os.path.join(eval_dir, 'rpn'))

    # noduleCADEvaluation(test_annotation_dir, data_dir, dataset.dataset.set_name, rpn_submission_path, os.path.join(eval_dir, 'dist/rpn'))

    iou_FROC(test_annotation_dir, series_uids_path,rpn_submission_path, os.path.join(eval_dir, 'iou/rpn'),fixed_prob_threshold=0.5)
    return eval_dir


def iou_FROC(annot_path, series_uids_path, pred_results_path, output_dir, iou_threshold=0.1, fixed_prob_threshold=0.5):
    FP_ratios = [0.125, 0.25, 0.5, 1, 2, 4, 8.]
    setup_logging(level='info', log_file=os.path.join(output_dir, 'log.txt'))  
    froc_out, fixed_out, (best_f1_score, best_f1_threshold),fps2_sens = nodule_evaluation(annot_path = annot_path,
                                                                                series_uids_path = series_uids_path, 
                                                                                pred_results_path = pred_results_path,
                                                                                output_dir = output_dir,
                                                                                iou_threshold = iou_threshold,
                                                                                fixed_prob_threshold=fixed_prob_threshold)
    fps, sens, thresholds, fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up, sens_points = froc_out
    fixed_tp, fixed_fp, fixed_fn, fixed_recall, fixed_precision, fixed_f1_score = fixed_out
    metrics = {'tp': fixed_tp,
                'fp': fixed_fp,
                'fn': fixed_fn,
                'recall': fixed_recall,
                'precision': fixed_precision,
                'f1_score': fixed_f1_score,
                'best_f1_score': best_f1_score,
                'best_f1_threshold': best_f1_threshold}
    mean_recall = np.mean(np.array(sens_points))
    metrics['froc_mean_recall'] = float(mean_recall)
    for fp_ratio, sens_p in zip(FP_ratios, sens_points):
        metrics['froc_{}_recall'.format(str(fp_ratio))] = sens_p
    return sens_points, mean_recall

def eval_single(net, input):
    with torch.no_grad():
        input = input.cuda().unsqueeze(0)
        logits = net.forward(input)
        logits = logits[0]

    masks = logits.cpu().data.numpy()
    masks = (masks > 0.5).astype(np.int32)
    return masks


if __name__ == '__main__':
    main()