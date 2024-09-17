from net.sanet import SANet
from net.MSANet import MsaNet
from net.TransUnet import TransUnet
from net.BiFPN import TransBiFPN
from net.TicNet import TicNet
import time
from dataset.collate import train_collate, test_collate, eval_collate
from dataset.bbox_reader import BboxReader
from utils.util import Logger
from single_config import train_config, datasets_info, net_config, config, test_config
import pprint
from torch.utils.data import DataLoader, ConcatDataset
from torch.autograd import Variable
import torch
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
import random
import traceback
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

from torch.optim.lr_scheduler import CosineAnnealingLR,_LRScheduler, SequentialLR, StepLR
from ModelEvaluator import ModelEvaluator
import pandas as pd
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
this_module = sys.modules[__name__]

parser = argparse.ArgumentParser(description='PyTorch Detector')
parser.add_argument('--net', '-m', metavar='NET', default=train_config['net'],
                    help='neural net')
parser.add_argument('--epochs', default=train_config['epochs'], type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=train_config['batch_size'], type=int, metavar='N',
                    help='batch size')
parser.add_argument('--epoch-rcnn', default=train_config['epoch_rcnn'], type=int, metavar='NR',
                    help='number of epochs before training rcnn')
parser.add_argument('--ckpt', default=train_config['initial_checkpoint'], type=str, metavar='CKPT',
                    help='checkpoint to use')
parser.add_argument('--optimizer', default=train_config['optimizer'], type=str, metavar='SPLIT',
                    help='which split set to use')
parser.add_argument('--init-lr', default=train_config['init_lr'], type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=train_config['momentum'], type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', default=train_config['weight_decay'], type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--epoch-save', default=train_config['epoch_save'], type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--out-dir', default=train_config['out_dir'], type=str, metavar='OUT',
                    help='directory to save results of this training')
parser.add_argument('--train-set-list', default=datasets_info['train_list'], nargs='+', type=str,
                    help='train set paths list')
parser.add_argument('--val-set-list', default=datasets_info['val_list'], nargs='+', type=str,
                    help='val set paths list')
parser.add_argument('--data-dir', default=datasets_info['data_dir'], type=str, metavar='OUT',
                    help='path to load data')
parser.add_argument('--num-workers', default=train_config['num_workers'], type=int, metavar='N',
                    help='number of data loading workers')


parser.add_argument('--test_out_dir', default=test_config['out_dir'], type=str, metavar='OUT',
                    help='path to save the results')
parser.add_argument('--test_name', default=datasets_info['test_name'], type=str,
                    help='test set name')
parser.add_argument('--test_data_dir', default=datasets_info['data_dir'], type=str, metavar='OUT',
                    help='path to load data')
parser.add_argument('--test_annotation_dir', default=datasets_info['test_annotation_dir'], type=str, metavar='OUT',
                    help='path to load annotation')
parser.add_argument('--test_series_uids_path', default=datasets_info['series_uids_path'], type=str, metavar='OUT',
                    help='path to load annotation')
parser.add_argument('--augtype', default=datasets_info['augtype'], type=str, metavar='OUT',
                    help='augment type')

class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, start_lr, base_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.start_lr = start_lr
        self.base_lr = base_lr
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = (self.base_lr - self.start_lr) / float(self.warmup_epochs)
            return [self.start_lr + warmup_factor * self.last_epoch for _ in self.base_lrs]
        else:
            return self.base_lrs
def main():
    # ----------------------
    # 1. Load training configuration
    # ----------------------
    args = parser.parse_args()
    net = args.net
    initial_checkpoint = args.ckpt
    out_dir = args.out_dir
    weight_decay = args.weight_decay
    momentum = args.momentum
    optimizer_name = args.optimizer
    init_lr = args.init_lr
    epochs = args.epochs
    epoch_save = args.epoch_save
    epoch_rcnn = args.epoch_rcnn
    batch_size = args.batch_size
    train_set_list = args.train_set_list
    val_set_list = args.val_set_list
    data_dir = args.data_dir
    label_types = config['label_types']
    augtype = config['augtype']
    

    # ----------------------
    # 2. Load dataset
    # ----------------------
    train_dataset_list = []
    val_dataset_list = []
    for i in range(len(train_set_list)):
        set_name = train_set_list[i]
        label_type = label_types[i]

        if label_type == 'bbox':
            train_dataset = BboxReader(data_dir, set_name, augtype, config, mode='train')

        train_dataset_list.append(train_dataset)

    for i in range(len(val_set_list)):
        set_name = val_set_list[i]
        label_type = label_types[i]

        if label_type == 'bbox':
            val_dataset = BboxReader(data_dir, set_name, augtype, config, mode='val')
            eval_dataset = BboxReader(data_dir, args.test_name, augtype, config, mode='eval')
        val_dataset_list.append(val_dataset)
            
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=train_collate,drop_last=True,persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=train_collate,drop_last=True,persistent_workers=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False,
                              num_workers=args.num_workers, pin_memory=False, collate_fn=train_collate)
    # ----------------------
    # 3. Initilize network
    # ----------------------
    net = getattr(this_module, net)(net_config)
    net = net.cuda()
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    start_epoch = 0
    # ----------------------
    # 4. optimizer
    # ----------------------
    optimizer = getattr(torch.optim, optimizer_name)
    if optimizer_name == "SGD":

        optimizer = optimizer(net.parameters(), lr=init_lr, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_name == "Adam" or optimizer_name == "AdamW":
        optimizer = optimizer(net.parameters(), lr=init_lr, weight_decay=weight_decay)
    # ----------------------
    # 5. scheduler
    # ----------------------
    warmup_epochs = 20  # Number of warm-up epochs
    start_lr = 0.00001  # Initial learning rate for warm-up
    base_lr = init_lr  # Base learning rate after warm-up
    step_size = 50  # Number of epochs to wait before decreasing the learning rate
    gamma = 0.1  # Multiplicative factor of learning rate decay
    # Create the warm-up scheduler
    warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs, start_lr, base_lr)

    # Create the step LR scheduler
    step_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Combine the schedulers
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, step_scheduler], milestones=[warmup_epochs])

    # ----------------------
    # 6. Load checkpoint
    # ----------------------
    if initial_checkpoint:
        print('[Loading model from %s]' % initial_checkpoint)
        checkpoint = torch.load(initial_checkpoint)
        if not ('model.ckpt' in initial_checkpoint):
            start_epoch = checkpoint['epoch']

        state = net.state_dict()
        state.update(checkpoint['state_dict'])

        try:
            net.load_state_dict(state, strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        except:
            print('Load something failed!')
            traceback.print_exc()

    #----------------------
    # 7. log
    #----------------------
    model_out_dir = os.path.join(out_dir, 'model')
    tb_out_dir = os.path.join(out_dir, 'runs')
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    logfile = os.path.join(out_dir, 'log_train')
    sys.stdout = Logger(logfile)

    print('[Training configuration]')
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print('[Model conf  iguration]')
    pprint.pprint(net_config)
    print('[start_epoch %d, out_dir %s]' % (start_epoch, out_dir))
    print('[length of train loader %d, length of valid loader %d]' % (len(train_loader), len(val_loader)))

    # Write graph to tensorboard for visualization
    writer = SummaryWriter(tb_out_dir)
    train_writer = SummaryWriter(os.path.join(tb_out_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(tb_out_dir, 'val'))
    # writer.add_graph(net, (torch.zeros((16, 1, 128, 128, 128)).cuda(), [[]], [[]], [[]], [torch.zeros((16, 128, 128, 128))]), verbose=False)
    best_loss = np.inf
    val_loss = 10

    #----------------------
    # 8. training
    #----------------------
    for i in tqdm(range(start_epoch, epochs + 1), desc='Total'):
        scheduler.step()
        lr =  optimizer.param_groups[0]['lr']
        if i >= epoch_rcnn:
            net.use_rcnn = True
        else:
            net.use_rcnn = False

        print('[epoch %d, lr %f, use_rcnn: %r]' % (i, lr, net.use_rcnn))
        train_loss = train(net, train_loader, optimizer, i, train_writer)
        if i >= 20: 
            val_loss = validate(net, val_loader, i, val_writer)
            
        state_dict = net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        evaluator = ModelEvaluator()
        if i % epoch_save == 0 and i >= 0:
            torch.save({
                'epoch': i,
                'out_dir': out_dir, 
                'state_dict': state_dict,
                'optimizer' : optimizer.state_dict()},
                os.path.join(model_out_dir, 'epoch_%03d_loss_%.6f.ckpt' % (i, val_loss)))
            evaluator.eval(net,eval_loader,out_dir,i)
            evaluator.froc(i,val_writer,args.test_annotation_dir,args.test_series_uids_path,"combine_27mm")
            evaluator.froc(i,val_writer,"E:\\desktop\\training_data\\My_SAnet\\Data\\crop_test_27.csv",args.test_series_uids_path,"27mm")
        
            
        elif val_loss < best_loss and i >= 130:
            best_loss = val_loss
            torch.save({
                'epoch': i,
                'out_dir': out_dir,
                'state_dict': state_dict,
                'optimizer' : optimizer.state_dict()},
                os.path.join(model_out_dir, 'best_%03d.ckpt'% i))
            
    writer.close()
    train_writer.close()
    val_writer.close()


def train(net, train_loader, optimizer, epoch, writer):
    net.set_mode('train')
    s = time.time()
    rpn_cls_loss, rpn_reg_loss = [], []
    rpn_cls_p_loss, rpn_cls_n_loss = [], []
    rcnn_cls_loss, rcnn_reg_loss = [], []
    total_loss = [] 
    rpn_stats = []
    rcnn_stats = []
    scaler = GradScaler()
    for j, (input, truth_box, truth_label) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Train %d' % epoch):
    
        input = Variable(input).cuda()
        truth_box = np.array(truth_box)
        truth_label = np.array(truth_label)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            net(input, truth_box, truth_label)

            loss, rpn_stat, rcnn_stat = net.loss()
        if loss.data.isnan():
            print('Loss is nan')
        optimizer.zero_grad()   
        #loss.backward()
        scaler.scale(loss).backward()
        #optimizer.step()           
        scaler.step(optimizer)
        scaler.update()

        rpn_cls_loss.append(net.rpn_cls_loss.cpu().data.item())
        rpn_reg_loss.append(net.rpn_reg_loss.cpu().data.item())
        rcnn_cls_loss.append(net.rcnn_cls_loss.cpu().data.item())
        rcnn_reg_loss.append(net.rcnn_reg_loss.cpu().data.item())
        total_loss.append(loss.cpu().data.item())
        
        rpn_stats.append(np.asarray(torch.Tensor(rpn_stat).cpu(), np.float32))
        if net.use_rcnn:
            rcnn_stats.append(np.asarray(torch.Tensor(rcnn_stat).cpu(), np.float32))
            del rcnn_stat

        del input, truth_box, truth_label   
        del net.rpn_proposals, net.detections
        del net.total_loss, net.rpn_cls_loss, net.rpn_reg_loss, net.rcnn_cls_loss, net.rcnn_reg_loss
        del net.rpn_logits_flat, net.rpn_deltas_flat
        del rpn_stat

        if net.use_rcnn:
            del net.rcnn_logits, net.rcnn_deltas

        torch.cuda.empty_cache()

    rpn_stats = np.asarray(rpn_stats, np.float32)

    print('Train Epoch %d, iter %d, total time %f, loss %f' % (epoch, j, time.time() - s, np.average(total_loss)))
    print('rpn_cls %f, rpn_reg %f, rcnn_cls %f, rcnn_reg %f' % \
        (np.average(rpn_cls_loss), np.average(rpn_reg_loss),
            np.average(rcnn_cls_loss), np.average(rcnn_reg_loss)
            ))
    print('rpn_stats: tpr %f, tnr %f, total pos %d, total neg %d, p_loss %.4f, n_loss %.4f' % (
        100.0 * np.sum(rpn_stats[:, 0]) / np.sum(rpn_stats[:, 1]),
        100.0 * np.sum(rpn_stats[:, 2]) / np.sum(rpn_stats[:, 3]),
        np.sum(rpn_stats[:, 1]),
        np.sum(rpn_stats[:, 3]),
        np.mean(rpn_stats[:, 4]),
        np.mean(rpn_stats[:, 5])
        ))
    
    

    # Write to tensorboard
    writer.add_scalar('loss', np.average(total_loss), epoch)
    writer.add_scalar('rpn_cls', np.average(rpn_cls_loss), epoch)
    writer.add_scalar('rpn_reg', np.average(rpn_reg_loss), epoch)
    writer.add_scalar('rpn_p_loss',  np.mean(rpn_stats[:, 4]), epoch)
    writer.add_scalar('rpn_n_loss',  np.mean(rpn_stats[:, 5]), epoch)

    writer.add_scalar('rpn_tpr', 100.0 * np.sum(rpn_stats[:, 0]) / np.sum(rpn_stats[:, 1]), epoch)
    writer.add_scalar('rpn_tnr', 100.0 * np.sum(rpn_stats[:, 2]) / np.sum(rpn_stats[:, 3]), epoch)
    # if net.use_rcnn:
        
    #     #rcnn_stats = np.asarray([stat[:-1] for stat in rcnn_stats], np.float32)
    #     rcnn_stats = np.asarray(rcnn_stats, np.float32)
        
    #     print('rcnn_stats: tpr %f, tnr %f, total pos %d, total neg %d, p_loss %.4f, n_loss %.4f' % (
    #     100.0 * np.sum(rcnn_stats[:, 0]) / np.sum(rcnn_stats[:, 1]),
    #     100.0 * np.sum(rcnn_stats[:, 2]) / np.sum(rcnn_stats[:, 3]),
    #     np.sum(rcnn_stats[:, 1]),
    #     np.sum(rcnn_stats[:, 3]),
    #     np.mean(rcnn_stats[:, 4]),
    #     np.mean(rcnn_stats[:, 5])
    #     ))
    #     writer.add_scalar('rcnn_cls', np.average(rcnn_cls_loss), epoch)
    #     writer.add_scalar('rcnn_reg', np.average(rcnn_reg_loss), epoch)
    #     writer.add_scalar('rcnn_p_loss',  np.mean(rcnn_stats[:, 4]), epoch)
    #     writer.add_scalar('rcnn_n_loss',  np.mean(rcnn_stats[:, 5]), epoch)

    #     writer.add_scalar('rcnn_tpr', 100.0 * np.sum(rcnn_stats[:, 0]) / np.sum(rcnn_stats[:, 1]), epoch)
    #     writer.add_scalar('rcnn_tnr', 100.0 * np.sum(rcnn_stats[:, 2]) / np.sum(rcnn_stats[:, 3]), epoch)
    return np.average(total_loss)

def validate(net, val_loader, epoch, writer):
    net.set_mode('valid')
    rpn_cls_loss, rpn_reg_loss = [], []
    rcnn_cls_loss, rcnn_reg_loss = [], []
    total_loss = []
    rpn_stats = []
    rcnn_stats = []

    s = time.time()
    for j, (input, truth_box, truth_label) in tqdm(enumerate(val_loader), total=len(val_loader), desc='Val %d' % epoch):
        with torch.no_grad():
            input = Variable(input).cuda()
            truth_box = np.array(truth_box)
            truth_label = np.array(truth_label)

            # with autocast():
            net(input, truth_box, truth_label)       
            loss, rpn_stat, rcnn_stat = net.loss()
            #print(loss.data)
            # get network output
        rpn_cls_loss.append(net.rpn_cls_loss.cpu().data.item())
        rpn_reg_loss.append(net.rpn_reg_loss.cpu().data.item())
        rcnn_cls_loss.append(net.rcnn_cls_loss.cpu().data.item())
        rcnn_reg_loss.append(net.rcnn_reg_loss.cpu().data.item())

        total_loss.append(loss.cpu().data.item())
        rpn_stats.append(np.asarray(torch.Tensor(rpn_stat).cpu(), np.float32))
        if net.use_rcnn:
            rcnn_stats.append(np.asarray(torch.Tensor(rcnn_stat).cpu(), np.float32))
            del rcnn_stat

    rpn_stats = np.asarray(rpn_stats, np.float32)
    print('Val Epoch %d, iter %d, total time %f, loss %f' % (epoch, j, time.time()-s, np.average(total_loss)))
    print('rpn_cls %f, rpn_reg %f, rcnn_cls %f, rcnn_reg %f' % \
        (np.average(rpn_cls_loss), np.average(rpn_reg_loss),
            np.average(rcnn_cls_loss), np.average(rcnn_reg_loss)
            ))
    print('rpn_stats: tpr %f, tnr %f, total pos %d, total neg %d, p_loss %.4f, n_loss %.4f' % (
        100.0 * np.sum(rpn_stats[:, 0]) / np.sum(rpn_stats[:, 1]),
        100.0 * np.sum(rpn_stats[:, 2]) / np.sum(rpn_stats[:, 3]),
        np.sum(rpn_stats[:, 1]),
        np.sum(rpn_stats[:, 3]),
        np.mean(rpn_stats[:, 4]),
        np.mean(rpn_stats[:, 5])
        ))
    
    # Write to tensorboard
    writer.add_scalar('loss', np.average(total_loss), epoch)
    writer.add_scalar('rpn_cls', np.average(rpn_cls_loss), epoch)
    writer.add_scalar('rpn_reg', np.average(rpn_reg_loss), epoch)
    writer.add_scalar('rpn_p_loss',  np.mean(rpn_stats[:, 4]), epoch)
    writer.add_scalar('rpn_n_loss',  np.mean(rpn_stats[:, 5]), epoch)
    TP = np.sum(rpn_stats[:, 0])        
    FP = np.sum(rpn_stats[:, 3]) - np.sum(rpn_stats[:, 2])
    TN = np.sum(rpn_stats[:, 2])
    FN = np.sum(rpn_stats[:, 1]) - np.sum(rpn_stats[:, 0])
    writer.add_scalar('rpn_tpr', 100.0 * np.sum(rpn_stats[:, 0]) / np.sum(rpn_stats[:, 1]), epoch)
    writer.add_scalar('rpn_tnr', 100.0 * np.sum(rpn_stats[:, 2]) / np.sum(rpn_stats[:, 3]), epoch)
    writer.add_scalar('rpn_precision', 100.0 *TP / (TP + FP), epoch)
    writer.add_scalar('rpn_recall', 100.0 *TP / (TP + FN), epoch)
    writer.add_scalar('rpn_f1', 100.0 *2 * TP / (2 * TP + FP + FN), epoch)

    if net.use_rcnn:
        
        # #rcnn_stats = np.asarray([stat[:-1] for stat in rcnn_stats], np.float32)
        # rcnn_stats = np.asarray(rcnn_stats, np.float32)
        
        # print('rcnn_stats: tpr %f, tnr %f, total pos %d, total neg %d, p_loss %.4f, n_loss %.4f' % (
        # 100.0 * np.sum(rcnn_stats[:, 0]) / np.sum(rcnn_stats[:, 1]),
        # 100.0 * np.sum(rcnn_stats[:, 2]) / np.sum(rcnn_stats[:, 3]),
        # np.sum(rcnn_stats[:, 1]),
        # np.sum(rcnn_stats[:, 3]),
        # np.mean(rcnn_stats[:, 4]),
        # np.mean(rcnn_stats[:, 5])
        # ))
        # writer.add_scalar('rcnn_cls', np.average(rcnn_cls_loss), epoch)
        # writer.add_scalar('rcnn_reg', np.average(rcnn_reg_loss), epoch)
        # writer.add_scalar('rcnn_p_loss',  np.mean(rcnn_stats[:, 4]), epoch)
        # writer.add_scalar('rcnn_n_loss',  np.mean(rcnn_stats[:, 5]), epoch)

        # writer.add_scalar('rcnn_tpr', 100.0 * np.sum(rcnn_stats[:, 0]) / np.sum(rcnn_stats[:, 1]), epoch)
        # writer.add_scalar('rcnn_tnr', 100.0 * np.sum(rcnn_stats[:, 2]) / np.sum(rcnn_stats[:, 3]), epoch)
        
        # TP = np.sum(rcnn_stats[:, 0])        
        # FP = np.sum(rcnn_stats[:, 3]) - np.sum(rcnn_stats[:, 2])
        # TN = np.sum(rcnn_stats[:, 2])
        # FN = np.sum(rcnn_stats[:, 1]) - np.sum(rcnn_stats[:, 0])
        # writer.add_scalar('rcnn_tpr', 100.0 * np.sum(rcnn_stats[:, 0]) / np.sum(rcnn_stats[:, 1]), epoch)
        # writer.add_scalar('rcnn_tnr', 100.0 * np.sum(rcnn_stats[:, 2]) / np.sum(rcnn_stats[:, 3]), epoch)
        # writer.add_scalar('rcnn_precision', 100.0 *TP / (TP + FP), epoch)
        # writer.add_scalar('rcnn_recall', 100.0 *TP / (TP + FN), epoch)
        # writer.add_scalar('rcnn_f1', 100.0 *2 * TP / (2 * TP + FP + FN), epoch)
        del input, truth_box, truth_label
        del net.rpn_proposals, net.detections
        del net.total_loss, net.rpn_cls_loss, net.rpn_reg_loss, net.rcnn_cls_loss, net.rcnn_reg_loss
        del rpn_stat
    if net.use_rcnn:
        del net.rcnn_logits, net.rcnn_deltas

    torch.cuda.empty_cache()
    return np.average(total_loss)

if __name__ == '__main__':
    main()






