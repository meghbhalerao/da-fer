import os
import sys
import time
import tqdm
import shutil
import argparse
import subprocess
import numpy as np
import pandas as pd
#sys.path.append("/home/megh/projects/fer/CD-FER-Benchmark/AGRA")
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

from Loss import HAFN, SAFN
from Utils import *

parser = argparse.ArgumentParser(description='Expression Classification Training')

parser.add_argument('--Log_Name', type=str, help='Log Name')
parser.add_argument('--OutputPath', type=str, help='Output Path')
parser.add_argument('--net', type=str, default='ResNet50', choices=['ResNet18', 'ResNet50', 'VGGNet', 'MobileNet'])
parser.add_argument('--pretrained', type=str, help='Resume_Model', default='None')
parser.add_argument('--GPU_ID', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

parser.add_argument('--useAFN', type=str2bool, default=False, help='whether to use AFN Loss')
parser.add_argument('--methodOfAFN', type=str, default='SAFN', choices=['HAFN', 'SAFN'])
parser.add_argument('--radius', type=float, default=25.0, help='radius of HAFN (default: 25.0)')
parser.add_argument('--deltaRadius', type=float, default=1.0, help='radius of SAFN (default: 1.0)')
parser.add_argument('--weight_L2norm', type=float, default=0.05, help='weight L2 norm of AFN (default: 0.05)')

parser.add_argument('--face_scale', type=int, default=112, help='Scale of face (default: 112)')
parser.add_argument('--source', type=str, default='RAF', choices=['RAF', 'AFED', 'MMI'])
parser.add_argument('--target', type=str, default='CK+', choices=['RAF', 'CK+', 'JAFFE', 'MMI', 'Oulu-CASIA', 'SFEW', 'FER2013', 'ExpW', 'AFED', 'WFED','AISIN'])
parser.add_argument('--train_batch', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--test_batch', type=int, default=64, help='input batch size for testing (default: 64)')
parser.add_argument('--useMultiDatasets', type=str2bool, default=False, help='whether to use MultiDataset')

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=10,help='number of epochs to train (default: 10)')
parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
parser.add_argument('--weight_decay', type=float, default=0.0005,help='SGD weight decay (default: 0.0005)')

parser.add_argument('--isTest', type=str2bool, default=False, help='whether to test model')
parser.add_argument('--showFeature', type=str2bool, default=False, help='whether to show feature')

parser.add_argument('--intra_gcn', type=str2bool, default=False, help='whether to use Intra-GCN')
parser.add_argument('--inter_gcn', type=str2bool, default=False, help='whether to use Inter-GCN')
parser.add_argument('--local_feat', type=str2bool, default=False, help='whether to use Local Feature')

parser.add_argument('--rand_mat', type=str2bool, default=False, help='whether to use Random Matrix')
parser.add_argument('--all1_mat', type=str2bool, default=False, help='whether to use All One Matrix')

parser.add_argument('--use_cov', type=str2bool, default=False, help='whether to use Cov')
parser.add_argument('--use_cluster', type=str2bool, default=False, help='whether to use Cluster')

parser.add_argument('--class_num', type=int, default=7, help='number of class (default: 7)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--use_gcn', type=str2bool, default=False, help='whether to use Graph Convolutional (AGRA) Network')


def Train(args, model, train_dataloader, optimizer, epoch, writer):
    """Train."""

    model.train()
    torch.autograd.set_detect_anomaly(True)

    acc, prec, recall = [AverageMeter() for i in range(args.class_num)], [AverageMeter() for i in range(args.class_num)], [AverageMeter() for i in range(args.class_num)]
    loss, global_cls_loss, local_cls_loss, afn_loss, data_time, batch_time =  AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    # Decay Learn Rate per Epoch
    if args.net in ['ResNet18', 'ResNet50']:
        if epoch <= 20:
            args.lr = 1e-4
        elif epoch <= 40:
            args.lr = 1e-5
        else:
            args.lr = 1e-6

    elif args.net == 'MobileNet':
        if epoch <= 20:
            args.lr = 1e-3
        elif epoch <= 40:
            args.lr = 1e-4
        elif epoch <= 60:
            args.lr = 1e-5
        else:
            args.lr = 1e-6

    elif args.net == 'VGGNet':
        if epoch <= 30:
            args.lr = 1e-3
        elif epoch <= 60:
            args.lr = 1e-4
        elif epoch <= 70:
            args.lr = 1e-5
        else:
            args.lr = 1e-6

    end = time.time()
    for step, (input, landmark, label) in enumerate(train_dataloader):

        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        data_time.update(time.time()-end)

        # Forward propagation
        end = time.time()
        feature, output, loc_output = model(input, landmark)
        batch_time.update(time.time()-end)

        # Compute Loss
        global_cls_loss_ = nn.CrossEntropyLoss()(output, label) 
        local_cls_loss_ = nn.CrossEntropyLoss()(loc_output, label) if args.local_feat else 0
        afn_loss_ = (HAFN(feature, args.weight_L2norm, args.radius) if args.methodOfAFN=='HAFN' else SAFN(feature, args.weight_L2norm, args.deltaRadius)) if args.useAFN else 0
        loss_ = global_cls_loss_ + local_cls_loss_ + (afn_loss_ if args.useAFN else 0)

        # Back Propagation
        optimizer.zero_grad()
        
        with torch.autograd.detect_anomaly():
            loss_.backward()

        optimizer.step()

        # Decay Learn Rate
        optimizer, lr = lr_scheduler_withoutDecay(optimizer, lr=args.lr, weight_decay=args.weight_decay) # optimizer = lr_scheduler(optimizer, num_iter*(epoch-1)+step, 0.001, 0.75, lr=args.lr, weight_decay=args.weight_decay)

        # Compute accuracy, recall and loss
        Compute_Accuracy(args, output, label, acc, prec, recall)
    
        # Log loss
        loss.update(float(loss_.cpu().data.item()))
        global_cls_loss.update(float(global_cls_loss_.cpu().data.item()))
        local_cls_loss.update(float(local_cls_loss_.cpu().data.item()) if args.local_feat else 0)
        afn_loss.update(float(afn_loss_.cpu().data.item()) if args.useAFN else 0)

        end = time.time()

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    writer.add_scalar('Accuracy', acc_avg, epoch)
    writer.add_scalar('Precision', prec_avg, epoch)
    writer.add_scalar('Recall', recall_avg, epoch)
    writer.add_scalar('F1', f1_avg, epoch)

    writer.add_scalar('Global_Cls_Loss', global_cls_loss.avg, epoch)
    writer.add_scalar('Local_Cls_Loss', local_cls_loss.avg, epoch)
    writer.add_scalar('AFN_Loss', afn_loss.avg, epoch)

    LoggerInfo = '''
    [Train]: 
    Epoch {0}
    Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
    Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})
    Learning Rate {1}\n'''.format(epoch, args.lr, data_time=data_time, batch_time=batch_time)

    LoggerInfo+=AccuracyInfo

    LoggerInfo+='''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}
    Total Loss {loss:.4f} Global Cls Loss {global_cls_loss:.4f} Local Cls Loss {local_cls_loss:.4f} AFN Loss {afn_loss:.4f}'''.format(acc_avg, prec_avg, recall_avg, f1_avg, loss=loss.avg, global_cls_loss=global_cls_loss.avg, local_cls_loss=local_cls_loss.avg,afn_loss=afn_loss.avg)

    print(LoggerInfo)

def Test(args, model, test_source_dataloader, test_target_dataloader, Best_Recall):
    """Test."""

    model.eval()
    torch.autograd.set_detect_anomaly(True)

    iter_source_dataloader = iter(test_source_dataloader)
    iter_target_dataloader = iter(test_target_dataloader)

    # Test on Source Domain
    acc, prec, recall = [AverageMeter() for i in range(args.class_num)], [AverageMeter() for i in range(args.class_num)], [AverageMeter() for i in range(args.class_num)]
    loss, global_cls_loss, local_cls_loss, afn_loss, data_time, batch_time =  AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    print("Starting Source Data Testing")
    for step, (input, landmark, label) in enumerate(iter_source_dataloader):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        data_time.update(time.time()-end)
        
        # Forward Propagation
        with torch.no_grad():
            end = time.time()
            feature, output, loc_output = model(input, landmark)
            batch_time.update(time.time()-end)
        
        # Compute Loss
        global_cls_loss_ = nn.CrossEntropyLoss()(output, label)
        local_cls_loss_ = nn.CrossEntropyLoss()(loc_output, label) if args.local_feat else 0
        afn_loss_ = (HAFN(feature, args.weight_L2norm, args.radius) if args.methodOfAFN=='HAFN' else SAFN(feature, args.weight_L2norm, args.deltaRadius)) if args.useAFN else 0
        loss_ = global_cls_loss_ + local_cls_loss_ + (afn_loss_ if args.useAFN else 0)

        # Compute accuracy, precision and recall
        Compute_Accuracy(args, output, label, acc, prec, recall)

        # Log loss
        loss.update(float(loss_.cpu().data.item()))
        global_cls_loss.update(float(global_cls_loss_.cpu().data.item()))
        local_cls_loss.update(float(local_cls_loss_.cpu().data.item()) if args.local_feat else 0)
        afn_loss.update(float(afn_loss_.cpu().data.item()) if args.useAFN else 0)

        end = time.time()

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    LoggerInfo = '''
    Test Source: 
    Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
    Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})
    Learning Rate {0}\n'''.format(args.lr, data_time=data_time, batch_time=batch_time)

    LoggerInfo+=AccuracyInfo

    LoggerInfo+='''Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}
    Total Loss {loss:.4f} Global Cls Loss {global_cls_loss:.4f} Local Cls Loss {local_cls_loss:.4f} AFN Loss {afn_loss:.4f}'''.format(acc_avg, prec_avg, recall_avg, f1_avg, loss=loss.avg,global_cls_loss=global_cls_loss.avg,local_cls_loss=local_cls_loss.avg, afn_loss=afn_loss.avg)
    print(LoggerInfo)

    # Save Checkpoints
    if recall_avg > Best_Recall:
        Best_Recall = recall_avg
        print("Recall Better than Best Recall")
        print('[Save] Best Recall: %.4f.' % Best_Recall)

        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), os.path.join(args.OutputPath, '{}.pkl'.format(args.Log_Name)))
        else:
            torch.save(model.state_dict(), os.path.join(args.OutputPath, '{}.pkl'.format(args.Log_Name)))

    # Test on Target Domain
    acc, prec, recall = [AverageMeter() for i in range(args.class_num)], [AverageMeter() for i in range(args.class_num)], [AverageMeter() for i in range(args.class_num)]
    loss, global_cls_loss, local_cls_loss, afn_loss, data_time, batch_time =  AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for step, (input, landmark, label) in enumerate(iter_target_dataloader):
        
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        data_time.update(time.time()-end)
        
        # Forward Propagation
        with torch.no_grad():
            end = time.time()
            feature, output, loc_output = model(input, landmark)
            batch_time.update(time.time()-end)
        
        # Compute Loss
        global_cls_loss_ = nn.CrossEntropyLoss()(output, label)
        local_cls_loss_ = nn.CrossEntropyLoss()(loc_output, label)  if args.local_feat else 0
        afn_loss_ = (HAFN(feature, args.weight_L2norm, args.radius) if args.methodOfAFN=='HAFN' else SAFN(feature, args.weight_L2norm, args.deltaRadius)) if args.useAFN else 0
        loss_ = global_cls_loss_ + local_cls_loss_ + (afn_loss_ if args.useAFN else 0)

        # Compute accuracy, precision and recall
        Compute_Accuracy(args, output, label, acc, prec, recall)

        # Log loss
        loss.update(float(loss_.cpu().data.item()))
        global_cls_loss.update(float(global_cls_loss_.cpu().data.item()))
        local_cls_loss.update(float(local_cls_loss_.cpu().data.item()) if args.local_feat else 0)
        afn_loss.update(float(afn_loss_.cpu().data.item()) if args.useAFN else 0)

        end = time.time()

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    LoggerInfo = '''
    Test Target: 
    Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
    Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})
    Learning Rate {0}\n'''.format(args.lr, data_time=data_time, batch_time=batch_time)

    LoggerInfo+=AccuracyInfo

    LoggerInfo+='''Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}
    Total Loss {loss:.4f} Global Cls Loss {global_cls_loss:.4f} Local Cls Loss {local_cls_loss:.4f} AFN Loss {afn_loss:.4f}'''.format(acc_avg, prec_avg, recall_avg, f1_avg, loss=loss.avg, global_cls_loss=global_cls_loss.avg, local_cls_loss=local_cls_loss.avg, afn_loss=afn_loss.avg)

    print(LoggerInfo)

    return Best_Recall

def main():
    """Main."""
 
    # Parse Argument
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # Experiment Information
    print('log : %s' % args.Log_Name, 'out-pth %s' % args.OutputPath, 'net: %s' % args.net, 'pretrained %s' % args.pretrained, 'dev %s' % args.GPU_ID )

    print('Use {} * {} Image'.format(args.face_scale, args.face_scale), 'source %s' % args.source, 'target %s' % args.target, 'train bs %d' % args.train_batch,'test bs %d' % args.test_batch)

    if args.showFeature:
        print('Show Visualization Result of Feature.')

    if args.isTest:
        print('Test Model.')
    else:
        print('Ep: %d' % args.epochs, 'LR %f' % args.lr, 'mu: %f' % args.momentum, 'WD: %f' % args.weight_decay)

        if args.useAFN:
            print('Use AFN Loss: %s' % args.methodOfAFN)
            if args.methodOfAFN=='HAFN':
                print('Radius of HAFN Loss: %f' % args.radius)
            else:
                print('Delta Radius of SAFN Loss: %f' % args.deltaRadius)
            print('Weight L2 nrom of AFN Loss: %f' % args.weight_L2norm)

    print('================================================')

    print('Num cls : %d' % args.class_num)
    if not args.local_feat:
        print('Only use global feature.')
    else:
        print('Use global feature and local feature.')

        if args.intra_gcn:
            print('Use Intra GCN.')
        if args.inter_gcn:
            print('Use Inter GCN.')

        if args.rand_mat and args.all1_mat:
            print('Wrong : Use RandomMatrix and AllOneMatrix both!')
            return None
        elif args.rand_mat:
            print('Use Random Matrix in GCN.')
        elif args.all1_mat:
            print('Use All One Matrix in GCN.')

        if args.use_cov and args.use_cluster:
            print('Wrong : Use Cov and Cluster both!')
            return None
        else:
            if args.use_cov:
                print('Use Mean and Cov.')
            else:
                print('Use Mean.') if not args.use_cluster else print('Use Mean in Cluster.')

    print('================================================')

    # Bulid Dataloder
    print("Building Train and Test Dataloader...")
    train_source_dataloader = BulidDataloader(args, flag1='train', flag2='source')
    train_target_dataloader = BulidDataloader(args, flag1='train', flag2='target')
    test_source_dataloader = BulidDataloader(args, flag1='test', flag2='source')
    test_target_dataloader = BulidDataloader(args, flag1='test', flag2='target')
    print('Done!')

    print('================================================')

    # Bulid Model
    print('Building Model...')
    model = BulidModel(args)
    print('Done!')

    print('================================================')

    # Init Mean
    if args.local_feat and args.intra_gcn and args.inter_gcn and not args.isTest:         
        if args.use_cov:
            print('Init Mean and Cov...')
            Initialize_Mean_Cov(args, model, True)
        else:
            if args.use_cluster:
                print('Initialize Mean in Cluster....')
                Initialize_Mean_Cluster(args, model, True)
            else:         
                print('Init Mean...')
                Initialize_Mean(args, model, True)

        torch.cuda.empty_cache()

        print('Done!')
        print('================================================')

    # Set Optimizer
    print('Building Optimizer...')
    param_optim = Set_Param_Optim(args, model)
    optimizer = Set_Optimizer(args, param_optim, args.lr, args.weight_decay, args.momentum)
    print('Done!')

    print('================================================')

    # Save Best Checkpoint
    Best_Recall = 0

    # Running Experiment
    print("Run Experiment...")
    writer = SummaryWriter(os.path.join(args.OutputPath, args.Log_Name))

    for epoch in range(1, args.epochs + 1):
        if args.showFeature and epoch%5 == 1:
            Visualization('{}_Source.pdf'.format(epoch), model, train_source_dataloader, useClassify=True, domain='Source')
            Visualization('{}_Target.pdf'.format(epoch), model, train_target_dataloader, useClassify=True, domain='Target')
            VisualizationForTwoDomain('{}_train'.format(epoch), model, train_source_dataloader, train_target_dataloader, useClassify=True, showClusterCenter=False)
            VisualizationForTwoDomain('{}_test'.format(epoch), model, test_source_dataloader, test_target_dataloader, useClassify=True, showClusterCenter=False)

        if not args.isTest:
            if args.use_cluster and epoch%10 == 0:
                Initialize_Mean_Cluster(args, model, True)
                torch.cuda.empty_cache()
            Train(args, model, train_source_dataloader, optimizer, epoch, writer)

        Best_Recall = Test(args, model, test_source_dataloader, test_target_dataloader, Best_Recall)

        torch.cuda.empty_cache()

    writer.close()

if __name__ == '__main__':
    main()
