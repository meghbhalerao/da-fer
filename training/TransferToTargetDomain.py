import os
import sys
import time
import tqdm
import shutil
import argparse
import subprocess
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from Loss import Entropy, DANN, CDAN, HAFN, SAFN, MME
from Utils import *

parser = argparse.ArgumentParser(description='Domain adaptation for Expression Classification')

parser.add_argument('--log', type=str, help='Log Name')
parser.add_argument('--out', type=str, help='Output Path')
parser.add_argument('--net', type=str, default='ResNet50', choices=['ResNet18', 'ResNet50', 'VGGNet', 'MobileNet'])
parser.add_argument('--pretrained', type=str, help='pretrained', default='None')
parser.add_argument('--dev', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

parser.add_argument('--use_dan', type=str2bool, default=False, help='whether to use DAN Loss')
parser.add_argument('--dan_method', type=str, default='CDAN-E', choices=['CDAN', 'CDAN-E', 'DANN','MME'])

parser.add_argument('--use_afn', type=str2bool, default=False, help='whether to use AFN Loss')
parser.add_argument('--afn_method', type=str, default='SAFN', choices=['HAFN', 'SAFN'])
parser.add_argument('--r', type=float, default=25.0, help='radius of HAFN (default: 25.0)')
parser.add_argument('--dr', type=float, default=1.0, help='radius of SAFN (default: 1.0)')
parser.add_argument('--w_l2', type=float, default=0.05, help='weight L2 norm of AFN (default: 0.05)')

parser.add_argument('--face_scale', type=int, default=112, help='Scale of face (default: 112)')
parser.add_argument('--source', type=str, default='RAF', choices=['RAF', 'AFED', 'MMI'])
parser.add_argument('--target', type=str, default='CK+', choices=['RAF', 'CK+', 'JAFFE', 'MMI', 'Oulu-CASIA', 'SFEW', 'FER2013', 'ExpW', 'AFED', 'WFED','AISIN'])
parser.add_argument('--train_batch', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--test_batch', type=int, default=64, help='input batch size for testing (default: 64)')
parser.add_argument('--multiple_data', type=str2bool, default=False, help='whether to use MultiDataset')

parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--lr_ad', type=float, default=0.01)

parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
parser.add_argument('--momentum', type=float, default=0.5,  help='SGD momentum (default: 0.5)')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='SGD weight decay (default: 0.0005)')

parser.add_argument('--isTest', type=str2bool, default=False, help='whether to test model')
parser.add_argument('--show_feat', type=str2bool, default=False, help='whether to show feature')

parser.add_argument('--intra_gcn', type=str2bool, default=False, help='whether to use Intra-GCN')
parser.add_argument('--inter_gcn', type=str2bool, default=False, help='whether to use Inter-GCN')
parser.add_argument('--local_feat', type=str2bool, default=False, help='whether to use Local Feature')

parser.add_argument('--rand_mat', type=str2bool, default=False, help='whether to use Random Matrix')
parser.add_argument('--all1_mat', type=str2bool, default=False, help='whether to use All One Matrix')

parser.add_argument('--use_cov', type=str2bool, default=False, help='whether to use Cov')

parser.add_argument('--class_num', type=int, default=7, help='number of class (default: 7)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--rand_layer', type=str2bool, default=False, help='whether to use random')
parser.add_argument('--use_cluster', type=str2bool, default=False, help='whether to use Cluster')
parser.add_argument('--method', type=str, default="CADA", help='Choose the method of the experiment')

def Train(args, model, ad_net, random_layer, train_source_dataloader, train_target_dataloader, optimizer, optimizer_ad, epoch, writer):
    """Train."""

    model.train()
    torch.autograd.set_detect_anomaly(True)

    acc, prec, recall = [AverageMeter() for i in range(args.class_num)], [AverageMeter() for i in range(args.class_num)], [AverageMeter() for i in range(args.class_num)]
    loss, global_cls_loss, local_cls_loss, afn_loss, dan_loss = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    data_time, batch_time = AverageMeter(), AverageMeter()

    if args.use_dan:
        num_ADNet = 0
        ad_net.train()

    # Decay Learn Rate per Epoch
    if epoch <= 20:
        args.lr, args.lr_ad = 0.0001, 0.001
    elif epoch <= 40:
        args.lr, args.lr_ad = 0.0001, 0.0001
    else:
        args.lr, args.lr_ad = 0.00001, 0.00001

    optimizer, lr = lr_scheduler_withoutDecay(optimizer, lr=args.lr)
    if args.use_dan:
        optimizer_ad, lr_ad = lr_scheduler_withoutDecay(optimizer_ad, lr=args.lr_ad)

    # Get Source/Target Dataloader iterator
    iter_source_dataloader = iter(train_source_dataloader)
    iter_target_dataloader = iter(train_target_dataloader)

    # len(data_loader) = math.ceil(len(data_loader.dataset)/batch_size)
    num_iter = len(train_source_dataloader) if (len(train_source_dataloader) > len(train_target_dataloader)) else len(train_target_dataloader)

    end = time.time()
    for batch_index in range(num_iter):
        try:
            data_source, landmark_source, label_source = iter_source_dataloader.next()
        except:
            iter_source_dataloader = iter(train_source_dataloader)
            data_source, landmark_source, label_source = iter_source_dataloader.next()

        try:
            data_target, landmark_target, label_target = iter_target_dataloader.next()
        except:
            iter_target_dataloader = iter(train_target_dataloader)
            data_target, landmark_target, label_target = iter_target_dataloader.next()
        
        data_time.update(time.time()-end)

        data_source, landmark_source, label_source = data_source.cuda(), landmark_source.cuda(), label_source.cuda()
        data_target, landmark_target, label_target = data_target.cuda(), landmark_target.cuda(), label_target.cuda()

        # Forward Propagation
        end = time.time()
        feature, output, loc_output = model(torch.cat((data_source, data_target), 0), torch.cat((landmark_source, landmark_target), 0), False)
        feat_target = feature[args.train_batch:,:]
        #feat_target, out_target, loc_out_target = model(data_target, landmark_target, False)
        #feat_source, out_source, loc_out_source = model(data_source, landmark_source, False)
        #feature = torch.cat((feat_source,feat_target),0)
        #output = torch.cat((out_target,out_source),0)
        #loc_output = torch.cat((loc_out_source,loc_out_target),0)
        batch_time.update(time.time()-end)

        # Compute Loss
        global_cls_loss_ = nn.CrossEntropyLoss()(output.narrow(0, 0, data_source.size(0)), label_source)
        local_cls_loss_ = nn.CrossEntropyLoss()(loc_output.narrow(0, 0, data_source.size(0)), label_source) if args.local_feat else 0

        afn_loss_ = (HAFN(feature, args.w_l2, args.r) if args.afn_method=='HAFN' else SAFN(feature, args.w_l2, args.dr)) if args.use_afn else 0
        
        if args.use_dan:
            softmax_output = nn.Softmax(dim=1)(output)
            if args.dan_method == 'CDAN-E':
                entropy = Entropy(softmax_output)
                dan_loss_ = CDAN([feature, softmax_output], ad_net, entropy, calc_coeff(num_iter*(epoch-1)+batch_index), random_layer)
            elif args.dan_method == 'CDAN':
                dan_loss_ = CDAN([feature, softmax_output], ad_net, None, None, random_layer)
            elif args.dan_method == 'DANN':
                dan_loss_ = DANN(feature, ad_net)
                print(dan_loss_)
            elif args.dan_method == "MME":
                    dan_loss_  = MME(model, feat_target,lamda=0.1,mode='minimax')
                    #dan_loss_.backward()
                    if epoch >=1000:
                        a, b, _, _, pl_loss = do_fixmatch(f, data_target_,label_target,landmark_target,model,0.975,nn.CrossEntropyLoss(reduce='none'))
                        sum_ = sum_ + a
                        sum_batch = sum_batch + b
        else:
            dan_loss_ = 0

        if epoch >= 1000    :
                    loss_ = global_cls_loss_ + local_cls_loss_ + pl_loss
        else:
            loss_ = global_cls_loss_ + local_cls_loss_
            
        if args.use_afn:
            loss_+=afn_loss_

        if args.use_dan:
            loss_+=dan_loss_

        # Log Adversarial Network Accuracy
        if args.use_dan:
            if args.dan_method=='CDAN' or args.dan_method=='CDAN-E': 
                softmax_output = nn.Softmax(dim=1)(output)
                if args.rand_layer:
                    random_out = random_layer.forward([feature, softmax_output])
                    adnet_output = ad_net(random_out.view(-1, random_out.size(1)))
                else:
                    op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
                    adnet_output = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
            elif args.dan_method=='DANN' or args.dan_method == 'MME': 
                adnet_output = ad_net(feature)

            adnet_output = adnet_output.cpu().data.numpy()
                
            adnet_output[adnet_output>0.5] = 1
            adnet_output[adnet_output<=0.5] = 0
            num_ADNet+=np.sum(adnet_output[:args.train_batch]) + (args.train_batch - np. sum(adnet_output[args.train_batch:]))

        # Back Propagation
        optimizer.zero_grad()
        if args.use_dan:
            optimizer_ad.zero_grad()
        
        with torch.autograd.detect_anomaly():
            loss_.backward()

        optimizer.step()
        if not args.dan_method == "MME":
            if args.use_dan:
                optimizer_ad.step()

        # Compute accuracy, precision and recall
        Compute_Accuracy(args, output.narrow(0, 0, data_source.size(0)), label_source, acc, prec, recall)

        # Log loss
        loss.update(float(loss_.cpu().data.item()))
        global_cls_loss.update(float(global_cls_loss_.cpu().data.item()))
        local_cls_loss.update(float(local_cls_loss_.cpu().data.item()) if args.local_feat else 0)
        afn_loss.update(float(afn_loss_.cpu().data.item()) if args.use_afn else 0)
        dan_loss.update(float(dan_loss_.cpu().data.item()) if args.use_dan else 0)

        writer.add_scalar('Glocal_Cls_Loss', float(global_cls_loss_.cpu().data.item()), num_iter*(epoch-1)+batch_index)
        writer.add_scalar('Local_Cls_Loss', float(local_cls_loss_.cpu().data.item()) if args.local_feat else 0, num_iter*(epoch-1)+batch_index)
        writer.add_scalar('AFN_Loss', float(afn_loss_.cpu().data.item()) if args.use_afn else 0, num_iter*(epoch-1)+batch_index)
        writer.add_scalar('DAN_Loss', float(dan_loss_.cpu().data.item()) if args.use_dan else 0, num_iter*(epoch-1)+batch_index)

        end = time.time()

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    writer.add_scalar('Accuracy', acc_avg, epoch)
    writer.add_scalar('Precision', prec_avg, epoch)
    writer.add_scalar('Recall', recall_avg, epoch)
    writer.add_scalar('F1', f1_avg, epoch)

    if args.use_dan:
        writer.add_scalar('AdversarialNetwork_Accuracy', num_ADNet/(2.0*args.train_batch*num_iter), epoch)
    
    LoggerInfo = '''
    [Tain]: 
    Epoch {0}
    Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
    Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})
    Learning Rate {1} Learning Rate(AdversarialNet) {2}\n'''.format(epoch, lr, lr_ad if args.use_dan else 0, data_time=data_time, batch_time=batch_time)

    LoggerInfo+=AccuracyInfo

    LoggerInfo+='''    AdversarialNet Acc {0:.4f} Acc_avg {1:.4f} Prec_avg {2:.4f} Recall_avg {3:.4f} F1_avg {4:.4f}
    Total Loss {loss:.4f} Global Cls Loss {global_cls_loss:.4f} Local Cls Loss {local_cls_loss:.4f} AFN Loss {afn_loss:.4f} DAN Loss {dan_loss:.4f}'''.format(num_ADNet/(2.0*args.train_batch*num_iter) if args.use_dan else 0, acc_avg, prec_avg, recall_avg, f1_avg, loss=loss.avg, global_cls_loss=global_cls_loss.avg, local_cls_loss=local_cls_loss.avg, afn_loss=afn_loss.avg if args.use_afn else 0, dan_loss=dan_loss.avg if args.use_dan else 0)
                                                                                
    print(LoggerInfo)

def Test(args, model, test_source_dataloader, test_target_dataloader, Best_Accuracy, Best_Recall, epoch, writer):
    """Test."""

    model.eval()
    torch.autograd.set_detect_anomaly(True)

    iter_source_dataloader = iter(test_source_dataloader)
    iter_target_dataloader = iter(test_target_dataloader)

    # Test on Source Domain
    acc, prec, recall = [AverageMeter() for i in range(args.class_num)], [AverageMeter() for i in range(args.class_num)], [AverageMeter() for i in range(args.class_num)]
    loss, data_time, batch_time =  AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for batch_index, (input, landmark, target) in enumerate(iter_source_dataloader):
        data_time.update(time.time()-end)

        input, landmark, target = input.cuda(), landmark.cuda(), target.cuda()
        
        with torch.no_grad():
            end = time.time()
            feature, output, loc_output = model(input, landmark, False, 'Source')
            batch_time.update(time.time()-end)
        
        loss_ = nn.CrossEntropyLoss()(output, target)

        # Compute accuracy, precision and recall
        Compute_Accuracy(args, output, target, acc, prec, recall)

        # Log loss
        loss.update(float(loss_.cpu().data.numpy()))

        end = time.time()

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    writer.add_scalar('Test_Recall_SourceDomain', recall_avg, epoch)
    writer.add_scalar('Test_Accuracy_SourceDomain', acc_avg, epoch)

    LoggerInfo = '''
    [Test (Source Domain)]: 
    Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
    Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})\n'''.format(data_time=data_time, batch_time=batch_time)

    LoggerInfo+=AccuracyInfo

    LoggerInfo+='''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}
    Loss {loss:.4f}'''.format(acc_avg, prec_avg, recall_avg, f1_avg, loss=loss.avg)

    print(LoggerInfo)

    # Test on Target Domain
    acc, prec, recall = [AverageMeter() for i in range(args.class_num)], [AverageMeter() for i in range(args.class_num)], [AverageMeter() for i in range(args.class_num)]
    loss, data_time, batch_time =  AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for batch_index, (input, landmark, target) in enumerate(iter_target_dataloader):
        data_time.update(time.time()-end)

        input, landmark, target = input.cuda(), landmark.cuda(), target.cuda()
        
        with torch.no_grad():
            end = time.time()
            feature, output, loc_output = model(input, landmark, False, 'Target')
            batch_time.update(time.time()-end)
        
        loss_ = nn.CrossEntropyLoss()(output, target)

        # Compute accuracy, precision and recall
        Compute_Accuracy(args, output, target, acc, prec, recall)

        # Log loss
        loss.update(float(loss_.cpu().data.numpy()))

        end = time.time()

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    writer.add_scalar('Test_Recall_TargetDomain', recall_avg, epoch)
    writer.add_scalar('Test_Accuracy_TargetDomain', acc_avg, epoch)

    LoggerInfo = '''
    [Test (Target Domain)]: 
    Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
    Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})\n'''.format(data_time=data_time, batch_time=batch_time)

    LoggerInfo+=AccuracyInfo

    LoggerInfo+='''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}
    Loss {loss:.4f}'''.format(acc_avg, prec_avg, recall_avg, f1_avg, loss=loss.avg)
    
    print(LoggerInfo)

    # Save Checkpoints
    if recall_avg > Best_Recall:
        Best_Recall = recall_avg
        print('[Save] Best Recall: %.4f.' % Best_Recall)

        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), os.path.join(args.out, '{}_Recall.pkl'.format(args.log)))
        else:
            torch.save(model.state_dict(), os.path.join(args.out, '{}_Recall.pkl'.format(args.log)))
    
    if acc_avg > Best_Accuracy:
        Best_Accuracy = acc_avg
        print('[Save] Best Accuracy: %.4f.' % Best_Accuracy)
        
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), os.path.join(args.out, '{}_Accuracy.pkl'.format(args.log)))
        else:
            torch.save(model.state_dict(), os.path.join(args.out, '{}_Accuracy.pkl'.format(args.log)))

    return Best_Accuracy, Best_Recall

def main():
    """Main."""

    # Parse Argument
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # Experiment Information
    print('Log Name: %s' % args.log)
    print('Output Path: %s' % args.out)
    print('Backbone: %s' % args.net)
    print('Resume Model: %s' % args.pretrained)
    print('CUDA_VISIBLE_DEVICES: %s' % args.dev)

    print('================================================')

    print('Use {} * {} Image'.format(args.face_scale, args.face_scale))
    print('SourceDataset: %s' % args.source)
    print('TargetDataset: %s' % args.target)
    print('Train Batch Size: %d' % args.train_batch)
    print('Test Batch Size: %d' % args.test_batch)

    print('================================================')
    
    if args.show_feat:
        print('Show Visualiza Result of Feature.')

    if args.isTest:
        print('Test Model.')
    else:
        print('Train Epoch: %d' % args.epochs)
        print('Learning Rate: %f' % args.lr)
        print('Momentum: %f' % args.momentum)
        print('Weight Decay: %f' % args.weight_decay)

        if args.use_afn:
            print('Use AFN Loss: %s' % args.afn_method)
            if args.afn_method=='HAFN':
                print('Radius of HAFN Loss: %f' % args.r)
            else:
                print('Delta Radius of SAFN Loss: %f' % args.dr)
            print('Weight L2 nrom of AFN Loss: %f' % args.w_l2)

        if args.use_dan:
            print('Use DAN Loss: %s' % args.dan_method)
            print('Learning Rate(Adversarial Network): %f' % args.lr_ad)

    print('================================================')

    print('Number of classes : %d' % args.class_num)
    if not args.local_feat:
        print('Only use global feature.')
    else:
        print('Use global feature and local feature.')

        if args.intra_gcn:
            print('Use Intra GCN.')
        if args.inter_gcn:
            print('Use Inter GCN.')

        if args.rand_mat and args.useAllOneMatrix:
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

    # Bulid Adversarial Network
    print('Building Adversarial Network...')
    random_layer, ad_net = BulidAdversarialNetwork(args, model.output_num(), args.class_num) if args.use_dan else (None, None)
    print('Done!')

    print('================================================')

    # Set Optimizer
    print('Building Optimizer...')
    param_optim = Set_Param_Optim(args, model)
    optimizer = Set_Optimizer(args, param_optim, args.lr, args.weight_decay, args.momentum)

    param_optim_ad = Set_Param_Optim(args, ad_net) if args.use_dan else None
    optimizer_ad = Set_Optimizer(args, param_optim_ad, args.lr, args.weight_decay, args.momentum) if args.use_dan else None
    print('Done!')

    print('================================================')

    # Init Mean
    if args.local_feat and args.intra_gcn and args.inter_gcn and not args.isTest:        
        if args.use_cov:
            print('Init Mean and Cov...')
            Initialize_Mean_Cov(args, model, False)
        else:
            if args.use_cluster:
                print('Initialize Mean in Cluster....')
                Initialize_Mean_Cluster(args, model, False)
            else:         
                print('Init Mean...')
                Initialize_Mean(args, model, False)

        torch.cuda.empty_cache()

        print('Done!')
        print('================================================')

    # Save Best Checkpoint
    Best_Accuracy, Best_Recall = 0, 0

    # Running Experiment
    print("Run Experiment...")
    writer = SummaryWriter(os.path.join(args.out, args.log))

    for epoch in range(1, args.epochs + 1):

        if args.show_feat and epoch%5 == 1:
            Visualization('{}_Source.pdf'.format(epoch), model, train_source_dataloader, useClassify=False, domain='Source')
            Visualization('{}_Target.pdf'.format(epoch), model, train_target_dataloader, useClassify=False, domain='Target')

            VisualizationForTwoDomain('{}_train'.format(epoch), model, train_source_dataloader, train_target_dataloader, useClassify=False, showClusterCenter=False)
            VisualizationForTwoDomain('{}_test'.format(epoch), model, test_source_dataloader, test_target_dataloader, useClassify=False, showClusterCenter=False)        

        if not args.isTest:
            if args.use_cluster and epoch%10 == 0:
                Initialize_Mean_Cluster(args, model, False)
                torch.cuda.empty_cache()
            Train(args, model, ad_net, random_layer, train_source_dataloader, train_target_dataloader, optimizer, optimizer_ad, epoch, writer)
            
        Best_Accuracy, Best_Recall = Test(args, model, test_source_dataloader, test_target_dataloader, Best_Accuracy, Best_Recall, epoch, writer)

    writer.close()

if __name__ == '__main__':
    main()
