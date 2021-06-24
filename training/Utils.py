import os
import sys
import time
import tqdm
import shutil
import argparse
import subprocess
import numpy as np
import pandas as pd

import sklearn
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from PIL import Image, ImageDraw

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from Dataset import MyDataset
from VGG import VGG, VGG_onlyGlobal
from ResNet import IR, IR_onlyGlobal
from MobileNet import MobileNetV2, MobileNetV2_onlyGlobal
from AdversarialNetwork import RandomLayer, AdversarialNetwork, calc_coeff

class AverageMeter(object):
    '''Computes and stores the sum, count and average'''
    def __init__(self):
        self.reset()

    def reset(self):    
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        self.val = val
        self.sum += val 
        self.count += count

        if self.count==0:
            self.avg = 0
        else:
            self.avg = float(self.sum) / self.count

def str2bool(input):
    if isinstance(input, bool):
       return input
    if input.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def Set_Param_Optim(args, model):
    """Set Parameters for optimization."""
    
    if isinstance(model, nn.DataParallel):
        return model.module.get_parameters()

    return model.get_parameters()

def Set_Optimizer(args, parameter_list, lr=0.001, weight_decay=0.0005, momentum=0.9):
    """Set Optimizer."""
    
    return optim.SGD(parameter_list, lr=lr, weight_decay=weight_decay, momentum=momentum)

def lr_scheduler(optimizer, iter_num, gamma, power, lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    lr = lr * (1 + gamma * iter_num) ** (-power)

    for param_group in optimizer.param_groups:
        
        if 'lr_mult' in param_group:
            param_group['lr'] = lr * param_group['lr_mult']
        else:
            param_group['lr'] = lr

        if 'decay_mult' in param_group:    
            param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        else:
            param_group['weight_decay'] = weight_decay

    return optimizer, lr

def lr_scheduler_withoutDecay(optimizer, lr=0.001, weight_decay=0.0005):
    """Learning rate without Decay."""

    for param_group in optimizer.param_groups:
        
        if 'lr_mult' in param_group:
            param_group['lr'] = lr * param_group['lr_mult']
        else:
            param_group['lr'] = lr

        if 'decay_mult' in param_group:    
            param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        else:
            param_group['weight_decay'] = weight_decay

    return optimizer, lr

def Compute_Accuracy(args, pred, target, acc, prec, recall):
    '''Compute the accuracy of all samples, the accuracy of positive samples, the recall of positive samples.'''

    pred = pred.cpu().data.numpy()
    pred = np.argmax(pred,axis=1)
    target = target.cpu().data.numpy()

    pred = pred.astype(np.int32).reshape(pred.shape[0],)
    target = target.astype(np.int32).reshape(target.shape[0],)

    for i in range(args.class_num):
        TP = np.sum((pred==i)*(target==i))
        TN = np.sum((pred!=i)*(target!=i))
        
        # Compute Accuracy of All --> TP+TN / All
        acc[i].update(np.sum(pred==target),pred.shape[0])
        
        # Compute Precision of Positive --> TP/(TP+FP)
        prec[i].update(TP,np.sum(pred==i))

        # Compute Recall of Positive --> TP/(TP+FN)
        recall[i].update(TP,np.sum(target==i))

def BulidModel(args):
    """Bulid Model."""
    if args.local_feat:
        if args.net == 'ResNet18':
            model = IR(18, args.intra_gcn, args.inter_gcn, args.rand_mat, args.all1_mat, args.use_cov, args.use_cluster,args.class_num)
        elif args.net == 'ResNet50':
            model = IR(50, args.intra_gcn, args.inter_gcn, args.rand_mat, args.all1_mat, args.use_cov, args.use_cluster,args.class_num)
            print("USING LOCAL FEATURES")
        elif args.net == 'VGGNet':
            model = VGG(args.intra_gcn, args.inter_gcn, args.rand_mat, args.all1_mat, args.use_cov, args.use_cluster, args.class_num)
        elif args.net == 'MobileNet':
            model = MobileNetV2(args.intra_gcn, args.inter_gcn, args.rand_mat, args.all1_mat, args.use_cov, args.use_cluster, args.class_num)
    else:
        if args.net == 'ResNet18':
            model = IR_onlyGlobal(18)
        elif args.net == 'ResNet50':
            model = IR_onlyGlobal(50)
        elif args.net == 'VGGNet':
            model = VGG_onlyGlobal()
        elif args.net == 'MobileNet':
            model = MobileNetV2_onlyGlobal()

    if args.pretrained != 'None':
        print('Resume Model: {}'.format(args.pretrained))
        checkpoint = torch.load(args.pretrained, map_location='cpu')

        model.load_state_dict(checkpoint, strict=True)
    else:
        print('No Resume Model')
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    #model = model.cuda()
    model.cuda()
    return model

def BulidAdversarialNetwork(args, model_output_num, class_num=7):
    """Bulid Adversarial Network."""

    if args.rand_layer:
        random_layer = RandomLayer([model_output_num, class_num], 1024)
        ad_net = AdversarialNetwork(1024, 512)
        random_layer.cuda()
        
    else:
        random_layer = None
        if args.dan_method=='DANN' or args.dan_method=='MME':
            ad_net = AdversarialNetwork(model_output_num, 128)
        else:
            ad_net = AdversarialNetwork(model_output_num * class_num, 512)

    ad_net.cuda()

    return random_layer, ad_net

def BulidDataloader(args, flag1='train', flag2='source'):
    """Bulid data loader."""

    assert flag1 in ['train', 'test'], 'Function BuildDataloader : function parameter flag1 wrong.'
    assert flag2 in ['source', 'target'], 'Function BuildDataloader : function parameter flag2 wrong.'

    # Set Transform
    trans = transforms.Compose([ 
            transforms.Resize((args.face_scale, args.face_scale)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ])
    target_trans = None

    # Basic Notes:
    # 0: Surprised
    # 1: Fear
    # 2: Disgust
    # 3: Happy
    # 4: Sad
    # 5: Angry
    # 6: Neutral

    dataPath_prefix = '../Dataset'

    data_imgs, data_labels, data_bboxs, data_landmarks = [], [], [], []
    if flag1 == 'train':
        if flag2 == 'source':
            list_patition_label = pd.read_csv(dataPath_prefix+'/%s/lists/image_list.txt'%(args.source), header=None, delim_whitespace=True)
            list_patition_label = np.array(list_patition_label)
            for index in range(list_patition_label.shape[0]):
                if list_patition_label[index,0][:5] == "train":
                    if not os.path.exists(dataPath_prefix+'/%s/boundingbox/'%(args.source)+list_patition_label[index,0][:-4] + '_boundingbox' + '.txt'):
                        continue
                    if not os.path.exists(dataPath_prefix+'/%s/landmarks_5/'%(args.source)+list_patition_label[index,0][:-4]+'.txt'):
                        continue
                    bbox = np.loadtxt(dataPath_prefix+'/%s/boundingbox/'%(args.source)+list_patition_label[index,0][:-4]+'_boundingbox.txt').astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/%s/landmarks_5/'%(args.source)+list_patition_label[index,0][:-3]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/%s/images/'%(args.source)+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index,1])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)
            
        if flag2 == 'target':
            list_patition_label = pd.read_csv(dataPath_prefix+'/%s/lists/image_list.txt'%(args.target), header=None, delim_whitespace=True)
            list_patition_label = np.array(list_patition_label)
            for index in range(list_patition_label.shape[0]):
                
                if list_patition_label[index,0][:5] == "train":
                    #if not os.path.exists(dataPath_prefix+'/%s/boundingbox/'%(args.target)+list_patition_label[index,0][:-3]+'txt'):
                        #continue
                    
                    if not os.path.exists(dataPath_prefix+'/%s/landmarks_5/'%(args.target)+list_patition_label[index,0][:-3]+'txt'):
                        #print(list_patition_label[index,0][:-3]+'txt')
                        continue
                    img = Image.open(dataPath_prefix + '/%s/images/'%(args.target)+list_patition_label[index,0]).convert('RGB')
                    ori_img_w, ori_img_h = img.size
                    #bbox = np.loadtxt(dataPath_prefix+'/%s/boundingbox/'%(args.target)+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/%s/landmarks_5/'%(args.target)+list_patition_label[index,0][:-3]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/%s/images/'%(args.target)+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index,1])
                    data_bboxs.append((0,0,ori_img_w,ori_img_h))
                    data_landmarks.append(landmark)
                        
    elif flag1 == 'test':
        if flag2 =='source':
            list_patition_label = pd.read_csv(dataPath_prefix+'/%s/lists/image_list.txt'%(args.source), header=None, delim_whitespace=True)
            list_patition_label = np.array(list_patition_label)
            for index in range(list_patition_label.shape[0]):
                if list_patition_label[index,0][:4] == "test":
                    if not os.path.exists(dataPath_prefix+'/%s/boundingbox/'%(args.source)+list_patition_label[index,0][:-4]+'_boundingbox.txt'):
                        print(list_patition_label[index,0][:-4]+'_boundingbox.txt')
                        continue
                    if not os.path.exists(dataPath_prefix+'/%s/landmarks_5/'%(args.source)+list_patition_label[index,0][:-3]+'txt'):
                        continue

                    bbox = np.loadtxt(dataPath_prefix+'/%s/boundingbox/'%(args.source) + list_patition_label[index,0][:-4]+'_boundingbox.txt').astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/%s/landmarks_5/'%(args.source)+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    data_imgs.append(dataPath_prefix+'/%s/images/'%(args.source)+ list_patition_label[index,0])
                    data_labels.append(list_patition_label[index,1])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)

        elif flag2=='target':
            list_patition_label = pd.read_csv(dataPath_prefix+'/%s/lists/image_list.txt'%(args.target), header=None, delim_whitespace=True)
            list_patition_label = np.array(list_patition_label)
            for index in range(list_patition_label.shape[0]):
                if list_patition_label[index,0][:4] == "test":
                    #if not os.path.exists(dataPath_prefix+'/%s/boundingbox/'%(args.target)+list_patition_label[index,0][:-3]+'txt'):
                        #continue
                    if not os.path.exists(dataPath_prefix+'/%s/landmarks_5/'%(args.target)+list_patition_label[index,0][:-3]+'txt'):
                        continue
                    img = Image.open(dataPath_prefix + '/%s/images/'%(args.target)+list_patition_label[index,0]).convert('RGB')
                    ori_img_w, ori_img_h = img.size
                    #bbox = np.loadtxt(dataPath_prefix+'/%s/boundingbox/'%(args.target)+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/%s/landmarks_5/'%(args.target) + list_patition_label[index,0][:-3]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/%s/images/'%(args.target)+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index,1])
                    data_bboxs.append((0,0,ori_img_w,ori_img_h))
                    data_landmarks.append(landmark)
        
    # DataSet Distribute
    distribute_ = np.array(data_labels)
    print(' %s %s dataset qty: %d' % ( flag1, flag2, len(data_imgs)))
    dataset_dist = []
    for i in range(args.class_num):
        dataset_dist.append(np.sum(distribute_==i))

    print("Dataset Distribution for %s classes is: "%(args.class_num), dataset_dist)

    # DataSet
    data_set = MyDataset(data_imgs, data_labels, data_bboxs, data_landmarks, flag1, trans, target_trans)

    # DataLoader
    if flag1=='train':
        data_loader = data.DataLoader(dataset=data_set, batch_size=args.train_batch, shuffle=True, num_workers=8, drop_last=True)
    elif flag1=='test':
        data_loader = data.DataLoader(dataset=data_set, batch_size=args.test_batch, shuffle=False, num_workers=8, drop_last=False)

    return data_loader

def Show_Accuracy(acc, prec, recall, class_num=7):
    """Compute average of accuaracy/precision/recall/f1"""

    # Compute F1 value    
    f1 = [AverageMeter() for i in range(class_num)]
    for i in range(class_num):
        if prec[i].avg==0 or recall[i].avg==0:
            f1[i].avg = 0
            continue
        f1[i].avg = 2*prec[i].avg*recall[i].avg/(prec[i].avg+recall[i].avg)
    
    # Compute average of accuaracy/precision/recall/f1
    acc_avg, prec_avg, recall_avg, f1_avg = 0, 0, 0, 0

    for i in range(class_num):
        acc_avg+=acc[i].avg
        prec_avg+=prec[i].avg
        recall_avg+=recall[i].avg
        f1_avg+=f1[i].avg

    acc_avg, prec_avg, recall_avg, f1_avg = acc_avg/class_num,prec_avg/class_num, recall_avg/class_num, f1_avg/class_num

    # Log Accuracy Infomation
    Accuracy_Info = ''
    
    Accuracy_Info+='Accuracy'
    for i in range(class_num):
        Accuracy_Info+=' {:.4f}'.format(acc[i].avg)
    Accuracy_Info+='\n'

    Accuracy_Info+='    Precision'
    for i in range(class_num):
        Accuracy_Info+=' {:.4f}'.format(prec[i].avg)
    Accuracy_Info+='\n'

    Accuracy_Info+='    Recall'
    for i in range(class_num):
        Accuracy_Info+=' {:.4f}'.format(recall[i].avg)
    Accuracy_Info+='\n'

    Accuracy_Info+='    F1'
    for i in range(class_num):
        Accuracy_Info+=' {:.4f}'.format(f1[i].avg)
    Accuracy_Info+='\n'

    return Accuracy_Info, acc_avg, prec_avg, recall_avg, f1_avg

def Initialize_Mean(args, model, useClassify=True):
    
    model.eval()
    
    source_data_loader = BulidDataloader(args, flag1='train', flag2='source')
    target_data_loader = BulidDataloader(args, flag1='train', flag2='target')
    
    # Source Mean
    mean = None

    for step, (input, landmark, label) in enumerate(source_data_loader):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, pred, loc_pred = model(input, landmark, useClassify, 'Source')

        if step==0:
            mean = torch.mean(feature,0)
        else:
            mean = step/(step+1) * torch.mean(feature,0) + 1/(step+1) * mean

    if isinstance(model, nn.DataParallel):
        model.module.SourceMean.init(mean)
    else:
        model.SourceMean.init(mean)

    # Target Mean
    mean = None

    for step, (input, landmark, label) in enumerate(target_data_loader):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, pred, loc_pred = model(input, landmark, useClassify, 'Target')

        if step==0:
            mean = torch.mean(feature,0)
        else:
            mean = step/(step+1) * torch.mean(feature,0) + 1/(step+1) * mean

    if isinstance(model, nn.DataParallel):
        model.module.TargetMean.init(mean)
    else:
        model.TargetMean.init(mean)

def Initialize_Mean_Cov(args, model, useClassify=True):
    
    model.eval()

    source_data_loader = BulidDataloader(args, flag1='train', flag2='source')
    target_data_loader = BulidDataloader(args, flag1='train', flag2='target')

    # Source Mean and Cov
    mean, cov = None, None

    for step, (input, landmark, label) in enumerate(source_data_loader):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, pred, loc_pred = model(input, landmark, useClassify, 'Source')

        if step==0:
            mean = torch.mean(feature,0)
            cov  = torch.mm((feature-mean).transpose(0,1), feature-mean) / (feature.size(0)-1)
        else:
            mean = step/(step+1) * torch.mean(feature,0) + 1/(step+1) * mean
            cov  = step/(step+1) * torch.mm((feature-mean).transpose(0,1), feature-mean) / (feature.size(0)-1) + 1/(step+1) * cov

    if isinstance(model, nn.DataParallel):
        model.module.SourceMean.init(mean, cov)
    else:
        model.SourceMean.init(mean, cov)

    # Target Mean and Cov
    mean, cov = None, None

    for step, (input, landmark, label) in enumerate(target_data_loader):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, pred, loc_pred = model(input, landmark, useClassify, 'Target')

        if step==0:
            mean = torch.mean(feature,0)
            cov  = torch.mm((feature-mean).transpose(0,1), feature-mean) / (feature.size(0)-1)
        else:
            mean = step/(step+1) * torch.mean(feature,0) + 1/(step+1) * mean
            cov  = step/(step+1) * torch.mm((feature-mean).transpose(0,1), feature-mean) / (feature.size(0)-1) + 1/(step+1) * cov

    if isinstance(model, nn.DataParallel):
        model.module.TargetMean.init(mean, cov)
    else:
        model.TargetMean.init(mean, cov)

def Initialize_Mean_Cluster(args, model, useClassify=True):

    model.eval()
    
    # Source Cluster of Mean
    Feature = []
    EndTime = time.time()
    source_data_loader = BulidDataloader(args, flag1='train', flag2='source')

    for step, (input, landmark, label) in enumerate(source_data_loader):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, pred, loc_pred = model(input, landmark, useClassify, 'Source')
        Feature.append(feature.cpu().data.numpy())
    Feature = np.vstack(Feature)

    # Using K-Means
    kmeans = KMeans(n_clusters=args.class_num, init='k-means++', algorithm='full')
    kmeans.fit(Feature)
    centers = torch.Tensor(kmeans.cluster_centers_).to('cuda' if torch.cuda.is_available else 'cpu')

    if isinstance(model, nn.DataParallel):
        model.module.SourceMean.init(centers)
    else:
        model.SourceMean.init(centers)

    print('[Source Domain] Cost time : %fs' % (time.time()-EndTime))

    # Target Cluster of Mean
    Feature = []
    EndTime = time.time()
    target_data_loader = BulidDataloader(args, flag1='train', flag2='target')

    for step, (input, landmark, label) in enumerate(target_data_loader):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, pred, loc_pred = model(input, landmark, useClassify, 'Target')
        Feature.append(feature.cpu().data.numpy())
    Feature = np.vstack(Feature)

    # Using K-Means
    kmeans = KMeans(n_clusters=args.class_num, init='k-means++', algorithm='full')
    kmeans.fit(Feature)
    centers = torch.Tensor(kmeans.cluster_centers_).to('cuda' if torch.cuda.is_available else 'cpu')

    if isinstance(model, nn.DataParallel):
        model.module.TargetMean.init(centers)
    else:
        model.TargetMean.init(centers)

    print('[Target Domain] Cost time : %fs' % (time.time()-EndTime))

def Visualization(figName, model, dataloader, useClassify=True, domain='Source'):
    '''Feature Visualization in Source/Target Domain.'''
    
    assert useClassify in [True, False], 'useClassify should be bool.'
    assert domain in ['Source', 'Target'], 'domain should be source or target.'

    model.eval()

    Feature, Label = [], []

    # Get Cluster
    for i in range(7):
        if domain=='Source':
            Feature.append(model.SourceMean.running_mean[i].cpu().data.numpy())
        elif domain=='Target':
            Feature.append(model.TargetMean.running_mean[i].cpu().data.numpy())
    Label.append(np.array([7 for i in range(7)]))

    # Get Feature and Label
    for step, (input, landmark, label) in enumerate(dataloader):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, output, loc_output = model(input, landmark, useClassify, domain)
        Feature.append(feature.cpu().data.numpy())
        Label.append(label.cpu().data.numpy())

    Feature = np.vstack(Feature)
    Label = np.concatenate(Label)

    # Using T-SNE
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=50, early_exaggeration=3)
    embedding = tsne.fit_transform(Feature)

    # Draw Visualization of Feature
    colors = {0:'red', 1:'blue', 2:'olive',  3:'green',  4:'orange',  5:'purple',  6:'darkslategray', 7:'black'}
    # labels = {0:'Surprised', 1:'Fear', 2:'Disgust',  3:'Happy',  4:'Sad',  5:'Angry',  6:'Neutral', 7:'Cluster'}
    labels = {0:'惊讶', 1:'恐惧', 2:'厌恶', 3:'开心', 4:'悲伤', 5:'愤怒', 6:'平静', 7:'聚类中心'}

    data_min, data_max = np.min(embedding, 0), np.max(embedding, 0)
    data_norm = (embedding - data_min) / (data_max - data_min)

    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(7):
        data_x, data_y = data_norm[Label==i][:,0], data_norm[Label==i][:,1]
        scatter = plt.scatter(data_x, data_y, c='', edgecolors=colors[i], s=5, label=labels[i], marker='^', alpha=0.6)
    scatter = plt.scatter(data_norm[Label==7][:,0], data_norm[Label==7][:,1], c=colors[7], s=20, label=labels[7], marker='^', alpha=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])
    
    plt.legend(handles=[mpatches.Patch(color=colors[i], label="{:s}".format(labels[i]) ) for i in range(8)],
               loc='upper left',
               #prop = {'size':8},
               prop = matplotlib.font_manager.FontProperties(fname='./simhei.ttf'), 
               bbox_to_anchor=(1.05,0.85),
               borderaxespad=0)
    plt.savefig(fname='{}'.format(figName), format="pdf", bbox_inches = 'tight')

def VisualizationForTwoDomain(figName, model, source_dataloader, target_dataloader, useClassify=True, showClusterCenter=True):
    '''Feature Visualization in Source and Target Domain.'''
    
    model.eval()

    Feature_Source, Label_Source, Feature_Target, Label_Target = [], [], [], []

    # Get Feature and Label in Source Domain
    if showClusterCenter:
        for i in range(7):
            Feature_Source.append(model.SourceMean.running_mean[i].cpu().data.numpy())
        Label_Source.append(np.array([7 for i in range(7)]))   

    for step, (input, landmark, label) in enumerate(source_dataloader):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, output, loc_output = model(input, landmark, useClassify, domain='Source')

        Feature_Source.append(feature.cpu().data.numpy())
        Label_Source.append(label.cpu().data.numpy())

    Feature_Source = np.vstack(Feature_Source)
    Label_Source = np.concatenate(Label_Source)

    # Get Feature and Label in Target Domain
    if showClusterCenter:
        for i in range(7):
            Feature_Target.append(model.TargetMean.running_mean[i].cpu().data.numpy())
        Label_Target.append(np.array([7 for i in range(7)]))

    for step, (input, landmark, label) in enumerate(target_dataloader):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, output, loc_output = model(input, landmark, useClassify, domain='Target')

        Feature_Target.append(feature.cpu().data.numpy())
        Label_Target.append(label.cpu().data.numpy())

    Feature_Target = np.vstack(Feature_Target)
    Label_Target = np.concatenate(Label_Target)

    # Sampling from Source Domain
    Feature_Temple, Label_Temple = [], []
    for i in range(8):
        num_source = np.sum(Label_Source==i)
        num_target = np.sum(Label_Target==i)

        num = num_source if num_source <= num_target else num_target 

        Feature_Temple.append(Feature_Source[Label_Source==i][:num])
        Label_Temple.append(Label_Source[Label_Source==i][:num]) 
 
    Feature_Source = np.vstack(Feature_Temple) 
    Label_Source = np.concatenate(Label_Temple)

    Label_Target+=8

    Feature = np.vstack((Feature_Source, Feature_Target))
    Label = np.concatenate((Label_Source, Label_Target))

    # Using T-SNE
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=50, early_exaggeration=3)
    embedding = tsne.fit_transform(Feature)

    # Draw Visualization of Feature
    colors = {0:'firebrick', 1:'aquamarine', 2:'goldenrod',  3:'cadetblue',  4:'saddlebrown',  5:'yellowgreen',  6:'navy'}
    labels = {0:'Surprised', 1:'Fear', 2:'Disgust',  3:'Happy',  4:'Sad',  5:'Angry',  6:'Neutral'}

    data_min, data_max = np.min(embedding, 0), np.max(embedding, 0)
    data_norm = (embedding - data_min) / (data_max - data_min)

    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(7):

        data_source_x, data_source_y = data_norm[Label==i][:,0], data_norm[Label==i][:,1]
        source_scatter = plt.scatter(data_source_x, data_source_y, color="none", edgecolor=colors[i], s=20, label=labels[i], marker="o", alpha=0.4, linewidth=0.5)
        
        data_target_x, data_target_y = data_norm[Label==(i+8)][:,0], data_norm[Label==(i+8)][:,1]
        target_scatter = plt.scatter(data_target_x, data_target_y, color=colors[i], edgecolor="none", s=30, label=labels[i], marker="x", alpha=0.6, linewidth=0.2)

        if i==0:
            source_legend = source_scatter
            target_legend = target_scatter

    if showClusterCenter:
        source_cluster = plt.scatter(data_norm[Label==7][:,0], data_norm[Label==7][:,1], c='black', s=20, label='Source Cluster Center', marker='^', alpha=1)
        target_cluster = plt.scatter(data_norm[Label==15][:,0], data_norm[Label==15][:,1], c='black', s=20, label='Target Cluster Center', marker='s', alpha=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])
    '''
    l1 = plt.legend(handles=[mpatches.Patch(color=colors[i], 
                    label="{:s}".format(labels[i]) ) for i in range(7)], 
                    loc='upper left', 
                    prop = {'size':8})
                    #bbox_to_anchor=(1.05,0.85), 
                    #borderaxespad=0)
    
    if showClusterCenter:
        plt.legend([source_legend, target_legend, source_cluster, target_cluster],
                   ['Source Domain', 'Target Domain', 'Source Cluster Center', 'Target Cluster Center'],
                   loc='lower left',
                   prop = {'size':7})
    else:
        plt.legend([source_legend, target_legend], ['Source Domain', 'Target Domain'], loc='lower left', prop = {'size':7})
    plt.gca().add_artist(l1)
    '''
    plt.savefig(fname='{}.pdf'.format(figName), format="pdf", bbox_inches='tight')



