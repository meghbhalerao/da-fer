import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def Entropy(input_):
    return torch.sum(-input_ * torch.log(input_ + 1e-5), dim=1)


def DANN(features, ad_net):

    '''
    Paper Link : https://papers.nips.cc/paper/7436-conditional-adversarial-domain-adaptation.pdf
    Github Link : https://github.com/thuml/CDAN
    '''

    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
    
    if not dc_target.is_cuda:
        dc_target = dc_target.cuda()

    return nn.BCELoss()(ad_out, dc_target)
    
def MME_loss(features, coeff = 1):
    '''
    Paper Link : https://arxiv.org/pdf/1904.06487.pdf
    Github Link : https://github.com/VisionLearningGroup/SSDA_MME
    '''
    ad_out = adentropy(features)
    ad_out.register_hook(grl_hook(coeff))
    return ad_out

def grl_hook(coeff):
    def fun1(grad):
        return - coeff * grad.clone()
    return fun1

def MME(model, feat, lamda=0.1, coeff=1.0):
    feat.register_hook(grl_hook(coeff))
    feat =  model.fc(feat)
    feat = F.softmax(feat)
    loss_adent = lamda * torch.mean(torch.sum(feat * (torch.log(feat + 1e-5)), 1))
    return loss_adent

def adentropy(feat, lamda=0.1):
    feat = F.softmax(feat)
    loss_adent = - lamda * torch.mean(torch.sum(feat * (torch.log(feat + 1e-5)), 1))
    return loss_adent


def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):

    '''
    Paper Link : https://papers.nips.cc/paper/7436-conditional-adversarial-domain-adaptation.pdf
    Github Link : https://github.com/thuml/CDAN
    '''

    feature = input_list[0]
    softmax_output = input_list[1].detach()
    
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))

    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
    
    if feature.is_cuda:
        dc_target = dc_target.cuda()

    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)

        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy * source_mask

        target_mask = torch.ones_like(entropy)
        target_mask[:feature.size(0)//2] = 0
        target_weight = entropy * target_mask

        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()

        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target) 

def HAFN(features, weight_L2norm, radius):
    
    '''
    Paper Link : https://arxiv.org/pdf/1811.07456.pdf
    Github Link : https://github.com/jihanyang/AFN
    '''

    return weight_L2norm * (features.norm(p=2, dim=1).mean() - radius) ** 2

def SAFN(features, weight_L2norm, deltaRadius):

    '''
    Paper Link : https://arxiv.org/pdf/1811.07456.pdf
    Github Link : https://github.com/jihanyang/AFN
    '''

    radius = features.norm(p=2, dim=1).detach()

    assert radius.requires_grad == False, 'radius\'s requires_grad should be False'
    
    return weight_L2norm * ((features.norm(p=2, dim=1) - (radius+deltaRadius)) ** 2).mean()

def do_fixmatch(f,data_target,label_target,landmark_target,model,thresh,criterion_pseudo,use_uncertainty = False):
    im_data_tu_weak_aug, im_data_tu_strong_aug = data_target[2].cuda(), data_target[1].cuda()
    # Getting predictions of weak and strong augmented unlabled examples
    feature_strong, output_strong, loc_output_strong = model(im_data_tu_strong_aug,landmark_target,False,'Target')
    with torch.no_grad():
        feature_weak, output_weak, loc_output_weak = model(im_data_tu_weak_aug,landmark_target,False,'Target')
    prob_weak_aug = F.softmax(output_weak,dim=1)
    mask_loss = prob_weak_aug.max(1)[0]>thresh
    pseudo_labels = output_weak.max(axis=1)[1]
    #try:
    if not use_uncertainty:
        loss_pseudo_unl = torch.mean(mask_loss.int() * criterion_pseudo(output_strong,pseudo_labels))
        mask_loss = mask_loss.int()
    elif use_uncertainty:
        loss_pseudo_unl = torch.mean(prob_weak_aug * mask_loss.int() * criterion_pseudo(output_strong,pseudo_labels))
    #print("Mask Loss is: ", mask_loss)
    #print("Pseudo labels are: ", pseudo_labels)
    #print("Actual target example Labels are: ", label_target)

    #print((pseudo_labels==label_target).int())

    a = (mask_loss * (pseudo_labels==label_target).int()).sum().data.item()
    #b = torch.ones_like(mask_loss * pseudo_labels).int().sum(≈).data.item()
    #print(torch.ones_like(mask_loss * pseudo_labels))
    b = mask_loss.int().sum().data.item()
    #loss_pseudo_unl.backward(retain_graph=True)
    #except:
    #pass
    if f:
        pass
        #sys.exit("This is a test run, exiting!")
    return a, b, pseudo_labels, mask_loss, loss_pseudo_unl

