import torch.nn as nn
import numpy as np
import torch
from copy import deepcopy


# def gradient_accumulate(old_model, model, agg_model, is_avg, num_workers):
#     # summing the model gradients into agg_model
#     # if args.is_ave==True, then average sum
#
#     for old_param, param, grad_param in zip(old_model.parameters(), model.parameters(), agg_model.parameters()):
#         # get model difference.
#         # param.grad.data = old_param.data - param.data
#         if is_avg:
#             # grad_param.grad.data += param.grad.data/num_workers
#             grad_param.grad.data += (old_param.data - param.data) / num_workers
#         else:
#             grad_param.grad.data += param.grad.data
#
#         # recover to old model.
#         param.data = deepcopy(old_param.data)
#
#     return None

def gradient_accumulate(model, agg_model, is_avg, num_workers):
    # summing the model gradients into agg_model
    # if args.is_ave==True, then average sum

    for param, grad_param in zip(model.parameters(), agg_model.parameters()):
        # get model difference.

        if is_avg:
            grad_param.grad.data += param.grad.data/num_workers
        else:
            grad_param.grad.data += param.grad.data

    return None


def ps_param_zero(agg_model):

    for p in agg_model.parameters():
        p.data.mul_(0)

    return None


def weight_accumulate(model, agg_model, num):

    for param, ps_param in zip(model.parameters(), agg_model.parameters()):
        ps_param.data += param.data / num
    return None


def weight_broadcast(model, agg_model):

    for param, ps_param in zip(model.parameters(), agg_model.parameters()):
        param.data = ps_param.data + 0

    return None


def model_retrieve(model, old_model):
    for param, old_param in zip(model.parameters, old_model.parameters):
        param.data = old_param.data + 0


def gradient_average(model, agg_model, scale):

    for param, param_ps in zip(model.parameters(), agg_model.parameters()):
        param.grad.data = param_ps.grad.data + 0
        param.data.add_(-scale, param.grad.data)

    old_model = deepcopy(model)
    return old_model


def sparse_grad(top_k, model, device):
    grad_flattened = torch.empty(0).to(device)

    for p in model.parameters():
        a = p.grad.data.flatten().to(device)
        grad_flattened = torch.cat((a, grad_flattened), 0)

    top_k_grads = torch.topk(grad_flattened.abs(), top_k)[0].to(device)
    grad_min_value = top_k_grads.min().to(device)

    for p in model.parameters():
        sparse_mask = p.grad.data.abs() >= grad_min_value
        p.grad.data = sparse_mask.float() * p.grad.data


def grad_init(model, x, y):
    # predictions
    y_hat = model(x)
    c = nn.CrossEntropyLoss()
    l = c(y_hat, y)
    l.backward()


def synch_weight(model, model_ps):
    for param, param_ps in zip(model.parameters(), model_ps.parameters()):
        param.data = param_ps.data + 0


# changing the learning rate of SGD

def lr_change(optim, epoch,num_cl,num_w):                    
    if epoch == 100:
        for c in range(num_cl):
            for n in range(num_w):
                for group in optim[c][n].param_groups:
                    group['lr'] = 0.025                   
    if epoch == 200:
        for c in range(num_cl):
            for n in range(num_w):
                for group in optim[c][n].param_groups:
                    group['lr'] = 0.0025
def warmup_lr(optim,num_cl,num_w,lr,period_ind,max_ind):
    if period_ind < (max_ind + 1):
        for c in range(num_cl):
            for n in range(num_w):
                for group in optim[c][n].param_groups:
                    group['lr'] = (lr * period_ind)/max_ind
                    
def lr_change_nc(optim, epoch, num_workers):                    
    if epoch == 50:
        for w in range(num_workers):
                for group in optim[w].param_groups:
                    group['lr'] = 0.025                   
    if epoch == 125:
        for w in range(num_workers):
                for group in optim[w].param_groups:
                    group['lr'] = 0.0025
def warmup_lr_nc(optim,num_workers,lr,period_ind,max_ind):
    if period_ind < (max_ind + 1):
        for w in range(num_workers):
            for group in optim[w].param_groups:
                group['lr'] = (lr * period_ind)/max_ind
