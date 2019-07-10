import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm

import nn_classes
import data_loader
import ps_functions
import SGD_custom2
from torch.optim.optimizer import Optimizer


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


def gradient_average(model, agg_model, scale):

    for param, param_ps in zip(model.parameters(), agg_model.parameters()):
        param.grad.data = param_ps.grad.data
        param.data.add_(-scale, param.grad.data)

    return None

for a in range(2):

    # select gpu
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # load data
    print('loadinf data...')
    mini_batch = 64
    trainloader, testloader = data_loader.CIFAR_data(mini_batch)

    for d in trainloader:
        # used in initilizing the gradiients
        x_init, y_init = d
        x_init, y_init = x_init.to(device), y_init.to(device)
        break

    num_workers = 35
    nets = [nn_classes.ResNet18() for n in range(num_workers)]
    [nets[n].to(device) for n in range(num_workers)]
    [ps_functions.grad_init(nets[n], x_init, y_init) for n in range(num_workers)]
    # model at PS for all-reduce purposes
    ps_model = nn_classes.ResNet18()
    ps_model.to(device)

    ps_functions.grad_init(ps_model, x_init, y_init)

    lr = 0.25
    scale = 1
    momentum = 0.9
    is_avg = True


    model_parameters = filter(lambda p: p.requires_grad, ps_model.parameters())
    params_number = sum([np.prod(p.size()) for p in model_parameters])

    p_sparce = .99  # sparsification param (uplink)

    top_k_params = int(np.ceil((1 - p_sparce) * params_number))

    # spars in downlink
    p_sparce_ps = 0
    top_k_ps = int(np.ceil((1 - p_sparce_ps) * params_number))

    weight_decay = 0.0001

    criterions = [nn.CrossEntropyLoss() for n in range(num_workers)]

    # optimizers = [define_optimizer(nets[n], lr, momentum, w_decay=weight_decay) for n in range(num_workers)]
    optimizers = [SGD_custom2.define_optimizer(nets[n], lr, momentum, w_decay=weight_decay) for n in range(num_workers)]
    #######################
    epochs = 300
    warm_up_epoch=5
    max_ind= 50000*warm_up_epoch/(mini_batch*num_workers)
    iter_ind = 1
    #######################
    [ps_functions.synch_weight(nets[n], ps_model) for n in range(num_workers)]
    #######################################
    ps_functions.warmup_lr_nc(optimizers,num_workers,lr,iter_ind,max_ind) #initialize lr for warmup phase
    # Result vector
    results=np.empty([1,150])
    res_ind=0
    #training
    for e in tqdm(range(epochs)):
        i = 0

        for data in trainloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # index of the worker
            index = i % num_workers
            optimizers[index].zero_grad()  # i verified that this also make model gradients zero
            # predictions
            preds = nets[index](inputs)
            loss = criterions[index](preds, labels)
            loss.backward()

            optimizers[index].step(device, top_k_params)
            # for param in nets[index].parameters():
            #     print(param.grad.data)
            i += 1
            if i == num_workers:
                i = 0
                iter_ind +=1
                ps_model.zero_grad()

                for n in range(num_workers):

                    gradient_accumulate(nets[n], ps_model, is_avg, num_workers)

                for n in range(num_workers):

                    gradient_average(nets[n], ps_model, lr)
            ##### cahange lr for next period #####################
            ps_functions.warmup_lr_nc(optimizers,num_workers,lr,iter_ind,max_ind)

        if (e % 2) == 0:
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = nets[0](images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the 10000 test images: %d %%' % (
                    100 * correct / total))
            results[0][res_ind]= 100 * correct / total
            res_ind+=1 
        ps_functions.lr_change_nc(optimizers, e, num_workers)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = nets[0](images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
    f=open('results_resnet_nocluster.txt','ab')
    np.savetxt(f,(results), fmt='%.5f', encoding='latin1')
    f.close()
