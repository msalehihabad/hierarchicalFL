import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm

import nn_classes
import SGD_custom2
import data_loader
import ps_functions



def gradient_average(model, agg_model, scale): #This is where we update model in each cluster based on averaged momentum that is 

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

    # number of clusters
    num_cl = 7
    # number of workers per cluster
    num_w_per_cluster = 5
    nets = [[nn_classes.ResNet18().to(device) for n in range(num_w_per_cluster)] for c in range(num_cl)]
    for c in range(num_cl):
        for n in range(num_w_per_cluster):
            ps_functions.grad_init(nets[c][n], x_init, y_init)

    # model at PS for all-reduce purposes
    ps_model = nn_classes.ResNet18().to(device)


    ps_functions.grad_init(ps_model, x_init, y_init)

    lr = 0.25
    scale = 1
    momentum = 0.9
    is_avg = True


    model_parameters = filter(lambda p: p.requires_grad, ps_model.parameters())
    params_number = sum([np.prod(p.size()) for p in model_parameters])

    p_sparce = 0.99  # sparsification param

    top_k_params = int(np.ceil((1 - p_sparce) * params_number))

    p_sparce_ps = 0
    top_k_ps = int(np.ceil((1 - p_sparce_ps) * params_number))

    weight_decay = 0.0001
    #weight_decay = 0

    criterions = [[nn.CrossEntropyLoss() for n in range(num_w_per_cluster)] for c in range(num_cl)]
    optimizers = [[SGD_custom2.define_optimizer(nets[c][n], lr, momentum, w_decay=weight_decay)
                   for n in range(num_w_per_cluster)] for c in range(num_cl)]
    #############
    epochs = 300
    period = 6
    iter_ind = 1
    #############
    warm_up_epoch=5
    max_ind= 50000*warm_up_epoch/(mini_batch*num_w_per_cluster*num_cl)
    ##############
    old_nets = deepcopy(nets)
    for c in range(num_cl):
        for n in range(num_w_per_cluster):
            ps_functions.grad_init(old_nets[c][n], x_init, y_init)
            ps_functions.synch_weight(nets[c][n], ps_model)

    ps_functions.warmup_lr(optimizers,num_cl,num_w_per_cluster,lr,iter_ind,max_ind) #initialize lr for warmup phase
    # Result vector
    results=np.empty([1,150])
    res_ind=0
    # training
    print('=======> training')
    for e in tqdm(range(epochs)):
        # user
        i = 0
        # cluster
        c = 0
        # period
        per = 0
        for data in trainloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # index of the worker
            index_user = i % num_w_per_cluster
            index_cluster = c % num_cl
            optimizers[index_cluster][index_user].zero_grad()  # i verified that this also make model gradients zero
            # predictions
            preds = nets[index_cluster][index_user](inputs)
            loss = criterions[index_cluster][index_user](preds, labels)
            loss.backward()

            optimizers[index_cluster][index_user].step(device, top_k_params)
            # for param in nets[index].parameters():
            #     print(param.grad.data)
            i += 1
            if i == num_w_per_cluster:

                ps_model.zero_grad()

                for n in range(num_w_per_cluster):
                    # ps_functions.gradient_accumulate(old_nets[c][n], nets[c][n], ps_model, is_avg, num_w_per_cluster)
                    ps_functions.gradient_accumulate(nets[c][n], ps_model, is_avg, num_w_per_cluster)

                # ps_functions.sparse_grad(top_k_ps, ps_model, device)

                for n in range(num_w_per_cluster):

                    # old_nets[c][n] = ps_functions.gradient_average(nets[c][n], ps_model, scale)
                    gradient_average(nets[c][n], ps_model, lr)


                i = 0
                c += 1

            if c == num_cl:
                c = 0
                per += 1
                iter_ind +=1
            if per == period:

                ps_functions.ps_param_zero(ps_model)

                for cl in range(num_cl):
                    # ps_functions.gradient_accumulate(old_nets[cl][0], nets[cl][0], ps_model, is_avg, num_cl)
                    ps_functions.weight_accumulate(nets[cl][0], ps_model, num_cl)

                # ps_functions.sparse_grad(top_k_ps, ps_model, device)

                for cl in range(num_cl):
                    for n in range(num_w_per_cluster):
                        # old_nets[cl][n] = ps_functions.gradient_average(nets[cl][n], ps_model, scale)
                        ps_functions.weight_broadcast(nets[cl][n], ps_model)


                per = 0
                i = 0
                c = 0
            ##### Change lr for next iteration during the warm up phase #####################
            ps_functions.warmup_lr(optimizers,num_cl,num_w_per_cluster,lr,iter_ind,max_ind)

        if e % 2 == 0:
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = nets[0][0](images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the 10000 test images: %d %%' % (
                    100 * correct / total))
            results[0][res_ind]= 100 * correct / total
            res_ind+=1 
        ps_functions.lr_change(optimizers, e, num_cl,num_w_per_cluster)
    correct = 0
    total = 0
    print('========> testing')
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = nets[0][0](images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    f=open('results_resnet_period_'+str(period)+'.txt','ab')
    np.savetxt(f,(results), fmt='%.5f', encoding='latin1')
    f.close()