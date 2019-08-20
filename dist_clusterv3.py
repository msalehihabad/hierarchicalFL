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


def deltaAcucumulate(DELTAmodel, refModel, num_workers):

    for paramDelta, paramRef in zip(DELTAmodel.parameters(), refModel.parameters()):
        # get model difference.
        paramRef.grad.data += paramDelta.data / num_workers

    return None


def gradient_average(model, agg_model, scale):

    for param, param_ps in zip(model.parameters(), agg_model.parameters()):
        param.grad.data = param_ps.grad.data
        param.data.add_(-scale, param.grad.data)

    return None

def sgdLocalstep(model, sparseDelta):

    for param, param_del in zip(model.parameters(), sparseDelta.parameters()):
        param.grad.data = param_del.data
        param.data.add_(1, param.grad.data)

    return None


# update method 1 in psudo algorithm
def sgdStep1(model, scale):
    for param_ps in model.parameters():
        param_ps.data.add_(-scale, param_ps.grad.data)

    return None

# update method 2 in psudo algorithm
def sgdStep2(modelMBS, modelTilde, modelError, scale, scaleBeta):
    for paramBS, paramTilde, paramError in zip(modelMBS.parameters(), modelTilde.parameters(), modelError.parameters()):

        diff = -scale * paramBS.grad.data + scaleBeta * paramError
        paramBS.data = paramTilde.data + diff

    return None


def updateDiff(delta_net, net1, net2):

    for d, w1, w2 in zip(delta_net.parameters(), net1.parameters(), net2.parameters()):
        d.data = w1.data - w2.data
    return None


def update_w_ref(modelRef, modelSparse):
    for param, paramSparse in zip(modelRef.parameters(), modelSparse.parameters()):
        param.data += paramSparse.grad.data + 0


def updateErrMBS(modelErr, modelRef, modelRefSparse):

    for d, dw1, dw2 in zip(modelErr.parameters(), modelRef.parameters(), modelRefSparse.parameters()):
        d.data = dw1.grad.data - dw2.grad.data
    return None

def updateWeightSBS(model_SBS, sparseDELTA, modelErr, model_ref, BETA2):

    for paramSBS, paramSparseDELTA, paramErr, paramRef in zip(
            model_SBS.parameters(),
            sparseDELTA.parameters(),
            modelErr.parameters(),
            model_ref.parameters()):
        paramSBS.data = paramRef.data + paramSparseDELTA.grad.data + BETA2 * paramErr.data

    return None


for a in range(2):
    # select gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    num_cl = 2
    # number of workers per cluster
    num_w_per_cluster = 3
    # weightsTilde = [[nn_classes.ResNet18().to(device) for n in range(num_w_per_cluster)] for c in range(num_cl)]
    weightsTilde = [[nn_classes.SIMPLE_CIFAR_NET().to(device) for n in range(num_w_per_cluster)] for c in range(num_cl)]

    for c in range(num_cl):
        for n in range(num_w_per_cluster):
            ps_functions.grad_init(weightsTilde[c][n], x_init, y_init)

    # model at PS for all-reduce purposes
    wref = nn_classes.SIMPLE_CIFAR_NET().to(device)
    wrefSparse = nn_classes.SIMPLE_CIFAR_NET().to(device)
    # error accumulated at MBS
    errMBS = nn_classes.SIMPLE_CIFAR_NET().to(device)
    updateDiff(errMBS, wref, wref)
    ps_functions.grad_init(wrefSparse, x_init, y_init)
    # weightMBS = nn_classes.ResNet18().to(device)
    ps_functions.grad_init(wref, x_init, y_init)


    weightsSBS = [nn_classes.SIMPLE_CIFAR_NET().to(device) for c in range(num_cl)]
    # weightsSBS = [nn_classes.ResNet18().to(device) for c in range(num_cl)]
    [ps_functions.grad_init(weightsSBS[c], x_init, y_init) for c in range(num_cl)]

    deltaWeight = [nn_classes.SIMPLE_CIFAR_NET().to(device) for c in range(num_cl)]
    # deltaWeight = [nn_classes.ResNet18().to(device) for c in range(num_cl)]
    # initialize deltaW to be zero
    [updateDiff(deltaWeight[c], wref, wref) for c in range(num_cl)]

    deltaWeightSparse = [nn_classes.SIMPLE_CIFAR_NET().to(device) for c in range(num_cl)]
    # deltaWeightSparse = [nn_classes.ResNet18().to(device) for c in range(num_cl)]

    # error of sparsigying deltaw
    epsSBS = [nn_classes.SIMPLE_CIFAR_NET().to(device) for c in range(num_cl)]
    # epsSBS = [nn_classes.ResNet18().to(device) for c in range(num_cl)]
    [updateDiff(epsSBS[c], wref, wref) for c in range(num_cl)]

    errSBS = [nn_classes.SIMPLE_CIFAR_NET().to(device) for c in range(num_cl)]
    [updateDiff(errSBS[c], wref, wref) for c in range(num_cl)]
    DELTASBS = [nn_classes.SIMPLE_CIFAR_NET().to(device) for c in range(num_cl)]
    [updateDiff(DELTASBS[c], wref, wref) for c in range(num_cl)]
    DELTASBSsparse = [nn_classes.SIMPLE_CIFAR_NET().to(device) for c in range(num_cl)]

    betaSBS = 1 / num_cl
    beta = 0.5
    BETAMBS = 0.5
    lr = 0.25
    scale = 1
    momentum = 0.9
    is_avg = True

    model_parameters = filter(lambda p: p.requires_grad, wref.parameters())
    params_number = sum([np.prod(p.size()) for p in model_parameters])

    p_sparce_ul = 0.99  # sparsification param
    p_sparce_dl = 0.9

    p_sparse_ul_sbs = 0.0
    p_sparse_dl_mbs = 0.0

    top_k_params_ul = int(np.ceil((1 - p_sparce_ul) * params_number))
    top_k_params_dl = int(np.ceil((1 - p_sparce_dl) * params_number))

    top_k_ul_sbs = int(np.ceil((1 - p_sparse_ul_sbs) * params_number))
    top_k_dl_mbs = int(np.ceil((1 - p_sparse_dl_mbs) * params_number))


    weight_decay = 0.0001
    #weight_decay = 0

    criterions = [[nn.CrossEntropyLoss() for n in range(num_w_per_cluster)] for c in range(num_cl)]
    optimizers = [[SGD_custom2.define_optimizer(weightsTilde[c][n], lr, momentum, w_decay=weight_decay)
                   for n in range(num_w_per_cluster)] for c in range(num_cl)]
    #############
    epochs = 300
    period = 6
    iter_ind = 1
    #############
    warm_up_epoch = 5
    max_ind= 50000*warm_up_epoch/(mini_batch*num_w_per_cluster*num_cl)
    ##############

    for c in range(num_cl):
        for n in range(num_w_per_cluster):
            ps_functions.synch_weight(weightsTilde[c][n], wref)

    ps_functions.warmup_lr(optimizers,num_cl,num_w_per_cluster,lr,iter_ind,max_ind) #initialize lr for warmup phase
    # Result vector
    results = np.empty([1, 150])
    res_ind = 0
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
            preds = weightsTilde[index_cluster][index_user](inputs)
            loss = criterions[index_cluster][index_user](preds, labels)
            loss.backward()

            optimizers[index_cluster][index_user].step(device, top_k_params_ul)

            i += 1
            if i == num_w_per_cluster:
                i == 0
                c += 1

            if c == num_cl:

                # line 17, updates w_n(t)
                for cl in range(num_cl):
                    weightsSBS[cl].zero_grad()

                    for n in range(num_w_per_cluster):
                        gradient_accumulate(weightsTilde[cl][n], weightsSBS[cl], is_avg, num_w_per_cluster)

                    sgdStep2(weightsSBS[cl], weightsTilde[cl][0], epsSBS[cl], lr, beta)

                iter_ind += 1
                if per == period:
                    per = 0
                    i = 0
                    c = 0
                    # global model update
                    wref.zero_grad()
                    # line 20-23
                    for cl in range(num_cl):
                        updateDiff(DELTASBS[cl], weightsSBS[cl], wref)
                        ps_functions.make_sparse(top_k_ul_sbs, DELTASBS[cl], DELTASBSsparse[cl], device)
                        deltaAcucumulate(DELTASBSsparse[cl], wref, num_cl)
                        updateDiff(errSBS[cl], DELTASBS[cl], DELTASBSsparse[cl])
                    # line 24, adds error to big delta
                    for errParam, param in zip(errMBS.parameters(), wref.parameters()):
                        param.grad.data = errParam.data * BETAMBS

                    for param, paramSparse in zip(wref.parameters(), wrefSparse.parameters()):
                        paramSparse.grad.data = param.grad.data + 0

                    ps_functions.sparse_grad(top_k_dl_mbs, wrefSparse, device)
                    # updates the error, line 26
                    updateErrMBS(errMBS, wref, wrefSparse)
                    # line 28-29
                    [updateWeightSBS(weightsSBS[cl], wrefSparse, errSBS[cl], betaSBS) for cl in range(num_cl)]
                    # line 27
                    update_w_ref(wref, wrefSparse)

                    # cluster model update

                    for cl in range(num_cl):
                        updateDiff(deltaWeight[cl], weightsSBS[cl], weightsTilde[cl][0])
                        ps_functions.make_sparse(top_k_params_dl, deltaWeight[cl], deltaWeightSparse[cl], device)
                        for n in range(num_w_per_cluster):
                            # updating w_tilde with sparsified deltaW
                            sgdLocalstep(weightsTilde[cl][n], deltaWeightSparse[cl])
                        # calculating error= delta_w - sparse(delta_w)
                        updateDiff(epsSBS[cl], deltaWeight[cl], deltaWeightSparse[cl])

                else:
                    # cluster model update
                    for cl in range(num_cl):
                        updateDiff(deltaWeight[cl], weightsSBS[cl], weightsTilde[cl][0])
                        ps_functions.make_sparse(top_k_params_dl, deltaWeight[cl], deltaWeightSparse[cl], device)
                        for n in range(num_w_per_cluster):
                            # updating w_tilde with sparsified deltaW
                            sgdLocalstep(weightsTilde[cl][n], deltaWeightSparse[cl])
                        # calculating error= delta_w - sparse(delta_w)
                        updateDiff(epsSBS[cl], deltaWeight[cl], deltaWeightSparse[cl])
                    per += 1
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
                    outputs = weightsTilde[0][0](images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the 10000 test images: %d %%' % (
                    100 * correct / total))
            results[0][res_ind] = 100 * correct / total
            res_ind += 1
        ps_functions.lr_change(optimizers, e, num_cl,num_w_per_cluster)
    correct = 0
    total = 0
    print('========> testing')
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = weightsTilde[0][0](images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    f=open('results_resnet_period_'+str(period)+'.txt','ab')
    np.savetxt(f, (results), fmt='%.5f', encoding='latin1')
    f.close()
