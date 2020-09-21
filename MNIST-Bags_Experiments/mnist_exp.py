from mnist_bags import MnistBags
from bags_nets import LeNet5
import numpy as np
from functions_ import *
import csv
import torch
import torch.utils.data as data_utils
import torch.nn.init as init
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from torchvision import datasets, transforms
from tqdm import tqdm
import time


keys = ["max","mean"]
choices = {"max":max,"mean":mean}
before = True
number_of_runs = 10

def get_loaders(mean_bag_length, num_bag,pos_inst, batch_size, dataset="mnist"):
    kwargs = {}
    if dataset == "mnist":
        train_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                       mean_bag_length=mean_bag_length,
                                                       var_bag_length=0,
                                                       num_bag=num_bag,
                                                       seed=98,
                                                       train=True),
                                             batch_size=batch_size,
                                             shuffle=False, **kwargs)

        test_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                      mean_bag_length=mean_bag_length,
                                                      var_bag_length=0,
                                                      num_bag=1000,
                                                      seed=98,
                                                      train=False),
                                            batch_size=batch_size,
                                            shuffle=False, **kwargs)
    else:
        pass
    return train_loader,test_loader

def init_func(m):
    pass

def run_sgl(mean_bag_length=10, num_bag=100, pos_inst=2, delta = 0.3, verbose=False):

    res_auc = []
    res_iou = []
    for it in range(number_of_runs):
        print('sgl',mean_bag_length)
        results = [0,0]
        k = 'max'
        before = True

        network = LeNet5().cuda()
        network.apply(init_func)
        optimizer = torch.optim.Adam(network.parameters(), 0.0001,
                                                betas=(0.9,0.999),
                                                weight_decay =0.0001)
        epochs = 300

        train_loader, test_loader = get_loaders(mean_bag_length=mean_bag_length, num_bag=num_bag,pos_inst=pos_inst,batch_size=64 ,dataset='mnist')
        pooling = choices[k]
        if before:
            loss_fn = torch.nn.BCELoss()
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()

        cur_time = time.time()
        for e in range(epochs):
            batch_idx = 0
            total_loss = 0
            network.train()
            with tqdm(total=len(train_loader), desc='Step at start {}; Training epoch {}/{}'.format(it, e, epochs)) as pbar:
                for batch_idx, data in enumerate(train_loader):

                    x = data[0].cuda().view(-1,1,32,32)
                    y = data[1][0].float().cuda()
                    y_instance = data[1][1].cuda()
                    outs = network(x); outs['out'] = outs['out'].view(-1,mean_bag_length,1,1)

                    # try:

                    loss, inst = sgl(outs['out'],y, pooling,delta, before)

                    optimizer.zero_grad()
                    (loss + inst).backward()
                    optimizer.step()
                    # except:
                    #     if before:
                    #         out,feat = pooling(outs['out'].sigmoid()),[]
                    #         pred_supervision = (out>0.5).cuda()
                    #     else:
                    #         out,feat = pooling(outs['out']),[]
                    #         pred_supervision = (out>0).cuda()
                    #
                    #     import pdb;pdb.set_trace()
                    #     print("error",out,y)
                    pbar.set_postfix(loss = '{:.2f},{:.2f}'.format(loss,inst))
                    pbar.update()
                    total_loss += loss/len(train_loader)

            targets, outputs, ious = [], [], []
            batch_idx = 0

            network.eval()
            with tqdm(total=len(test_loader), desc='Step at start {}; Training epoch {}/{}'.format(it, e, epochs)) as pbar:
                for batch_idx, data in enumerate(test_loader):
                    with torch.no_grad():
                        x = data[0].cuda().view(-1,1,32,32)
                        y = data[1][0].float().cuda()
                        y_instance = data[1][1].cuda()
                        outs = network(x); outs['out'] = outs['out'].view(-1,mean_bag_length,1,1)
                        if before:
                            out,feat = pooling(outs['out'].sigmoid()),[]

                        else:
                            out,feat = pooling(outs['out']).sigmoid().detach(),[]

                        out_instances = outs['out'].sigmoid() > 0.5

                        intersection = (out_instances.flatten() * y_instance.flatten()).sum()
                        union = out_instances.flatten().sum() + y_instance.flatten().sum()-intersection
                        iou = intersection.float()/union.float()
                        if iou == iou:
                            ious += [iou.cpu()]
                        outputs += [out.cpu()]
                        targets += [y.cpu()]

                        pbar.set_postfix(loss = " ")
                        pbar.update()

            outputs = torch.cat(outputs,0);targets = torch.cat(targets,0);fpr, tpr, _ = roc_curve(targets, outputs)
            a = auc(fpr, tpr)
            if a > results[0]:
                results = [a,np.mean(ious)]
            if verbose:
                print("iou is :",np.mean(ious))
                print('auc at epoch {} is {}'.format(e, a))
            else:
                clear_output()
        res_iou += [results[1]]
        res_auc += [results[0]]
    if verbose:
        print("a = np.array({})\nb = np.array({})".format(res_iou,res_auc))
    return res_iou,res_auc

def run_bil(mean_bag_length=10, num_bag=100, pos_inst=2, verbose=False):

    res_auc = []
    res_iou = []
    for it in range(number_of_runs):
        print('bil',mean_bag_length)
        results = [0,0]
        k = 'mean'
        before = False

        network = LeNet5().cuda()
        network.apply(init_func)
        optimizer = torch.optim.Adam(network.parameters(), 0.0001,
                                                betas=(0.9,0.999),
                                                weight_decay =0.0001)
        epochs = 300

        train_loader, test_loader = get_loaders(mean_bag_length=mean_bag_length, num_bag=num_bag,pos_inst=pos_inst,batch_size=64 ,dataset='mnist')
        pooling = choices[k]
        if before:
            loss_fn = torch.nn.BCELoss()
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()

        cur_time = time.time()
        for e in range(epochs):
            batch_idx = 0
            total_loss = 0
            network.train()
            with tqdm(total=len(train_loader), desc='Step at start {}; Training epoch {}/{}'.format(it, e, epochs)) as pbar:
                for batch_idx, data in enumerate(train_loader):
                    x = data[0].cuda().view(-1,1,32,32)
                    y = data[1][0].float().cuda()
                    y_instance = data[1][1].cuda()
                    outs = network(x); outs['out'] = outs['out'].view(-1,mean_bag_length,1,1)


                    try:
                        if before:
                            loss = cust_bce(pooling(outs['out'].sigmoid()),y,-1)
                        else:
                            loss = cust_bce(pooling(outs['out']).sigmoid(),y,-1)

                        pred_supervision = ((outs['out'].sigmoid()>0.5).float()*y).detach()
                        r = cust_bce(outs['out'].sigmoid(),pred_supervision,1)
                        d = ((loss.detach() - r)**2)
                        loss = (((loss + d)*y) + ((1-y)*r)).mean()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    except:
                        if before:
                            out,feat = pooling(outs['out'].sigmoid()),[]
                            pred_supervision = (out>0.5).detach()
                        else:
                            out,feat = pooling(outs['out']),[]
                            pred_supervision = (out>0).detach()

                        import pdb;pdb.set_trace()
                        print("error",out,y)
                    pbar.set_postfix(loss = '{:.2f}'.format(loss,))
                    pbar.update()
                    total_loss += loss/len(train_loader)

            targets, outputs, ious = [], [], []
            batch_idx = 0

            network.eval()
            with tqdm(total=len(test_loader), desc='Step at start {}; Training epoch {}/{}'.format(it, e, epochs)) as pbar:
                for batch_idx, data in enumerate(test_loader):
                    with torch.no_grad():
                        x = data[0].cuda().view(-1,1,32,32)
                        y = data[1][0].float().cuda()
                        y_instance = data[1][1].cuda()
                        outs = network(x); outs['out'] = outs['out'].view(-1,mean_bag_length,1,1)
                        if before:
                            out,feat = pooling(outs['out'].sigmoid()),[]

                        else:
                            out,feat = pooling(outs['out']).sigmoid().detach(),[]

                        out_instances = outs['out'].sigmoid() > 0.5

                        intersection = (out_instances.flatten() * y_instance.flatten()).sum()
                        union = out_instances.flatten().sum() + y_instance.flatten().sum()-intersection
                        iou = intersection.float()/union.float()
                        if iou == iou:
                            ious += [iou.cpu()]
                        outputs += [out.cpu()]
                        targets += [y.cpu()]

                        pbar.set_postfix(loss = " ")
                        pbar.update()

            outputs = torch.cat(outputs,0);targets = torch.cat(targets,0);fpr, tpr, _ = roc_curve(targets, outputs)
            a = auc(fpr, tpr)
            if a > results[0]:
                results = [a,np.mean(ious)]
            if verbose:
                print("iou is :",np.mean(ious))
                print('auc at epoch {} is {}'.format(e, a))
            else:
                clear_output()
        res_iou += [results[1]]
        res_auc += [results[0]]

    if verbose:
        print("a = np.array({})\nb = np.array({})".format(res_iou,res_auc))
    return res_iou,res_auc

def run_mean(mean_bag_length=10, num_bag=100, pos_inst=2,verbose=False):

    res_auc = []
    res_iou = []
    for it in range(number_of_runs):
        print('mean',mean_bag_length)
        results = [0,0]
        k = 'mean'
        before = False

        network = LeNet5().cuda()
        network.apply(init_func)
        optimizer = torch.optim.Adam(network.parameters(), 0.0001,
                                                betas=(0.9,0.999),
                                                weight_decay =0.0001)
        epochs = 300

        train_loader, test_loader = get_loaders(mean_bag_length=mean_bag_length, num_bag=num_bag,pos_inst=pos_inst,batch_size=64 ,dataset='mnist')
        pooling = choices[k]
        if before:
            loss_fn = torch.nn.BCELoss()
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()

        cur_time = time.time()
        for e in range(epochs):
            batch_idx = 0
            total_loss = 0
            network.train()
            with tqdm(total=len(train_loader), desc='Step at start {}; Training epoch {}/{}'.format(it, e, epochs)) as pbar:
                for batch_idx, data in enumerate(train_loader):
                    x = data[0].cuda().view(-1,1,32,32)
                    y = data[1][0].float().cuda()
                    y_instance = data[1][1].cuda()
                    outs = network(x); outs['out'] = outs['out'].view(-1,mean_bag_length,1,1)
                    try:

                        if before:
                            loss = cust_bce(pooling(outs['out'].sigmoid()),y)
                        else:
                            loss = cust_bce(pooling(outs['out']).sigmoid(),y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    except:
                        if before:
                            out,feat = pooling(outs['out'].sigmoid()),[]
                        else:
                            out,feat = pooling(outs['out']),[]
                        print("error",out,y)
                    pbar.set_postfix(loss = '{:.2f}'.format(loss,))
                    pbar.update()
                    total_loss += loss/len(train_loader)

            targets, outputs, ious = [], [], []
            batch_idx = 0

            network.eval()
            with tqdm(total=len(test_loader), desc='Step at start {}; Training epoch {}/{}'.format(it, e, epochs)) as pbar:
                for batch_idx, data in enumerate(test_loader):
                    with torch.no_grad():
                        x = data[0].cuda().view(-1,1,32,32)
                        y = data[1][0].float().cuda()
                        y_instance = data[1][1].cuda()
                        outs = network(x); outs['out'] = outs['out'].view(-1,mean_bag_length,1,1)
                        if before:
                            out,feat = pooling(outs['out'].sigmoid()),[]

                        else:
                            out,feat = pooling(outs['out']).sigmoid().detach(),[]

                        out_instances = outs['out'].sigmoid() > 0.5

                        intersection = (out_instances.flatten() * y_instance.flatten()).sum()
                        union = out_instances.flatten().sum() + y_instance.flatten().sum()-intersection
                        iou = intersection.float()/union.float()
                        if iou == iou:
                            ious += [iou.cpu()]
                        outputs += [out.cpu()]
                        targets += [y.cpu()]

                        pbar.set_postfix(loss = " ")
                        pbar.update()

            outputs = torch.cat(outputs,0);targets = torch.cat(targets,0);fpr, tpr, _ = roc_curve(targets, outputs)
            a = auc(fpr, tpr)
            if a > results[0]:
                results = [a,np.mean(ious)]
            if verbose:
                print("iou is :",np.mean(ious))
                print('auc at epoch {} is {}'.format(e, a))
            else:
                clear_output()
        res_iou += [results[1]]
        res_auc += [results[0]]
    if verbose:
        print("a = np.array({})\nb = np.array({})".format(res_iou,res_auc))
    return res_iou,res_auc

def run_max(mean_bag_length=10, num_bag=100, pos_inst=2,verbose=False):
    res_auc = []
    res_iou = []
    for it in range(number_of_runs):
        print('max',mean_bag_length)
        results = [0,0]
        k = 'max'
        before = True

        network = LeNet5().cuda()
        network.apply(init_func)

        optimizer = torch.optim.Adam(network.parameters(), 0.0001,
                                                betas=(0.9,0.999),
                                                weight_decay =0.0001)
        epochs = 300

        train_loader, test_loader = get_loaders(mean_bag_length=mean_bag_length, num_bag=num_bag,pos_inst=pos_inst,batch_size=64 ,dataset='mnist')
        pooling = choices[k]
        if before:
            loss_fn = torch.nn.BCELoss()
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()

        cur_time = time.time()
        for e in range(epochs):
            batch_idx = 0
            total_loss = 0
            network.train()
            with tqdm(total=len(train_loader), desc='Step at start {}; Training epoch {}/{}'.format(it, e, epochs)) as pbar:
                for batch_idx, data in enumerate(train_loader):
                    x = data[0].cuda().view(-1,1,32,32)
                    y = data[1][0].float().cuda()
                    y_instance = data[1][1].cuda()
                    outs = network(x); outs['out'] = outs['out'].view(-1,mean_bag_length,1,1)

                    try:

                        if before:
                            loss = cust_bce(pooling(outs['out'].sigmoid()),y)
                        else:
                            loss = cust_bce(pooling(outs['out']).sigmoid(),y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    except:
                        if before:
                            out,feat = pooling(outs['out'].sigmoid()),[]
                        else:
                            out,feat = pooling(outs['out']),[]
                        print("error",out,y)
                    pbar.set_postfix(loss = '{:.2f}'.format(loss,))
                    pbar.update()
                    total_loss += loss/len(train_loader)

            targets, outputs, ious = [], [], []
            batch_idx = 0

            network.eval()
            with tqdm(total=len(test_loader), desc='Step at start {}; Training epoch {}/{}'.format(it, e, epochs)) as pbar:
                for batch_idx, data in enumerate(test_loader):
                    with torch.no_grad():
                        x = data[0].cuda().view(-1,1,32,32)
                        y = data[1][0].float().cuda()
                        y_instance = data[1][1].cuda()
                        outs = network(x); outs['out'] = outs['out'].view(-1,mean_bag_length,1,1)
                        if before:
                            out,feat = pooling(outs['out'].sigmoid()),[]

                        else:
                            out,feat = pooling(outs['out']).sigmoid(),[]

                        out_instances = outs['out'].sigmoid() > 0.5

                        intersection = (out_instances.flatten() * y_instance.flatten()).sum()
                        union = out_instances.flatten().sum() + y_instance.flatten().sum()-intersection
                        iou = intersection.float()/union.float()
                        if iou == iou:
                            ious += [iou.cpu()]
                        outputs += [out.cpu()]
                        targets += [y.cpu()]

                        pbar.set_postfix(loss = " ")
                        pbar.update()

            outputs = torch.cat(outputs,0);targets = torch.cat(targets,0);fpr, tpr, _ = roc_curve(targets, outputs)
            a = auc(fpr, tpr)
            if a > results[0]:
                results = [a,np.mean(ious)]
            if verbose:
                print("iou is :",np.mean(ious))
                print('auc at epoch {} is {}'.format(e, a))
            else:
                clear_output()
        res_iou += [results[1]]
        res_auc += [results[0]]

    if verbose:
        print("a = np.array({})\nb = np.array({})".format(res_iou,res_auc))
    return res_iou,res_auc

def run_mmm(mean_bag_length=10, num_bag=100, pos_inst=2, verbose=False):
    res_auc = []
    res_iou = []
    for it in range(number_of_runs):
        print('mmm',mean_bag_length)
        results = [0,0]
        k = 'max'
        before = True

        network = LeNet5().cuda()
        network.apply(init_func)

        optimizer = torch.optim.Adam(network.parameters(), 0.0001,
                                                betas=(0.9,0.999),
                                                weight_decay =0.0001)
        epochs = 300

        train_loader, test_loader = get_loaders(mean_bag_length=mean_bag_length, num_bag=num_bag,pos_inst=pos_inst,batch_size=64 ,dataset='mnist')
        pooling = choices[k]
        if before:
            loss_fn = torch.nn.BCELoss()
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()

        cur_time = time.time()
        for e in range(epochs):
            batch_idx = 0
            total_loss = 0
            network.train()
            with tqdm(total=len(train_loader), desc='Step at start {}; Training epoch {}/{}'.format(it, e, epochs)) as pbar:
                for batch_idx, data in enumerate(train_loader):
                    x = data[0].cuda().view(-1,1,32,32)
                    y = data[1][0].float().cuda()
                    y_instance = data[1][1].cuda()
                    outs = network(x); outs['out'] = outs['out'].view(-1,mean_bag_length,1,1)
                    try:

                        if before:
                            loss = (cust_bce(torch.max(outs['out'].sigmoid(),1)[0].squeeze(),y) +\
                                cust_bce(torch.min(outs['out'].sigmoid(),1)[0].squeeze(),0) +\
                                cust_bce(torch.mean(outs['out'].sigmoid(),1), torch.tensor(0.5)*y))/3
                        else:
                            loss = (cust_bce(pooling(outs['out']).sigmoid(),y)+\
                                cust_bce(torch.min(outs['out'],1)[0].sigmoid().squeeze(),0) +\
                                cust_bce(torch.mean(outs['out'],1).sigmoid(), torch.tensor(0.5)*y))/3
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    except:
                        import pdb;pdb.set_trace()
                        if before:
                            out,feat = pooling(outs['out'].sigmoid()),[]
                        else:
                            out,feat = pooling(outs['out']),[]

                        print("error",out,y)
                    pbar.set_postfix(loss = '{:.2f}'.format(loss,))
                    pbar.update()
                    total_loss += loss/len(train_loader)

            targets, outputs, ious = [], [], []
            batch_idx = 0

            network.eval()
            with tqdm(total=len(test_loader), desc='Step at start {}; Training epoch {}/{}'.format(it, e, epochs)) as pbar:
                for batch_idx, data in enumerate(test_loader):
                    with torch.no_grad():
                        x = data[0].cuda().view(-1,1,32,32)
                        y = data[1][0].float().cuda()
                        y_instance = data[1][1].cuda()
                        outs = network(x); outs['out'] = outs['out'].view(-1,mean_bag_length,1,1)
                        if before:
                            out,feat = pooling(outs['out'].sigmoid()),[]

                        else:
                            out,feat = pooling(outs['out']).sigmoid().detach(),[]

                        out_instances = outs['out'].sigmoid() > 0.5

                        intersection = (out_instances.flatten() * y_instance.flatten()).sum()
                        union = out_instances.flatten().sum() + y_instance.flatten().sum()-intersection
                        iou = intersection.float()/union.float()
                        if iou == iou:
                            ious += [iou.cpu()]
                        outputs += [out.cpu()]
                        targets += [y.cpu()]

                        pbar.set_postfix(loss = " ")
                        pbar.update()

            outputs = torch.cat(outputs,0);targets = torch.cat(targets,0);fpr, tpr, _ = roc_curve(targets, outputs)
            a = auc(fpr, tpr)
            if a > results[0]:
                results = [a,np.mean(ious)]
            if verbose:
                print("iou is :",np.mean(ious))
                print('auc at epoch {} is {}'.format(e, a))
            else:
                clear_output()
        res_iou += [results[1]]
        res_auc += [results[0]]
    if verbose:
        print("a = np.array({})\nb = np.array({})".format(res_iou,res_auc))

    return res_iou,res_auc

def run(method='sgl',currrent=10,delta=None):
    methods = {'sgl':run_sgl,"max":run_max,"mean":run_mean,"mmm":run_mmm,'bil':run_bil}
    def run_iteration(method, cur,cur2, delta=None):
        if delta is None:
            a,b = methods[method](cur,cur2,verbose=True)
        else:
            a,b = methods[method](cur,cur2, delta,verbose=True)
        a += [[],np.array(a).mean(),np.array(a).max()-np.array(a).mean(),np.array(a).mean()-np.array(a).min()]
        b += [[],np.array(b).mean(),np.array(b).max()-np.array(b).mean(),np.array(b).mean()-np.array(b).min()]
        file = open('{}_instances.csv'.format(current), 'a+');file.write(';'.join(['{}_{}_{}_delta{}_auc'.format(method,cur,cur2,delta)]+[str(l) for l in a])+'\n');file.close()
        file = open('{}_instances.csv'.format(current), 'a+');file.write(';'.join(['{}_{}_{}_delta{}_iou'.format(method,cur,cur2,delta)]+[str(l) for l in b]) + '\n');file.close()

    run_iteration(method,current,50,)
    run_iteration(method,current,100)
    run_iteration(method,current,150,)
    run_iteration(method,current,200,)
    run_iteration(method,current,300,)
    run_iteration(method,current,500,)




import argparse

parser = argparse.ArgumentParser(description='Start MNIST-Runs. --method <choices> --instances <instances>')
parser.add_argument('--method', type=str, choices=['sgl','max','mean','mmm','bil',],
                    help='an integer for the accumulator')
parser.add_argument('--instances', type=int,
                    default=10,
                    help='sum the integers (default: find the max)')
parser.add_argument('--delta', type=float,
                    default=None,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()

current = args.instances
method = args.method
delta = args.delta
run(method,current,delta)
