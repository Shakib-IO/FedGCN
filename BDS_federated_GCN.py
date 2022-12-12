from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from models import GCN

import torch_geometric
import torch_sparse

import matplotlib.pyplot as plt
from sklearn import metrics

from utils import get_plot
import scipy.sparse as sp

from data_process import generate_data, load_data, load_data_for_amazon, load_data_fraudre
from train_func import test, train, eval_metrics, Lhop_Block_matrix_train, FedSage_train

import importlib
import data_process
importlib.reload(data_process)

def get_K_hop_neighbors(adj_matrix, index, K):
    adj_matrix = adj_matrix + torch.eye(adj_matrix.shape[0],adj_matrix.shape[1])  #make sure the diagonal part >= 1
    hop_neightbor_index=index
    for i in range(K):
        hop_neightbor_index=torch.unique(torch.nonzero(adj[hop_neightbor_index])[:,1])
    return hop_neightbor_index


def normalize(mx):  #adj matrix
    
    mx = mx + torch.eye(mx.shape[0],mx.shape[1])
    
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return torch.tensor(mx)

def setdiff1d(t1, t2):
    
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    #intersection = uniques[counts > 1]
    return difference

def intersect1d(t1, t2):
    
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    #difference = uniques[counts == 1]
    intersection = uniques[counts > 1]
    return intersection

#define model

#for compare 2-10 layer performance in appendix
#iterations = 400
#Adam, lr = 0.01

def BDS_federated_GCN(K, features, adj, labels, idx_train, idx_val, idx_test, iid_percent, sample_rate =0.5, L_hop=1, num_layers=2):
        # K: number of models
        #choose adj matrix
        #multilayer_GCN:n*n
        #define model
        global_model = GCN(nfeat=features.shape[1],
                nhid=args_hidden,
                nclass=labels.max().item() + 1,
                dropout=args_dropout,
                NumLayers=num_layers)
        global_model.reset_parameters()
        models=[]
        for i in range(K):
            models.append(GCN(nfeat=features.shape[1],
                nhid=args_hidden,
                nclass=labels.max().item() + 1,
                dropout=args_dropout,
                NumLayers=num_layers))
        if args_cuda:
                for i in range(K):
                    models[i]=models[i].cuda()
                global_model=global_model.cuda()
                features = features.cuda()
                adj = adj.cuda()
                labels = labels.cuda()
                idx_train = idx_train.cuda()
                idx_val = idx_val.cuda()
                idx_test = idx_test.cuda()
        #optimizer and train
        optimizers=[]
        for i in range(K):
            optimizers.append(optim.SGD(models[i].parameters(),
                              lr=args_lr, weight_decay=args_weight_decay))
        # Train model
        
        row, col, edge_attr = adj.t().coo()
        edge_index = torch.stack([row, col], dim=0)
        
        
        split_data_indexes=[]
        
        nclass=labels.max().item() + 1
        split_data_indexes = []
        non_iid_percent = 1 - float(iid_percent)
        iid_indexes = [] #random assign
        shuffle_labels = [] #make train data points split into different devices
        for i in range(K):
            current = torch.nonzero(labels == i).reshape(-1)
            current = current[np.random.permutation(len(current))] #shuffle
            shuffle_labels.append(current)
                
        average_device_of_class = K // nclass
        if K % nclass != 0: #for non-iid
            average_device_of_class += 1
        for i in range(K):  
            label_i= i // average_device_of_class    
            labels_class = shuffle_labels[label_i]

            average_num= int(len(labels_class)//average_device_of_class * non_iid_percent)
            split_data_indexes.append((labels_class[average_num * (i % average_device_of_class):average_num * (i % average_device_of_class + 1)]))
        
        if args_cuda:
            iid_indexes = setdiff1d(torch.tensor(range(len(labels))).cuda(), torch.cat(split_data_indexes))
        else:
            iid_indexes = setdiff1d(torch.tensor(range(len(labels))), torch.cat(split_data_indexes))
        
        iid_indexes = iid_indexes[np.random.permutation(len(iid_indexes))]
        
        for i in range(K):  #for iid
            label_i= i // average_device_of_class
            labels_class = shuffle_labels[label_i]

            average_num= int(len(labels_class)//average_device_of_class * (1 - non_iid_percent))
            split_data_indexes[i] = list(split_data_indexes[i]) + list(iid_indexes[:average_num])
                    
            iid_indexes = iid_indexes[average_num:]
            
        communicate_indexes = []
        in_com_train_data_indexes = []

        for i in range(K):
            if args_cuda:
                split_data_indexes[i] = torch.tensor(split_data_indexes[i]).cuda()
            else:
                split_data_indexes[i] = torch.tensor(split_data_indexes[i])
                
            split_data_indexes[i] = split_data_indexes[i].sort()[0]
            
            #communicate_index=get_K_hop_neighbors(adj, split_data_indexes[i], L_hop) #normalized adj
            
            communicate_index = torch_geometric.utils.k_hop_subgraph(split_data_indexes[i],L_hop,edge_index)[0]
            communicate_indexes.append(communicate_index)
            communicate_indexes[i] = communicate_indexes[i].sort()[0]
            
            inter = intersect1d(split_data_indexes[i], idx_train)  ###only count the train data of nodes in current server(not communicate nodes)   
            in_com_train_data_indexes.append(torch.searchsorted(communicate_indexes[i], inter).clone())   #local id in block matrix

            
            
        #assign global model weights to local models at initial step
        for i in range(K):
            models[i].load_state_dict(global_model.state_dict())
        
        for t in range(args_iterations):
            acc_trains=[]
            for i in range(K):
                for epoch in range(args_epochs):
                    diff = setdiff1d(split_data_indexes[i], communicate_indexes[i])
                    sample_index = torch.cat((split_data_indexes[i], diff[torch.randperm(len(diff))[:int(len(diff) * sample_rate)]])).clone()

                    sample_index = sample_index.sort()[0]
                    
                    inter = intersect1d(split_data_indexes[i], idx_train)  ###only count the train data of nodes in current server(not communicate nodes)
                    in_sample_train_data_index = torch.searchsorted(sample_index, inter).clone()   #local id in block matrix

                    if len(in_sample_train_data_index) == 0:
                        continue
                    try:
                        adj[sample_index][:,sample_index]
                    except: #adj is empty
                        continue
                    
                    acc_train = FedSage_train(epoch, models[i], optimizers[i], 
                                                        features, adj, labels, sample_index, in_sample_train_data_index)
                acc_trains.append(acc_train)
                
            states=[]
            gloabl_state=dict()
            for i in range(K):
                states.append(models[i].state_dict())
            # Average all parameters
            for key in global_model.state_dict():
                gloabl_state[key] = in_com_train_data_indexes[0].shape[0] * states[0][key]
                count_D=in_com_train_data_indexes[0].shape[0]
                for i in range(1,K):
                    gloabl_state[key] += in_com_train_data_indexes[i].shape[0] * states[i][key]
                    count_D += in_com_train_data_indexes[i].shape[0]
                gloabl_state[key] /= count_D

            global_model.load_state_dict(gloabl_state)
            
            # Testing
            
            loss_train, acc_train = test(global_model, features, adj, labels, idx_train)
            print(t,'\t',"train",'\t',loss_train,'\t',acc_train)
            
            loss_val, acc_val = test(global_model, features, adj, labels, idx_val) #validation
            print(t,'\t',"val",'\t',loss_val,'\t',acc_val)
            

            a = open(dataset_name+'_IID_'+str(iid_percent)+'_' + str(L_hop) +'hop_BDS_federated_' + str(num_layers) + 'layer_GCN_iter_'+str(args_iterations)+'_epoch_'+str(args_epochs)+'_device_num_'+str(K),'a+')

            
            a.write(str(t)+'\t'+"train"+'\t'+str(loss_train)+'\t'+str(acc_train)+'\n')
            a.write(str(t)+'\t'+"val"+'\t'+str(loss_val)+'\t'+str(acc_val)+'\n')
            a.close()
            for i in range(K):
                models[i].load_state_dict(gloabl_state)
        #test  
        loss_test, acc_test= test(global_model, features, adj, labels, idx_test)
        print(t,'\t',"test",'\t',loss_test,'\t',acc_test)
        a = open(dataset_name+'_IID_'+str(iid_percent)+'_' + str(L_hop) +'hop_BDS_federated_' + str(num_layers) + 'layer_GCN_iter_'+str(args_iterations)+'_epoch_'+str(args_epochs)+'_device_num_'+str(K),'a+')
        a.write(str(t)+'\t'+"test"+'\t'+str(loss_test)+'\t'+str(acc_test)+'\n')
        a.close()
        print("save file as",dataset_name+'_IID_'+str(iid_percent)+'_' + str(L_hop) +'hop_BDS_federated_' + str(num_layers) + 'layer_GCN_iter_'+str(args_iterations)+'_epoch_'+str(args_epochs)+'_device_num_'+str(K))
        
        del global_model
        del features 
        del adj
        del labels
        del idx_train
        del idx_val
        del idx_test
        while(len(models)>=1):
            del models[0]
        
        return loss_test, acc_test
    

#for testing
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    #'cora', 'citeseer', 'pubmed', 'amazon' #simulate #other dataset twitter, 
    dataset_name="amazon" #'ogbn-arxiv'

    if dataset_name == 'simulate':
        number_of_nodes=200
        class_num=3
        link_inclass_prob=10/number_of_nodes  #when calculation , remove the link in itself
        #EGCN good when network is dense 20/number_of_nodes  #fails when network is sparse. 20/number_of_nodes/5

        link_outclass_prob=link_inclass_prob/20


        features, adj, labels, idx_train, idx_val, idx_test = generate_data(number_of_nodes,  class_num, link_inclass_prob, link_outclass_prob)               
    else:
        features, adj, labels, idx_train, idx_val, idx_test = load_data_fraudre(dataset_name)
        class_num = labels.max().item() + 1
        dataset_name="amazon" #'ogbn-arxiv'
    if dataset_name in ['simulate', 'cora', 'citeseer', 'pubmed','amazon']:
        args_hidden = 16
    else:
        args_hidden = 256

    args_dropout = 0.5
    args_lr = 1.0
    args_weight_decay = 5e-4     #L2 penalty
    args_epochs = 3
    args_no_cuda = False
    args_cuda = not args_no_cuda and torch.cuda.is_available()
    #class_num = 3
    args_device_num = class_num #split data into args_device_num parts
    args_iterations = 500
    
    # Call the Function
    for args_epochs in [3]:
        for args_random_assign in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            for i in range(10):
                BDS_federated_GCN(class_num, features, adj, labels, idx_train, idx_val, idx_test, args_random_assign)
            
main()