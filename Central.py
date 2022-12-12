#import all the libraries and functions

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

import scipy.sparse as sp

from utils import get_plot

from data_process import generate_data, load_data, load_data_for_amazon, load_data_fraudre
from train_func import test, train, eval_metrics, Lhop_Block_matrix_train, FedSage_train


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

#Define model
#for compare 2-10 layer performance in appendix
#iterations = 400
#Adam, lr = 0.01

def centralized_GCN(features, adj, labels, idx_train, idx_val, idx_test, num_layers):
        model = GCN(nfeat=features.shape[1],
                nhid=args_hidden,
                nclass=labels.max().item() + 1,
                dropout=args_dropout,
                NumLayers=num_layers)
        model.reset_parameters()
        if args_cuda:
                #from torch_geometric.nn import DataParallel
                #model = DataParallel(model)
                #model= torch.nn.DataParallel(model)
                model=model#.cuda()
                
                #features= torch.nn.DataParallel(features)
                
                features = features#.cuda()
                
                #edge_index= torch.nn.DataParallel(edge_index)
                
                adj = adj#.cuda()
                labels = labels#.cuda()
                idx_train = idx_train#.cuda()
                idx_val = idx_val#.cuda()
                idx_test = idx_test#.cuda()
        #optimizer and train
        
        optimizer = optim.SGD(model.parameters(),
                              lr=args_lr, weight_decay=args_weight_decay)
        
        
        #optimizer = optim.Adam(model.parameters(),
        #                      lr=args_lr, weight_decay=args_weight_decay)
        # Train model
        best_val=0
        for t in range(args_iterations): #make to equivalent to federated
            loss_train, acc_train=train(t, model, optimizer, features, adj, labels, idx_train)
            # validation
            loss_train, acc_train= test(model, features, adj, labels, idx_train) #train after backward
            print(t,"train",loss_train,acc_train)
            loss_val, acc_val= test(model, features, adj, labels, idx_val) #validation
            print(t,"val",loss_val,acc_val)
            
            #Save the training data into a File
            a = open(dataset_name+'_IID_'+'centralized_' + str(num_layers) + 'layer_GCN_iter_'+str(args_iterations),'a+')
            a.write(str(t)+'\t'+"train"+'\t'+str(loss_train)+'\t'+str(acc_train)+'\n')
            a.write(str(t)+'\t'+"val"+'\t'+str(loss_val)+'\t'+str(acc_val)+'\n')
            a.close()
            
        #test 
        loss_test, acc_test= test(model, features, adj, labels, idx_test)
        print("\n")
        print(t,'\t',"test",'\t',loss_test,'\t',acc_test)

        #evaluation metrics -> Precision, Recall, Balanced Accuracy, and F1-score. 
        precision_sc, recalL_sc, balanced_acc_sc, f1_test = eval_metrics(model, features, adj, labels, idx_test)
        print(f"{precision_sc}, {recalL_sc}, {balanced_acc_sc}, {f1_test}".format(precision_sc,recalL_sc,balanced_acc_sc,f1_test))

        
        # Save the testing data into a File
        # a = open(dataset_name+'_IID_'+'centralized_' + str(num_layers) + 'layer_GCN_iter_'+str(args_iterations),'a+')
        # a.write(str(t)+'\t'+"test"+'\t'+str(loss_test)+'\t'+str(acc_test)+'\n')
        # a.close()
        #print("save file as",dataset_name+'_IID_'+'centralized_' + str(num_layers) + 'layer_GCN_iter_'+str(args_iterations))

        # Save the Evalution Metrics into a file
        a = open(dataset_name+'_IID_'+'centralized_' + str(num_layers) + 'layer_GCN_iter_'+str(args_iterations),'a+')
        a.write(str(t)+'\t'+"test"+'\t'+str(loss_test)+'\t'+str(acc_test)+'\n')
        a.write(str(t)+'\t'+str(precision_sc) + '\t'+str(recalL_sc) +'\t' +str(balanced_acc_sc) +'\t' +str(f1_test)) 
        a.close()
        print("save file as",dataset_name+'_IID_'+'centralized_' + str(num_layers) + 'layer_GCN_iter_'+str(args_iterations))

        del model
        del features 
        del adj
        del labels
        del idx_train
        del idx_val
        del idx_test
        
        return loss_test, acc_test, f1_test

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
    centralized_GCN(features, adj, labels, idx_train, idx_val, idx_test, num_layers = 2)

main()