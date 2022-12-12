#setting of data generation

import torch
import random
import sys
import pickle as pkl
import numpy as np
import scipy.sparse as sp
#from scipy.sparse.linalg.eigen.arpack import eigsh
import networkx as nx
import torch_geometric
import torch_sparse


def generate_data(number_of_nodes, class_num, link_inclass_prob, link_outclass_prob):
    
    adj=torch.zeros(number_of_nodes,number_of_nodes) #n*n adj matrix

    labels=torch.randint(0,class_num,(number_of_nodes,)) #assign random label with equal probability
    labels=labels.to(dtype=torch.long)
    #label_node, speed up the generation of edges
    label_node_dict=dict()

    for j in range(class_num):
            label_node_dict[j]=[]

    for i in range(len(labels)):
        label_node_dict[int(labels[i])]+=[int(i)]


    #generate graph
    for node_id in range(number_of_nodes):
                j=labels[node_id]
                for l in label_node_dict:
                    if l==j:
                        for z in label_node_dict[l]:  #z>node_id,  symmetrix matrix, no repeat
                            if z>node_id and random.random()<link_inclass_prob:
                                adj[node_id,z]= 1
                                adj[z,node_id]= 1
                    else:
                        for z in label_node_dict[l]:
                            if z>node_id and random.random()<link_outclass_prob:
                                adj[node_id,z]= 1
                                adj[z,node_id]= 1
                              
    adj=torch_geometric.utils.dense_to_sparse(torch.tensor(adj))[0]

    #generate feature use eye matrix
    features=torch.eye(number_of_nodes,number_of_nodes)
    
    #seprate train,val,test
    idx_train = torch.LongTensor(range(number_of_nodes//5))
    idx_val = torch.LongTensor(range(number_of_nodes//5, number_of_nodes//2))
    idx_test = torch.LongTensor(range(number_of_nodes//2, number_of_nodes))

    return features.float(), adj, labels, idx_train, idx_val, idx_test
    


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
  
    

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx



def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    """For Cora, Citeseer, Pubmed"""
    if dataset_str in ['cora', 'citeseer', 'pubmed']:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        number_of_nodes=adj.shape[0]


        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        #features = normalize(features) #cannot converge if use SGD, why??????????
        #adj = normalize(adj)    # no normalize adj here, normalize it in the training process


        features=torch.tensor(features.toarray()).float()
        adj = torch.tensor(adj.toarray())
        adj = torch_sparse.tensor.SparseTensor.from_edge_index(torch_geometric.utils.dense_to_sparse(adj)[0])
        #edge_index=torch_geometric.utils.dense_to_sparse(torch.tensor(adj.toarray()))[0]
        labels=torch.tensor(labels)
        labels=torch.argmax(labels,dim=1)

    """for ogbn arxiv, product, mag 
        https://ogb.stanford.edu/docs/nodeprop/
    """    
    elif dataset_str in ['ogbn-arxiv', 'ogbn-products', 'ogbn-mag', 'ogbn-papers100M']: #'ogbn-mag' is heteregeneous
        #from ogb.nodeproppred import NodePropPredDataset
        from ogb.nodeproppred import PygNodePropPredDataset
        # Download and process data at './dataset/.'
        #dataset = NodePropPredDataset(name = dataset_str, root = 'dataset/')
        dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=torch_geometric.transforms.ToSparseTensor())

        split_idx = dataset.get_idx_split()
        idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        data = dataset[0]
        
        features = data.x #torch.tensor(graph[0]['node_feat'])
        labels = data.y.reshape(-1) #torch.tensor(graph[1].reshape(-1))
        adj = data.adj_t.to_symmetric()
        #edge_index = torch.tensor(graph[0]['edge_index'])
        #adj = torch_geometric.utils.to_dense_adj(torch.tensor(graph[0]['edge_index']))[0]
    return features.float(), adj, labels, idx_train, idx_val, idx_test


"""For Amazon Data"""
from scipy.io import loadmat
import pickle
from sklearn.model_selection import train_test_split

def load_data_for_amazon(data):
    """
	Load graph, feature, and label given dataset name:
    returns: home and single-relation graphs, feature, label
	"""
    prefix = '/home/shakib/Work/Implementation/FedGCN-3/Experiment_on_New_Dataset/data/'
    if data == 'amazon':
        data_file = loadmat('/home/shakib/Work/Implementation/FedGCN-3/Experiment_on_New_Dataset/data/amazon.mat')
        labels = data_file['label'].flatten()
        feat_data = data_file['features'].todense().A
		# load the preprocessed adj_lists
        with open(prefix + 'amz_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        with open(prefix + 'amz_upu_adjlists.pickle', 'rb') as file:
            relation1 = pickle.load(file)
        file.close()
        with open(prefix + 'amz_usu_adjlists.pickle', 'rb') as file:
            relation2 = pickle.load(file)
        file.close()
        with open(prefix + 'amz_uvu_adjlists.pickle', 'rb') as file:
            relation3 = pickle.load(file)
        
        # Changes
        adj = [homo, relation1, relation2, relation3]
        #adj = np.array(adj)
        #adj = [relation1]
        adj = torch.vstack(adj, dim=1).squeeze(0)
        # adj = np.array(adj)
        #adj = torch.tensor(adj)
        adj = torch_sparse.tensor.SparseTensor.from_edge_index(torch_geometric.utils.dense_to_sparse(adj)[0])

        feat_data = torch.tensor(feat_data).float()
        labels=torch.tensor(labels)
        labels=torch.argmax(labels,dim=1)

        index = list(range(3305, len(labels)))
        idx_train, idx_test, y_train, y_test = train_test_split(index, labels[3305:], stratify=labels[3305:],
															test_size=0.60, random_state=2, shuffle=True)
        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_test, labels[3305:], stratify=labels[3305:],
															test_size=0.60, random_state=2, shuffle=True)                                                   


    return feat_data.float(), adj, labels, idx_train, idx_valid, idx_test

from collections import defaultdict

def sparse_to_adjlist(sp_matrix, filename):
	"""
	Transfer sparse matrix to adjacency list
	:param sp_matrix: the sparse matrix
	:param filename: the filename of adjlist
	"""
	# add self loop
	homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
	# create adj_list
	adj_lists = defaultdict(set)
	edges = homo_adj.nonzero()
	for index, node in enumerate(edges[0]):
		adj_lists[node].add(edges[1][index])
		adj_lists[edges[1][index]].add(node)
	with open(filename, 'wb') as file:
		pickle.dump(adj_lists, file)
	file.close()

def sparse_to_adjlist(sp_matrix):

	"""Transfer sparse matrix to adjacency list"""

	#add self loop
	homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
	#creat adj_list
	adj_lists = defaultdict(set)
	edges = homo_adj.nonzero()
	
	for index, node in enumerate(edges[0]):
		adj_lists[node].add(edges[1][index])
		adj_lists[edges[1][index]].add(node)
	adj_lists = {keya:random.sample(adj_lists[keya],10) if len(adj_lists[keya])>=10 else adj_lists[keya] for i, keya in enumerate(adj_lists)}

	return adj_lists

#from fast_ml.model_development import train_valid_test_split
def load_data_fraudre(data):
    
    if data == 'amazon':
        amz = loadmat('/home/shakib/Work/Implementation/FedGCN-3/Experiment_on_New_Dataset/data/amazon.mat')
        homo = sparse_to_adjlist(amz['homo'])
        relation1 = sparse_to_adjlist(amz['net_upu'])
        relation2 = sparse_to_adjlist(amz['net_usu'])
        relation3 = sparse_to_adjlist(amz['net_uvu'])
        
        feat_data = amz['features'].toarray()
        labels = amz['label'].flatten()

         # Changes
        adj = [homo, relation1, relation2, relation3]

        #adj = [relation1]
        #adj = np.vstack(adj)
        # adj = np.array(adj)
        #adj = torch.tensor(adj).view(-1)
        # adj = torch_sparse.tensor.SparseTensor.from_edge_index(torch_geometric.utils.dense_to_sparse(adj))
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(adj[0]))
        adj = torch.tensor(adj.toarray())
        adj = torch_sparse.tensor.SparseTensor.from_edge_index(torch_geometric.utils.dense_to_sparse(adj)[0])
        
        
        feat_data = torch.tensor(feat_data).float()
        labels=torch.tensor(labels)
        #labels=torch.argmax(labels,dim=1)

        index = list(range(3305, len(labels)))
        idx_train, idx_test, y_train, y_test = train_test_split(index, labels[3305:], stratify=labels[3305:],
                                                            test_size=0.60, random_state=2, shuffle=True)
        # idx_valid, idx_test, y_valid, y_test = train_test_split(idx_test, labels[1000:1100], stratify=labels[10000:11000],
        #                                                      test_size=0.60, random_state=2, shuffle=True)                                                   
        # idx_train, idy_train, idx_valid, idy_valid, idx_test, idy_test = train_valid_test_split(index, labels[3305:], 
        #                                                                    train_size=0.8, valid_size=0.1, test_size=0.1)
        
        idx_train = torch.LongTensor(idx_train)
        #idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        idx_train = idx_train.type(torch.LongTensor)
        # idx_test = idx_test.type(torch.LongTensor)

    return feat_data, adj, labels, idx_train, idx_train, idx_test




