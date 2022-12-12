# The file contains the main GCN model.
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, NumLayers):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(nfeat, nhid, normalize=True, cached=True))
        for _ in range(NumLayers - 2):
            self.convs.append(
                GCNConv(nhid, nhid, normalize=True, cached=True))
            self.convs.append(
                GCNConv(float(nhid), float(nclass), normalize=True, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)
    
'''   

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, NumLayers):
        super(GCN, self).__init__()
        
        self.dropout = dropout
        self.gcns = nn.ModuleList()
        self.num_layers = NumLayers
        #first layer
        if NumLayers <= 1:
            self.gcns.append(GCNConv(nfeat, nclass, normalize=False))
        else:
            self.gcns.append(GCNConv(nfeat, nhid, normalize=False))
            #remaining layers                 
            for i in range(1, NumLayers-1):
                self.gcns.append(GCNConv(nhid, nhid, normalize=False))
            self.gcns.append(GCNConv(nhid, nclass, normalize=False))
    
    def reset_parameters(self):
        for gcn in self.gcns:
            gcn.reset_parameters()
            
    def forward(self, x, adj): #sparse adj
        if self.num_layers > 1:
            for i in range(self.num_layers-1):                     
                x = F.relu(self.gcns[i](x, adj))
                x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcns[-1](x, adj)
        return F.log_softmax(x, dim=1)
''' 
