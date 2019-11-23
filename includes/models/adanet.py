import torch
from torch import nn
from torch.nn import functional as F

from .model import Model

class Adanet(Model):
    def __init__(self, name, loss_fn, input_dim, B):
        super(Adanet, self).__init__(name, loss_fn)

        self.depth = 1
        self.B = B                                                  # number of neurons in each subnetwork layer
        self.layers = [nn.Parameter(torch.randn(input_dim, B))]
        self.in_dims = [input_dim, B]                               # stores the inputs dims for each layer in the network 

    def create_subnetworks(self, depth, B):
        sub = SubNetwork(self.depth, self.B, self.in_dims)
        subA = sub.getA()
        subB = sub.getB()
        return subA, subB

    def add_subnetwork(self, sub_net):                              # returns and appends the subnet to the network
        _network = [nn.Parameter(torch.cat((layer.weight, sub_layer.weight), 1)) \
            for layer, sub_layer in zip(self.layers, sub_net)]
        
        if(len(self.layers) != len(sub_net)):
            _network.append(sub_net[-1])
        return _network
    
    def update_network(self, sub_net):                              # updates the current network
        self.layers = [nn.Parameter(torch.cat((layer.weight, sub_layer.weight), 1)) \
            for layer, sub_layer in zip(self.layers, sub_net)]
        self.in_dims[1:] += self.B
        
        if(len(self.layers) != len(sub_net)):
            self.layers.append(sub_net[-1])
            self.in_dims.append(self.B)
            self.depth += 1

    def train_subnetwork(self):
        raise NotImplementedError

class SubNetwork():
    def __init__(self, depth, B, in_dims):
        self.B = B
        self.depth = depth
        self.in_dims = in_dims
    
    def getA(self):
        sub_net = [nn.Parameter(torch.normal(_dim, self.B)) for _dim in self.in_dims[:-1]]
        return sub_net
    def getB(self):
        sub_net = [nn.Parameter(torch.normal(_dim, self.B)) for _dim in self.in_dims]
        return sub_net