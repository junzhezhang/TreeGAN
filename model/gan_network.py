import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.gcn import TreeGCN

from math import ceil

class Discriminator(nn.Module):
    def __init__(self, batch_size, features):
        # import pdb; pdb.set_trace()
        self.batch_size = batch_size
        self.layer_num = len(features)-1
        super(Discriminator, self).__init__()

        self.fc_layers = nn.ModuleList([])
        for inx in range(self.layer_num):
            self.fc_layers.append(nn.Conv1d(features[inx], features[inx+1], kernel_size=1, stride=1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        #jz below code got problem, linearity, and final sigmoid,  
        #jz TODO final softmax/sigmoid needed?
        # self.final_layer = nn.Sequential(nn.Linear(features[-1], features[-1]),
        #                                  nn.Linear(features[-1], features[-2]),
        #                                  nn.Linear(features[-2], features[-2]),
        #                                  nn.Linear(features[-2], 1))
        
        # follow the r-GAN discriminator
        # jz NOTE below got Sigmoid function
        self.final_layer = nn.Sequential(
                    nn.Linear(features[-1], 128),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(128, 64),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(64, 1),
                    nn.Sigmoid())

    def forward(self, f):
        
        # feat shape (B,3,2048)
        feat = f.transpose(1,2)
        vertex_num = feat.size(2)

        for inx in range(self.layer_num):
            feat = self.fc_layers[inx](feat)
            feat = self.leaky_relu(feat)
        # import pdb; pdb.set_trace()
        # feat shape (B,dimension,2048) --> out (B,dimension)
        out = F.max_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)
        # out (B,1)
        out = self.final_layer(out) # (B, 1)
        # import pdb; pdb.set_trace()
        return out


class Generator(nn.Module):
    def __init__(self, batch_size, features, degrees, support):
        self.batch_size = batch_size
        self.layer_num = len(features)-1
        assert self.layer_num == len(degrees), "Number of features should be one more than number of degrees."
        self.pointcloud = None
        super(Generator, self).__init__()
        
        vertex_num = 1
        self.gcn = nn.Sequential()
        for inx in range(self.layer_num):
            #jz NOTE last layer activation False
            if inx == self.layer_num-1:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(self.batch_size, inx, features, degrees, 
                                            support=support, node=vertex_num, upsample=True, activation=False))
            else:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(self.batch_size, inx, features, degrees, 
                                            support=support, node=vertex_num, upsample=True, activation=True))
            vertex_num = int(vertex_num * degrees[inx])

    def forward(self, tree):
        # shape of feat[i] (B,nodes,features)
        # [torch.Size([64, 1, 96]), torch.Size([64, 1, 256]), torch.Size([64, 2, 256]), torch.Size([64, 4, 256]), 
        # torch.Size([64, 8, 128]), torch.Size([64, 16, 128]), torch.Size([64, 32, 128]), torch.Size([64, 2048, 3])]
        feat = self.gcn(tree)
        
        self.pointcloud = feat[-1]
        # import pdb; pdb.set_trace()
        return self.pointcloud

    def getPointcloud(self):
        return self.pointcloud[-1]