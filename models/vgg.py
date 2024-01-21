
"""vgg in pytorch
[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision
from utils.losses import DualKLLoss, Entropy


cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    # 'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=100):
        super().__init__()

        self.num_classes = num_classes

        vgg = torchvision.models.vgg16_bn(pretrained=True)
        self.features = list(vgg.features.children())[:44]

        self.features = nn.Sequential(*self.features)


        self.embedding = nn.Sequential(
                            nn.Linear(512*7*7, 4096), # 512*8*8 for DR, 512*7*7 for others, 512x1x1 for lung
                            nn.ReLU(),
                            nn.Dropout(),
                            nn.Linear(4096, 4096),
                            nn.ReLU(),
                            nn.Dropout(),
                            )

        self.classifier = nn.Sequential(
                            # nn.Dropout(),
                            # nn.Linear(4096, 4096),
                            # nn.ReLU(),
                            # nn.Dropout(),
                            nn.Linear(4096, num_classes),
                            # nn.Softplus()
                            )

        self.dualkl = DualKLLoss(32, self.num_classes)
        self.ent = Entropy()
        self.unimodal = Unimodal(num_classes = self.num_classes, input_channels = self.num_classes, dist_type = 'Binomial')

    def forward(self, x, label_graph = False, label = False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        emb = self.embedding(x)
        
        emb_matrix = self._embedding(emb, norm = True)

        output = self.classifier(emb)
        

        dualkl_loss, lamb = self.dualkl(emb_matrix, label_graph, label)
        dualkl_loss += self.ent(lamb)

        return output, emb_matrix, emb, dualkl_loss, lamb


    def _embedding(self, emb, norm = True):
        matrix = torch.zeros((emb.size(0), emb.size(0))).cuda()

        if norm:

            sigma = 1
            dist = self._euclideanDist(emb, emb)

            matrix = F.softmax(- dist / (2 * sigma * sigma), dim = 1)
        else:
            tau = 1
            dist = self._euclideanDist(emb, emb)

            denominator = torch.sum(dist - torch.eye(dist.size(0)).cuda(), dim = 1)
            dist = dist / denominator.unsqueeze(1)
            matrix = F.softmax(- dist / tau, dim = 1)

        return matrix

    def _euclideanDist(self, t1, t2):
        dim = len(t1.size())
        sigma = 1
        if dim == 2:
            N, d = t1.size()
            M, _ = t2.size()
            dist = torch.sum(t1 ** 2, -1).unsqueeze(1) - 2 * torch.mm(t1, t2.t()) + torch.sum(t2 ** 2, -1).unsqueeze(0)
            return dist
        elif dim == 3:
            B,N,_=t1.size()
            _,M,_=t2.size()
            dist = -2 * torch.matmul(t1, t2.permute(0, 2, 1))
            dist += torch.sum(t1 ** 2, -1).view(B, N, 1)
            dist += torch.sum(t2 ** 2, -1).view(B, 1, M)
            dist=torch.sqrt(dist)
            return dist
    



def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))

def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16_bn(num_classes):
    return VGG(make_layers(cfg['D'], batch_norm=True), num_classes = num_classes)

def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))




