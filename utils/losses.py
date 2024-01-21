from __future__ import division

import torch
import torch.nn as nn
import json
import os
import sys
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
from shutil import copyfile

class SORD_Loss(_Loss):
    def __init__(self, ranks = None, loss_type = 'ce', metric = 'mae'):
        super(SORD_Loss, self).__init__()

        self.ranks = torch.FloatTensor(ranks).unsqueeze(0).cuda()
        self.loss_type = loss_type
        self.metric = metric

        if loss_type == 'ce':
            self._loss = CELoss()
        elif loss_type == 'kl':
            # self._loss = KLLoss()
            self._loss = nn.KLDivLoss()

    def _sord(self, target):
        # convert target to SORD version
        ranks = self.ranks.expand(target.size(0), self.ranks.size(1)) # [16, 3]
        target = target.unsqueeze(1).expand(target.size(0), self.ranks.size(1)) # [16, 3]
        target = target.type(torch.cuda.FloatTensor)

        # x1 = 1 - torch.exp(-self._metric_loss(target, ranks)) # [16, 3]
        # x2 = torch.sum(1 - torch.exp(-self._metric_loss(target, ranks)), dim = 1).unsqueeze(1) # [16, 1]
        x1 = torch.exp(-self._metric_loss(target, ranks)) # [16, 3]
        x2 = torch.sum(torch.exp(-self._metric_loss(target, ranks)), dim = 1).unsqueeze(1) # [16, 1]
        x2 = x2.expand(x1.size())

        new_target = x1 / x2

        return new_target

    def _metric_loss(self, x, y):
        # metric loss \phi
        if self.metric == 'mae':
            return torch.abs(x - y) # MAE
        elif self.metric == 'mse':
            return torch.sqrt((x - y)**2) # MSE

    def forward(self, pred, target, passive_mask = None):

        new_target = self._sord(target)
        if passive_mask is not None:
            new_target = new_target * passive_mask

        ent = - torch.sum(new_target * torch.log(new_target))
        if self.loss_type == 'ce':
            cls_loss = self._loss(F.softmax(pred, dim = 1), new_target) - ent
            # cls_loss = self._loss(pred, new_target)

        if self.loss_type == 'kl':
            eps = 1e-8
            pred = F.log_softmax(pred, dim = 1)
            new_target = F.log_softmax(new_target, dim = 1)

            # cls_loss = torch.sum(- pred * (torch.log(pred+eps) - torch.log(new_target+eps)))
            # cls_loss = self._loss(pred, 1 - new_target)
            cls_loss = self._loss(pred, new_target)

        return cls_loss


class CELoss(_Loss):

    def forward(self, pred, label):
        label = label.expand(pred.size())
        offset = 1e-10
        bs = label.size(0)
        # loss = - torch.sum(torch.log(pred_matrix + offset).mul(label) + torch.log(1 - pred_matrix + offset).mul(1 - label)) / bs
        loss = - torch.sum(torch.log(pred + offset).mul(label))
        return loss



class Entropy(_Loss):

    def forward(self, pred):
        offset = 0.0000001
        return - torch.mean(pred.mul(torch.log(pred + offset)))


class MeanVarianceLoss(nn.Module):

    def __init__(self, lambda_1, lambda_2, start_age, end_age):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.start_age = start_age
        self.end_age = end_age

    def forward(self, input, target):

        N = input.size()[0]
        target = target.type(torch.FloatTensor).cuda()
        m = nn.Softmax(dim=1)
        p = m(input)
        # mean loss
        a = torch.arange(self.start_age, self.end_age + 1, dtype=torch.float32).cuda()
        mean = torch.squeeze((p * a).sum(1, keepdim=True), dim=1)
        mse = (mean - target)**2
        mean_loss = mse.mean() / 2.0

        # variance loss
        b = (a[None, :] - mean[:, None])**2
        variance_loss = (p * b).sum(1, keepdim=True).mean()
        
        return self.lambda_1 * mean_loss, self.lambda_2 * variance_loss



class DualKLLoss(nn.Module):
    def __init__(self, batch_size, num_classes):
        super(DualKLLoss, self).__init__()
        self.lamb = nn.Parameter(torch.ones(num_classes, batch_size))
        # self.lamb = nn.Parameter(torch.FloatTensor(num_classes, batch_size).fill_(0.01))
        # self.lamb = nn.Parameter(F.sigmoid(torch.ones(num_classes, 1).normal_(mean = 0.1, std = 1)).cuda().repeat(1, batch_size))
        # self.lamb = nn.Parameter(torch.ones(batch_size, 1).repeat(1, batch_size))
        self.batch_size = batch_size

    def forward(self, emb_matrix, label_matrix, label):
        unique_label, inverse_indices = torch.unique(label, sorted = True, return_inverse = True)

        if self.batch_size != emb_matrix.shape[0]:
            bs = emb_matrix.size(0)
        else:
            bs = self.batch_size

        kl = 0
        for i in unique_label:
            selects = torch.where(label == i)
            mean_label = torch.mean(label_matrix[selects], dim = 0, keepdim = True)
            mean_emb = torch.mean(emb_matrix[selects], dim = 0, keepdim = True)

            tau_emb_lamb = mean_emb * torch.exp(- self.lamb[i, :bs]/0.5)
            tau_emb_lamb = tau_emb_lamb / tau_emb_lamb.view(tau_emb_lamb.shape[0], -1).sum(dim = 1)

            tau_label_lamb = mean_label * torch.exp(self.lamb[i, :bs])
            tau_label_lamb = tau_label_lamb / tau_label_lamb.view(tau_label_lamb.shape[0], -1).sum(dim = 1)

            kl += F.kl_div(tau_emb_lamb.log(), tau_label_lamb, reduction = 'batchmean')

        return kl, self.lamb




        














