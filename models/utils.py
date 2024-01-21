import torch
import torch.nn.functional as F
import torch.nn as nn


def adjacent_matrix(labels):

    labels = labels.type(torch.cuda.FloatTensor)
    matrix = torch.zeros((labels.size(0), labels.size(0))).cuda()


    tau = 1
    x1, x2 = labels.unsqueeze(1), labels.unsqueeze(0)
    numerator = torch.exp(-torch.pow(x1 - x2, 2) / tau) # numerator: diagonal == 1
    denominator = torch.sum(numerator, dim = 1).unsqueeze(1)
    matrix = numerator / denominator

    matrix = torch.where(matrix > 1e-6, matrix, torch.zeros(matrix.size()).cuda())

    return matrix, const




















