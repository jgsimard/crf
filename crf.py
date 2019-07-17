import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from permuthohedral_lattice import PermutohedralLattice as pl
import permuthohedral_lattice


def _diagonal_initializer(num_classes):
    return np.eye(num_classes, dtype=np.float32)


def _potts_model_initializer(num_classes):
    return -1 * _diagonal_initializer(num_classes)

class CRF(nn.Module):
    def __init__(self,
                 num_classes = 10,
                 num_spatial_dimentions=3,
                 num_iterations=5,
                 theta_aplha=1.0,
                 theta_beta=1.0,
                 theta_gamma=8.0,
                 trainable = True):
        self.num_classes = num_classes
        self.num_iterations = num_iterations
        self.num_spatial_dimensions = num_spatial_dimentions
        self.theta_aplha = theta_aplha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.pl = pl.apply
        self.spatial_weights = Variable(torch.from_numpy(_diagonal_initializer(num_classes)), requires_grad=trainable)
        self.bilateral_weights = Variable(torch.from_numpy(_diagonal_initializer(num_classes)), requires_grad=trainable)
        self.compatibility_matrix = Variable(torch.from_numpy(_potts_model_initializer(num_classes)), requires_grad=trainable)


    def forward(self, unaries, feat):
        """
        :param unaries: BxCxN
        :param feat: Bx(S+F)xN
        :return:
        """
        q_values = unaries
        for i in range(self.num_iterations):
            q_values = F.softmax(q_values, dim=1)
            # spatial filtering
            spatial_out = self.pl(feat[:,:self.num_spatial_dimensions, :] / self.theta_gamma, q_values)

            # bilateral filtering
            temp_feat = feat
            temp_feat[:,:self.num_spatial_dimensions, :] /=self.theta_aplha
            temp_feat[:, self.num_spatial_dimensions:, :] /= self.theta_beta
            bilateral_out = self.pl(temp_feat, q_values)

            message_passing = torch.mm(self.spatial_weights, spatial_out) \
                              + torch.mm(self.bilateral_weights, bilateral_out)

            pairwise = torch.mm(self.compatibility_matrix, message_passing)

            q_values = unaries - pairwise

        return q_values





