# -*- coding: utf-8 -*-
"""


Created on Thu May  6 09:51:21 2021


Authors:  Martin Voigt Vejling, Simone Birk Bols Thomsen and
          Kristian Søgaard
E-Mails: {mvejli17, sbbt17, ksagaa17}@student.aau.dk

In this script the Baseline model is defined as described in the report

        Fourier Neural Operators for Sound Field Reconstruction
                        in the Time Domain
               - Chapter 8: Numerical Experiments

This model class can be used to train a model in the script
train_FNO_network_2d.py.

The script has been developed using Python 3.6 with the libraries numpy, scipy,
torch 1.7.0, and torchsummary.


"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import operator
from functools import reduce
from functools import partial

torch.manual_seed(0)
np.random.seed(0)

# Complex multiplication
def compl_mul2d(a, b):
    # (batch, in_channel, x, t), (in_channel, out_channel, x, t) -> (batch, out_channel, x, t)
    op = partial(torch.einsum, "bixt,ioxt->boxt")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)

################################################################
# 2d fourier layers
################################################################

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes_x, modes_t):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        This class is not used in the model however is included in order
        to initialize the FNO and the baseline model with the same initial
        parameters.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes_t = modes_t

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes_x, self.modes_t, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes_x, self.modes_t, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.rfft(x, 2, normalized=True, onesided=True)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-2), x.size(-1)//2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes_x, :self.modes_t] = \
            compl_mul2d(x_ft[:, :, :self.modes_x, :self.modes_t], self.weights1)

        out_ft[:, :, -self.modes_x:, :self.modes_t] = \
            compl_mul2d(x_ft[:, :, -self.modes_x:, :self.modes_t], self.weights2)

        # Return to physical space
        x = torch.irfft(out_ft, 2, normalized=True, onesided=True, signal_sizes=(x.size(-2), x.size(-1)))
        return x

class SimpleBlock2d(nn.Module):
    def __init__(self, modes_x, modes_t, width, q_width):
        super(SimpleBlock2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0,
           self.fc1, and self.fc2.
        2. 4 layers of the local linear transformation u' = W u.
            W defined by self.w.
        3. Project from the channel space to the output space by self.fc3 and
           self.fc4.

        Input: The N_a initial sound pressure measurements + 3 locations 
                (p(t_1, x), ..., p(t_{N_a}, x), x, t_u).
                It's a constant function in time, except for the last index.
        Input shape: (batchsize, x=64, y=64, t=40, c=28)
        Output: The sound pressure measurements in the target.
        Output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes_x = modes_x
        self.modes_t = modes_t
        self.width = width
        self.q_width = q_width
        self.fc0 = nn.Linear(28, self.width)
        self.fc1 = nn.Linear(self.width, self.width)
        self.fc2 = nn.Linear(self.width, self.width)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes_x, self.modes_t)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes_x, self.modes_t)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes_x, self.modes_t)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes_x, self.modes_t)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc3 = nn.Linear(self.width, self.q_width)
        self.fc4 = nn.Linear(self.q_width, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_t = x.shape[1], x.shape[2]

        x = self.fc0(x) #(batchsize, x, t, width)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2) #(batchsize, width, x, t)


        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_t)
        x = self.bn0(x2)
        x = F.relu(x)
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_t)
        x = self.bn1(x2)
        x = F.relu(x)
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_t)
        x = self.bn2(x2)
        x = F.relu(x)
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_t)
        x = self.bn3(x2)


        x = x.permute(0, 2, 3, 1) #(batchsize, x, t, width)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x


class Net2d(nn.Module):
    def __init__(self, modes_x, modes_t, width, q_width):
        super(Net2d, self).__init__()

        """
        A wrapper function.
        """

        self.conv1 = SimpleBlock2d(modes_x, modes_t, width, q_width)


    def forward(self, x):
        x = self.conv1(x)
        return x.squeeze()


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

    def summary_of_model(self):
        summary(self.conv1)
