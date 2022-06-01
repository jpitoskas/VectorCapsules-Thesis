import torch.nn as nn
import torch
import math
import numpy as np
#
from utils import squash

class CapsLayer(nn.Module):
    def __init__(self, input_caps, input_dim, output_caps, output_dim, routing_module):
        super(CapsLayer, self).__init__()
        self.input_dim = input_dim
        self.input_caps = input_caps
        self.output_dim = output_dim
        self.output_caps = output_caps
        self.weights = nn.Parameter(torch.Tensor(input_caps, input_dim, output_caps * output_dim))
        # print(input_caps, input_dim, output_caps, output_dim)
        self.routing_module = routing_module
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_caps)
        self.weights.data.uniform_(-stdv, stdv)

    def __str__(self):
        str = f"{self.__class__.__name__}(\n \
            input caps: {self.input_caps}\n \
            output_caps = {self.output_caps}\n \
            output_dim = {self.output_dim}\n \
            )"
        return str

    def forward(self, caps_output):
        # print("DigitCaps")
        caps_output = caps_output.unsqueeze(2)
        # print("after unsqueeze(2)",caps_output.shape)
        u_predict = caps_output.matmul(self.weights)
        # print("matmul with weights", self.weights.shape)
        # print("after matmul", u_predict.shape)
        u_predict = u_predict.view(u_predict.size(0), self.input_caps, self.output_caps, self.output_dim)
        # print("reshape before routing", u_predict.shape)
        # exit()
        # print("Before Routing", u_predict.shape)
        v = self.routing_module(u_predict)

        return v

class ConvCapsules2d(nn.Module):
    '''Convolutional Capsule Layer'''
    def __init__(self, in_caps, out_caps, output_dim, kernel_size, stride, padding=0, share_W_ij=False, coor_add=False):
        super().__init__()

        self.B = in_caps
        self.C = out_caps
        self.K = kernel_size
        self.S = stride
        self.padding = padding
        self.output_dim = output_dim

        self.share_W_ij = share_W_ij # share the transformation matrices across (F*F)
        self.coor_add = coor_add # embed coordinates

        # Out ← [1, B, C, 1, P, P, 1, 1, K, K]
        # NEW ← [1, B, C, 1, D, 1, 1, K, K]
        self.W_ij = torch.empty(1, self.B, self.C, 1, self.output_dim, 1, 1, self.K, self.K)

        # Xavier weight init
        fan_in = self.B * self.K*self.K * self.output_dim # in_caps types * receptive field size
        fan_out = self.C * self.K*self.K * self.output_dim # out_caps types * receptive field size
        std = np.sqrt(2. / (fan_in + fan_out))
        bound = np.sqrt(3.) * std

        # Uniform
        self.W_ij = nn.Parameter(self.W_ij.uniform_(-bound, bound))


        if self.padding != 0:
            if isinstance(self.padding, int):
                self.padding = [self.padding]*4

    def __str__(self):
        str = f"{self.__class__.__name__}(\n \
            input capsules: {self.B}\n \
            output_caps = {self.C}\n \
            output_dim = {self.output_dim}\n \
            kernel_size = {self.K}\n \
            stride = {self.S}\n \
            coordinate addition = {self.coor_add}\n \
            )"
        return str


    def forward(self, poses): # ([?, B, F, F], [?, B, P, P, F, F]) ← In
                                           #                [?, B, D, F, F]← NEW


        if self.padding != 0:
#             activations = F.pad(activations, self.padding) # [1,1,1,1]
            poses = F.pad(poses, self.padding + [0]*4) # [0,0,1,1,1,1]

        if self.share_W_ij: # share the matrices over (F*F), if class caps layer
            self.K = poses.shape[-1] # out_caps (C) feature map size

        self.F = (poses.shape[-1] - self.K) // self.S + 1 # featuremap size

        # Out ← [?, B, P, P, F', F', K, K] ← [?, B, P, P, F, F]
        # NEW ←                              [?, B, D, F, F]
        poses = poses.unfold(3, size=self.K, step=self.S).unfold(4, size=self.K, step=self.S)
        # Out ← [?, B, 1, P, P, 1, F', F', K, K] ← [?, B, P, P, F', F', K, K]
        # NEW ← [?, B, 1, D, 1, F', F', K, K]    ← [?, B, D, F', F', K, K]
        poses = poses.unsqueeze(2).unsqueeze(4)


        # Out ← [?, B, C, P, P, F', F', K, K] ← ([?, B, 1, P, P, 1, F', F', K, K] * [1, B, C, 1, P, P, 1, 1, K, K])
        # NEW ← [?, B, C, D, F', F', K, K] ← ([?, B, 1, D, 1, F', F', K, K] * [1, B, C, 1, D, 1, 1, K, K])
        V_ji = (poses * self.W_ij) # matmul equiv.
        # V_ji = V_ji.sum(dim=4)
        V_ji = V_ji.sum(dim=3)

        # Out ← [?, B, C, P*P, 1, F', F', K, K] ← [?, B, C, P, P, F', F', K, K]
        # NEW ← [?, B, C, D, 1, F', F', K, K] ← [?, B, C, D, F', F', K, K]
        V_ji = V_ji.reshape(poses.size(0), self.B, self.C, self.output_dim, 1, *V_ji.shape[-4:-2], self.K, self.K)

        if self.coor_add:
            if V_ji.shape[-1] == 1: # if class caps layer (featuremap size = 1)
                self.F = self.K # 1->4

            # coordinates = torch.arange(self.F, dtype=torch.float32) / self.F
            coordinates = torch.arange(self.F, dtype=torch.float32).add(1.) / (self.F*10)
            i_vals = torch.zeros(self.output_dim,self.F,1).to(V_ji)
            j_vals = torch.zeros(self.output_dim,1,self.F).to(V_ji)
            i_vals[-1,:,0] = coordinates
            j_vals[-2,0,:] = coordinates


            if V_ji.shape[-1] == 1: # if class caps layer
                # Out ← [?, B, C, D, 1, 1, 1, K=F, K=F] (class caps)
                V_ji = V_ji + (i_vals + j_vals).reshape(1,1,1,self.output_dim,1,1,1,self.F,self.F)
                return V_ji.squeeze(4)

            # Out ← [?, B, C, D, 1, F, F, K, K]
            V_ji = V_ji + (i_vals + j_vals).reshape(1,1,1,self.output_dim,1,self.F,self.F,1,1)

        return V_ji.squeeze(4)

class PrimaryCapsLayer(nn.Module):
    def __init__(self, input_channels, output_caps, output_dim, kernel_size, stride, padding=0, dilation=1):
        super(PrimaryCapsLayer, self).__init__()
        self.input_channels = input_channels
        self.output_caps = output_caps
        self.output_dim = output_dim
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)

        self.conv = nn.Conv2d(input_channels, output_caps * output_dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.input_channels = input_channels
        self.output_caps = output_caps
        self.output_dim = output_dim

    def __str__(self):
        str = f"{self.__class__.__name__}(\n \
            input channels: {self.input_channels}\n \
            output_caps = {self.output_caps}\n \
            output_dim = {self.output_dim}\n \
            kernel_size = {self.kernel_size[0]}\n \
            stride = {self.stride[0]}\n \
            )"
        return str

    def forward(self, input):
        out = self.conv(input)
        N, C, H, W = out.size()
        out = out.view(N, self.output_caps, self.output_dim, H, W)

        # will output N x OUT_CAPS x OUT_DIM
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        # exit()
        out = squash(out)
        # exit()
        return out

class FlattenCapsLayer(nn.Module):
    def __init__(self):
        super(FlattenCapsLayer, self).__init__()

    def forward(self, input):
        out = input.reshape(input.size(0), -1, input.size(4))

        return out
