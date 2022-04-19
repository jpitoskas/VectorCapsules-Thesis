import torch.nn as nn
import torch
import math
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
        # print("After Routing", v.shape)
        # exit()
        return v


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

    def forward(self, input):
        out = self.conv(input)
        N, C, H, W = out.size()
        out = out.view(N, self.output_caps, self.output_dim, H, W)

        # will output N x OUT_CAPS x OUT_DIM
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        out = out.view(out.size(0), -1, out.size(4))
        out = squash(out)
        return out
