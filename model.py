import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from capslayers import PrimaryCapsLayer, CapsLayer
from routing import  AgreementRouting, WeightedAverageRouting, DropoutWeightedAverageRouting, SubsetRouting, RansacRouting


class CapsNet(nn.Module):
    def __init__(self, args):
        routing_iterations = args.routing_iterations
        # n_classes = args.n_classes
        super(CapsNet, self).__init__()
        self.channels = args.channels
        self.H = args.Hin
        self.W = args.Win
        self.extra_conv = args.extra_conv

        # self.conv1 = nn.Conv2d(in_channels=args.channels, out_channels=256, kernel_size=9, stride=1)
        self.conv1 = nn.Conv2d(in_channels=args.channels, out_channels=256, kernel_size=9, stride=1)
        self.update_output_shape(self.conv1)

        if args.extra_conv:
            self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1)
            self.update_output_shape(self.conv2)


        # exit()
        self.primaryCaps = PrimaryCapsLayer(input_channels=self.channels, output_caps=32, output_dim=8, kernel_size=9, stride=2)  # outputs 6*6
        self.update_output_shape(self.primaryCaps)
        # self.num_primaryCaps = 32 * 6 * 6
        # routing_module = AgreementRouting(self.num_primaryCaps, n_classes, routing_iterations)
        # self.routing_module = WeightedAverageRouting()
        if args.routing_module == "AgreementRouting":
            self.routing_module = AgreementRouting(self.num_primaryCaps, args.n_classes, args.routing_iterations)
        elif args.routing_module == "WeightedAverageRouting":
            self.routing_module = WeightedAverageRouting()
        elif args.routing_module == "DropoutWeightedAverageRouting":
            self.routing_module = DropoutWeightedAverageRouting(p=args.dropout_probability)
        elif args.routing_module == "SubsetRouting":
            self.routing_module = SubsetRouting(sub=args.subset_fraction)
        elif args.routing_module == "RansacRouting":
            self.routing_module = RansacRouting(H=args.n_hypotheses, sub=args.subset_fraction)

        # self.routing_module = RansacRouting()

        # self.convCaps = CapsLayer(input_caps=self.num_capsules, input_dim=self.caps_dim, output_caps=16,
        #                             output_dim=16, routing_module=self.routing_module)
        # self.update_output_shape(self.convCaps)
        print("input caps", self.num_capsules)
        print("input dim", self.caps_dim)
        print("output caps", args.n_classes)
        print("output dim", 16)

        print(self.num_capsules, self.H)
        # exit()

        self.digitCaps = CapsLayer(input_caps=self.num_capsules, input_dim=self.caps_dim, output_caps=args.n_classes,
                                   output_dim=16, routing_module=self.routing_module)
        self.update_output_shape(self.digitCaps)

    def update_output_shape(self, layer):
        if isinstance(layer, nn.Conv2d):
            self.channels = layer.out_channels
            self.H = math.floor((self.H + 2*layer.padding[0]-layer.dilation[0]*(layer.kernel_size[0]-1)-1)/layer.stride[0] + 1)
            self.W = math.floor((self.W + 2*layer.padding[1]-layer.dilation[1]*(layer.kernel_size[1]-1)-1)/layer.stride[1] + 1)
        elif isinstance(layer, PrimaryCapsLayer):
            self.channels = layer.output_caps
            self.H = math.floor((self.H + 2*layer.padding[0]-layer.dilation[0]*(layer.kernel_size[0]-1)-1)/layer.stride[0] + 1)
            self.W = math.floor((self.W + 2*layer.padding[1]-layer.dilation[1]*(layer.kernel_size[1]-1)-1)/layer.stride[1] + 1)
            self.num_capsules = self.channels * self.H * self.W
            self.caps_dim = layer.output_dim
        elif isinstance(layer, CapsLayer):
            self.num_capsules = layer.output_caps
            self.caps_dim = layer.output_dim

    def forward(self, x):
        # print("\nInput:", input.shape)
        x = self.conv1(x)
        x = F.relu(x)
        if self.extra_conv:
            x = self.conv2(x)
            x = F.relu(x)
        # print("after 1st conv", x.shape)
        x = self.primaryCaps(x)
        # print("after PrimaryCaps", x.shape)
        # exit()

        # x = self.convCaps(x)

        x = self.digitCaps(x)
        # print(x.shape)
        # exit()
        probs = x.pow(2).sum(dim=2).sqrt()
        return x, probs


class ReconstructionNet(nn.Module):
    def __init__(self, args, n_dim=16):
        super(ReconstructionNet, self).__init__()
        self.n_dim = n_dim
        self.n_classes = args.n_classes
        self.fc1 = nn.Linear(self.n_dim * self.n_classes, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, args.channels * args.Hin * args.Win)

    def forward(self, x, target):
        mask = Variable(torch.zeros((x.size()[0], self.n_classes)), requires_grad=False)
        if next(self.parameters()).is_cuda:
            mask = mask.cuda()
        mask.scatter_(1, target.view(-1, 1), 1.)
        mask = mask.unsqueeze(2)
        x = x * mask
        x = x.view(-1, self.n_dim * self.n_classes)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class CapsNetWithReconstruction(nn.Module):
    def __init__(self, capsnet, reconstruction_net):
        super(CapsNetWithReconstruction, self).__init__()
        self.capsnet = capsnet
        self.reconstruction_net = reconstruction_net

    def forward(self, x, target):
        x, probs = self.capsnet(x)
        reconstruction = self.reconstruction_net(x, target)
        return reconstruction, probs


class MarginLoss(nn.Module):
    def __init__(self, m_pos, m_neg, lambda_):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_

    def forward(self, lengths, targets, size_average=True):
        t = torch.zeros(lengths.size()).long()
        if targets.is_cuda:
            t = t.cuda()
        t = t.scatter_(1, targets.data.view(-1, 1), 1)
        targets = Variable(t)
        losses = targets.float() * F.relu(self.m_pos - lengths).pow(2) + \
                 self.lambda_ * (1. - targets.float()) * F.relu(lengths - self.m_neg).pow(2)
        return losses.mean() if size_average else losses.sum()


# class VectorCaps(nn.Module):
#
#     if args.with_reconstruction:
#         reconstruction_model = ReconstructionNet(args, model.digitCaps.output_dim)
#         reconstruction_alpha = 0.0005
#         model = CapsNetWithReconstruction(model, reconstruction_model)
