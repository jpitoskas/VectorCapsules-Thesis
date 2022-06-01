import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging


from capslayers import PrimaryCapsLayer, CapsLayer, FlattenCapsLayer, ConvCapsules2d
from routing import  AgreementRouting, WeightedAverageRouting, DropoutWeightedAverageRouting, SubsetRouting, ConvSubsetRouting, RansacRouting
from routing import ConvWeightedAverageRouting
from utils import squash


class CapsNet(nn.Module):
    def __init__(self, args):
        routing_iterations = args.routing_iterations
        # n_classes = args.n_classes
        super(CapsNet, self).__init__()
        self.channels = args.channels
        self.H = args.Hin
        self.W = args.Win
        self.extra_conv = args.extra_conv
        # self.extra_caps = args.extra_caps
        self.extra_caps_layer = args.extra_caps_layer
        self.extra_convCaps_layer = args.extra_convCaps_layer

        # self.conv1 = nn.Conv2d(in_channels=args.channels, out_channels=256, kernel_size=9, stride=1)
        self.conv1 = nn.Conv2d(in_channels=args.channels, out_channels=256, kernel_size=9, stride=1)
        logging.info(self.conv1)
        self.update_output_shape(self.conv1)

        if args.extra_conv:
            self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1)
            self.update_output_shape(self.conv2)



        self.primaryCaps = PrimaryCapsLayer(input_channels=self.channels, output_caps=32, output_dim=8, kernel_size=9, stride=2)  # outputs 6*6
        logging.info(self.primaryCaps)
        self.update_output_shape(self.primaryCaps)

        if self.extra_convCaps_layer:
            self.convCaps1 = ConvCapsules2d(in_caps=self.num_capsules, out_caps=32, output_dim=16,
                                            kernel_size=3, stride=2)
            logging.info(self.convCaps1)
            self.convRouting1 = self.get_routing(args, forConvCaps=True)
            self.update_output_shape(self.convCaps1)

            if self.extra_convCaps_layer == 2:
                self.convCaps2 = ConvCapsules2d(in_caps=self.num_capsules, out_caps=32, output_dim=16,
                                                kernel_size=3, stride=1)
                logging.info(self.convCaps2)
                self.convRouting2 = self.get_routing(args, forConvCaps=True)
                self.update_output_shape(self.convCaps2)

        self.flattenCaps = FlattenCapsLayer()
        self.update_output_shape(self.flattenCaps)

        if self.extra_caps_layer:
            self.caps1 = CapsLayer(input_caps=self.num_capsules, input_dim=self.caps_dim, output_caps=64,
                                        output_dim=16, routing_module=self.get_routing(args, forConvCaps=False))
            logging.info(self.caps1)
            self.update_output_shape(self.caps1)

            if self.extra_caps_layer == 2:
                self.caps2 = CapsLayer(input_caps=self.num_capsules, input_dim=self.caps_dim, output_caps=32,
                                            output_dim=16, routing_module=self.get_routing(args, forConvCaps=False))
                logging.info(self.caps2)
                self.update_output_shape(self.caps2)











        # self.num_primaryCaps = 32 * 6 * 6
        # routing_module = AgreementRouting(self.num_primaryCaps, n_classes, routing_iterations)
        # self.routing_module = WeightedAverageRouting()

            # if args.routing_module == "AgreementRouting":
            #     self.routing_module = AgreementRouting(self.num_primaryCaps, args.n_classes, args.routing_iterations)
            # elif args.routing_module == "WeightedAverageRouting":
            #     self.routing_module = WeightedAverageRouting()
            # elif args.routing_module == "DropoutWeightedAverageRouting":
            #     self.routing_module = DropoutWeightedAverageRouting(p=args.dropout_probability)
            # elif args.routing_module == "SubsetRouting":
            #     self.routing_module = SubsetRouting(sub=args.subset_fraction, metric=args.metric)
            # elif args.routing_module == "RansacRouting":
            #     self.routing_module = RansacRouting(H=args.n_hypotheses, sub=args.subset_fraction, metric=args.metric)


            # if args.extra_caps ==  "caps" or not args.extra_caps:
            #     self.flattenCaps = FlattenCapsLayer()
            #     self.update_output_shape(self.flattenCaps)

        # self.routing_module = RansacRouting()

        # self.convCaps = CapsLayer(input_caps=self.num_capsules, input_dim=self.caps_dim, output_caps=16,
        #                             output_dim=16, routing_module=self.routing_module)
        # self.update_output_shape(self.convCaps)

        # print("input caps", self.num_capsules)
        # print("input dim", self.caps_dim)
        # print("output caps", args.n_classes)
        # print("output dim", 16)

        # print(self.num_capsules, self.H)
        # exit()

            # if args.extra_caps == "convCaps":
            #
            #     if args.routing_module == "AgreementRouting":
            #         self.extra_routing_module = AgreementRouting(self.num_primaryCaps, args.n_classes, args.routing_iterations)
            #     elif args.routing_module == "WeightedAverageRouting":
            #         self.extra_routing_module = ConvWeightedAverageRouting()
            #         # self.extra_routing_module = WeightedAverageRouting()
            #     elif args.routing_module == "DropoutWeightedAverageRouting":
            #         self.extra_routing_module = DropoutWeightedAverageRouting(p=args.dropout_probability)
            #     elif args.routing_module == "SubsetRouting":
            #         self.extra_routing_module = SubsetRouting(sub=args.subset_fraction, metric=args.metric)
            #     elif args.routing_module == "RansacRouting":
            #         self.extra_routing_module = RansacRouting(H=args.n_hypotheses, sub=args.subset_fraction, metric=args.metric)
            #
            #
            #     self.convCaps = ConvCapsules2d(in_caps=self.num_capsules, out_caps=32, output_dim=16,
            #                                     kernel_size=3, stride=1)
            #     logging.info(self.convCaps)
            #     self.extra_routing_module = ConvWeightedAverageRouting()
            #     self.update_output_shape(self.convCaps)
            #
            #
            #     self.flattenCaps = FlattenCapsLayer()
            #     self.update_output_shape(self.flattenCaps)



            # elif args.extra_caps == "caps":
            #
            #     if args.routing_module == "AgreementRouting":
            #         self.extra_routing_module = AgreementRouting(self.num_primaryCaps, args.n_classes, args.routing_iterations)
            #     elif args.routing_module == "WeightedAverageRouting":
            #         self.extra_routing_module = WeightedAverageRouting()
            #     elif args.routing_module == "DropoutWeightedAverageRouting":
            #         self.extra_routing_module = DropoutWeightedAverageRouting(p=args.dropout_probability)
            #     elif args.routing_module == "SubsetRouting":
            #         self.extra_routing_module = SubsetRouting(sub=args.subset_fraction, metric=args.metric)
            #     elif args.routing_module == "RansacRouting":
            #         self.extra_routing_module = RansacRouting(H=args.n_hypotheses, sub=args.subset_fraction, metric=args.metric)
            #
            #     self.caps = CapsLayer(input_caps=self.num_capsules, input_dim=self.caps_dim, output_caps=64,
            #                                 output_dim=16, routing_module=self.extra_routing_module)
            #     logging.info(self.caps)
            #     self.update_output_shape(self.caps)



        self.digitCaps = CapsLayer(input_caps=self.num_capsules, input_dim=self.caps_dim, output_caps=args.n_classes,
                                    output_dim=16, routing_module=self.get_routing(args, forConvCaps=False))
        logging.info(self.digitCaps)
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
            self.num_capsules = self.channels
            self.caps_dim = layer.output_dim
        elif isinstance(layer, CapsLayer):
            self.num_capsules = layer.output_caps
            self.caps_dim = layer.output_dim
        elif isinstance(layer, FlattenCapsLayer):
            self.num_capsules = self.num_capsules * self.H * self.W
        elif isinstance(layer, ConvCapsules2d):
            self.H = math.floor((self.H + 2*layer.padding-(layer.K-1)-1)/layer.S + 1)
            self.W = math.floor((self.W + 2*layer.padding-(layer.K-1)-1)/layer.S + 1)
            self.num_capsules = layer.C
            self.caps_dim = layer.output_dim

    def get_routing(self, args, forConvCaps):
        # forConvCaps is boolean if layer before routing is ConvCaps or not

        if args.routing_module == "AgreementRouting":
            if forConvCaps:
                raise NotImplementedError("ConvAgreementRouting")
            else:
                routing_module = AgreementRouting(self.num_primaryCaps, args.n_classes, args.routing_iterations)
        elif args.routing_module == "WeightedAverageRouting":
            if forConvCaps:
                routing_module = ConvWeightedAverageRouting()
            else:
                routing_module = WeightedAverageRouting()
        elif args.routing_module == "DropoutWeightedAverageRouting":
            if forConvCaps:
                raise NotImplementedError("ConvDropoutWeightedAverageRouting")
            else:
                routing_module = DropoutWeightedAverageRouting(p=args.dropout_probability)
        elif args.routing_module == "SubsetRouting":
            if forConvCaps:
                routing_module = ConvSubsetRouting(sub=args.subset_fraction, metric=args.metric)
            else:
                routing_module = SubsetRouting(sub=args.subset_fraction, metric=args.metric)
        elif args.routing_module == "RansacRouting":
            if forConvCaps:
                raise NotImplementedError("ConvRansacRouting")
            else:
                routing_module = RansacRouting(H=args.n_hypotheses, sub=args.subset_fraction, metric=args.metric)

        return routing_module


    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)

        if self.extra_conv:
            x = self.conv2(x)
            x = F.relu(x)

        x = self.primaryCaps(x)

        if self.extra_convCaps_layer:
            x = x.permute(0,1,4,2,3)

            x = self.convCaps1(x)
            x = self.convRouting1(x)

            if self.extra_convCaps_layer == 2:
                x = self.convCaps2(x)
                x = self.convRouting2(x)

            x = x.permute(0,1,3,4,2)


        x = self.flattenCaps(x)

        if self.extra_caps_layer:
            x = self.caps1(x)
            if self.extra_caps_layer == 2:
                x = self.caps2(x)





        # if self.extra_caps == "convCaps":
        #     x = x.permute(0,1,4,2,3)
        #     x = self.convCaps(x)
        #     x = self.extra_routing_module(x)
        #     x = x.permute(0,1,3,4,2)

        #
        # x = self.flattenCaps(x)
        #
        # if self.extra_caps == "caps":
        #     x = self.caps(x)

        x = self.digitCaps(x)
        x = squash(x)


        probs = x.pow(2).sum(dim=2).sqrt()
        return x, probs


class ConvCapsNet(nn.Module):
    def __init__(self, args):
        # routing_iterations = args.routing_iterations
        # n_classes = args.n_classes
        super(ConvCapsNet, self).__init__()
        self.channels = args.channels
        self.H = args.Hin
        self.W = args.Win
#         self.extra_conv = args.extra_conv
#         self.extra_caps = args.extra_caps

        self.conv1 = nn.Conv2d(in_channels=args.channels, out_channels=32, kernel_size=5, stride=2)
        logging.info(self.conv1)
        self.update_output_shape(self.conv1)

#         if args.extra_conv:
#             self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1)
#             self.update_output_shape(self.conv2)




        self.primaryCaps = PrimaryCapsLayer(input_channels=self.channels, output_caps=32, output_dim=16, kernel_size=1, stride=1)  # outputs 6*6
        logging.info(self.primaryCaps)
        self.update_output_shape(self.primaryCaps)




        self.convCaps1 = ConvCapsules2d(in_caps=self.num_capsules, out_caps=32, output_dim=16,
                                        kernel_size=3, stride=2)
        logging.info(self.convCaps1)
        self.convRouting1 = self.get_routing(args, forConvCaps=True)
        self.update_output_shape(self.convCaps1)


        self.convCaps2 = ConvCapsules2d(in_caps=self.num_capsules, out_caps=32, output_dim=16,
                                        kernel_size=3, stride=1)
        logging.info(self.convCaps2)
        self.convRouting2 = self.get_routing(args, forConvCaps=True)
        self.update_output_shape(self.convCaps2)


        self.digitCaps = ConvCapsules2d(in_caps=self.num_capsules, out_caps=args.n_classes, output_dim=16,
                                        kernel_size=1, stride=1, share_W_ij=True, coor_add=True)
        logging.info(self.digitCaps)
        self.digitRouting = self.get_routing(args, forConvCaps=True)
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
            self.num_capsules = self.channels
            self.caps_dim = layer.output_dim
        elif isinstance(layer, CapsLayer):
            self.num_capsules = layer.output_caps
            self.caps_dim = layer.output_dim
        elif isinstance(layer, FlattenCapsLayer):
            self.num_capsules = self.num_capsules * self.H * self.W
        elif isinstance(layer, ConvCapsules2d):
            self.H = math.floor((self.H + 2*layer.padding-(layer.K-1)-1)/layer.S + 1)
            self.W = math.floor((self.W + 2*layer.padding-(layer.K-1)-1)/layer.S + 1)
            self.num_capsules = layer.C
            self.caps_dim = layer.output_dim

    def get_routing(self, args, forConvCaps):
        # forConvCaps is boolean if layer before routing is ConvCaps or not

        if args.routing_module == "AgreementRouting":
            if forConvCaps:
                raise NotImplementedError("ConvAgreementRouting")
            else:
                routing_module = AgreementRouting(self.num_primaryCaps, args.n_classes, args.routing_iterations)
        elif args.routing_module == "WeightedAverageRouting":
            if forConvCaps:
                routing_module = ConvWeightedAverageRouting()
            else:
                routing_module = WeightedAverageRouting()
        elif args.routing_module == "DropoutWeightedAverageRouting":
            if forConvCaps:
                raise NotImplementedError("ConvDropoutWeightedAverageRouting")
            else:
                routing_module = DropoutWeightedAverageRouting(p=args.dropout_probability)
        elif args.routing_module == "SubsetRouting":
            if forConvCaps:
                routing_module = ConvSubsetRouting(sub=args.subset_fraction, metric=args.metric)
            else:
                routing_module = SubsetRouting(sub=args.subset_fraction, metric=args.metric)
        elif args.routing_module == "RansacRouting":
            if forConvCaps:
                raise NotImplementedError("ConvRansacRouting")
            else:
                routing_module = RansacRouting(H=args.n_hypotheses, sub=args.subset_fraction, metric=args.metric)

        return routing_module


    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.primaryCaps(x)



        x = x.permute(0,1,4,2,3)

        x = self.convCaps1(x)       # OUT = [?, B, C, D, F, F, K, K])
        x = self.convRouting1(x)    # OUT = [?, C, D, F, F])


        x = self.convCaps2(x)
        x = self.convRouting2(x)

        x = self.digitCaps(x)
        x = self.digitRouting(x)

        x = x.squeeze(-1).squeeze(-1)


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

class ReconstructionLoss(nn.Module):
    def __init__(self, reconstruction_alpha):
        super(ReconstructionLoss, self).__init__()
        self.reconstruction_alpha = reconstruction_alpha


    def forward(self, output, data):
        reconstruction_loss = F.mse_loss(output, data.view(-1, output.size(1)))
        reconstruction_loss = self.reconstruction_alpha * reconstruction_loss
        return reconstruction_loss

class MixedLoss(nn.Module):
    def __init__(self, margin_loss, reconstruction_loss):
        super(MixedLoss, self).__init__()
        self.margin_loss = margin_loss
        self.reconstruction_loss = reconstruction_loss

    def forward(self, lengths, targets, output, data, size_average=True):
        loss = self.margin_loss(lengths, targets, size_average) + self.reconstruction_loss(output, data)
        return loss
