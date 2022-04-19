import torch.nn as nn
import math
import torch

from utils import squash

class AgreementRouting(nn.Module):
    def __init__(self, input_caps, output_caps, n_iterations):
        super(AgreementRouting, self).__init__()
        self.n_iterations = n_iterations
        self.b = nn.Parameter(torch.zeros((input_caps, output_caps)))

    def forward(self, u_predict):
        batch_size, input_caps, output_caps, output_dim = u_predict.size()

        c = F.softmax(self.b)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        v = squash(s)

        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, input_caps, output_caps))
            for r in range(self.n_iterations):
                v = v.unsqueeze(1)
                b_batch = b_batch + (u_predict * v).sum(-1)

                c = F.softmax(b_batch.view(-1, output_caps)).view(-1, input_caps, output_caps, 1)
                s = (c * u_predict).sum(dim=1)
                v = squash(s)

        return v


class WeightedAverageRouting(nn.Module):
    def __init__(self):
        super(WeightedAverageRouting, self).__init__()
        # self.b = nn.Parameter(torch.zeros((input_caps, output_caps)))

    def forward(self, u_predict):
        batch_size, input_caps, output_caps, output_dim = u_predict.size()

        # s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        # print(u_predict.size())
        u_predict_norm = u_predict.norm(p=2, dim=3, keepdim=True)
        # print(u_predict_norm.max())
        u_weighted = u_predict*u_predict_norm
        u_weighted_sum = u_weighted.sum(dim=1)
        u_weighted_average = u_weighted_sum / u_predict_norm.sum(dim=1)
        # print("u", u_predict.size())
        # print("u_norm", u_predict_norm.size())
        # print("u_weighted", u_weighted.size())
        # print("u_weighted_sum", u_weighted_sum.size())
        # print("u_weighted_average", u_weighted_average.size())
        # print("")
        # print(out.size())
        # exit()
        # v = squash(s)

        v = u_weighted_average

        return v


class DropoutWeightedAverageRouting(nn.Module):
    def __init__(self, p):
        super(DropoutWeightedAverageRouting, self).__init__()

        self.p = p
        # self.b = nn.Parameter(torch.zeros((input_caps, output_caps)))

    def forward(self, u_predict):

        batch_size, input_caps, output_caps, output_dim = u_predict.size()

        be = torch.bernoulli(torch.full(u_predict.size()[:-1], 1-self.p).unsqueeze(-1)).to(u_predict)
        u_predict = torch.div(be*u_predict, 1-self.p)


        u_predict_norm = u_predict.norm(p=2, dim=3, keepdim=True)
        u_weighted = u_predict*u_predict_norm
        u_weighted_sum = u_weighted.sum(dim=1)
        u_weighted_average = u_weighted_sum / u_predict_norm.sum(dim=1)
        # print("u_weighted", u_weighted.size())
        # print("u_weighted_sum", u_weighted_sum.size())
        # print("u_weighted_average", u_weighted_average.size())
        # print("")
        # print(out.size())
        # exit()
        # v = squash(s)

        v = squash(u_weighted_average)

        # v = squash(v)

        return v


class RansacRouting(nn.Module):
    def __init__(self, n_hypothesis=10, p=0.3):
        super(RansacRouting, self).__init__()
        self.n_hypothesis = n_hypothesis
        self.p = p
        # self.b = nn.Parameter(torch.zeros((input_caps, output_caps)))

    def forward(self, u_predict):
        batch_size, input_caps, output_caps, output_dim = u_predict.size()
        M = math.floor(self.p * input_caps)

        u_predict = u_predict.permute(0, 2, 1, 3)
        print(u_predict.shape)
        # bl = torch.randint(low=0, high=2, size=(5,5))
        print(bl)


        exit()

        # for j in range(output_caps):



        # s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        # print(u_predict.size())
        u_predict_norm = u_predict.norm(p=2, dim=3, keepdim=True)
        u_weighted = u_predict*u_predict_norm
        u_weighted_sum = u_weighted.sum(dim=1)
        u_weighted_average = u_weighted_sum / u_predict_norm.sum(dim=1)
        # print("u", u_predict.size())
        # print("u_norm", u_predict_norm.size())
        # print("u_weighted", u_weighted.size())
        # print("u_weighted_sum", u_weighted_sum.size())
        # print("u_weighted_average", u_weighted_average.size())
        # print("")
        # print(out.size())
        # exit()
        # v = squash(s)

        v = u_weighted_average

        return v
