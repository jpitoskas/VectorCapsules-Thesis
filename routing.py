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

    def forward(self, u_predict):

        batch_size, input_caps, output_caps, output_dim = u_predict.size()

        be = torch.bernoulli(torch.full(u_predict.size()[:-1], 1-self.p).unsqueeze(-1)).to(u_predict)
        u_predict = torch.div(be*u_predict, 1-self.p)

        u_predict_norm = u_predict.norm(p=2, dim=3, keepdim=True)
        u_weighted = u_predict*u_predict_norm
        u_weighted_sum = u_weighted.sum(dim=1)
        u_weighted_average = u_weighted_sum / u_predict_norm.sum(dim=1)

        v = squash(u_weighted_average)

        return v




class SubsetRouting(nn.Module):
    def __init__(self, sub=0.8):
        super(SubsetRouting, self).__init__()
        self.sub = sub

    def forward(self, u_predict):

        batch_size, input_caps, output_caps, output_dim = u_predict.size()
        subset_size = math.ceil(self.sub*input_caps)

        v = self.capsule_weighted_average(u_predict)

        if self.sub > 0:
            losses = torch.norm(v.unsqueeze(1) - u_predict, p=2, dim=3)
            loss_threshold, _ = losses.kthvalue(k=subset_size, dim=1)
            loss_threshold = loss_threshold.unsqueeze(1)
            loss_threshold = loss_threshold.expand(losses.size())
            condition = torch.le(losses, loss_threshold)
            choose = torch.where(condition, torch.ones_like(losses), torch.zeros_like(losses))
            choose = choose.unsqueeze(3)
            u_predict = u_predict*choose
        elif sel.sub == 0:
            choose = torch.zeros(batch_size, input_caps, output_caps).unsqueeze(3).to(u_predict)
            u_predict = u_predict*choose
            u_predict += torch.finfo(torch.float64).eps

        v = self.capsule_weighted_average(u_predict)

        return v

    def capsule_weighted_average(self, u_predict):

        u_predict_norm = u_predict.norm(p=2, dim=3, keepdim=True)
        u_weighted = u_predict*u_predict_norm
        u_weighted_sum = u_weighted.sum(dim=1)
        u_weighted_average = u_weighted_sum / u_predict_norm.sum(dim=1)

        return u_weighted_average




class RansacRouting(nn.Module):
    def __init__(self, H=10, sub=0.8):
        super(RansacRouting, self).__init__()
        self.H = H
        self.sub = sub

    def forward(self, u_predict):

        r = self.optimal_hypotheses_matrix(u_predict)
        u_predict = u_predict*r
        v = self.capsule_weighted_average(u_predict)

        return v

    def optimal_hypotheses_matrix(self, V):

        batch_size, input_caps, output_caps, output_dim = V.size()
        subset_size = math.ceil(self.sub*input_caps)

        sample = torch.rand(batch_size, input_caps, output_caps, self.H).topk(subset_size, dim=1).indices.to(V)
        mask = torch.zeros(batch_size, input_caps, output_caps, self.H).to(V)
        r = mask.scatter_(dim=1, index=sample.type(torch.int64), value=1).to(V)

        V_exp = V.unsqueeze(-1).expand(batch_size, input_caps, output_caps, output_dim, self.H)
        Vr = V_exp*r.unsqueeze(3)

        Vr_norm = Vr.norm(p=2, dim=3, keepdim=True)

        Mu = torch.sum(Vr_norm*Vr, dim=1) / Vr_norm.sum(dim=1)

        losses = torch.norm(V_exp - Mu.unsqueeze(1) , p=2, dim=3).sum(dim=1)
        min_losses, min_losses_idxs = losses.min(dim=2)

        r = r.permute(0,2,3,1).reshape(-1, self.H, input_caps)
        min_losses_idxs = min_losses_idxs.reshape(-1,1)
        ar = torch.arange(r.shape[0]).unsqueeze(-1)

        final_r = r[ar, min_losses_idxs].reshape(batch_size, output_caps, 1, input_caps).permute(0,3,1,2)

        return final_r

    def capsule_weighted_average(self, u_predict):

        u_predict_norm = u_predict.norm(p=2, dim=3, keepdim=True)
        u_weighted = u_predict*u_predict_norm
        u_weighted_sum = u_weighted.sum(dim=1)
        u_weighted_average = u_weighted_sum / u_predict_norm.sum(dim=1)

        return u_weighted_average
