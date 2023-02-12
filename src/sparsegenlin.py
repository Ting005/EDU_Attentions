import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.entmax.activations import sparsemax, entmax15
from torch.autograd import Variable


class Sparsegen_lin(torch.nn.Module):
    def __init__(self, lam, data_driven=False, normalized=True):
        super(Sparsegen_lin, self).__init__()
        self.lam = lam
        self.data_driven = data_driven
        self.normalized = normalized

    def forward(self, input, rev_mask):
        bs = input.data.size()[0]
        dim = input.data.size()[1]
        dtype = torch.cuda.FloatTensor
        z = input.type(dtype)
        # sort z
        z_sorted = torch.sort(z, descending=True)[0]
        # calculate k(z)
        z_cumsum = torch.cumsum(z_sorted, dim=1)
        k = Variable(torch.arange(1, dim + 1).unsqueeze(0).repeat(bs, 1)).to(z.device)
        z_check = torch.gt(1 - self.lam + k * z_sorted, z_cumsum)

        # because the z_check vector is always [1,1,...1,0,0,...0] finding the
        # (index + 1) of the last `1` is the same as just summing the number of 1.
        k_z = torch.sum(z_check.float(), 1)
        # calculate tau(z)
        # pdb.set_trace()
        # applying mask here ?
        mask = 1 - rev_mask.float()
        # tausum = torch.sum(z_check.float() * z_sorted, 1)
        # mask out '-inf'
        # pdb.set_trace()
        z[torch.isinf(z)] = .0
        z_sorted[torch.isinf(z_sorted)] = .0
        tausum = torch.sum(z_check.float() * z_sorted, 1)
        tau_z = (tausum - 1 + self.lam) / k_z
        # pdb.set_trace()
        prob = z.sub(tau_z.view(bs, 1).repeat(1, dim)).clamp(min=0).type(dtype)
        prob = prob * mask
        if self.normalized:
            prob /= (1 - self.lam)
        # pdb.set_trace()
        # # modify with inf mask
        # # torch.isinf
        # for i_idx, i_len in enumerate(len_s.cpu().int().tolist()):
        #     bs, dim = 1, i_len
        #     z_i = z[i_idx, 0: i_len].unsqueeze(dim=0)
        #     z_i_sorted = torch.sort(z_i, descending=True)[0]
        #     z_i_cumsum = torch.cumsum(z_i_sorted, dim=1)
        #     k = Variable(torch.arange(1, dim + 1).unsqueeze(0).repeat(bs, 1))
        #     z_i_check = torch.gt(1 - self.lam + k * z_i_sorted, z_i_cumsum)
        #     k_i_z = torch.sum(z_i_check.float(), 1)
        #     pdb.set_trace()
        #     tausum_i = torch.sum(z_i_check.float() * z_i_sorted, 1)
        #     tau_z_i = (tausum_i - 1 + self.lam) / k_i_z
        #     prob_i = z_i.sub(tau_z_i.view(bs, 1).repeat(1, dim)).clamp(min=0).type(dtype)
        #     prob_i /= (1 - self.lam)
        #     output_prob[i_idx, 0: i_len] = prob_i
        return prob