import torch
import torch.nn as nn


class Mask(nn.Module):
    def __init__(self, num, K=2, sample_ratio=0.2, if_tem=False):
        super(Mask, self).__init__()
        if if_tem:
            self.M = nn.Parameter(torch.FloatTensor(K, num, num))
        else:
            self.M = nn.Parameter(torch.FloatTensor(K, num))
        self.reset_parameters()
        
        num_sample = int(num * sample_ratio)
        self.row_idx = torch.arange(0, num).unsqueeze(-1).repeat(1, num_sample)
        self.K = K if sample_ratio > 0 else 1
        self.if_tem = if_tem
        self.num = num
        self.num_sample = num_sample
        
    def reset_parameters(self):
        nn.init.uniform_(self.M)

    def forward(self):
        P = torch.softmax(self.M if self.if_tem else self.M.unsqueeze(1).repeat(1, self.num, 1), dim=-1) # prob. (k, num, num)
        M = torch.ones(self.K, self.num, self.num, dtype=torch.float).to(self.M.device) # masks
        log_p = 0
        for i in range(self.K):
            S = torch.multinomial(P[i], num_samples=self.num_sample) # action
            M[i, self.row_idx, S[i]] = 0.
            log_p += torch.sum(P[i, self.row_idx, S[i]])

        return M, log_p
    