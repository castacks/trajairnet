"""Modified from https://github.com/alexmonti19/dagnet"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
#         print(input.shape,self.W.shape)
        h = torch.mm(input, self.W)  # matrix multiplication of the matrices
        N = h.size()[0]
#         print("N",N)
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
#         print("a_input",a_input.shape)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
#         print("E",e.shape)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = e #torch.where(adj > 0, e, zero_vec)
        # print("attn_",attention,attention.shape)
        attention = F.softmax(attention, dim=0)
        h_prime = torch.matmul(attention, h)
        # print("attn_",attention)
        # print("h",h)
        # print("h_",h_prime)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime