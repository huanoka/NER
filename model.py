import torch
import torch.nn as nn
from torch.nn.modules import LSTMCell
from transformers import BertTokenizer, BertPreTrainedModel


class NERModel(nn.Module):
    def __init__(self, config):
        super(NERModel).__init__()
        self.bert = BertPreTrainedModel.from_pretrained('bert-base-chinese')


class LSTMLayer(nn.Module):
    # Ct: Cell_state: 不会输出， 记录cell记忆的东西
    # Ht: Hidden State: 本cell输出的hidden_state
    # Xt: 当前输入的向量
    # Xt: [B * EMB_D] Ht: [B * HID_D]  ==> Wf:[ , emb_d + hid_d]    Ft: []
    # Ct: []
    # W_f W_i, W_c, W_v: [emb_d + hid_d, hid_d]

    def __init__(self, e_dim, h_dim, sigma):
        super(LSTMLayer).__init__()
        self.W_f = nn.Parameter(torch.rand(e_dim + h_dim, h_dim))
        self.b_f = nn.Parameter(torch.rand(h_dim))
        self.W_i = nn.Parameter(torch.rand(e_dim + h_dim, h_dim))
        self.b_i = nn.Parameter(torch.rand(h_dim))
        self.W_c = nn.Parameter(torch.rand(e_dim + h_dim, h_dim))
        self.b_c = nn.Parameter(torch.rand(h_dim))
        self.W_o = nn.Parameter(torch.rand(e_dim + h_dim, h_dim))
        self.b_o = nn.Parameter(torch.rand(h_dim))
        self.c = torch.rand(h_dim)
        self.sigma = sigma

    def forward(self, x_input: torch.Tensor):
        c_pre = self.c
        concated = torch.concat((x_input, c_pre))
        f = self.sigma * (torch.mul(concated, self.W_f) + self.b_f)
        i = self.sigma * (torch.mul(concated, self.W_i) + self.b_i)
        c = torch.tanh_(torch.mul(concated, self.W_c) + self.b_c)
        o = self.sigma * (torch.mul(concated, self.W_o) + self.b_o)
        c = torch.mul(f, c_pre) + torch.mul(i, c)
        self.c = c
        h_out = torch.mul(o, torch.tanh_(c))
        return h_out
