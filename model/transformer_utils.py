# Created by Patrick Kao
import math

import torch
from torch import nn

# from http://juditacs.github.io/2018/12/27/masked-attention.html
def generate_padding_mask(sequence, lengths):
    len_tensor = torch.tensor(lengths)
    batch_size = sequence.shape[0]
    maxlen = sequence.shape[1]

    idx = torch.arange(maxlen).unsqueeze(0).expand(batch_size, -1)
    len_expanded = len_tensor.unsqueeze(1).expand(-1, maxlen)
    mask = idx > len_expanded  # transformer mask should be true if padding
    return mask == 1


# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """

        :param x: shape time x batch x dim
        :return:
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
