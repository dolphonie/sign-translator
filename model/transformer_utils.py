# Created by Patrick Kao


import torch


# from http://juditacs.github.io/2018/12/27/masked-attention.html
def generate_padding_mask(sequence, lengths):
    len_tensor = torch.tensor(lengths)
    batch_size = sequence.shape[0]
    maxlen = sequence.shape[1]

    idx = torch.arange(maxlen).unsqueeze(0).expand(batch_size)
    len_expanded = len_tensor.unsqueeze(1).expand(maxlen)
    mask = idx > len_expanded  # transformer mask should be true if padding
    return mask == 1
