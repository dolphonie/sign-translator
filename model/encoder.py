# Created by Patrick Kao
import torch
from torch import nn

from model.transformer_utils import generate_padding_mask, PositionalEncoding


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, config.encoder_layers)
        self.pos_embed = PositionalEncoding(config.frame_embed_dim)

    def forward(self, frame_embeddings, lengths):
        """

        :param frame_embeddings: Shape: batch x time x feature
        :param lengths:
        :return:
        """
        padding_mask = generate_padding_mask(frame_embeddings, lengths).to(frame_embeddings.device)
        # transformer_encoder wants batch second
        encoder_input = frame_embeddings.permute(1, 0, 2)
        encoder_input = self.pos_embed(encoder_input)
        encoded = self.transformer_encoder(src=encoder_input,
                                           src_key_padding_mask=padding_mask)
        return encoded, padding_mask
