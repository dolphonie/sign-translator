# Created by Patrick Kao
from torch import nn

from model.transformer_utils import generate_padding_mask


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, config.encoder_layers)

    def forward(self, frame_embeddings, lengths):
        """

        :param frame_embeddings: Shape: batch x time x feature
        :param lengths:
        :return:
        """
        padding_mask = generate_padding_mask(frame_embeddings, lengths)

        # transformer_encoder wants batch second
        encoded = self.transformer_encoder(src=frame_embeddings.permute(1, 0, 2),
                                           src_key_padding_mask=padding_mask)
        return encoded, padding_mask
