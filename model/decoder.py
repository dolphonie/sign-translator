# Created by Patrick Kao
# Created by Patrick Kao
import torch
from torch import nn

from model.language_model import LanguageModel


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, config.decoder_layers)
        self.language_model = LanguageModel()

    def forward(self, encoder_output, encoder_padding, target_sequence):
        """

        :param frame_embeddings: Shape: batch x time x feature
        :param lengths:
        :return:
        """
        tokens, attention_padding = self.language_model.tokenize(target_sequence)
        # tokens shape: batch x sequence_len
        attention_padding = ~attention_padding  # pytorch uses reverse convention
        target_embeddings = None

        # target mask: prevent decoder from looking into future
        max_target_len = tokens.shape[1]
        tgt_mask = torch.triu(
            torch.ones(max_target_len, max_target_len)) == 1  # upper because pytorch backward
        # transformer_decoder wants batch second
        decoded = self.transformer_decoder(tgt=None,
                                           memory=encoder_output,
                                           tgt_mask=tgt_mask,
                                           tgt_key_padding_mask=attention_padding,
                                           memory_key_padding_mask=encoder_padding)
        return decoded
