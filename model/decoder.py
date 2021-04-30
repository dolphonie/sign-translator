# Created by Patrick Kao
# Created by Patrick Kao
import copy

import numpy as np
import torch
from torch import nn

from model.language_model import LanguageModel
from model.transformer_utils import PositionalEncoding


def embedding_to_logits(embedding, embed_matrix):
    """

    :param embedding: time x batch x embed
    :param embed_matrix: vocab x embed
    :return:
    """
    # get word logits from embedding space
    embed_broadcastable = embed_matrix.unsqueeze(0).unsqueeze(0)  # 1x1x vocab x embed
    # without broadcasting: want vocab x embed @ embed x 1
    logits = embed_broadcastable @ embedding.unsqueeze(-1)
    return logits


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, config.decoder_layers)
        self.language_model = LanguageModel()
        self.lm_dim = self.language_model.model.config.n_embd
        # start with same word embeddng as language model but clone for training
        self.word_embedding = copy.deepcopy(self.language_model.token_embedding_layer())
        word_embed_dim = self.word_embedding.embedding_dim
        self.to_embed_bridge = nn.Linear(self.lm_dim, word_embed_dim)
        self.pos_embed = PositionalEncoding(self.lm_dim + self.word_embedding.embedding_dim)

    def forward(self, encoder_output, encoder_padding, target_sequence):
        """

        :param frame_embeddings: Shape: batch x time x feature
        :param lengths:
        :return:
        """
        input_ids, attention_padding_normal = self.language_model.tokenize(target_sequence)
        # input_ids shape: batch x sequence_len
        attention_padding = ~attention_padding_normal  # pytorch uses reverse convention
        tokens_embed = self.word_embedding(input_ids)  # batch x seq x embed
        tokens_embed = tokens_embed.permute(1, 0, 2)  # seq x batch x embed
        embed_dim = tokens_embed.shape[-1]

        # target mask: prevent decoder from looking into future
        batch_size, max_target_len = input_ids.shape
        tgt_mask = torch.triu(
            torch.ones(max_target_len, max_target_len)) == 1  # upper because pytorch backward
        # transformer_decoder wants batch second

        decoder_outputs = torch.zeros(max_target_len, batch_size, embed_dim).type_as(encoder_output)
        built_seq = torch.zeros(max_target_len, batch_size, embed_dim + self.lm_dim)
        built_seq = built_seq.type_as(encoder_output)

        # setup first word
        built_seq[0, :, :embed_dim] = tokens_embed[0, :, :]
        # teacher forcing sometimes
        teacher_forcing = np.random.rand() < self.config.teacher_forcing_probability

        lm_past = None
        lm_attention = None
        for i in range(1, max_target_len):
            decoder_input_pos = self.pos_embed(built_seq)
            trans_out = self.transformer_decoder(tgt=decoder_input_pos,
                                                 memory=encoder_output,
                                                 tgt_mask=tgt_mask,
                                                 tgt_key_padding_mask=attention_padding,
                                                 memory_key_padding_mask=encoder_padding)
            # trans_out shape: time x batch x embed
            decoded = self.to_embed_bridge(trans_out)
            decoded_time = decoded[1]  # embedding of predicted output

            decoder_outputs[i] = decoded_time

            # lm features shape: batch x 1 x hidden
            lm_features, lm_past, lm_attention = self.language_model.forward(
                input_ids=input_ids[:, i:i + 1],
                attention_mask=attention_padding_normal[:, i:i + 1],
                past_key_values=lm_past,
                past_attention_mask=lm_attention)
            built_seq[i, :, :embed_dim] = tokens_embed[i]
            built_seq[i, :, embed_dim:] = lm_features.permute(1, 0, 2)

        embed_matrix = self.word_embedding.weight.data.detach()  # shape: vocab x embed_dim
        logits = embedding_to_logits(decoder_outputs, embed_matrix)

        return logits, input_ids
