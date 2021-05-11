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
    :return: time x batch x vocab
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

        self.language_model = LanguageModel()
        # start with same word embeddng as language model but clone for training
        self.word_embedding = copy.deepcopy(self.language_model.token_embedding_layer())

        # freeze lm
        self.language_model.requires_grad_(False)
        self.word_embedding.requires_grad_(False)

        self.word_embed_dim = self.word_embedding.embedding_dim
        self.lm_dim = self.language_model.model.config.n_embd
        self.transformer_dim = self.word_embed_dim + self.lm_dim
        self.dec_to_embed_bridge = nn.Linear(self.transformer_dim, self.word_embed_dim)
        self.pos_embed = PositionalEncoding(self.lm_dim + self.word_embedding.embedding_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.transformer_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, config.decoder_layers)
        self.enc_to_dec_bridge = nn.Linear(512, self.transformer_dim)

    def forward(self, encoder_output, encoder_padding, target_sequence):
        """

        :param encoder_output: Shape: time x batch x feature
        :param lengths:
        :return: output_logits: shape out_length x batch x vocab_size
        """
        device = encoder_output.device
        input_ids, attention_padding_normal = self.language_model.tokenize(target_sequence)
        input_ids = input_ids.to(device)
        attention_padding_normal = attention_padding_normal.to(device)
        # input_ids shape: batch x sequence_len
        attention_padding = ~attention_padding_normal.type(
            torch.bool)  # pytorch uses reverse convention
        tokens_embed = self.word_embedding(input_ids)  # batch x seq x embed
        tokens_embed = tokens_embed.permute(1, 0, 2)  # seq x batch x embed

        # target mask: prevent decoder from looking into future
        batch_size, max_target_len = input_ids.shape
        # upper because diagonals and future to be masked off
        tgt_mask = torch.triu(
            torch.ones(max_target_len, max_target_len)).type(torch.bool)
        tgt_mask[0, 0] = False  # first word attends to first token otherwise output all nan
        tgt_mask = tgt_mask.to(device)
        # transformer_decoder wants batch second

        decoder_outputs = torch.zeros(max_target_len, batch_size, self.word_embed_dim).type_as(
            encoder_output)
        built_seq = torch.zeros(max_target_len, batch_size, self.transformer_dim)
        built_seq = built_seq.type_as(encoder_output)

        # setup first word (SOS token)
        built_seq[0, :, :self.word_embed_dim] = tokens_embed[0, :, :]
        # teacher forcing sometimes
        teacher_forcing = np.random.rand() < self.config.teacher_forcing_probability

        memory_converted = self.enc_to_dec_bridge(encoder_output)
        lm_past = None
        lm_attention = None
        for i in range(1, max_target_len):
            decoder_input_pos = self.pos_embed(built_seq)
            # TODO: even without pos_embeddings, trans_out[0] changes between runs, maybe dropout?
            trans_out = self.transformer_decoder(tgt=decoder_input_pos,
                                                 memory=memory_converted,
                                                 tgt_mask=tgt_mask,
                                                 tgt_key_padding_mask=attention_padding,
                                                 memory_key_padding_mask=encoder_padding)
            # trans_out shape: time x batch x embed
            decoded = self.dec_to_embed_bridge(trans_out)
            decoded_time = decoded[i]  # embedding of predicted output

            decoder_outputs[i] = decoded_time

            # lm features shape: batch x 1 x hidden
            lm_features, lm_past, lm_attention = self.language_model.forward(
                input_ids=input_ids[:, i:i + 1],
                attention_mask=attention_padding_normal[:, i:i + 1],
                past_key_values=lm_past,
                past_attention_mask=lm_attention)
            built_seq[i, :, :self.word_embed_dim] = tokens_embed[i]
            built_seq[i, :, self.word_embed_dim:] = lm_features.permute(1, 0, 2)

        embed_matrix = self.word_embedding.weight.data.detach()  # shape: vocab x embed_dim
        logits = embedding_to_logits(decoder_outputs, embed_matrix).squeeze(-1)

        return logits, input_ids
