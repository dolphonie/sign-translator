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


class GreedyDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.language_model = LanguageModel()
        # start with same word embeddng as language model but clone for training
        self.word_embedding = copy.deepcopy(self.language_model.token_embedding_layer())

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
        :param encoder_padding: padding of encoder
        :param target_sequence: either None (teacher forcing prob = 0), or target_sequence
        :return:
            output_logits: shape out_length x batch x vocab_size
            output_mask: shape out_length x batch, 1 for important and 0 for disregard
            labels_tokenized: tokenized target sequence shape batch x in_length, or None
            labels_mask: shape batch x in_length, 1 for important and 0 for disregard
        """
        device = encoder_output.device
        batch_size = encoder_output.shape[1]

        if target_sequence is None:
            teacher_forcing = False
            max_target_len = self.config.max_decode_len
            input_ids, attention_padding_normal, attention_padding, tokens_embed = None, None, \
                                                                                   None, None
        else:
            teacher_forcing = np.random.rand() < self.config.teacher_forcing_probability
            input_ids, attention_padding_normal = self.language_model.tokenize(target_sequence)
            input_ids = input_ids.to(device)
            attention_padding_normal = attention_padding_normal.to(device)
            # input_ids shape: batch x sequence_len
            attention_padding = ~attention_padding_normal.type(
                torch.bool)  # pytorch uses reverse convention
            tokens_embed = self.word_embedding(input_ids)  # batch x seq x embed
            tokens_embed = tokens_embed.permute(1, 0, 2)  # seq x batch x embed
            max_target_len = input_ids.shape[1]

        # target mask: prevent decoder from looking into future
        # upper because diagonals and future to be masked off
        tgt_mask = ~torch.tril(torch.ones(max_target_len, max_target_len)).bool()
        tgt_mask = tgt_mask.to(device)
        # transformer_decoder wants batch second

        # decoder_outputs = torch.zeros(max_target_len, batch_size, self.word_embed_dim).type_as(
        #     encoder_output)
        embed_matrix = self.word_embedding.weight.data.detach()  # shape: vocab x embed_dim
        all_logits = torch.zeros(max_target_len, batch_size, embed_matrix.shape[0]).type_as(
            encoder_output)
        built_seq = torch.zeros(max_target_len, batch_size, self.transformer_dim)
        built_seq = built_seq.type_as(encoder_output)
        decoded_mask = torch.zeros(max_target_len, batch_size).to(device=device, dtype=torch.bool)

        # setup first word (SOS token)
        start_token = torch.LongTensor([[self.language_model.tokenizer.eos_token_id]]).to(device)
        all_logits[0, :, start_token] = 1
        built_seq[0, :, :self.word_embed_dim] = self.word_embedding(start_token)
        decoded_mask[0, :] = 1

        memory_converted = self.enc_to_dec_bridge(encoder_output)
        lm_past = None
        lm_attention = None
        for i in range(1, max_target_len):
            decoder_input_pos = self.pos_embed(built_seq[:i])
            # TODO: even without pos_embeddings, trans_out[0] changes between runs, maybe dropout?
            # TODO: I'm only passing the shortest necessary sequence to the decoder for now...
            #  unclear if this does
            #       speed it up or actually make it worse LOL
            trans_out = self.transformer_decoder(tgt=decoder_input_pos,
                                                 memory=memory_converted,
                                                 tgt_mask=tgt_mask[:i, :i],
                                                 tgt_key_padding_mask=attention_padding[:,
                                                                      :i] if teacher_forcing else
                                                 ~decoded_mask[:i].T,
                                                 memory_key_padding_mask=encoder_padding)
            # trans_out shape: time x batch x embed  /TODO: not embed, but transformer_dim?
            decoded = self.dec_to_embed_bridge(
                trans_out[i - 1])  # todo: confirm, shoudln't need non-i time values?
            # decoded shape: batch x embed
            # decoded_time = decoded[i]  # embedding of predicted output

            logits = embedding_to_logits(decoded, embed_matrix).squeeze(-1).squeeze(
                0)  # batch x vocab
            all_logits[i] = logits
            greedy_input_ids = torch.argmax(logits, dim=1,
                                            keepdim=True)  # get highest probability token greedily
            greedy_attention = greedy_input_ids != self.language_model.model.config.eos_token_id
            # greedy_input_ids shape: batch x 1

            # lm features shape: batch x 1 x hidden
            lm_features, lm_past, lm_attention = self.language_model.forward(
                input_ids=input_ids[:, i:i + 1] if teacher_forcing else greedy_input_ids,
                attention_mask=attention_padding_normal[:,
                               i:i + 1] if teacher_forcing else greedy_attention,
                past_key_values=lm_past,
                past_attention_mask=lm_attention)
            built_seq[i, :, :self.word_embed_dim] = tokens_embed[i] if teacher_forcing else decoded
            built_seq[i, :, self.word_embed_dim:] = lm_features[:, 0]
            decoded_mask[i] = torch.logical_and(greedy_attention.flatten(),
                                                decoded_mask[i - 1]).bool()

        return all_logits, decoded_mask, input_ids, attention_padding_normal
