# Created by Patrick Kao
import datetime
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, Tensor

from model.decoders.greedy_decoder import GreedyDecoder
from model.decoder import Decoder
from model.encoder import Encoder
from model.pretrain_videocnn import get_pretrained_cnn, transform_frames_for_pretrain


class SignTranslatorNoLightning(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.video_encoder = get_pretrained_cnn()
        self.encoder = Encoder(config)
        self.decoder = GreedyDecoder(config)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, frames: Tensor, lengths: List[int], labels: List[str],
                labels_id: Tensor = None):
        """

        :param frames: batch x time x channels x height x width
        :param lengths: length batch
        :param labels: length batch
        :return:
        """
        start = datetime.datetime.now()
        if labels_id is not None:
            labels = np.asarray(labels)
            labels = labels[labels_id.detach().cpu()]
            labels = list(labels)

        print(f"Labels elapsed: {datetime.datetime.now() - start}")

        start = datetime.datetime.now()
        frame_embed = self.video_encoder(frames)  # batch x time x out_dim
        print(f"CNN elapsed {datetime.datetime.now() - start}")
        start = datetime.datetime.now()
        encoder_output, encoder_padding = self.encoder(frame_embeddings=frame_embed,
                                                       lengths=lengths)
        output_logits, output_mask, labels_tokenized, labels_mask = self.decoder(encoder_output=encoder_output,
                                                                                 encoder_padding=encoder_padding,
                                                                                 target_sequence=labels)
        print(f"Encdec elapsed {datetime.datetime.now() - start}")
        return output_logits, output_mask, labels_tokenized, labels_mask

    def training_step(self, batch):
        frames, lengths, labels, labels_id = batch
        frames_tr = transform_frames_for_pretrain(frames)
        output_logits, output_mask, labels_tokenized, labels_mask = self.forward(frames=frames_tr,
                                                                                 lengths=lengths,
                                                                                 labels=labels,
                                                                                 labels_id=labels_id)
        loss = self.masked_loss(output_logits, labels_tokenized, labels_mask)
        return loss

    def masked_loss(self, output_logits, labels_tokenized, labels_mask):
        # trim everything to min seq len
        min_seq_len = min(output_logits.shape[0], labels_tokenized.shape[1])
        output_logits = output_logits[:min_seq_len]
        labels_tokenized = labels_tokenized[:, :min_seq_len]
        labels_mask = labels_mask[:, :min_seq_len]
        # no need to shift as in GPT2 objective, since each logit corresponds to the prediction
        # for the corresponding word
        # output_logits shape: (seq_len, batch, vocab)
        logits_contig = output_logits.permute(1, 2, 0).contiguous()  # want (batch, vocab, seq_len)
        labels_contig = labels_tokenized.contiguous()  # (batch, seq_len)
        # Flatten the tokens
        loss = self.loss_fn(logits_contig, labels_contig)  # (batch, seq_len)
        loss = torch.mean(loss * labels_mask)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.config.lr)
        return optimizer
