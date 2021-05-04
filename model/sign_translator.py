# Created by Patrick Kao

from typing import List

import pytorch_lightning as pl
import torch
from torch import nn, Tensor

from model.decoder import Decoder
from model.encoder import Encoder
from model.pretrain_videocnn import get_pretrained_cnn, transform_frames_for_pretrain


class SignTranslator(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.video_encoder = get_pretrained_cnn()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, frames: Tensor, lengths: List[int], labels: List[str]):
        """

        :param frames: batch x time x channels x height x width
        :param lengths: length batch
        :param labels: length batch
        :return:
        """
        frame_embed = self.video_encoder(frames)  # batch x time x out_dim
        encoder_output, encoder_padding = self.encoder(frame_embeddings=frame_embed,
                                                       lengths=lengths)
        output_logits, labels_tokenized = self.decoder(encoder_output=encoder_output,
                                                       encoder_padding=encoder_padding,
                                                       target_sequence=labels)
        return output_logits, labels_tokenized

    def training_step(self, batch, batch_idx):
        frames, lengths, labels = batch

        frames_tr = transform_frames_for_pretrain(frames)
        output_logits, labels_tokenized = self.forward(frames=frames_tr,
                                                       lengths=lengths,
                                                       labels=labels, )
        # no need to shift as in GPT2 objective, since each logit corresponds to the prediction
        # for the corresponding word
        logits_contig = output_logits.permute(1, 0, 2).contiguous()  # want batch first
        labels_contig = labels_tokenized.contiguous()
        # Flatten the tokens
        loss = self.loss_fn(logits_contig.view(-1, logits_contig.size(-1)), labels_contig.view(-1))
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.config.lr)
        return optimizer
