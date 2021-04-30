# Created by Patrick Kao
import pytorch_lightning as pl
import torch
from torch import nn

from model.decoder import Decoder
from model.encoder import Encoder
from model.pretrain_videocnn import get_pretrained_cnn


class SignTranslator(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.video_encoder = get_pretrained_cnn()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        frames, lengths, labels = x
        frame_embed = self.video_encoder(frames)
        encoder_output, encoder_padding = self.encoder(frame_embeddings=frame_embed,
                                                       lengths=lengths)
        output_logits, labels_tokenized = self.decoder(encoder_output=encoder_output,
                                                       encoder_padding=encoder_padding,
                                                       target_sequence=labels)

        shift_logits = output_logits[:-1].permute(1, 0, 2).contiguous()
        shift_labels = labels_tokenized[1:].permute(1, 0, 2).contiguous()
        # Flatten the tokens
        loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.config.lr)
        return optimizer
