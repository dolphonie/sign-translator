# Created by Patrick Kao
from pytorch_lightning import Trainer
from torch import nn

from config import Config
from data.lrs3 import LRS3DataModule
from model.sign_translator import SignTranslator

data = LRS3DataModule(Config)
model = SignTranslator(Config)

trainer = Trainer(gpus=-1)
trainer.fit(model, data)