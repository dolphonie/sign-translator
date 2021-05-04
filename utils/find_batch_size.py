# Created by Patrick Kao
from pytorch_lightning import Trainer
from pytorch_lightning.tuner.tuning import Tuner
from torch import nn

from config import Config
from data.lrs3 import LRS3DataModule
from model.sign_translator import SignTranslator

if __name__ == "__main__":
    data = LRS3DataModule(Config)
    model = SignTranslator(Config)
    trainer = Trainer()
    tuner = Tuner(trainer)

    # Invoke method
    new_batch_size = tuner.scale_batch_size(model, datamodule=data)

    print(f"Biggest batch that fits: {new_batch_size}")
