# Created by Patrick Kao
import argparse
import os

from pytorch_lightning.callbacks import ModelCheckpoint


def remove_slurm_vars():
    for k, v in os.environ.items():
        if "SLURM" in k:
            print(f"Deleting env variable {k}")
            del os.environ[k]


if __name__ == '__main__':
    remove_slurm_vars()

    from pytorch_lightning import Trainer

    from config import Config, LRS2Config
    from data.lrs3 import LRSDataModule
    from model.sign_translator import SignTranslator

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lrs3", action="store_true", help="Use the LRS3 Dataset params")
    args = parser.parse_args()

    config = LRS2Config
    if args.lrs3:
        config = Config
    data = LRSDataModule(config)
    model = SignTranslator(config).load_from_checkpoint(
        "lightning_logs/version_0/checkpoints/epoch=0-step=29999.ckpt", config=config)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          save_top_k=200,
                                          mode='min',
                                          save_last=True)
    trainer = Trainer(**config.trainer_params, callbacks=[checkpoint_callback])
    trainer.fit(model, data)
    trainer.test()
