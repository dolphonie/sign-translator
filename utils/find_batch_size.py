# Created by Patrick Kao
import argparse
import os

from pytorch_lightning.tuner.tuning import Tuner


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
    parser.add_argument("-l", "--lrs2", action="store_true", help="Use the LRS2 Dataset params")
    args = parser.parse_args()

    config = Config
    if args.lrs2:
        config = LRS2Config
    data = LRSDataModule(config)
    model = SignTranslator(config)
    trainer = Trainer(**Config.trainer_params)
    tuner = Tuner(trainer)

    # Invoke method
    new_batch_size = tuner.scale_batch_size(model, datamodule=data, init_val=4)

    print(f"Biggest batch that fits: {new_batch_size}")
