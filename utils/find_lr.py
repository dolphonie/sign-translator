# Created by Patrick Kao
import argparse
import os
import sys

cur_file = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file, ".."))
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
    parser.add_argument("-l", "--lrs3", action="store_true", help="Use the LRS3 Dataset params")
    args = parser.parse_args()

    config = LRS2Config
    if args.lrs3:
        config = Config
    data = LRSDataModule(config)
    model = SignTranslator(config)

    trainer = Trainer(**config.trainer_params)

    # Invoke method
    lr_finder = trainer.tuner.lr_find(model)

    # Plot with
    fig = lr_finder.plot(suggest=True)
    fig.savefig("lr.png")

    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()

    # update hparams of the model
    model.hparams.lr = new_lr

    print(new_lr)
    print(lr_finder.results)
