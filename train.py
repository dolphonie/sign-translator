# Created by Patrick Kao
import argparse
import os


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
    trainer.fit(model, data)
    print(f"Counts {data.train_dataset.get_included_excluded_counts()}")
