# Created by Patrick Kao
import os

if __name__ == '__main__':
    def remove_slurm_vars():
        for k, v in os.environ.items():
            if "SLURM" in k:
                print(f"Deleting env variable {k}")
                del os.environ[k]


    remove_slurm_vars()

    import argparse

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

    trainer = Trainer(**config.trainer_params)
    trainer.fit(model, data)
