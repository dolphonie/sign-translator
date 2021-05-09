# Created by Patrick Kao
import os


def remove_slurm_vars():
    for k, v in os.environ.items():
        if "SLURM" in k:
            print(f"Deleting env variable {k}")
            del os.environ[k]
            
remove_slurm_vars()

if __name__ == '__main__':
    from pytorch_lightning import Trainer

    from config import Config
    from data.lrs3 import LRSDataModule
    from model.sign_translator import SignTranslator

    data = LRSDataModule(Config)
    model = SignTranslator(Config)

    trainer = Trainer(**Config.trainer_params)
    trainer.fit(model, data)
