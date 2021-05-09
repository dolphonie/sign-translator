# Created by Patrick Kao
from config import Config
from data.lrs3 import LRSDataModule

dm = LRSDataModule(Config)
dm.prepare_data()

# splits/transforms
dm.setup('fit')
for batch in dm.train_dataloader():
    print(batch[1])
