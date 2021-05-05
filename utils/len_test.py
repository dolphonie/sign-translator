# Created by Patrick Kao
from config import Config
from data.lrs3 import LRS3DataModule

dm = LRS3DataModule(Config)
dm.prepare_data()

# splits/transforms
dm.setup('fit')
for batch in dm.train_dataloader():
    print(batch[1])
