# Created by Patrick Kao
from config import Config
from data.lrs3 import LRSDataModule

dm = LRSDataModule(Config)
dm.prepare_data()

# splits/transforms
dm.setup('fit')
max = 0
for i, batch in enumerate(dm.train_dataloader()):
    max = i

print(max)
counts = dm.train_dataset.get_included_excluded_counts()
print(counts)
print(counts[0] / counts[1])
