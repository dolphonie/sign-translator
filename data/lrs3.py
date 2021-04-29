# Created by Patrick Kao
import datetime
import os.path
from pathlib import Path
from typing import Optional, Any, Union, List

import dill
from params_proto.neo_proto import PrefixProto
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from data import loader_utils
from data.loader_utils import collate_batch, get_frame_text


class LRS3WholeDataSet(Dataset):
    def __init__(self, dataset_directory: str, transform=None, start_str="Text:  "):
        self.frames = []
        self.texts = []

        self.dataset_directory = dataset_directory
        self.transform = transform
        self.start_str = start_str

        self.populate_data()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return {"frames": self.frames[idx], "text": self.texts[idx]}

    def populate_data(self):
        for (dirpath, dirnames, filenames) in os.walk(self.dataset_directory):
            for filename in filenames:
                if filename.endswith('.txt'):
                    base_local = filename.replace(".txt", "")
                    base_path = os.sep.join([dirpath, base_local])
                    frames, text = get_frame_text(base_path, self.start_str, self.transform)
                    self.frames.append(frames)
                    self.texts.append(text)


class LRS3LazyDataSet(Dataset):
    def __init__(self, dataset_directory: str, transform=None, start_str="Text:  "):
        self.data_list = loader_utils.crawl_directory_one_nest(dataset_directory)
        self.transform = transform
        self.start_str = start_str

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        file_path = self.data_list[item]
        frames, text = get_frame_text(file_path, self.start_str, self.transform)
        return {"frames": frames, "text": text}


class LRS3DataModule(LightningDataModule):
    def __init__(self, config: PrefixProto.__class__):
        super().__init__()
        self.config = config

    def setup(self, stage: Optional[str] = None):
        data_dir = self.config.dataset_dir

        load_filename = self.config.serialize_dataset_path
        if os.path.isfile(load_filename):
            [self.train_dataset, self.val_dataset, self.test_dataset] = dill.load(
                open(load_filename, 'rb'))
        else:
            print("Serializing data.")
            self.train_dataset = None  # LRS3DataSet(dataset_directory=os.path.join(data_dir,
            # "pretrain"))
            self.val_dataset = None  # LRS3DataSet(dataset_directory=os.path.join(data_dir,
            # "trainval"))
            start = datetime.datetime.now()
            self.test_dataset = self.config.dataset_class(dataset_directory=os.path.join(data_dir, "test"))
            print(datetime.datetime.now() - start)

            Path(load_filename).parent.mkdir(parents=True, exist_ok=True)
            dill.dump([self.train_dataset, self.val_dataset, self.test_dataset],
                      open(load_filename, mode='wb'))

    def train_dataloader(self) -> Any:
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size,
                          collate_fn=collate_batch)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_dataset, batch_size=self.config.batch_size)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_dataset, batch_size=self.config.batch_size)
