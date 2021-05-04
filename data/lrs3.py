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
    def __init__(self, dataset_directory: str, transform=None, start_str="Text:  ",
                 load_file_list=None):

        self.dataset_directory = dataset_directory
        self.transform = transform
        self.start_str = start_str

        self.data_list = load_file_list if load_file_list is not None else self.populate_data()

    def __len__(self):
        return len(self.data_list[0])

    def __getitem__(self, idx):
        return {"frames": self.data_list[0][idx], "text": self.data_list[1][idx]}

    def populate_data(self):
        frames_list = []
        texts_list = []
        for (dirpath, dirnames, filenames) in os.walk(self.dataset_directory):
            for filename in filenames:
                if filename.endswith('.txt'):
                    base_local = filename.replace(".txt", "")
                    base_path = os.sep.join([dirpath, base_local])
                    frames, text = get_frame_text(base_path, self.start_str, self.transform)
                    frames_list.append(frames)
                    texts_list.append(text)

        data_list = [frames_list, texts_list]
        return data_list


class LRS3LazyDataSet(Dataset):
    def __init__(self, dataset_directory: str = None, transform=None, start_str="Text:  ",
                 load_file_list=None):
        assert (dataset_directory and (not load_file_list)) or (
                (not dataset_directory) and load_file_list)
        self.data_list = load_file_list if load_file_list is not None else \
            loader_utils.crawl_directory_one_nest(dataset_directory)
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
            [train_list, val_list, test_list] = dill.load(
                open(load_filename, 'rb'))
            self.train_dataset = self.config.dataset_class(
                load_file_list=train_list)
            self.val_dataset = self.config.dataset_class(
                load_file_list=val_list)
            self.test_dataset = self.config.dataset_class(
                load_file_list=test_list)
        else:
            print("Serializing data.")
            start = datetime.datetime.now()
            self.train_dataset = self.config.dataset_class(
                dataset_directory=os.path.join(data_dir, "pretrain"))
            self.val_dataset = self.config.dataset_class(
                dataset_directory=os.path.join(data_dir, "trainval"))
            self.test_dataset = self.config.dataset_class(
                dataset_directory=os.path.join(data_dir, "test"))
            print(f"Completed serialization in: {datetime.datetime.now() - start}")

            Path(load_filename).parent.mkdir(parents=True, exist_ok=True)
            dill.dump([self.train_dataset.data_list, self.val_dataset.data_list,
                       self.test_dataset.data_list],
                      open(load_filename, mode='wb'))

    def train_dataloader(self) -> Any:
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size,
                          collate_fn=collate_batch)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_dataset, batch_size=self.config.batch_size)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_dataset, batch_size=self.config.batch_size)
