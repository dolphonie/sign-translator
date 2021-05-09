# Created by Patrick Kao
import os.path
import datetime

from typing import Optional

from data.lrs3 import LRSLazyDataSet, LRS3DataModule


class LRSFileLazyDataset(LRSLazyDataSet):
    def __init__(self, file_list: str, dataset_directory: str, transform=None, start_str="Text:  "):
        super().__init__(transform=transform, start_str=start_str, load_file_list=[])  # don't crawl
        self.data_list = []
        with open(file_list, "r") as file:
            for line in file.readlines():
                self.data_list.append(os.path.join(dataset_directory, line))

class LRS2DataModule(LRS3DataModule):
    def setup(self, stage: Optional[str] = None):
        data_dir = self.config.dataset_dir

        load_filename = self.config.serialize_dataset_path
        if os.path.isfile(load_filename):
            [self.train_dataset, self.val_dataset, self.test_dataset] = dill.load(
                open(load_filename, 'rb'))
        else:
            print("Serializing data.")
            start = datetime.datetime.now()
            self.train_dataset = self.config.dataset_class(
                dataset_directory=os.path.join(data_dir, self.config.train_dir))
            self.val_dataset = self.config.dataset_class(
                dataset_directory=os.path.join(data_dir, self.config.val_dir))
            self.test_dataset = self.config.dataset_class(
                dataset_directory=os.path.join(data_dir, self.config.test_dir))

            # apply wrappers
            self.train_dataset = self.config.wrapper_func(self.train_dataset)
            self.val_dataset = self.config.wrapper_func(self.val_dataset)
            self.test_dataset = self.config.wrapper_func(self.test_dataset)

            # TODO: save in way that tolerates changes to dataset class
            Path(load_filename).parent.mkdir(parents=True, exist_ok=True)
            dill.dump([self.train_dataset, self.val_dataset,
                       self.test_dataset],
                      open(load_filename, mode='wb'))
            print(f"Completed serialization in: {datetime.datetime.now() - start}")