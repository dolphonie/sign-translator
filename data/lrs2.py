# Created by Patrick Kao
import datetime
import os.path
from typing import Optional

from data.lrs3 import LRSLazyDataSet, LRSDataModule


class LRSFileLazyDataset(LRSLazyDataSet):
    def __init__(self, file_list: str, dataset_directory: str, transform=None, start_str="Text:  "):
        super().__init__(transform=transform, start_str=start_str, load_file_list=[])  # don't crawl
        cur_file = os.path.dirname(os.path.realpath(__file__))
        dataset_directory = os.path.join(cur_file, "..", dataset_directory)
        self.data_list = []
        with open(os.path.join(dataset_directory, "..", file_list), "r") as file:
            for line in file.read().splitlines():
                # test set has "NF" or "MV" after line in file list, remove
                self.data_list.append(os.path.join(dataset_directory, line.split(" ")[0]))

