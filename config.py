# Created by Patrick Kao
from params_proto.neo_proto import PrefixProto

from data.lrs3 import LRS3WholeDataSet, LRS3LazyDataSet


class Config(PrefixProto):
    dataset_dir = "dataset_dir/lrs3"
    batch_size = 4
    serialize_dataset_path = "datasets.dill"
    lr = 1e-3
    dataset_class = LRS3LazyDataSet
