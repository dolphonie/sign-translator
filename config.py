# Created by Patrick Kao
import torch.cuda
from params_proto.neo_proto import PrefixProto

from data.lrs2 import LRSFileLazyDataset
from data.lrs3 import LRSLazyDataSet
from data.maxlen_wrapper import MaxLenWrapper


def wrapper_func(dataset):
    return MaxLenWrapper(dataset, max_len=120)

def no_wrapper(dataset):
    return dataset

class Config(PrefixProto):
    dataset_dir = "dataset_dir/lrs3"
    train_dir = "pretrain"
    val_dir = "trainval"
    test_dir = "test"
    additional_train_dir = None
    train_kwargs = {}
    test_kwargs = {}
    val_kwargs = {}
    additional_train_kwargs = {}

    batch_size = 2
    serialize_dataset_path = "datasets.dill"
    lr = 1e-3

    # data
    dataset_class = LRSLazyDataSet
    wrapper_func = wrapper_func

    trainer_params = {
        "gpus": -1 if torch.cuda.is_available() else None,
    }
    # model params
    encoder_layers = 6
    decoder_layers = 6
    teacher_forcing_probability = 0.8
    frame_embed_dim = 1


class LRS2Config(Config):
    serialize_dataset_path = "datasets_lrs2.dill"
    dataset_dir = "dataset_dir/lrs2/mvlrs_v1"
    train_dir = "main"
    val_dir = "main"
    test_dir = "main"
    additional_train_dir = "pretrain"
    train_kwargs = {"file_list": "train.txt"}
    test_kwargs = {"file_list": "test.txt"}
    val_kwargs = {"file_list": "val.txt"}
    additional_train_kwargs = {"file_list": "pretrain.txt"}

    dataset_class = LRSFileLazyDataset