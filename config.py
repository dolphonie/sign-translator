# Created by Patrick Kao
import torch.cuda
from params_proto.neo_proto import PrefixProto

from data.lrs2 import LRSFileLazyDataset
from data.lrs3 import LRSLazyDataSet, LRSWholeDataSet
from data.maxlen_wrapper import MaxLenWrapper, MaxLenWrapperIterable


def wrapper_func(dataset):
    return MaxLenWrapper(dataset, max_len=120)


def no_wrapper(dataset):
    return dataset


def iterable_wrapper(dataset, **kwargs):
    return MaxLenWrapperIterable(dataset, max_len=200, **kwargs)


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

    batch_size = 1
    serialize_dataset_path = "datasets.dill"
    lr = 1.3182567385564076e-05

    # data
    dataset_class = LRSLazyDataSet
    wrapper_func = iterable_wrapper
    num_epochs = 5
    num_dataloader_workers = 4

    trainer_params = {
        "gpus": -1 if torch.cuda.is_available() else None,
        "accelerator": "dp",
        "val_check_interval": 25000,
        "limit_val_batches": 3000,
        "max_epochs": num_epochs,
    }
    # model params
    encoder_layers = 6
    decoder_layers = 6
    beam_search_width = 3
    max_decode_len = 45
    teacher_forcing_probability = 0.8
    frame_embed_dim = 1

    visualize_freq = 10


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


class WholeConfig(LRS2Config):
    serialize_dataset_path = "dataset_lrs2_whole.dill"
    dataset_class = LRSWholeDataSet
    train_kwargs = {}
    test_kwargs = {}
    val_kwargs = {}
