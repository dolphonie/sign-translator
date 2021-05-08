# Created by Patrick Kao
import torch.cuda
from params_proto.neo_proto import PrefixProto

from data.lrs3 import LRS3LazyDataSet
from data.maxlen_wrapper import MaxLenWrapper


def wrapper_func(dataset):
    return MaxLenWrapper(dataset, max_len=120)


class Config(PrefixProto):
    dataset_dir = "dataset_dir/lrs3"
    batch_size = 1
    serialize_dataset_path = "datasets.dill"
    lr = 1e-3

    # data
    dataset_class = LRS3LazyDataSet
    wrapper_func = lambda x: x

    trainer_params = {
        "gpus": -1 if torch.cuda.is_available() else None,
    }
    # model params
    encoder_layers = 6
    decoder_layers = 6
    teacher_forcing_probability = 0.8
    frame_embed_dim = 1
