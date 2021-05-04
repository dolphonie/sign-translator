# Created by Patrick Kao
import os

import torch
from torchvision import transforms

from model.pretrain_cnn_files.model import VideoModel


class PretrainedArgs:
    se = False
    border = False
    n_class = 500


def load_missing(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict.keys() and v.size() == model_dict[k].size()}
    missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]

    print('loaded params/tot params:{}/{}'.format(len(pretrained_dict), len(model_dict)))
    print('miss matched params:', missed_params)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def get_pretrained_cnn(weights_path="../pretrain_models/lrw-cosine-lr-acc-0.85080.pt"):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    weights_path_adjusted = os.path.join(file_dir, weights_path)
    video_model = VideoModel(PretrainedArgs)
    print('load weights')
    weight = torch.load(weights_path_adjusted, map_location=torch.device('cpu'))
    load_missing(video_model, weight.get('video_model'))
    return video_model.video_cnn


def transform_frames_for_pretrain(frames):
    """

    :param frames: batch x time x h x w x 3
    :return:
    """
    gray = transforms.Grayscale()
    to_ret = frames.permute(0, 1, 4, 2, 3).type(torch.float)/255  # want batch x time x channels x h x w
    to_ret = gray(to_ret)

    return to_ret
