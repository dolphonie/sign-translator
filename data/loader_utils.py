# Created by Patrick Kao
import os

import torch.nn.utils.rnn
from torchvision.io import read_video


def crawl_directory_walk(directory):
    list_of_files = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.txt'):
                extension_removed = filename.replace(".txt", "")
                file_path = os.sep.join([dirpath, extension_removed])
                list_of_files.append(os.path.abspath(file_path))

    return list_of_files


def crawl_directory_one_nest(directory):
    list_of_files = []
    with os.scandir(directory) as it:
        for entry in it:
            if entry.is_dir():
                list_of_files.extend(crawl_directory_one_nest(entry.path))
            elif entry.is_file():
                filename = entry.path
                if filename.endswith('.txt'):
                    extension_removed = filename.replace(".txt", "")  # already absolute
                    list_of_files.append(os.path.abspath(extension_removed))

    return list_of_files


def get_frame_text(base_path, start_str, transform=None):
    video_path = f"{base_path}.mp4"
    text_path = f"{base_path}.txt"
    frames = read_video(video_path)
    if transform:
        frames = transform(frames)

    frames = frames[0]  # discard audio
    with open(text_path, "r") as file:
        raw_text = file.readlines()

    text = None
    for line in raw_text:
        if line.startswith(start_str):
            text = line[len(start_str):]
            break

    assert text is not None, "File read failed"
    return frames, text


def collate_batch(batch):
    frames = [el["frames"] for el in batch]
    text = [el["text"] for el in batch]
    lengths = torch.tensor([el.shape[0] for el in frames])
    padded_frames = torch.nn.utils.rnn.pad_sequence(frames, batch_first=True)
    # for dataparallel, lists are not scattered so let each gpu keep track of which labels it needs
    text_indices = torch.arange(padded_frames.shape[0])
    return padded_frames, lengths, text, text_indices
