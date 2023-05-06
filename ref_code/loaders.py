import os
import torch
from torch.utils.data import Dataset, DataLoader

from timelens.common import (hybrid_storage, os_tools, transformers, image_sequence, event)
"""
Create custom datasets/loaders to load for training/inference
"""

"""
each call of next outputs the following:
    3 sequential images, img0 and img2 are query iamges, img1 is the target image
    both event sequences are reversed
    2 event sequences, from img 0->1 and img 1->2
    the second event sequence is reversed
"""
class TrainDataset():
    def __init__(self, root_event_folder, root_image_folder, event_file_template="*.npz", img_file_template="*.png"):
        root_event_folder = os.path.abspath(root_event_folder)
        root_image_folder = os.path.abspath(root_image_folder)

        # get an iterator that outputs triplets of all images
        self.img_sequence = image_sequence.ImageSequence.from_folder(
            folder=root_image_folder,
            image_file_template=img_file_template,
            timestamps_file="timestamp.txt"
        )

        self.event_sequence = event.EventSequence.from_folder(
            folder=root_event_folder,
            image_height=self.img_sequence._height,
            image_width=self.img_sequence._width,
            event_file_template=event_file_template
        )

        self.length = len(self.img_sequence._timestamps) - 2

    """
    returns [[img0, img2], [EventSequence_01, EventSequence_12], img1]
    """
    def get_triplet(self, index):
        if index >= self.length:
            ValueError("Train dataset index out of range: " % index)

        images = [self.img_sequence.__getitem__(i) for i in [index, index+1, index+2]]
        times = [self.img_sequence._timestamps[i] for i in [index, index+1, index+2]]

        event_subsequence_itr = self.event_sequence.make_sequential_iterator(times)
        event_subsequences = [e for e in event_subsequence_itr]

        # print("01: ", times[0], " | ", times[1], " || ", event_subsequences[0].start_time(), event_subsequences[0].end_time())
        # print("12: ", times[1], " | ", times[2], " || ", event_subsequences[1].start_time(), event_subsequences[1].end_time())

        return [[images[0], images[2]], event_subsequences, images[1]]


# t = TrainDataset(r"/media/aneesh/Cherry passport vibes/hs-ergb/hsergb/close/test/baloon_popping/events_aligned", r"/media/aneesh/Cherry passport vibes/hs-ergb/hsergb/close/test/baloon_popping/images_corrected/")

# queryimg, es, res = t.get_triplet(0)
# print(queryimg)
# print(es)
# print(es[0].start_time(), es[0].end_time())
# print(es[1].start_time(), es[1].end_time())
# print(res)