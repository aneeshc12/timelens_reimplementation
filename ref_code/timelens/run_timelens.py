import os

import click
import numpy as np
import sys
from os.path import dirname, join
import torch

sys.path.append(dirname(dirname(__file__)))
import torch as th
from timelens import attention_average_network
from timelens.common import (
    hybrid_storage,
    image_sequence,
    os_tools,
    pytorch_tools,
    transformers
)
from torchvision import transforms

""" MAIN FUNCTION """
def _interpolate(
        network,
        transform_list,
        interframe_events_iterator,
        boundary_frames_iterator,
        number_of_frames_to_interpolate,
        output_folder
):
    
    output_frames, output_timestamps = [], []
    pytorch_tools.set_fastest_cuda_mode()
    combined_iterator = zip(boundary_frames_iterator, interframe_events_iterator)
    """
    iterate over boundary images and events

    boundary frames are (1,2),(2,3),(3,4) etc
    """
    
    counter = 0
    for (left_frame, right_frame), event_sequence in combined_iterator:
        """
        iterate over every pair of consecutive keyframe images
        process all events between them
        """

        print("Counter: %04d" % counter)
        output_timestamps += list(
            np.linspace(
                event_sequence.start_time(),
                event_sequence.end_time(),
                2 + number_of_frames_to_interpolate,
            )
        )[:-1]
        """
        append all but the last timestamp (last one included in the next pair)
        """

        iterator_over_splits = event_sequence.make_iterator_over_splits(
            number_of_frames_to_interpolate
        )
        """
        return an iterator containing the event sequence split into two parts,
        beginning to tau, end to tau as pairs

        (t_start->t_0 sequence, t_0->t_end sequence)
        (t_start->t_1 sequence, t_1->t_end sequence)
        (t_start->t_2 sequence, t_2->t_end sequence)
        etc
        """
        
        output_frames.append(left_frame)
        output_frames[-1].save(join(output_folder, "{:06d}.png".format(counter)))
        counter += 1
        """
        save initial image
        """

        for split_index, (left_events, right_events) in enumerate(iterator_over_splits):
            """
            iterate over all splits
            """

            print("Events left: ", len(left_events._features), "Events right: ", len(right_events._features))
            example = _pack_to_example(
                left_frame,
                right_frame,
                left_events,
                right_events,
                float(split_index + 1.0) / (number_of_frames_to_interpolate + 1.0),
            )
            """
            pack the left, right keyframe into an example struct
                " left_image, right_image, left_events, right_events, right_weight "
                first two are the keyframes, 
                next two are the event sequence split acc to timestamps
                last one is the ["middle"]["weight"] thing
            """
            
            example = transformers.apply_transforms(example, transform_list)
            """
            1. convert all PIL images to tensors, exmaple contains a dict storing everything (things with image in the name converted to torch tensors)
            2.  reverse event stream in before packet
                ["before"]["events"] contains an event_stream class
            3. convert all event streams in before (now reversed) and after to voxel grids, store them in ["voxel_grid"]
            """
            example = transformers.collate([example])
            """
            collate all tensors and voxel grids from all 'before', 'middle' and 'after' into one torch.stack, to be processed by the unet 
            """
            example = pytorch_tools.move_tensors_to_cuda(example)

            with torch.no_grad():
                frame, _ = network.run_fast(example)
            """
            run the attention network over the example struct

            in AttentionAverage

            returns weighted average of the generated image for this split
            """
    
            interpolated = th.clamp(
                frame.squeeze().cpu().detach(), 0, 1,
            )
            output_frames.append(transforms.ToPILImage()(interpolated))
            output_frames[-1].save(join(output_folder, "{:06d}.png".format(counter)))
            counter += 1

            """
            retrieve generated image for this split, 
            save with appropriate name
            """
        """
        repeat for all splits
        """

    output_frames.append(right_frame)
    output_frames[-1].save(join(output_folder, "{:06d}.png".format(counter)))
    counter += 1
    """
    save the last image
    """

    return output_frames, output_timestamps


def _load_network(checkpoint_file):
    network = attention_average_network.AttentionAverage()
    network.from_legacy_checkpoint(checkpoint_file)
    network.cuda()
    network.eval()

    # print(network.eval())
    # exit(0)

    return network
"""
load network
"""

def _pack_to_example(left_image, right_image, left_events, right_events, right_weight):
    return {
        "before": {"rgb_image": left_image, "events": left_events},
        "middle": {"weight": right_weight},
        "after": {"rgb_image": right_image, "events": right_events},
    }
"""
initialises the example struct
simply dict of dicts that everything gets shoved into
"""


def run_recursively(
        checkpoint_file,
        root_event_folder,
        root_image_folder,
        root_output_folder,
        number_of_frames_to_skip,
        number_of_frames_to_insert,
):
    (root_image_folder, root_event_folder, root_output_folder) = [os.path.abspath(folder) for folder in [root_image_folder, root_event_folder, root_output_folder]
    ]

    # here we initialize the remapping function for events
    remapping_maps = None

    transform_list = transformers.initialize_transformers()
    """
    1. convert all PIL images to tensors, exmaple contains a dict storing everything (things with image in the name converted to torch tensors)
    2.  reverse event stream in before packet
        ["before"]["events"] contains an event_stream class
    3. convert all event streams in before (now reversed) and after to voxel grids, store them in ["voxel_grid"]
    """
    
    network = _load_network(checkpoint_file)

    """
    get and iterate over all subfolders
    """
    leaf_image_folders = os_tools.find_leaf_folders(root_image_folder)
    for leaf_image_folder in leaf_image_folders:
        relative_path = os.path.relpath(leaf_image_folder, root_image_folder)
    
        print("Processing {}".format(relative_path))
    
        leaf_event_folder = os.path.join(root_event_folder, relative_path)
        leaf_output_folder = os.path.join(root_output_folder, relative_path)
        """
        get event and output folders for each
        """
    
        storage = hybrid_storage.HybridStorage.from_folders(
            leaf_event_folder, leaf_image_folder, "*.npz", "*.png"
        )
        """
        create a class that makes the imgs and events into imageseuqeucne and eventseuence classes respectively
        """
    
        interframe_events_iterator = storage.make_interframe_events_iterator(
            number_of_frames_to_skip
        )
        """
        convert eventseq into an iterator
        """
    
        boundary_frames_iterator = storage.make_boundary_frames_iterator(
            number_of_frames_to_skip
        )
        """
        convert imageseq into an iterator
        """
    
        print("Processing {}".format(leaf_output_folder))
        os.makedirs(leaf_output_folder, exist_ok=True)

        output_frames, output_timestamps = _interpolate(
            network,
            transform_list,
            interframe_events_iterator,
            boundary_frames_iterator,
            number_of_frames_to_insert,
            leaf_output_folder
        )
        """
        MAIN FRAME CREATION FUNCTION
        takes:
            loaded network weights,
            transform list (img2tensor, flipper, voxeliser) (all processing happens in _interpolate())
            event itr
            image itr
            no of extra frames to generate
            ouput folder name
        """

        output_image_sequence = image_sequence.ImageSequence(
            output_frames, output_timestamps
        )

        input_image_sequence = storage._images.skip_and_repeat(number_of_frames_to_skip, number_of_frames_to_insert)
        output_image_sequence.to_folder(leaf_output_folder, file_template="frame_{:06d}.png")
        output_image_sequence.to_video(os.path.join(leaf_output_folder, "interpolated.mp4"))
        input_image_sequence.to_video(os.path.join(leaf_output_folder, "input.mp4"))
        """
        convert created frames and timestamps to an img sequence and store acc to given params
        """


@click.command()
@click.argument("checkpoint_file", type=click.Path(exists=True))
@click.argument("root_event_folder", type=click.Path(exists=True))
@click.argument("root_image_folder", type=click.Path(exists=True))
@click.argument("root_output_folder", type=click.Path(exists=False))
@click.argument("number_of_frames_to_skip", default=1)
@click.argument("number_of_frames_to_insert", default=1)
def main(
        checkpoint_file,
        root_event_folder,
        root_image_folder,
        root_output_folder,
        number_of_frames_to_skip,
        number_of_frames_to_insert,
):
    run_recursively(
        checkpoint_file,
        root_event_folder,
        root_image_folder,
        root_output_folder,
        number_of_frames_to_skip,
        number_of_frames_to_insert,
    )


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
