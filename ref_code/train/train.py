# imports
import os
from ..timelens.common import transformers, hybrid_storage, os_tools
from ..timelens.attention_average_network import AttentionAverage

# load data, datasets, dataloaders


# define models
whole_model = AttentionAverage()


# define training params
"""
# loop
    # train synthesis network
        # give it frame triplets
        # generate a center frame from the first and last img
        # L1 loss on the generated and center img
    
    # freeze synth net
        
    # train optical flow net
        # give it frame triplets
        # get forward and backwards optical flows
        # warp to get both forward and backward images
        # l1 loss on both wrt the center image
    
    # freeze of net

    # train refinement net
        # give it triplets
        # get optical flows, warp images, synthesis img
        # l1 loss on both forward and backwards images wrt center
        
    # freeze refinement net

    # train attention net
        
"""


# training loop
def trainLoop(checkpoint_file, root_event_folder, root_image_folder, root_output_folder, number_of_frames_to_skip, number_of_frames_to_insert,):
    (root_image_folder, root_event_folder, root_output_folder) = \
    [os.path.abspath(folder) for folder in [root_image_folder, root_event_folder, root_output_folder]]


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


#       run forward
#       calculate respective costs
#       backprop


