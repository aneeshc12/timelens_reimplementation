## MOST NOTES WRITTEN IN run_timelens.py
## STACK TRACE IS AS FOLLOWS

run_timelens -> main()
    run_timelens -> _interpolate()
        attnavg() -> run

## A LOT OF THINGS INVOLVING EXAMPLE HAPPEN IN PLACE

## FORWARD METHODS FOR EACH MODULE RETURN ALL OUTPUTS NEEDED TO CALCULATE LOSS

# Loading event data
- loading event cam data in `event.py`
    - EventSequence loads raw data from npz files, sotres it as numpy arrays

- converted to a voxel based representation in `representation.py`

# architecture
- unet architecture defined in `unet.py`

    ## attention_average
    - 

# actually running it
in `run_timelens.py`
- calls AttentionAverage().from_legacy_checkpoint
- uses `hybrid_storage.py` to load images and npzs together
- _interpolate() generates new frames

1. load checkpoint
2. find all images/event data for each folder
3. interpolate between all frames for the given number of frames
4. convert to an image sequence and save

# eventual anatomy of example

example["middle"]["before_refined_warped"],
example["middle"]["after_refined_warped"],
example["middle"]["before_refined_warped_invalid"],
example["middle"]["after_refined_warped_invalid"],
example["before"]["residual_flow"],
example["after"]["residual_flow"],
example["middle"]["fusion"]

# training details from the paper
or training, we use the Adam
optimizer[12] with standard settings, batches of size 4 and
learning rate 104, which we decrease by a factor of 10 ev-
ery 12 epoch. We train each module for 27 epoch. For the
training, we use large dataset with synthetic events gener-
ated from Vimeo90k septuplet dataset [44] using the video to
events method [6], based on the event simulator from [31].
We train the network by adding and training modules
one by one, while freezing the weights of all previously
trained modules. We train modules in the following or-
der: synthesis-based interpolation, warping-based interpo-
lation, warping refinement, and attention averaging mod-
ules. We adopted this training because end-to-end training
from scratch does not converge, and fine-tuning of the en-
tire network after pretraining only marginally improved the
results. We supervise our network with perceptual [45] and
L1 losses as shown in Fig. 3b, 3d, 3e and 3c. We fine-tune
our network on real data module-by-module in the order
of training. To measure the quality of interpolated images
we use structural similarity (SSIM) [39] and peak signal to
noise ratio (PSNR) metrics