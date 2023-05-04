- event cameras asynchronously detect changes in each pixels log light intensity, async for each pixel

# interpolation by synthesis

   - Use a synthesis UNet and the voxel grid event cam data to interpolate *I_tau* given *I_0* and *I_1*
   - estimates a new frame by directly fusing the input information from the boundary keyframes and the event sequence
   - trained over 0.1 \* perceptual loss and 1 \* L1 loss (no paper given)
   - perceptual loss involves functions based on losses between high level extracted from both images by pretrained networks, described in `Perceptua Losses for Real-Time Style Transfer and Super-Resolution`-> feature and tyle reconstruction loss

# warping based interpolation

Use the method from `Unsupervised Event-based Learning of Optical Flow, Depth, and Egomotion` to encode **event cam data** estimate optical flow from event cam data

   - encode all data in a **voxel based volume**
   - pass the volume through an NN to predict motion, to reduce motion blur
   - novel **loss function** proposed that **measures motion blur**

   - representation discretizes time, accumulates events linearly
   - two networks trained to remove optical flow (we only use the optical flow one)

- The optical flow synthesis method uses this representation and **differential interpolation** to compute optical flow between keyframes and uses this to warp keyframes
   - keyframes are warped from the starting boundary to the target and the end boundary to the target
   - event sequence polarity flipped to get the "end to middle" interpolated image
   - spatial transformer network used to get optical flow (? need to verify)
   - trained over only l1 loss

   - warps to an intermediate value tau, not sure how

# warping refinement

   - refines the optical flow warping images by taking both optical flow and synthesis images. 
   - based on the assumption that the synthesized image is cloes to ground truth
   - estimate residual optical flow from beginning-to-mid and end-to-mid optical flow imgs,
   - also inpaints occluded regions with nearby region data
   - trained over only l1 loss

# attention averaging
   - takes in the entire volume of all predicted images, synthesized, both refined optical flow images, the optical flow, and an interpoaltion constant tau (bilinear interplolation position for the intermediate frame)
   - gets pixel by pixel weights for synthesized and both refined images, in a unet architrecture
   - optical flow only used to determine weights, not included in average
   - these weights are used to calculate a weighted average over the entire image
   - trained over 0.1 \* perceptual loss and 1 \* L1 loss (no paper given)

# Extra stuff
- Multi Vehicle Stereo Event Camera dataset