- event cameras asynchronously detect changes in each pixels log light intensity, async for each pixel

Use the method from `Unsupervised Event-based Learning of Optical Flow, Depth, and Egomotion` to encode **event cam data** estimate optical flow from event cam data

   - encode all data in a **voxel based volume**
   - pass the volume through an NN to predict motion, to reduce motion blur
   - novel **loss function** proposed that **measures motion blur**

   - representation discretizes time, accumulates events linearly
   - two networks trained to remove optical flow (we only use the optical flow one)

- The optical flow synthesis method uses this representation and **differential interpolation** to compute optical flow between keyframes and uses this to warp keyframes
   - keyframes are warped from the starting boundary to the target and the end boundary to the target
   - spatial transformer network used to get optical flow (? need to verify)

# Extra stuff
- Multi Vehicle Stereo Event Camera dataset