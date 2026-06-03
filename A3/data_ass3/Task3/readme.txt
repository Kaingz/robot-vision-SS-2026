Image data and ground truth disparities come from the KITTI data set.
Below is an excerpt of the documentation of the KITTI data sets which describes 
the data format of the GT images.

Data format:
============

Disparity and flow values range [0..256] and [-512..+512] respectively. For
both image types documented MATLAB and C++ I/O functions are provided
within this development kit in the folders matlab and cpp. If you want to
use your own code instead, you need to follow these guidelines:

Disparity maps are saved as uint16 PNG images, which can be opened with
either MATLAB or libpng++. A 0 value indicates an invalid pixel (ie, no
ground truth exists, or the estimation algorithm didn't produce an estimate
for that pixel). Otherwise, the disparity for a pixel can be computed by
converting the uint16 value to float and dividing it by 256.0:

disp(u,v)  = ((float)I(u,v))/256.0;
valid(u,v) = I(u,v)>0;
