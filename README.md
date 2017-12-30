# Computer Vision assignment

This assignment makes use of the SIFT descriptor [1].

Specifically, descriptors from two images are gotten and afterwards they are matched following two methods (brute force and knnMatch). 
Lastly, a mosaic is created by using descriptors and homographies. 

OpenCV (artificial vision library) is used in this assignment.  


# Use

Set folder "image" and file "main.py" in the same path.

Run main.py and see results. 

In folder "images" you can find the images used by main.py

In case you would like to change the images you work with, you need to:

1. Add the new images to the "image" folder.
2. Change the image's paths in the "main" section in main.py


# References
[1] Lowe, D. (2004). Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision, 60(2), pp.91-110.
