# Image-Representation-and-Retrieval-Methods
Simple image representations and use kNN classification method to obtain categories of the query images, gabor filters to detect different orientations in the images, SIFT feature vectors extraction by using a built-in function and BoW descriptor extraction


### FILES

* part_1.py (Gabor filter bank)
* part_2.py (SIFT)
* part_3.py (BoW)
* part_4.py (BoW with spatial tiling)



### COMMON FUNCTIONS

find_five_most_similar() : For achieve finding five most similar image experiment
> Open comment lines 92,93 (part_1.py)
> Open comment lines 91,92 (part_2.py)
> Open comment lines 104,105 (part_3.py)
> Open comment lines 131,132 (part_4.py)



### DEPENDENCIES

`pip install opencv-python
...
import cv2 (part_1.py, part_2.py, part_3.py, part_4.py)`


`pip install numpy
...
import numpy as np (part_1.py, part_2.py, part_3.py, part_4.py)`


`pip install scipy
...
from scipy.spatial import distance (part_1.py, part_2.py, part_3.py, part_4.py)`


`pip install sklearn
...
from sklearn.preprocessing import normalize (part_1.py)
from scipy.spatial import distance (part_2.py)
from sklearn.cluster import KMeans (part_3.py, part_4.py)`

