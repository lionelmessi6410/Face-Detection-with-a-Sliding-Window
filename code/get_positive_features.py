import numpy as np
import os
import random
from cyvlfeat.hog import hog
from skimage.io import imread
from tqdm import tqdm
import glob

# you can implement your own data augmentation functions

def get_positive_features(train_path_pos, feature_params):
    """
    FUNC: This function should return all positive training examples (faces)
        from 36x36 images in 'train_path_pos'. Each face should be converted
        into a HoG template according to 'feature_params'. For improved performances,
        try mirroring or warping the positive training examples.
    ARG:
        - train_path_pos: a string; directory that contains 36x36 images
                          of faces.
        - feature_params: a dict; with keys,
                          > template_size: int (probably 36); the number of
                            pixels spanned by each train/test template.
                          > hog_cell_size: int (default 6); the number of pixels
                            in each HoG cell.
                          Template size should be evenly divisible by hog_cell_size.
                          Smaller HoG cell sizez tend to work better, but they
                          make things slower because the feature dimenionality
                          increases and more importantly the step size of the
                          classifier decreases at test time.
    RET:
        - features_pos: (N,D) ndarray; N is the number of faces and D is the
                        template dimensionality, which would be,
                        (template_size/hog_cell_size)^2 * 31,
                        if you're using default HoG parameters.
    """
    #########################################
    ##          you code here              ##
    #########################################
    
    print("START: get_positive_features")
    
    temp_size = feature_params['template_size']
    cell_size = feature_params['hog_cell_size']
    
    features_pos = []
    for filename in glob.glob(train_path_pos + '/*.jpg'):
        img = imread(filename)
        img_mirror = img[:, ::-1]
        hog_features = hog(img, cell_size=cell_size)
        hog_features_mirror = hog(img_mirror, cell_size=cell_size)
        hog_features = hog_features.flatten()
        hog_features_mirror = hog_features_mirror.flatten()
        features_pos.append(hog_features)
        features_pos.append(hog_features_mirror)
    
    features_pos = np.asarray(features_pos)
    
    print("DONE: get_positive_features")
    print("Shape:", features_pos.shape)
    
    #########################################
    ##          you code here              ##
    #########################################

    return features_pos