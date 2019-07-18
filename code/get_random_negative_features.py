import numpy as np
import os
import random
from cyvlfeat.hog import hog
from skimage.io import imread
from skimage.transform import pyramid_gaussian
from skimage.transform import resize
from skimage import color
from tqdm import tqdm
import glob
import imgaug as ia
from imgaug import augmenters as iaa

# you may implement your own data augmentation functions

def get_random_negative_features(non_face_scn_path, feature_params, num_samples):
    '''
    FUNC: This funciton should return negative training examples (non-faces) from
        any images in 'non_face_scn_path'. Images should be converted to grayscale,
        because the positive training data is only available in grayscale. For best
        performance, you should sample random negative examples at multiple scales.
    ARG:
        - non_face_scn_path: a string; directory contains many images which have no
                             faces in them.
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
        - features_neg: (N,D) ndarray; N is the number of non-faces and D is 
                        the template dimensionality, which would be, 
                        (template_size/hog_cell_size)^2 * 31,
                        if you're using default HoG parameters.
        - neg_examples: TODO
    '''
    #########################################
    ##          you code here              ##
    #########################################
    
    print("START: get_random_negative_features")

    temp_size = feature_params['template_size']
    cell_size = feature_params['hog_cell_size']
    
    features_neg = []
    
    i = 0
    while i < num_samples:
        for filename in glob.glob(non_face_scn_path + '/*.jpg'):
            if i >= num_samples:
                break
            img = imread(filename)
            img = color.rgb2gray(img)
#             img = resize(img, (temp_size, temp_size))
            
            sometimes = lambda aug: iaa.Sometimes(0.5, aug)
            
            # Define our sequence of augmentation steps that will be applied to every image
            # All augmenters with per_channel=0.5 will sample one value _per image_
            # in 50% of all cases. In all other cases they will sample new values
            # _per channel_.
            seq = iaa.Sequential(
                [
                    # apply the following augmenters to most images
                    iaa.Fliplr(0.5), # horizontally flip 50% of all images
                    iaa.Flipud(0.5), # vertically flip 50% of all images
                    # crop images by -5% to 10% of their height/width
                    sometimes(iaa.CropAndPad(
                        percent=(-0.05, 0.1),
                        pad_mode=ia.ALL,
                        pad_cval=(0, 255)
                    )),
                    sometimes(iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                        rotate=(-45, 45), # rotate by -45 to +45 degrees
                        shear=(-16, 16), # shear by -16 to +16 degrees
                        order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                        cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                        mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    )),
                ],
                random_order=True
            )

            seq_det = seq.to_deterministic()
            img_aug = seq_det.augment_images([img])[0]
            
            rand_x = np.random.randint(img_aug.shape[0]-temp_size)
            rand_y = np.random.randint(img_aug.shape[1]-temp_size)
            img_aug = img_aug[rand_x:rand_x+temp_size, rand_y:rand_y+temp_size]
            
            hog_features = hog(img_aug, cell_size=cell_size)
            hog_features = hog_features.flatten()
            features_neg.append(hog_features)
            i+=1
    
    features_neg = np.asarray(features_neg)
    neg_examples = num_samples
    
    print("DONE: get_random_negative_features")
    print("Shape:", features_neg.shape)
    
    #########################################
    ##          you code here              ##
    #########################################
            
    return features_neg, neg_examples