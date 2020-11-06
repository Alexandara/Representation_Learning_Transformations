import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import cv2
import random
from random import seed
from random import randint
from numpy import savetxt
from PIL import Image
import os
from skimage import data, transform

__author__ = "Abien Fred Agarap"

np.random.seed(1)
tf.random.set_seed(1)
batch_size = 128
epochs = 10
learning_rate = 1e-2
intermediate_dim = 64
original_dim = 784

(training_features, _), _ = tf.keras.datasets.fashion_mnist.load_data()
training_features = training_features / np.max(training_features)

"""
training_features = training_features.reshape(training_features.shape[0],
                                              training_features.shape[1] * training_features.shape[2])
training_features = training_features.astype('float32')
"""

training_dataset = tf.data.Dataset.from_tensor_slices(training_features)
training_dataset = training_dataset.batch(batch_size)
training_dataset = training_dataset.shuffle(training_features.shape[0])
training_dataset = training_dataset.prefetch(batch_size * 4)

# From https://mmas.github.io/linear-transformations-numpy
def linear_transformation(src, a):
    M, N = src.shape
    points = np.mgrid[0:N, 0:M].reshape((2, M*N))
    new_points = np.linalg.inv(a).dot(points).round().astype(int)
    x, y = new_points.reshape((2, M, N), order='F')
    indices = x + N*y
    return np.take(src, indices, mode='wrap')

# From https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions/48097478
def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out

def shearImage(img,sx,sy):
    trans = transform.AffineTransform(shear=-sx)
    img = transform.warp(img, trans, order=1, preserve_range=True, mode='constant')

    trans = transform.AffineTransform(shear=-sy)
    img = np.swapaxes(img, 0, 1)
    img = transform.warp(img, trans, order=1, preserve_range=True, mode='constant')
    img = np.swapaxes(img, 0, 1)

    return img

def generateNonRigid():
    transforms = randint(1,3)
    for i in range(len(training_features)):
        transformation = training_features[i]
        if transforms == 3:
            # zoom
            # assign the zooms equal to eachother
            zoom = 1+random.random()
            transformation = clipped_zoom(transformation,zoom)
            # sheer
            degree = random.random()
            transformation = shearImage(transformation,sx=degree,sy=degree)
        elif transforms == 2:
            # zoom
            # assign the zooms equal to eachother
            zoom = 1+random.random()
            transformation = clipped_zoom(transformation,zoom)
        elif transforms == 1:
             # sheer
            degree = random.random()
            transformation = shearImage(transformation,sx=degree,sy=degree)
        
        training_features[i] = (training_features[i]/2) + (transformation/2)
        
        img = Image.fromarray(np.uint8(training_features[i]*255),'L')
        
        img.save('C:\\Users\\gunne\\Desktop\\CS791\\NonRigid\\'+str(i)+'.jpg')

        
        if(i%10==0):
            print(str(i) + "\n")

if __name__ == "__main__":
    generateNonRigid()