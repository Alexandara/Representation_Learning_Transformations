import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
from random import seed
from random import randint
from numpy import savetxt
from PIL import Image
import os

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

def generateRigid():
    transforms = randint(1,7)
    for i in range(len(training_features)):
        transformation = training_features[i]
        if transforms == 7:
            # translation
            x = randint(0,27)
            y = randint(0,27)
            transformation = np.roll(transformation, x, axis=0)
            transformation = np.roll(transformation, y, axis=1)
            # rotation
            degree = randint(1,360)
            radian = np.radians(degree)
            a = np.array([[np.cos(radian), -np.sin(radian)],
                          [np.sin(radian), np.cos(radian)]])
            transformation = linear_transformation(transformation, a)
            # reflection
            degree = randint(1, 360)
            radian = np.radians(degree)
            a = np.array([[np.cos(radian), np.sin(radian)],
                          [np.sin(2 * radian), -np.cos(radian)]])
            transformation = linear_transformation(transformation, a)
        elif transforms == 6:
            # translation
            x = randint(0, 27)
            y = randint(0, 27)
            transformation = np.roll(transformation, x, axis=0)
            transformation = np.roll(transformation, y, axis=1)
            # rotation
            degree = randint(1, 360)
            radian = np.radians(degree)
            a = np.array([[np.cos(radian), -np.sin(radian)],
                          [np.sin(radian), np.cos(radian)]])
            transformation = linear_transformation(transformation, a)
        elif transforms == 5:
            # translation
            x = randint(0, 27)
            y = randint(0, 27)
            transformation = np.roll(transformation, x, axis=0)
            transformation = np.roll(transformation, y, axis=1)
            # reflection
            degree = randint(1, 360)
            radian = np.radians(degree)
            a = np.array([[np.cos(radian), np.sin(radian)],
                          [np.sin(2 * radian), -np.cos(radian)]])
            transformation = linear_transformation(transformation, a)
        elif transforms == 4:
            # rotation
            degree = randint(1, 360)
            radian = np.radians(degree)
            a = np.array([[np.cos(radian), -np.sin(radian)],
                          [np.sin(radian), np.cos(radian)]])
            transformation = linear_transformation(transformation, a)
            # reflection
            degree = randint(1, 360)
            radian = np.radians(degree)
            a = np.array([[np.cos(radian), np.sin(radian)],
                          [np.sin(2 * radian), -np.cos(radian)]])
            transformation = linear_transformation(transformation, a)
        elif transforms == 3:
            # translation
            x = randint(0, 27)
            y = randint(0, 27)
            transformation = np.roll(transformation, x, axis=0)
            transformation = np.roll(transformation, y, axis=1)
        elif transforms == 2:
            # rotation
            degree = randint(1, 360)
            radian = np.radians(degree)
            a = np.array([[np.cos(radian), -np.sin(radian)],
                          [np.sin(radian), np.cos(radian)]])
            transformation = linear_transformation(transformation, a)
        else:
            # reflection
            degree = randint(1, 360)
            radian = np.radians(degree)
            a = np.array([[np.cos(radian), np.sin(radian)],
                          [np.sin(2 * radian), -np.cos(radian)]])
            transformation = linear_transformation(transformation, a)
        training_features[i] = (training_features[i]/2) + (transformation/2)
        
        img = Image.fromarray(np.uint8(training_features[i]*255),'L')
        
        img.save('C:\\Users\\gunne\\Desktop\\CS791\\Rigid\\'+str(i)+'.jpg')

        
        if(i%10==0):
            print(str(i) + "\n")

if __name__ == "__main__":
    generateRigid()