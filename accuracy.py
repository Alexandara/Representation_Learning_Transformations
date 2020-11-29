import numpy as np
import time

import matplotlib.pyplot as plt
import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


"""
N = total dimensions of representation (for 8x8 representation n = 64)
clusters = 2 x N array representing the two cluster centers of k means
data = NumImages x 2 array containing flattened representations and labels 
        (the "2" contains an N dimension flattened representation and a size 2 one hot array for labels)
        
Data is as follows:
data[x] = set of one flattened representation with labels
data[x][0] = N size flattened representation
data[x][1] = one hot array with labels
"""
def accuracy(clusters, data):
    cluster1 = 0
    cluster2 = 0
    for i in range(len(data)):
        dist1 = 0
        dist2 = 0
        for j in range(len(data[i][0])):
            dist1 = dist1 + ((data[i][0][j] + clusters[0][j]))**2
            dist2 = dist2 + ((data[i][0][j] + clusters[1][j])) ** 2
        dist1 = math.sqrt(dist1)
        dist2 = math.sqrt(dist2)
        if dist1 > dist2:
            if data[i][1][0] == 0:
                cluster1 = cluster1 - 1
            else:
                cluster1 = cluster1 + 1
        else:
            if data[i][1][0] == 0:
                cluster2 = cluster2 - 1
            else:
                cluster2 = cluster2 + 1
    # 1 is for label = [1, 0]
    # -1 is for label = [0,1]
    if cluster1 >= cluster2:
        cluster1 = 1
        cluster2 = -1
    else:
        cluster1 = -1
        cluster2 = 1
    acc = 0
    for i in range(len(data)):
        dist1 = 0
        dist2 = 0
        for j in range(len(data[i][0])):
            dist1 = dist1 + ((data[i][0][j] + clusters[0][j])) ** 2
            dist2 = dist2 + ((data[i][0][j] + clusters[1][j])) ** 2
        dist1 = math.sqrt(dist1)
        dist2 = math.sqrt(dist2)
        if dist1 > dist2: # If belongs to cluster1
            if data[i][1][0] == 0: # If non-rigid (label = [0, 1]
                if cluster1 < 0:
                    acc = acc + 1
            else: # if rigid (label = [1,0])
                if cluster1 > 0:
                    acc = acc + 1
        else: # If belongs to cluster2
            if data[i][1][0] == 0: # If non-rigid (label = [0, 1]
                if cluster2 < 0:
                    acc = acc + 1
            else: # if rigid (label = [1,0])
                if cluster2 > 0:
                    acc = acc + 1
    acc = acc/len(data)
    return acc

if __name__ == '__main__':
    #data[0] = image and labels
    # data[0][0] = image
    # data[0][1] = one hot array with labels
    # data[0][0][0] = full flattened representation
    data = np.array([[np.array([2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                                          1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]), np.array([1,0])],
            [np.array([3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                                 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]), np.array([1,0])]])
    # clusters[0] = first cluster center
    # clusters[1] = second cluster center
    clusters = np.array([np.array([4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]),
           np.array([5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])])


    print("Shape of centroids: "+str(np.shape(clusters)))
    print("Shape of data: "+str(np.shape(data)))
    print("Shape of latent image: "+str(np.shape(data[0][0])))
    print("Shape of latent label: "+str(np.shape(data[0][1])))
      
    acc = accuracy(clusters, data)
    print(acc)