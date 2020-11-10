import numpy as np
import time

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from pykeops.torch import LazyTensor

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

batch_size = 512
epochs = 5
learning_rate = 1e-3

use_cuda = torch.cuda.is_available()
dtype = 'float32' if use_cuda else 'float64'
torchtype = {'float32': torch.float32, 'float64': torch.float64}

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=121
        )
        self.encoder_output_layer = nn.Linear(
            in_features=121, out_features=121
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=121, out_features=121
        )
        self.decoder_output_layer = nn.Linear(
            in_features=121, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code1 = self.encoder_output_layer(activation)
        code = torch.relu(code1)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)

        return reconstructed, code1

def KMeans(x, K=10, Niter=10, verbose=True):
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # K-means loop:
    # - x  is the point cloud,
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    start = time.time()
    c = x[:K, :].clone()  # Simplistic random initialization
    x_i = LazyTensor(x[:, None, :])  # (Npoints, 1, D)

    for i in range(Niter):

        c_j = LazyTensor(c[None, :, :])  # (1, Nclusters, D)
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        Ncl = torch.bincount(cl).type(torchtype[dtype])  # Class weights
        for d in range(D):  # Compute the cluster centroids with torch.bincount:
            c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl

    end = time.time()

    if verbose:
        print("K-means example with {:,} points in dimension {:,}, K = {:,}:".format(N, D, K))
        print('Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n'.format(
                Niter, end - start, Niter, (end-start) / Niter))

    return cl, c

if __name__ == '__main__':
    start = time.time()
    data = np.load("alexis_training_data.npy")
    end = time.time()
    # data[0] = image and labels
    # data[0][0] = image
    # data[0][1] = one hot array with labels
    # data[0][0][0] = one line of image

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    #train_dataset = torchvision.datasets.MNIST(root="~/torch_datasets", train=True, transform=transform, download=True)
    data2 = []
    for i in range(len(data)):
        data2.append(data[i][0])
    train_dataset = torch.Tensor(list(data2))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = AE(input_shape=784).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # mean-squared error loss
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        loss = 0
        for batch_features in train_loader: #Used to have ,_ in front of batch_features
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            batch_features = batch_features.view(-1, 784).to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs, _ = model(batch_features)

            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(train_loader)

        # display the epoch training loss
        print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))

    #test_dataset = torchvision.datasets.MNIST(
     #   root="~/torch_datasets", train=False, transform=transform, download=True
    #)

    data3 = []
    for i in range(len(data)):
        #for j in range(len(data[i][0])):
        data3.append(data[i][0])
    test_dataset = torch.Tensor(list(data2)) # GUNNER THIS NEEDS TO BE A SUBSET OF TESTING DATA

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=10, shuffle=False
    )

    test_examples = None

    with torch.no_grad():
        for batch_features in test_loader:
            batch_features = batch_features[0]
            test_examples = batch_features.view(-1, 784)
            reconstruction, representation = model(test_examples)
            break

    with torch.no_grad():
        number = 1
        plt.figure(figsize=(1, 3))
        for index in range(number):
            # display original
            ax = plt.subplot(3, number, index + 1)
            plt.imshow(test_examples[index].numpy().reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display hidden layer?
            ax = plt.subplot(3, number, index + 1 + number)
            plt.imshow(representation[index].numpy().reshape(11,11))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(3, number, index + 2 + number)
            plt.imshow(reconstruction[index].numpy().reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        #plt.show() # UNCOMMENT TO VIEW AUTOENCODER EXAMPLE

    data4 = []
    for i in range(len(data)):
        rep = []
        temp = []
        for j in range(len(data[i][0])):
            for x in range(len(data[i][0][j])):
                temp.append(data[i][0][j][x])
        temp2 = torch.Tensor(list(temp))
        _, rep = model(temp2)
        data4.append(list(rep))
    #print(np.shape(np.array(data4)))
    #START OF KMEANS SECTION (GUNNER)
    # Right now it is using randomly generated data (x) instead of our representations (data4)
    # KMeans throws an error
    N, D, K = 10000, 2, 50
    x = torch.randn(N, D, dtype=torchtype[dtype]) / 6 + .5
    cl, c = KMeans(x, K=K, Niter=10, verbose=True)
