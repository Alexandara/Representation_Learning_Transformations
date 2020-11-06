# Autoencoder Based Representations of Rigid andNon-Rigid Transformations for Classification

This project creates a representation of sets of two images representing transformations in a reduced feature space.  This representation can then be graphed based on the reduced features and thenclustered for an easy and fast classification of image transformations..

  - Preprocess images
  - *(TODO) Feed images into autoencoder*
  - *(TODO) Extract hidden layer & feed into k-means classifier*

# Preprocessing Images

  - Extract `Images.zip` into working directory
  - Create a conda virtual environment
  - Install dependencies from `requirements.txt`
  - Open `data_preprocessing.ipynb`  and run the notebook to generate `training_data.npy`

Reading `training_data.npy`

  - 120,000 image/label pairs
  - Images are 28x28; 0-255 grayscale 
  - Labels are one-hot vectors 
    * [1,0] is Rigid ; [0,1] is Nonrigid
    
![Nonrigid Image](https://i.imgur.com/IySuxbs.png)
*NonRigid Image with One-hot vector [0,1]*

# (*TODO) Feeding Images into Autoencoder

# (*TODO) Extracting hidden layer & feeding into K-means