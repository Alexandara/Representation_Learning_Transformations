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
    
![Nonrigid Image](data:image/png;base64,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)
*NonRigid Image with One-hot vector [0,1]*

# (*TODO) Feeding Images into Autoencoder

# (*TODO) Extracting hidden layer & feeding into K-means