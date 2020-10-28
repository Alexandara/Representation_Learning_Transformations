import numpy as np
import tensorflow as tf
import autoencoder
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from random import seed
from random import randint
from numpy import savetxt
from tensorflow.contrib.factorization import KMeans
import glob
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

__author__ = "Abien Fred Agarap, Alexis Tudor, and Gunner Stone"

def uploadData(): # GUNNER You'll need to mess with this to get our data in
    np.random.seed(1)
    tf.random.set_seed(1)
    batch_size = 128
    epochs = 10
    learning_rate = 1e-2
    intermediate_dim = 64 # GUNNER This needs to be checked
    original_dim = 784 # This needs to be checked (but I think it's correct)

    filelist = glob.glob('C:\\Users\\gunne\\Desktop\\CS791\\Representation_Learning_Transformations-main\\Rigid\\*.jpg')
    

    training_features = np.array([np.array(Image.open(fname)) for fname in filelist])
    #(training_features, _), _ = tf.keras.datasets.fashion_mnist.load_data() # instead of raw data, it should be the transformed data
    training_features = training_features / np.max(training_features)
    training_features = training_features.reshape(training_features.shape[0],
                                                  training_features.shape[1] * training_features.shape[2])
    training_features = training_features.astype('float32')

    training_dataset = tf.data.Dataset.from_tensor_slices(training_features)
    training_dataset = training_dataset.batch(batch_size)
    training_dataset = training_dataset.shuffle(training_features.shape[0])
    training_dataset = training_dataset.prefetch(batch_size * 4)

    labels = [0]*60000 # The labels should be 0 for rigid, 1 for non-rigid

    return intermediate_dim, original_dim, learning_rate, epochs, training_features, training_dataset, labels

# https://wizardforcel.gitbooks.io/tensorflow-examples-aymericdamien/content/2.6_kmeans.html
def kmeans(representations, labels):
    # Parameters
    num_steps = 50  # Total steps to train
    batch_size = 1024  # The number of samples per batch
    k = 2  # The number of clusters
    num_classes = 2  # Rigid vs. non-rigid
    num_features = 64  # GUNNER Each image WAS 28x28 pixels, check what the autoencoder spits out as a representation?

    print(type(representations))

    print(type(representations))
    print(type(representations))
    print(type(representations))
    print(type(representations))
    print(type(representations))

    print(np.shape(representations))
    print(np.shape(representations))
    print(np.shape(representations))
    print(np.shape(representations))
    print(np.shape(representations))
    
    # Input images
    X = tf.placeholder(tf.float32, shape=[None, num_features])
    # Labels (for assigning a label to a centroid and testing)
    Y = tf.placeholder(tf.float32, shape=[None, num_classes])

    # K-Means Parameters
    kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine',
                    use_mini_batch=True)

    # Build KMeans graph
    (all_scores, cluster_idx, scores, cluster_centers_initialized, init_op, train_op) = kmeans.training_graph()
    cluster_idx = cluster_idx[0]  # fix for cluster_idx being a tuple
    avg_distance = tf.reduce_mean(scores)

    # Initialize the variables (i.e. assign their default value)
    init_vars = tf.global_variables_initializer()

    # Start TensorFlow session
    sess = tf.Session()

    # Run the initializer
    sess.run(init_vars, feed_dict={X: representations})
    sess.run(init_op, feed_dict={X: representations})

    # Training
    for i in range(1, num_steps + 1):
        _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                             feed_dict={X: representations})
        if i % 10 == 0 or i == 1:
            print("Step %i, Avg Distance: %f" % (i, d))

    # Assign a label to each centroid
    # Count total number of labels per centroid, using the label of each training
    # sample to their closest centroid (given by 'idx')
    counts = np.zeros(shape=(k, num_classes))
    for i in range(len(idx)):
        counts[idx[i]] += labels[i]
    # Assign the most frequent label to the centroid
    labels_map = [np.argmax(c) for c in counts]
    labels_map = tf.convert_to_tensor(labels_map)

    # Evaluation ops
    # Lookup: centroid_id -> label
    cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
    # Compute accuracy
    correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Test Model
    test_x, test_y = representations, labels
    print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))

if __name__ == "__main__":
    intermediate_dim, original_dim, learning_rate, epochs, training_features, training_dataset, labels = uploadData()
    #generateRigid()
    autoencoder = autoencoder.Autoencoder(intermediate_dim=intermediate_dim,original_dim=original_dim)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    print("Checkpoint 1")

    def loss(model, original):
        reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(model(original), original)))
        return reconstruction_error


    def train(loss, model, opt, original):
        with tf.GradientTape() as tape:
            gradients = tape.gradient(loss(model, original), model.trainable_variables)
        gradient_variables = zip(gradients, model.trainable_variables)
        opt.apply_gradients(gradient_variables)

    writer = tf.summary.create_file_writer('tmp')

    print("Checkpoint 2")

    with writer.as_default():
        with tf.summary.record_if(True):
            for epoch in range(epochs):
                print("Epoch #"+str(epoch))
                for step, batch_features in enumerate(training_dataset):
                    train(loss, autoencoder, opt, batch_features)
                    loss_values = loss(autoencoder, batch_features)
                    original = tf.reshape(batch_features, (batch_features.shape[0], 28, 28, 1))
                    reconstructed = tf.reshape(autoencoder(tf.constant(batch_features)),
                                                (batch_features.shape[0], 28, 28, 1))
                    tf.summary.scalar('loss', loss_values, step=step)
                    tf.summary.image('original', original, max_outputs=10, step=step)
                    tf.summary.image('reconstructed', reconstructed, max_outputs=10, step=step)
    print("--Has finished Autoencoder--")
    representations = autoencoder.code
    print("--Working on Kmeans--")
    kmeans(representations, labels)
