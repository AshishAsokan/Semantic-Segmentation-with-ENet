import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Defining dataset path
camvid = 'evaluation/CamVid'

def decode_image(input_image, height, width):

    """
    Function to decode the One-Hot encoded image

    Returns:
        out: Semantically segmented output for the given encoded input
    """

    # Determining the classes of the pixels
    pred_val = tf.argmax(input_image, -1)

    # Casting to float32 for reshaping
    out = tf.cast(pred_val, dtype=tf.float32)

    # Reshaping to specified height and width
    out = tf.reshape(out, shape=[height, width])
    return out

def process_data(images, labels, height, width, num_classes):

    """
    Function to decode the training images and labels for the dataset

    Parameters:
        images : Tensor containing the train images
        labels : Tensor containing the train labels

    Returns:
        [Tuple] : containing the decoded training images and annotations

    """

    image_list = []
    label_list = []

    for names in zip(images, labels):

        # Reading Image and label
        image = tf.io.read_file(names[0])
        label = tf.io.read_file(names[1])

        # Decoding image and label
        image = tf.image.decode_png(image, channels = 3)
        label = tf.image.decode_png(label)

        # Resizing image and label
        image = tf.image.resize(image, size = (height, width))
        image = tf.cast(tf.convert_to_tensor(image, dtype = tf.float32), dtype = tf.int32)
        label = tf.image.resize(label, size = (height, width))

        # One-hot encoding the class labels
        label = tf.cast(tf.convert_to_tensor(label, dtype = tf.float32), dtype = tf.int32)
        label = tf.reshape(label, shape = [height, width])
        label = tf.one_hot(label, num_classes, axis = -1)

        # # Displaying labels
        # image_out = decode_image(label, height, width)
        # plt.imshow(image_out)
        # plt.show()

        image_list.append(image)
        label_list.append(label)

    # Creating tensor for the datasets
    train_x = tf.convert_to_tensor(tf.stack(image_list), dtype = tf.float32)
    train_y = tf.convert_to_tensor(tf.stack(label_list))
    return (train_x, train_y)

def train_valid_data(set_type : str):

    """
    Prepares the dataset by reading the appropriate directory.

    Parameters:
        set_type : Folder to read images (train or valid)

    Returns:
        dataset: Final decoded dataset (either train or validation)
    """

    # Storing all file names in a list
    train_images = [os.path.join(camvid, set_type + '/images', image) for image in os.listdir(camvid + '/' + set_type + '/images')]
    train_labels = [os.path.join(camvid, set_type + '/labels', image) for image in os.listdir(camvid + '/' + set_type + '/labels')]

    # Converting images to tensors
    images_ten = tf.convert_to_tensor(train_images)
    labels_ten = tf.convert_to_tensor(train_labels)

    # Creating the dataset
    dataset = process_data(images_ten, labels_ten, 512, 512, 32)
    return dataset

def ret_dataset():

    """
    Simple function to return the train and validation sets

    Returns:
        [Tuple]: Containing training set and validation set
    """

    train_set = train_valid_data('train')
    valid_set = train_valid_data('valid')
    return (train_set, valid_set)

if __name__ == '__main__':

    ret_dataset()