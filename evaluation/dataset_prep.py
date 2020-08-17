import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Defining dataset path
camvid = 'Evaluation/CamVid'

def process_data(images, labels):

    """
    Function to decode the training images and labels for the dataset

    Parameters:
        images : Tensor containing the train images
        labels : Tensor containing the train labels

    Returns:
        Tuple containing the decoded training images and annotations

    """

    image_list = []
    label_list = []

    for names in zip(images, labels):

        # Reading Image and label
        image = tf.io.read_file(names[0])
        label = tf.io.read_file(names[1])

        # Decoding image and label
        image = tf.image.decode_image(image, channels = 3)
        label = tf.image.decode_image(label, channels = 3)

        image_list.append(image)
        label_list.append(label)

    # Creating tensor for the datasets
    train_x = tf.convert_to_tensor(tf.stack(image_list))
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
    # input_queue = tf.data.Dataset.from_tensor_slices((images_ten, labels_ten))

    # Creating the dataset
    dataset = process_data(images_ten, labels_ten)
    return dataset

def ret_dataset():

    """
    Simple function to return the train and validation sets

    Returns:
        [Tuple]: Containing training set and validation set
    """

    train_set = train_valid_data('train')
    valid_set = train_valid_data('valid')

    print(train_set[0].shape)
    print(train_set[1].shape)
    print(valid_set[0].shape)
    print(valid_set[1].shape)
    return (train_set, valid_set)

ret_dataset()