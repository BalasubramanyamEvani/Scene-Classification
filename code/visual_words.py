import os
import multiprocessing
from os.path import join, isfile
import random
import numpy as np
import scipy.ndimage
import skimage.color
from PIL import Image
from sklearn.cluster import KMeans


def extract_filter_responses(opts, img):
    """
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3) or (H,W,4) with range [0, 1]
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    """
    filter_scales = opts.filter_scales
    if len(img.shape) != 3:
        img = img[:, :, np.newaxis]
    if img.shape[2] > 3:
        img = img[:, :, :3]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    img = skimage.color.rgb2lab(img)
    F = 4 * len(filter_scales)
    filter_responses = np.zeros((img.shape[0], img.shape[1], img.shape[2] * F))
    index = 0
    for scale in filter_scales:
        for channel in range(0, 3):
            filter_responses[:, :, index * 3 + channel] = scipy.ndimage.gaussian_filter(
                img[:, :, channel], sigma=scale
            )
        index += 1
        for channel in range(0, 3):
            filter_responses[
                :, :, index * 3 + channel
            ] = scipy.ndimage.gaussian_laplace(img[:, :, channel], sigma=scale)
        index += 1
        for channel in range(0, 3):
            filter_responses[:, :, index * 3 + channel] = scipy.ndimage.gaussian_filter(
                img[:, :, channel], sigma=scale, order=(0, 1)
            )
        index += 1
        for channel in range(0, 3):
            filter_responses[:, :, index * 3 + channel] = scipy.ndimage.gaussian_filter(
                img[:, :, channel], sigma=scale, order=(1, 0)
            )
        index += 1
    return filter_responses


def compute_dictionary_one_image(args):
    """
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    """
    opts, data_dir_img, feat_dir_img = args
    with Image.open(data_dir_img) as img:
        img = np.array(img).astype(np.float32) / 255
        filter_responses = extract_filter_responses(opts, img)
        h, w, nf = filter_responses.shape
        # want 1 x alpha x 3F, 1 since only 1 image processed
        reshaped_filter_responses = np.reshape(filter_responses, (h * w, nf))
        rand_indices = random.sample(range(0, h * w), opts.alpha)
        filter_responses = reshaped_filter_responses[rand_indices, :]
        with open(feat_dir_img, "wb") as f:
            np.save(f, filter_responses)
        return feat_dir_img


def compute_dictionary(opts, n_worker=1):
    """
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel

    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    """
    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()

    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    mp_inputs = [
        (
            opts,
            join(data_dir, img_path),
            join(feat_dir, "_".join(os.path.splitext(img_path)[0].split("/")) + ".npy"),
        )
        for img_path in train_files
    ]
    feature_files = []
    with multiprocessing.Pool(n_worker) as pool:
        for result in pool.imap_unordered(compute_dictionary_one_image, mp_inputs):
            feature_files.append(result)

    assert len(feature_files) == len(mp_inputs), print(
        "Not all images processed while creating BOW"
    )

    features = []
    for feature_file in feature_files:
        features.append(np.load(feature_file))

    features = np.concatenate(features, axis=0)

    # example code snippet to save the dictionary
    kmeans = KMeans(n_clusters=K).fit(features)
    dictionary = kmeans.cluster_centers_

    np.save(join(out_dir, "dictionary.npy"), dictionary)


def get_visual_words(opts, img, dictionary):
    """
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    """
    h, w = img.shape[0], img.shape[1]
    wordmap = np.zeros((h, w))
    filter_response = extract_filter_responses(opts, img)
    filter_response = np.reshape(
        filter_response,
        (filter_response.shape[0] * filter_response.shape[1], filter_response.shape[2]),
    )
    distances = scipy.spatial.distance.cdist(
        filter_response, dictionary, metric="euclidean"
    )
    wordmap = np.array([np.argmin(pixel) for pixel in distances]).reshape(h, w)
    return wordmap
