import os
import math
import multiprocessing
from os.path import join
from copy import copy
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score
import visual_words


def get_feature_from_wordmap(opts, wordmap):
    """
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    """
    K = opts.K
    bins = np.arange(0, K + 1)
    histogram = np.histogram(wordmap, bins=bins, density=True)[0]
    return histogram


def get_feature_from_wordmap_SPM(opts, wordmap):
    """
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape K*(4^(L+1) - 1) / 3
    """
    K = opts.K
    L = opts.L
    h, w = wordmap.shape
    blocks = 2**L
    row_step = h // blocks
    col_step = w // blocks
    total_blocks = blocks * blocks
    histograms = []
    hist_matrix = np.zeros((blocks, blocks, K))
    for block_num in range(total_blocks):
        col = block_num % blocks
        row = block_num // blocks
        local_map = wordmap[
            row * row_step : (row * row_step) + row_step,
            col * col_step : (col * col_step) + col_step,
        ]
        # breakpoint()
        hist_matrix[row, col, :] = get_feature_from_wordmap(opts, local_map)
        histograms.append(0.5 * hist_matrix[row, col, :])

    prev = hist_matrix
    prev_blocks = blocks
    for level in range(L - 1, -1, -1):
        weight = math.pow(2, level - L - 1) if level > 1 else math.pow(2, -L)
        curr_blocks = prev_blocks // 2
        curr_row_step = prev_blocks // curr_blocks
        curr_col_step = prev_blocks // curr_blocks
        curr = np.zeros((curr_blocks, curr_blocks, K))
        total_blocks = curr_blocks * curr_blocks
        for block_num in range(total_blocks):
            col = block_num % curr_blocks
            row = block_num // curr_blocks
            local_map = prev[
                row * curr_row_step : (row * curr_row_step) + curr_row_step,
                col * curr_col_step : (col * curr_col_step) + curr_col_step,
            ]
            # breakpoint()
            curr[row, col, :] = np.sum(local_map, axis=(0, 1)) / (
                curr_row_step * curr_col_step
            )
            histograms.append(weight * curr[row, col, :])
        prev_blocks = curr_blocks
        prev = curr

    # breakpoint()
    histograms = np.asarray(histograms).flatten()
    return histograms


def get_image_feature(args):
    """
    Extracts the spatial pyramid matching feature.

    [input]
    * arg.opts      : options
    * arg.img_path  : path of image file to read
    * arg.dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K*(4^(L+1) - 1) / 3)
    """
    opts, img_path, dictionary = args
    with Image.open(img_path) as img:
        img = np.array(img).astype(np.float32) / 255
        wordmap = visual_words.get_visual_words(opts, img, dictionary)
        spm_feats = get_feature_from_wordmap_SPM(opts, wordmap)
        return spm_feats


def distance_to_set(word_hist, histograms):
    """
    Compute distance between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    """
    minima = np.minimum(word_hist, histograms)
    sim = np.sum(minima, axis=1)
    return sim


def build_recognition_system(opts, n_worker=1):
    """
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    """
    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, "train_labels.txt"), np.int32)
    dictionary = np.load(join(out_dir, "dictionary.npy"))

    mp_inputs = [
        (opts, join(data_dir, img_path), dictionary) for img_path in train_files
    ]
    features = []
    with multiprocessing.Pool(n_worker) as pool:
        for result in pool.imap_unordered(get_image_feature, mp_inputs):
            features.append(result)

    features = np.stack(features, axis=0)

    # example code snippet to save the learned system
    np.savez_compressed(
        join(out_dir, "trained_system.npz"),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )


def evaluate_one_image(args):
    """
    Evaluates one image
    [input]
    * args.img_features: test image features
    * train_features: train img SPM features
    * train_labels: train labels

    [output]
    * label: test label
    """
    img_features, train_features, train_labels = args
    sims = distance_to_set(img_features, train_features)
    label = train_labels[np.argmax(sims)]
    return label


def evaluate_recognition_system(opts, subset=None, n_worker=1):
    """
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * subset      : to filter out test images whose file name starts with subset
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    """
    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, "trained_system.npz"))
    dictionary = trained_system["dictionary"]
    train_features = trained_system["features"]
    train_labels = trained_system["labels"]

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system["SPM_layer_num"]
    test_files = open(join(data_dir, "test_files.txt")).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, "test_labels.txt"), np.int32)

    if subset is not None:
        indices = [i for i, file in enumerate(test_files) if file.startswith(subset)]
        test_files = [test_files[i] for i in indices]
        test_labels = [test_labels[i] for i in indices]

    mp_inputs = [
        (test_opts, join(data_dir, img_path), dictionary) for img_path in test_files
    ]
    next_mp_inputs = []
    with multiprocessing.Pool(n_worker) as pool:
        for result in pool.imap_unordered(get_image_feature, mp_inputs):
            next_mp_inputs.append((result, train_features, train_labels))

    # breakpoint()
    ypred = []
    with multiprocessing.Pool(n_worker) as pool:
        for result in pool.imap_unordered(evaluate_one_image, next_mp_inputs):
            ypred.append(result)

    ypred = np.asarray(ypred)

    if subset is not None:
        print("=" * 20)
        indices = np.arange(len(test_labels))
        indices = indices[ypred != test_labels]
        print(f"Showing Misclassification for {subset}, total: {len(indices)}")
        for ind in indices:
            print(
                f"misclassified file name = {test_files[ind]} | predicted = {ypred[ind]} | true label = {test_labels[ind]}"
            )
        print("=" * 20)

    conf = confusion_matrix(test_labels, ypred)
    acc = accuracy_score(test_labels, ypred)
    return conf, acc
