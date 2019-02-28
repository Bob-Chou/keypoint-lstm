import os

import numpy as np
import threading
import tensorflow as tf


def dense(x, z_dim, name, activation=None, reuse=None, regularizer=None):
    """Normal dense layer"""
    x_dim = x.shape[-1]
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable("w", (x_dim, z_dim), regularizer=regularizer)
        b = tf.get_variable("b", z_dim)
        z = tf.add(tf.matmul(x, w), b)
        if activation is not None:
            z = activation(z)
    return z


def random_batch_generator(package, batch_size, max_step=None, max_epoch=None):
    """Get random batch from dataset

    :param package: a list of data to be shuffle
    :param batch_size: size of batch
    :param max_step: maximum step, if given, max_epoch will be ignored
    :param max_epoch: maximum epoch, will be ignored if max_step is given
    :return: a list of bathes of each data in package
    """
    package = list(map(lambda x: np.asarray(x), package))
    max_idx = package[0].shape[0]

    if max_epoch is None and max_step is None:
        raise ValueError("Should give either max_epoch or max_step")
    else:
        if max_step is None:
            max_step = max_epoch * max_idx // batch_size

    offset = 0
    random_idxs = list(range(max_idx))
    np.random.shuffle(random_idxs)
    for step in range(max_step):
        if offset + batch_size > max_idx:
            batch1 = list(map(lambda x: x[random_idxs[offset:]], package))
            np.random.shuffle(random_idxs)
            batch2 = list(map(lambda x:
                              x[random_idxs[:batch_size - max_idx + offset]],
                              package))
            batch = list(map(lambda x: np.concatenate(x), zip(batch1, batch2)))
            offset = batch_size - max_idx + offset
        else:
            batch = list(map(lambda x: x[random_idxs[offset:offset + batch_size]],
                             package))
            offset += batch_size
        yield batch


def walk_dataset(root, data_ext, valid_list=[]):
    """Get path of all samples in dataset directory

    :param root: root directory of dataset
    :param data_ext: extention of data file, e.g. data_ext=".txt"
    :param valid_list: a list contains marks to pick validation set,
                       e.g. valid_list=["s01s02", "s01s03"]
    :return: train_set, valid_set
    """
    samples = []
    dataset = [[], []]
    for root, _, files in os.walk(root, topdown=False):
        samples.extend([os.path.join(root, f) for f in files
                         if os.path.splitext(f)[-1] == data_ext.lower()])
    for s in samples:
        dataset[int(any([v in s for v in valid_list]))].append(s)

    return dataset[0], dataset[1]


def load_batch_parallel(file_list, shape):
    """Load skeleton data vector from disk parrallel

    :param file_list: a list contains all the file path to be loaded
    :param shape: resized shape in [n_frame, n_point, n_dim], e.g. [10, 30, 3]
    :return: a np.darray with the assigned shape
    """
    if len(file_list) == 0:
        return None

    def _load_data_load_data_parallel(file, idx, shape, out):
        n_frame = shape[0]
        with open(file) as f:
            data_array = [list(map(float, line.strip().split(",")))
                          for line in f.readlines()]
            data_array = np.reshape(np.asarray(data_array)[:, 1:],
                                    [-1, shape[1], shape[2]])
        skip_length = data_array.shape[0] // n_frame
        equal_length = skip_length * n_frame
        start = np.random.randint(data_array.shape[0] - equal_length
                                  + skip_length)
        skip_idxs = [start + i * skip_length for i in range(n_frame)]
        out[idx] = (data_array[skip_idxs], skip_idxs)

    collection = list(range(len(file_list)))
    threads = []
    for i, file in enumerate(file_list):
        thread = threading.Thread(
            target=_load_data_load_data_parallel,
            args=(file, i, shape, collection))
        thread.start()
        threads.append(thread)

    for th in threads:
        th.join()
    threads.clear()

    batch, skip_idxs = list(zip(*collection))

    return np.asarray(batch)


