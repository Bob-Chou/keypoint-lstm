import os

import numpy as np
import threading
import tensorflow as tf
from configparser import ConfigParser


def dense(x, z_dim, name, activation=None, reuse=None, regularizer=None):
    """Normal dense layer

    :param x: input tensor
    :param z_dim: output dimension
    :param name: name of operation
    :param activation: activation function
    :param reuse: reuse tf.Variable
    :param regularizer: regularizer of parameters
    :return: the output of dense layer
    """
    x_dim = x.shape[-1]
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable("w", (x_dim, z_dim), regularizer=regularizer)
        b = tf.get_variable("b", 1)
        z = tf.matmul(x, w) + b
        if activation is not None:
            z = activation(z)
    return z


def attn_dense(x, h, attn_dim, name, activation=tf.nn.tanh, reuse=False,
               regularizer=None, rescale=False):
    """Spatio attention

    :param x: input x
    :param h: input h from lstm
    :param attn_dim: dimension of attention
    :param name: name of operation
    :param activation: activation function
    :param reuse: reuse tf.Variable
    :param regularizer: regularizer of parameters
    :param rescale: whether to rescale the attention
    :return: attention tensor
    """
    x_dim = x.shape[-1]
    h_dim = h.shape[-1]
    with tf.variable_scope(name, reuse=reuse):
        wxs = tf.get_variable("wxs", (x_dim, attn_dim), regularizer=regularizer)
        whs = tf.get_variable("whs", (h_dim, attn_dim), regularizer=regularizer)
        bs = tf.get_variable("bs", 1)
        attn = tf.matmul(x, wxs) + tf.matmul(h, whs) + bs

        if rescale:
            us = tf.get_variable("us", 1, regularizer=regularizer)
            bus = tf.get_variable("bus", 1, regularizer=regularizer)
        else:
            us = 1
            bus = 0

        if activation is not None:
            attn = activation(attn)

    return us * attn + bus


def add_summary(summary_list, collection_list):
    """Add summary to assigned collection

    :param summary_list: a list of tf summary object
    :param collection_list: a list contains all the collection for all summary
                            objects to be added
    :return: None
    """
    list(map(lambda x: x.extend(summary_list), collection_list))


def filter_vars(var_list, freeze_list):
    """Return the filtered variable list

    :param var_list: all variable list
    :param freeze_list: key words for variables to be left out
    :return:
    """
    return [v for v in var_list if all([k not in v.name for k in freeze_list])]


def pick_vars(var_list, pick_list):
    """Return the picked variable list

    :param var_list: all variable list
    :param pick_list: key words for variables to be picked out
    :return:
    """
    return [v for v in var_list if any([k in v.name for k in pick_list])]


def get_tensor(g, name):
    """Get a tensor in the graph by name

    :param g:
    :param name:
    :return: the tensor
    """
    return g.get_tensor_by_name(name)


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


def load_batch_parallel(file_list, shape, as_flow=False):
    """Load skeleton data vector from disk parrallel

    :param file_list: a list contains all the file path to be loaded
    :param shape: resized shape in [n_frame, n_point, n_dim], e.g. [10, 30, 3]
    :param as_flow: whether to return the flow formatted vector
    :return: a np.darray with the assigned shape
    """
    if len(file_list) == 0:
        return None

    def _load_data_parallel(file, idx, shape, out):
        n_frame = shape[0] + int(as_flow)
        with open(file) as f:
            data_array = [list(map(float, line.strip().split(",")))
                          for line in f.readlines()]
            data_array = np.reshape(np.asarray(data_array)[:, 1:],
                                    [-1, shape[1], shape[2]])
        real_length = data_array.shape[0]
        if real_length >= n_frame:
            skip_length = data_array.shape[0] // n_frame
            equal_length = skip_length * n_frame
            start = np.random.randint(data_array.shape[0] - equal_length
                                      + skip_length)
            selected_idxs = [start + i * skip_length for i in range(n_frame)]
        else:
            data_array = np.pad(data_array,
                                [[0, n_frame-real_length], [0, 0], [0, 0]],
                                "edge")
            selected_idxs = list(range(real_length)) + \
                            [-1 for _ in range(n_frame - real_length)]
        out[idx] = (data_array[selected_idxs], selected_idxs)

    collection = list(range(len(file_list)))
    threads = []
    for i, file in enumerate(file_list):
        thread = threading.Thread(
            target=_load_data_parallel,
            args=(file, i, shape, collection))
        thread.start()
        threads.append(thread)

    for th in threads:
        th.join()
    threads.clear()

    batch, idxs = list(map(np.asarray, zip(*collection)))

    if as_flow:
        batch = batch[:, 1:, ...] - batch[:, :-1, ...]
        idxs = idxs[:, :-1]

    return batch, idxs

def get_config(filename):
    """Read and parse the config file

    :param filename: path to the config file
    :return: configure
    """
    class Config(object):
        pass

    reader = ConfigParser()
    reader.read(os.path.expanduser(filename))
    config = Config()

    config.data = Config()
    config.data.root = reader.get("data", "root")
    config.data.ext = reader.get("data", "ext")
    config.data.as_flow = reader.getboolean("data", "as_flow")
    config.data.mean = list(map(float,
                                reader.get("data", "mean").split(",")))
    config.data.std = list(map(float,
                               reader.get("data", "std").split(",")))
    config.model = Config()
    config.model.n_frame = reader.getint("model", "n_frame")
    config.model.n_class = reader.getint("model", "n_class")
    config.model.n_keypoint = reader.getint("model", "n_keypoint")
    config.model.n_dim = reader.getint("model", "n_dim")
    config.model.n_lstm = reader.getint("model", "n_lstm")
    config.model.n_hidden = reader.getint("model", "n_hidden")
    config.model.learning_rate = reader.getfloat("model", "learning_rate")
    config.model.lambda1 = reader.getfloat("model", "lambda1")
    config.model.lambda2 = reader.getfloat("model", "lambda2")
    config.model.lambda3 = reader.getfloat("model", "lambda3")
    config.model.use_bn = reader.getboolean("model", "use_bn")
    config.model.use_res = reader.getboolean("model", "use_res")


    config.train = Config()
    config.train.batch_size = reader.getint("train", "batch_size")
    config.train.n_pretrain_epoch = reader.getint("train", "n_pretrain_epoch")
    config.train.n_finetune_epoch = reader.getint("train", "n_finetune_epoch")
    config.train.valid_set = reader.get("train", "valid_set").split(",")
    config.train.valid_per_step = reader.getint("train", "valid_per_step")
    config.train.save_per_step = reader.getint("train", "save_per_step")

    return config
