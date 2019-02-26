import os
import json
import re

import threading
import queue
import numpy as np
import tensorflow as tf
import imageio
from PIL import Image
from PIL import ImageDraw



def dense(x, z_dim, name, activation=None, reuse=None, regularizer=None):
    """Normal dense layer"""
    x_dim = x.shape[-1]
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable("w", (x_dim, z_dim), regularizer=regularizer)
        b = tf.get_variable("b", z_dim)
        z = tf.add(tf.matmul(x, w), b)
        if activation is not None:
            z = activation(z)
        if not reuse:
            summarize(w)
            summarize(b)
    return z


def attn_dense(x, h, z_dim, name, activation=None, reuse=None,
               regularizer=None):
    """Attention dense layer"""
    x_dim = x.shape[-1]
    h_dim = h.shape[-1]
    with tf.variable_scope(name, reuse=reuse):
        wx = tf.get_variable("wx", (x_dim, z_dim), regularizer=regularizer)
        wh = tf.get_variable("wh", (h_dim, z_dim), regularizer=regularizer)
        b = tf.get_variable("b", z_dim)
        z = tf.add(tf.add(tf.matmul(x, wx), tf.matmul(h, wh)), b)
        if activation is not None:
            z = activation(z)
        if not reuse:
            summarize(wx)
            summarize(wh)
            summarize(b)
    return z


def summarize(var):
    """Define summaries of a tensor."""
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def get_random_batch(package, random_idxs, offset, batch_size):
    """Get random batch from dataset

    :param x: data
    :param y: label
    :param random_idxs: reference to index list, will be shuffle automatically
    :param offset: current offset, batch will start here
    :param batch_size: size of batch
    :return: batch of data, batch of label, updated offset
    """
    package = list(map(lambda x: np.asarray(x), package))
    max_idx = package[0].shape[0]
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
    return batch, offset

def draw_debug_image(feature, filepath):
    """Draw debug image of input

    :param feature: data in the format of (keypoints, 2)
    :param filepath: image path of data
    :return: debug map of feature
    """
    img = Image.open(filepath)
    drawer = ImageDraw.Draw(img)
    r = 10
    pairs = [
        (0, 1),
        (0, 15),
        (0, 16),
        (1, 2),
        (1, 5),
        (1, 8),
        (2, 3),
        (3, 4),
        (5, 6),
        (6, 7),
        (8, 9),
        (8, 12),
        (9, 10),
        (10, 11),
        (11, 22),
        (11, 24),
        (12, 13),
        (13, 14),
        (14, 19),
        (14, 21),
        (15, 17),
        (16, 18),
        (19, 20),
        (22, 23)]
    # draw keypoints
    for point in feature:
        x, y = point
        drawer.ellipse((x-r, y-r, x+r, y+r), fill=(255, 0, 0))
    threads = []
    # draw lines
    for pair in pairs:
        start = pair[0]
        end = pair[1]
        drawer.line([feature[start][0], feature[start][1], feature[end][0],
                   feature[end][1]], fill=(255, 0, 0), width=5)
    return img


def write_debug_gif(features, filepaths, tar_path, fps=10):
    """Draw debug gif of input

    :param features: array of feature in the format of (frames, keypoints, 2)
    :param filepaths: array of image path
    :param tar_path: save path of generated gif
    :return: none
    """
    def _draw_debug_image_parrallel(feature, filepath, idx, out):
        out[idx] = draw_debug_image(feature, filepath)

    debug_images = list(range(features.shape[0]))
    threads = []
    for idx, (feature, filepath) in enumerate(zip(features, filepaths)):
        thread = threading.Thread(
            target=_draw_debug_image_parrallel,
            args=(feature, filepath, idx, debug_images))
        thread.start()
        threads.append(thread)
    for th in threads:
        th.join()
    imageio.mimsave(tar_path, debug_images, 'GIF', duration=1/float(fps))
    # writeGif(tar_path, debug_images, duration=1/float(fps), subRectangles=False)


class FeatureParser(object):
    """docstring for FeatureParser"""
    def __init__(self,
                 feature_dir,
                 image_dir,
                 data_format=(15, 25, 2),
                 image_size=(1280, 720),
                 crop=(0, 0, 1280, 720)):
        super(FeatureParser, self).__init__()
        self.data_format = data_format
        self.image_size = image_size
        self.crop = (max(0, crop[0]),
                     max(0, crop[1]),
                     min(image_size[0], crop[2]),
                     min(image_size[1], crop[3]))
        self.motions_path = []
        self.labels = []
        self.label_to_name = {}
        self.name_to_label = {}
        self.map_to_image = lambda p: os.path.join(image_dir,"/"\
                                                        .join((p[:-15]+".jpg")\
                                                              .split("/")[-3:]))
        feature_dirs = os.listdir(feature_dir)
        for feature_name in feature_dirs:
            abs_feature_dir = os.path.join(feature_dir, feature_name)
            if not os.path.isdir(abs_feature_dir):
                continue
            label = len(self.name_to_label)
            self.name_to_label[feature_name] = label
            self.label_to_name[label] = feature_name
            subject_dirs = os.listdir(abs_feature_dir)
            for motion_name in subject_dirs:
                abs_motion_dir = os.path.join(abs_feature_dir, motion_name)
                if not os.path.isdir(abs_motion_dir):
                    continue
                self.motions_path.append(abs_motion_dir)
                self.labels.append(label)

    def normalize(self, data_array):
        wcen = (self.crop[2] + self.crop[0]) / 2
        hcen = (self.crop[3] + self.crop[1]) / 2
        wmax = (self.crop[2] - self.crop[0])
        hmax = (self.crop[3] - self.crop[1])
        m = np.asarray([wcen, hcen])
        r = np.asarray([wmax, hmax])
        return (data_array - m) / r

    def denormalize(self, data_array, resize=1):
        wcen = (self.crop[2] + self.crop[0]) / 2
        hcen = (self.crop[3] + self.crop[1]) / 2
        wmax = (self.crop[2] - self.crop[0])
        hmax = (self.crop[3] - self.crop[1])
        m = np.asarray([wcen, hcen])
        r = np.asarray([wmax, hmax])
        return data_array * r + m

    def get_motion_vector(self, motion_dirs, labels):

        def _load_single_frame_parrallel(n_points, n_dims, path, idx, out):
            """ load frame vector with multi-thread"""
            with open(path) as fp:
                frame_vector = json.load(fp)["people"][0]["pose_keypoints_2d"]
                frame_vector = np.asarray(frame_vector).reshape(-1,
                                                                n_dims + 1)
                frame_vector[frame_vector[:, -1] < 0.1] = [(self.crop[2] + self.crop[0]) / 2, (self.crop[3] + self.crop[1]) / 2, 0]
                out[idx] = frame_vector

        def _append_single_motion_parrallel(lock, n_points, n_dims, motion_dir,
                                            label, out):
            """ load motion vector with multi-thread"""
            p = re.compile("^\d+_keypoints\.json$")
            frame_paths = [os.path.join(motion_dir, m)
                           for m in os.listdir(motion_dir) if bool(p.match(m))]
            frame_paths = sorted(frame_paths, key=lambda x: int(x[-18:-15]))
            motion_vector = list(range(len(frame_paths)))
            threads = []
            for frame_path in frame_paths:
                frame_id = int(os.path.basename(frame_path).split("_")[0]) - 1
                thread = threading.Thread(
                    target=_load_single_frame_parrallel,
                    args=(n_points, n_dims, frame_path, frame_id,
                          motion_vector))
                thread.start()
                threads.append(thread)
            for th in threads:
                th.join()
            motion_vector = np.asarray(motion_vector)[..., :-1]
            if lock.acquire():
                out.append((motion_vector, label, frame_paths))
                lock.release()

        lock = threading.Lock()
        collection = []
        threads = []

        _, n_points, n_dims = self.data_format
        for j, (motion_dir, label) in enumerate(zip(motion_dirs, labels)):
            thread = threading.Thread(
                target=_append_single_motion_parrallel,
                args=(lock, n_points, n_dims, motion_dir, label, collection))
            thread.start()
            threads.append(thread)
        for th in threads:
            th.join()
        threads.clear()
        _x, _y, _paths = tuple(map(list, tuple(zip(*collection))))
        _x = np.asarray(_x)
        _y = np.asarray(_y, dtype=np.int32)
        return _x, _y, _paths


    def batch_parser(self, step=1, batch_size=16,
                     random_skip=False, external=False):
        """An iterator to parse the data list and return the data batch

        :param step: maximum step of generating data batches
        :param batch_size: size of batch
        :param random_skip: whether adopt random skip size for skip-frame
        :param external: whether use external mode (load file when training)
        :return: batch of data, batch of label
        """
        def _append_skip_frame_parrallel(lock, n_frames, motion_vector,
                                         label, path, out):
            skip_length = motion_vector.shape[0] // n_frames
            equal_length = skip_length * n_frames
            start = np.random.randint(motion_vector.shape[0] - equal_length
                                      + skip_length)
            skip_vector = [start + i * skip_length for i in range(n_frames)]
            skip_path = [path[start + i * skip_length] for i in range(n_frames)]
            if lock.acquire():
                out.append((motion_vector[skip_vector], label, skip_path))
                lock.release()

        lock = threading.Lock()
        threads = []
        batch_array = []

        # if not external mode, load all the data into memory
        if not external:
            x, y, paths = self.get_motion_vector(self.motions_path, self.labels)

        offset = 0
        random_idxs = np.arange(y.shape[0])
        np.random.shuffle(random_idxs)

        for i in range(step):
            # load data from paths if use external mode
            if external:
                (_data, _label), offset = get_random_batch([x, y], random_idxs,
                                                         offset, batch_size)
                _x, _y, _paths = self.get_motion_vector(_data, _label)
            # if not external mode, just refer to the fetched batch
            else:
                (_x, _y, _paths), offset = get_random_batch([x, y, paths], random_idxs,
                                                  offset, batch_size)

            # map feature paths to original image path
            _paths = [[self.map_to_image(pp) for pp in p] for p in _paths]

            # extract skip frame
            for j, (_xx, _yy, _p) in enumerate(zip(_x, _y, _paths)):
                thread = threading.Thread(
                    target=_append_skip_frame_parrallel,
                    args=(lock, self.data_format[0], _xx, _yy, _p, batch_array))
                thread.start()
                threads.append(thread)
            for th in threads:
                th.join()
            threads.clear()
            _xx, _yy, _paths = tuple(map(list, tuple(zip(*batch_array))))
            _xx = np.asarray(_xx)
            _yy = np.asarray(_yy)
            _xx = self.normalize(_xx)
            yield np.asarray(_xx), np.asarray(_yy, dtype=np.int32), _paths


if __name__ == "__main__":
    feature_parser = FeatureParser(feature_dir="../dataset/feature",
                                   image_dir="../dataset/frame",
                                   data_format=(30, 25, 2))
    for xx, yy, paths in feature_parser.batch_parser(step=1, batch_size=8):
        xx = feature_parser.denormalize(xx)
        write_debug_gif(xx[0], paths[0], "/Users/bob/Academic/graduation-project/debug.gif")
        break









