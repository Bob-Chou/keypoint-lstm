import numpy as np
import tensorflow as tf

from utils import *


class STModel(object):

    """docstring for STModel"""
    # protobuf

    def __init__(self, n_frame, n_keypoint, n_dim=2, n_hidden=128, n_class=2):
        super(STModel, self).__init__()
        # network architecture
        self.graph = tf.Graph()
        self.x = None
        self.y = None
        self.y_one_hot = None
        self.keep_prob = None
        self.n_keypoint = n_keypoint
        self.n_frame = n_frame
        self.n_dim = n_dim
        self.n_hidden = n_hidden
        self.n_class = n_class
        self.lstm_outputs = []
        self.pred = None
        # keep prob
        self.keep_prob = 0.5
        # loss
        self.lambda1 = 1e-3
        self.lambda3 = 5e-4
        self.loss = None
        self.l0 = None
        self.l1 = None
        # train
        self.lr = 1e-2
        self.step = None
        # summary
        self.summary = None
        self.accuracy = None
        self.saver = None
        if tf.gfile.Exists('./summary'):
            tf.gfile.DeleteRecursively('./summary')
        tf.gfile.MakeDirs('./summary/train')
        tf.gfile.MakeDirs('./summary/test')
        self.train_writter = tf.summary.FileWriter('./summary/train')
        self.test_writter = tf.summary.FileWriter('./summary/test')
        # construct the network
        self._build_network()

    def _build_network(self):

        with self.graph.as_default():

            # regularizer
            with tf.variable_scope("l1_regularizer"):
                reg1 = tf.contrib.layers.l2_regularizer(scale=self.lambda1)
                reg3 = tf.contrib.layers.l1_regularizer(scale=self.lambda3)

            # input layer
            with tf.variable_scope("input"):
                self.x = tf.placeholder(tf.float32, shape=(None, self.n_frame,
                                        self.n_keypoint, self.n_dim),
                                        name="keypoint_holder")
                self.y = tf.placeholder(tf.int32, shape=(None,),
                                        name="label_holder")
                self.batch = tf.placeholder(tf.int32, (), name="batch_size")
                # self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
                self.y_one_hot = tf.one_hot(self.y, self.n_class)

            # lstm
            with tf.variable_scope("lstm"):
                n_input_dim = self.n_dim * self.n_keypoint
                lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=n_input_dim,
                                                    forget_bias=1.0,
                                                    name="basic_lstm_cell",
                                                    state_is_tuple=True)

                lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell,
                                                          input_keep_prob=1.0,
                                                          output_keep_prob=self.keep_prob)

                init_state = lstm_cell.zero_state(self.batch, dtype=tf.float32)
                xt = tf.reshape(self.x,
                                (-1, self.n_frame, self.n_dim *
                                 self.n_keypoint))
                outputs, _ = tf.nn.dynamic_rnn(lstm_cell, inputs=xt,
                                               initial_state=init_state,
                                               time_major=False)



            # output
            with tf.variable_scope("output"):
                logits = dense(outputs[:, -1, :], self.n_class, name="output_dense",
                               activation=None, reuse=tf.AUTO_REUSE)
                self.pred = tf.nn.softmax(logits, axis=-1)
                correct = tf.equal(tf.argmax(self.pred, -1),
                                   tf.argmax(self.y_one_hot, -1))
                self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
                tf.summary.scalar('accuracy', self.accuracy)

            # loss
            with tf.variable_scope("loss"):
                # cross-entropy
                self.l0 = tf.losses.softmax_cross_entropy(self.y_one_hot, logits)
                # spatio-attention l2 regularization
                # s_reg = 1 - self.n_keypoint * tf.add_n(s_attns) / self.n_frame
                # self.l1 = tf.contrib.layers.apply_regularization(reg1, [s_reg])
                # total loss
                self.loss = tf.losses.get_total_loss()
                tf.summary.scalar('loss', self.loss)
                tf.summary.scalar('loss0', self.l0)
                # tf.summary.scalar('loss1', self.l1)

            # optimizer
            with tf.variable_scope("optimizer"):
                optimizer = tf.train.AdadeltaOptimizer(self.lr)
                self.global_step = tf.get_variable("global_step", initializer=0,
                                                   trainable=False)
                self.step = optimizer.minimize(self.loss,
                                               global_step=self.global_step)

            # merged summary
            self.summary = tf.summary.merge_all()
            self.saver = tf.train.Saver()
            # [print(v) for v in tf.trainable_variables()]

    def train(self, train_data, valid_data=None, selected_list=None):
        gpu_options = tf.GPUOptions(allow_growth=True,
                                    per_process_gpu_memory_fraction=0.7)
        with tf.Session(graph=self.graph,
                        config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            for x, y in train_data:
                if selected_list is not None:
                    x = x[:, :, selected_list, :]
                curr_step = sess.run(self.global_step)
                # assert x.shape[0] == y.shape[0]
                summary, _ = sess.run([self.summary, self.step],
                                      feed_dict={self.x: x,
                                                 self.y: y,
                                                 self.batch: x.shape[0]})
                self.train_writter.add_summary(summary, curr_step)
                if curr_step % 100 == 0 and valid_data is not None:
                    _x, _y = valid_data
                    summary, = sess.run([self.summary],
                                        feed_dict={self.x: _x,
                                                   self.y: _y,
                                                   self.batch: _x.shape[0]})
                    self.test_writter.add_summary(summary, curr_step)
                if curr_step % 1000 == 0:
                    self.saver.save(sess, "./checkpoint/st_lstm.ckpt",
                                    global_step=curr_step)


if __name__ == "__main__":
    n_frame = 10
    n_keypoint = 30
    n_dim = 3
    batch_size = 16
    max_epoch = 10000
    n_class = 8

    root = "/home/bob/Workspace/undergrad-project/dataset/SBU"
    train_set, valid_set = walk_dataset(root, ".txt", valid_list=["s05"])
    train_label = [int(sample.split("/")[-3]) - 1 for sample in train_set]
    valid_label = [int(sample.split("/")[-3]) - 1 for sample in valid_set]
    valid_set = load_batch_parallel(valid_set, [n_frame, n_keypoint, n_dim])
    valid_set = (valid_set - [0.5, 0.5, 1]) / [0.5, 0.5, 1.]

    def training_data_generator(package, shape, batch_size, max_step=None,
                                max_epoch=None):
        for batch in random_batch_generator(package, batch_size,
                                            max_step=max_step,
                                            max_epoch=max_epoch):
            batch[0] = load_batch_parallel(batch[0], shape)
            batch[0] = (batch[0] - [0.5, 0.5, 1]) / [0.5, 0.5, 1.]
            yield batch

    training_data = training_data_generator([train_set, train_label],
                                            [n_frame, n_keypoint, n_dim],
                                            batch_size,
                                            max_epoch=max_epoch)

    model = STModel(n_frame, n_keypoint, n_dim=n_dim, n_hidden=64,
                    n_class=n_class)
    model.train(training_data, valid_data=[valid_set, valid_label])




