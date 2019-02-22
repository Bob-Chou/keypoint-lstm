import numpy as np
import tensorflow as tf

from model_utils import *


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
        # loss
        self.lambda1 = 1e-3
        self.lambda2 = 1e-4
        self.lambda3 = 5e-4
        self.loss = None
        # train
        self.lr = 1e-2
        self.step = None
        # summary
        self.summary = None
        self.accuracy = None
        self.saver = None
        self.train_writter = tf.summary.FileWriter('./train_summary')
        self.test_writter = tf.summary.FileWriter('./test_summary')
        # construct the network
        self._build_network()

    def _build_network(self):

        with self.graph.as_default():

            # regularizer
            with tf.variable_scope("l1_regularizer"):
                reg1 = tf.contrib.layers.l2_regularizer(scale=self.lambda1)
                reg2 = tf.contrib.layers.l2_regularizer(scale=self.lambda2)
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
                lstm_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden,
                                                    name="lstm_cell",
                                                    dtype=tf.float32)
                state = lstm_cell.zero_state(self.batch, dtype=tf.float32)

            # attention
            s_attns = []
            t_attns = []
            for i in range(self.n_frame):
                xt = tf.reshape(self.x[:, i, :, :],
                                (-1, self.n_dim * self.n_keypoint))

                # spatio-attention
                with tf.variable_scope("spatial_attention"):
                    s_attn = attn_dense(xt, state.h, self.n_keypoint,
                                        name="attention_dense",
                                        activation=tf.nn.tanh,
                                        reuse=(i!=0), regularizer=reg3)
                    with tf.variable_scope("attention_linear",
                                           reuse=(i!=0)):
                        us = tf.get_variable("us", 1)
                        bus = tf.get_variable("bus", (self.n_keypoint,))
                        s_attn = tf.nn.softmax(tf.add(tf.multiply(us, s_attn),
                                                      bus))
                    # record for regularizer
                    s_attns.append(s_attn)
                    s_attn = tf.expand_dims(s_attn, -1)

                # temporal-attention
                with tf.variable_scope("temporal_attention"):
                    t_attn = attn_dense(xt, state.h, 1, name="attention_dense",
                                        activation=tf.nn.relu,
                                        reuse=(i!=0), regularizer=reg3)
                    # record for regularizer
                    t_attns.append(t_attn)

                # add spatio-attention to input sequence
                xt = tf.reshape(xt, (-1, self.n_keypoint, self.n_dim))
                xs = tf.reduce_sum(tf.multiply(s_attn, xt), axis=1)
                xt = tf.reshape(self.x[:, i, :, :],
                                (-1, self.n_dim * self.n_keypoint))
                # run lstm for a single timestep
                out, state = lstm_cell(xs, state)
                # add temporal-attention to output
                out = dense(out, self.n_class, name="output_dense",
                            activation=None, reuse=tf.AUTO_REUSE,
                            regularizer=reg3)
                self.lstm_outputs.append(tf.multiply(out, t_attn))

            # output
            with tf.variable_scope("output"):
                logits = tf.add_n(self.lstm_outputs)
                self.pred = tf.nn.softmax(logits, axis=-1)
                correct = tf.equal(tf.argmax(self.pred, -1),
                                   tf.argmax(self.y_one_hot, -1))
                self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
                # _, self.accuracy = tf.metrics.accuracy(
                #     labels=tf.argmax(self.y_one_hot, axis=1),
                #     predictions=tf.argmax(self.pred, axis=1))
                tf.summary.scalar('accuracy', self.accuracy)

            # loss
            with tf.variable_scope("loss"):
                # cross-entropy
                _ = tf.losses.softmax_cross_entropy(self.y_one_hot, logits)
                # spatio-attention l2 regularization
                s_reg = tf.reduce_mean(
                    1 - self.n_keypoint * tf.add_n(s_attns) / self.n_frame,
                    axis=0)
                _ = tf.contrib.layers.apply_regularization(reg1, [s_reg])
                # temporal-attention l2 regularization
                t_reg = tf.reduce_mean(tf.add_n(t_attns) / self.n_frame, axis=0)
                _ = tf.contrib.layers.apply_regularization(reg2, [t_reg])
                # total loss
                self.loss = tf.losses.get_total_loss()
                tf.summary.scalar('loss', self.loss)

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
            sess.run(tf.local_variables_initializer())
            for x, y, _ in train_data:
                if selected_list is not None:
                    x = x[:, :, selected_list, :]
                curr_step = sess.run(self.global_step)
                # assert x.shape[0] == y.shape[0]
                summary, l, acc, _ = sess.run([self.summary, self.loss,
                                         self.accuracy, self.step],
                                         feed_dict={
                                             self.x: x,
                                             self.y: y,
                                             self.batch: x.shape[0]})
                print(curr_step, l, acc)
                self.train_writter.add_summary(summary, curr_step)
                if curr_step % 100 == 0 and valid_data is not None:
                    _x, _y = valid_data
                    summary = sess.run([self.summary],
                                       feed_dict={
                                           self.x: _x,
                                           self.y: _y,
                                           self.batch: _x.shape[0]})
                    self.test_writter.add_summary(summary, curr_step)
                if curr_step % 1000 == 0:
                    self.saver.save(sess, "./checkpoint/st_lstm.ckpt",
                                    global_step=curr_step)


if __name__ == "__main__":
    n_frame = 30
    n_keypoint = 25
    n_selected_keypoint = 7
    selected_list = [1,2,3,4,5,6,7]
    st_model = STModel(n_frame, n_selected_keypoint, n_class=4)
    feature_parser = FeatureParser(feature_dir="../dataset/feature",
                                   image_dir="../dataset/frame",
                                   data_format=(n_frame, n_keypoint, 2))
    st_model.train(feature_parser.batch_parser(step=10000, batch_size=32),
                   selected_list=selected_list)
