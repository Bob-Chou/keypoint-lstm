import numpy as np
import tensorflow as tf

from utils import *


class STModel(object):

    """docstring for STModel"""
    # protobuf

    def __init__(self, config):
        super(STModel, self).__init__()
        self.name = config.name
        # network architecture
        self.graph = tf.Graph()
        self.x = None
        self.y = None
        self.y_one_hot = None
        self.keep_prob = None
        self.n_keypoint = config.n_keypoint
        self.n_frame = config.n_frame
        self.n_dim = config.n_dim
        self.n_hidden = config.n_hidden
        self.n_class = config.n_class
        self.logits = None
        self.pred = None
        # keep prob
        self.keep_prob = 0.5
        # loss
        self.lambda1 = config.lambda1
        self.lambda2 = config.lambda2
        self.lambda3 = config.lambda3
        self.loss = None
        self.l0 = None
        self.l1 = None
        # train
        self.lr = config.learning_rate
        self.step = None
        # summary
        self.train_summary = None
        self.test_summary = None
        self.accuracy = None
        self.saver = None
        summary_dir = "./summary/{}".format(self.name)
        if tf.gfile.Exists(summary_dir):
            tf.gfile.DeleteRecursively(summary_dir)
        tf.gfile.MakeDirs(os.path.join(summary_dir, "train"))
        tf.gfile.MakeDirs(os.path.join(summary_dir, "test"))
        self.train_writter = tf.summary.FileWriter(
            os.path.join(summary_dir, "train"))
        self.test_writter = tf.summary.FileWriter(
            os.path.join(summary_dir, "test"))
        # construct the network
        self._build_network()

    def _build_network(self):
        train_summaries = []
        test_summaries = []
        with self.graph.as_default():

            # input layer
            with tf.variable_scope("input"):
                self.x = tf.placeholder(tf.float32, shape=(None, self.n_frame,
                                        self.n_keypoint, self.n_dim),
                                        name="feature")
                self.y = tf.placeholder(tf.int32, shape=(None,),
                                        name="label")
                self.mask = tf.placeholder(tf.int32, shape=(None,),
                                           name="mask")
                self.batch = tf.placeholder(tf.int32, [], name="batch_size")
                # self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
                self.y_one_hot = tf.one_hot(self.y, self.n_class)

            # regularizer
            with tf.variable_scope("l1_regularizer"):
                reg1 = tf.contrib.layers.l2_regularizer(
                    scale=self.lambda1 / tf.cast(self.batch, tf.float32))
                reg2 = tf.contrib.layers.l2_regularizer(
                    scale=self.lambda2 / tf.cast(self.batch, tf.float32))
                reg3 = tf.contrib.layers.l1_regularizer(scale=self.lambda3)

            # convert mask
            mask_mat = tf.expand_dims(
                tf.cast(tf.sequence_mask(self.mask, self.n_frame), tf.float32),
                axis=-1, name="mask")
            frame_len_mat = tf.expand_dims(
                tf.cast(self.mask, tf.float32), axis=-1, name="length")

            # lstm
            with tf.variable_scope("lstm"):
                lstm_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden,
                                                    name="lstm_cell",
                                                    dtype=tf.float32)
                state = lstm_cell.zero_state(self.batch, dtype=tf.float32)

            # run lstm through time steps manually
            s_attns = []
            t_attns = []
            lstm_outputs = []

            for i in range(self.n_frame):
                reuse = i != 0

                xt = tf.reshape(self.x[:, i, :, :],
                                (-1, self.n_dim * self.n_keypoint))

                # spatio-attention
                with tf.variable_scope("spatio_attention"):
                    s_attn = attn_dense(xt, state.h, self.n_keypoint,
                                        name="attention_dense",
                                        reuse=reuse, regularizer=reg3,
                                        rescale=True)
                    s_attn = tf.nn.softmax(s_attn, axis=-1)
                    # record for regularizer
                    s_attns.append(s_attn)
                    s_attn = tf.expand_dims(s_attn, -1)

                # temporal-attention
                with tf.variable_scope("temporal_attention"):
                    t_attn = attn_dense(xt, state.h, 1, name="attention_dense",
                                        activation=tf.nn.relu,
                                        reuse=reuse, regularizer=reg3)
                    # record for regularizer
                    t_attns.append(t_attn)

                with tf.variable_scope("spatio_attention"):
                    # add spatio-attention to input sequence
                    xt = s_attn * tf.reshape(xt,
                                             (-1, self.n_keypoint, self.n_dim))
                    xt = tf.reshape(xt, (-1, self.n_dim * self.n_keypoint))

                # run lstm for a single timestep
                out, state = lstm_cell(xt, state)

                # add temporal-attention to output
                with tf.variable_scope("temporal_attention"):
                    out = dense(out, self.n_class, name="output_dense",
                                activation=None, reuse=reuse,
                                regularizer=reg3)

                lstm_outputs.append(out * t_attn)

            # stack the tensor
            s_attn_mat = tf.stack(s_attns, axis=1,
                                  name="spatio_attention_stack")
            t_attn_mat = tf.stack(t_attns, axis=1,
                                  name="temporal_attention_stack")
            out_mat = tf.stack(lstm_outputs, axis=1, name="output_stack")

            # output
            with tf.variable_scope("output"):
                self.logits = tf.reduce_sum(out_mat * mask_mat,
                                            axis=1) / frame_len_mat
                self.pred = tf.nn.softmax(self.logits, axis=-1)
                correct = tf.equal(tf.argmax(self.pred, -1),
                                   tf.argmax(self.y_one_hot, -1))
                self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

                add_summary([tf.summary.scalar('accuracy', self.accuracy)],
                            [train_summaries, test_summaries])

            # loss
            with tf.variable_scope("loss"):
                # cross-entropy
                self.l0 = tf.losses.softmax_cross_entropy(self.y_one_hot,
                                                          self.logits)
                # spatio-attention l2 regularization
                s_reg = tf.reduce_sum(self.n_keypoint * s_attn_mat * mask_mat,
                                      axis=1) / frame_len_mat - 1
                self.l1 = tf.contrib.layers.apply_regularization(reg1, [s_reg])
                # temporal-attention l2 regularization
                t_reg = tf.reduce_sum(t_attn_mat * mask_mat,
                                      axis=-1) / tf.math.sqrt(frame_len_mat)
                self.l2 = tf.contrib.layers.apply_regularization(reg2, [t_reg])
                # total loss
                self.loss = tf.losses.get_total_loss()
                add_summary(
                    [tf.summary.scalar('loss: total', self.loss),
                     tf.summary.scalar('loss: cross-entropy', self.l0),
                     tf.summary.scalar('loss: spatio-attention', self.l1),
                     tf.summary.scalar('loss: temporal-attention', self.l2)],
                    [train_summaries])

            # optimizer
            with tf.variable_scope("optimizer"):
                optimizer = tf.train.AdadeltaOptimizer(self.lr)
                self.global_step = tf.get_variable("global_step", initializer=0,
                                                   trainable=False)
                self.step = optimizer.minimize(self.loss,
                                               global_step=self.global_step)

            # merged summary
            self.train_summary = tf.summary.merge(train_summaries)
            self.test_summary = tf.summary.merge(test_summaries)
            self.saver = tf.train.Saver()
            # print("=================Architecture=================")
            # for n in self.graph.as_graph_def().node:
            #     print(n.name)
            print("================Trainable Vars================")
            for v in tf.trainable_variables():
                print(v)
            print("==============================================")

    def train(self, train_data, valid_data=None, selected_list=None,
              valid_per=100, save_per=1000):
        ckpt_dir = "./checkpoint/{}".format(self.name)
        if tf.gfile.Exists(ckpt_dir):
            tf.gfile.DeleteRecursively(ckpt_dir)

        gpu_options = tf.GPUOptions(allow_growth=True,
                                    per_process_gpu_memory_fraction=0.7)
        with tf.Session(graph=self.graph,
                        config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            tf.logging.info("Please monitor the training process with:\n"
                            "tensorboard --logdir=./summary")
            for (x, y), m in train_data:
                if selected_list is not None:
                    x = x[:, :, selected_list, :]
                curr_step = sess.run(self.global_step)
                summary, _ = sess.run([self.train_summary, self.step],
                                      feed_dict={self.x: x,
                                                 self.y: y,
                                                 self.batch: x.shape[0],
                                                 self.mask: m})
                self.train_writter.add_summary(summary, curr_step)
                if curr_step % valid_per == 0 and valid_data is not None:
                    _x, _y, _m = valid_data
                    summary, = sess.run([self.test_summary],
                                        feed_dict={self.x: _x,
                                                   self.y: _y,
                                                   self.batch: _x.shape[0],
                                                   self.mask: _m})
                    self.test_writter.add_summary(summary, curr_step)
                if curr_step % save_per == 0:
                    self.saver.save(sess, os.path.join(ckpt_dir, "model.ckpt"),
                                    global_step=curr_step)


if __name__ == "__main__":
    config = get_config("./config.ini")

    def training_data_generator(package, shape, batch_size, max_step=None,
                                max_epoch=None):
        for batch in random_batch_generator(package, batch_size,
                                            max_step=max_step,
                                            max_epoch=max_epoch):
            batch[0], mask = load_batch_parallel(batch[0], shape)
            batch[0] = (batch[0] - config.data.mean) / config.data.std
            yield batch, np.sum(mask != -1, axis=-1)

    model = STModel(config.model)

    train_set, valid_set = walk_dataset(config.data.root, config.data.ext,
                                        valid_list=config.train.valid_set)
    train_label = [int(sample.split("/")[-3]) - 1 for sample in train_set]
    valid_label = [int(sample.split("/")[-3]) - 1 for sample in valid_set]
    valid_set, valid_mask = load_batch_parallel(valid_set,
                                          [config.model.n_frame,
                                           config.model.n_keypoint,
                                           config.model.n_dim])
    valid_set = (valid_set - config.data.mean) / config.data.std

    training_data = training_data_generator([train_set, train_label],
                                            [config.model.n_frame,
                                             config.model.n_keypoint,
                                             config.model.n_dim],
                                            config.train.batch_size,
                                            max_epoch=config.train.n_epoch)

    model.train(training_data,
                valid_data=[valid_set,
                            valid_label,
                            np.sum(valid_mask != -1, axis=-1)],
                valid_per=config.train.valid_per_step,
                save_per=config.train.save_per_step)
