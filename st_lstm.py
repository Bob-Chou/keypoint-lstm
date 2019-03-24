import numpy as np
import tensorflow as tf

from utils import *

tf.logging.set_verbosity(tf.logging.INFO)


class _Graph(object):
    pass


class STModel(object):

    """docstring for STModel"""
    # protobuf

    def __init__(self, config, n_lstm=1, use_spatio_attn=True,
                 use_temporal_attn=True, name="unnamed"):
        super(STModel, self).__init__()

        self.name = name
        # network architecture
        self.n_keypoint = config.n_keypoint
        self.n_frame = config.n_frame
        self.n_dim = config.n_dim
        self.n_hidden = config.n_hidden
        self.n_class = config.n_class
        self.n_lstm = n_lstm
        self.use_temporal_attn = use_temporal_attn
        self.use_spatio_attn = use_spatio_attn
        self._use_bn = config.use_bn
        self._use_res = config.use_res

        # network pipeline
        self.graph = None
        self.x = None
        self.y = None
        self.mask = None
        self.batch = None
        self.global_step = None
        self.accuracy = None
        self.pred = None
        # loss
        self.lambda1 = config.lambda1
        self.lambda2 = config.lambda2
        self.lambda3 = config.lambda3
        self.loss = None
        # train
        self.lr = config.learning_rate
        # summary
        self.all_vars = []
        self.train_summary = None
        self.test_summary = None
        summary_dir = "./summary/{}".format(self.name)
        if tf.gfile.Exists(summary_dir):
            tf.gfile.DeleteRecursively(summary_dir)
        tf.gfile.MakeDirs(os.path.join(summary_dir, "train"))
        tf.gfile.MakeDirs(os.path.join(summary_dir, "test"))
        self.train_writter = tf.summary.FileWriter(
            os.path.join(summary_dir, "train"))
        self.test_writter = tf.summary.FileWriter(
            os.path.join(summary_dir, "test"))
        self._build_network()

    def _build_network(self):
        train_summaries = []
        test_summaries = []
        self.graph = tf.Graph()
        with self.graph.as_default():

            # input layer
            with tf.variable_scope("input"):
                self.x = tf.placeholder(tf.float32,
                                        [None, self.n_frame, self.n_keypoint,
                                         self.n_dim],
                                        name="feature")
                self.y = tf.placeholder(tf.int32, [None, ], name="label")
                self.mask = tf.placeholder(tf.int32, [None, ], name="mask")
                self.batch = tf.placeholder(tf.int32, [], name="batch_size")
                self.training = tf.placeholder(tf.bool, [], name="training")
                y_one_hot = tf.one_hot(self.y, self.n_class, name="one_hot")

            if self._use_bn:
                with tf.variable_scope("batchnorm"):
                    feature = tf.layers.batch_normalization(
                        self.x, training=self.training)
            else:
                feature = self.x

            # regularizer
            with tf.variable_scope("regularizer"):
                reg1 = tf.contrib.layers.l1_regularizer(
                    scale=self.lambda1 / tf.cast(self.batch, tf.float32))
                reg2 = tf.contrib.layers.l1_regularizer(
                    scale=self.lambda2 / tf.cast(self.batch, tf.float32))
                reg3 = tf.contrib.layers.l1_regularizer(scale=self.lambda3)

            # convert mask
            mask_mat = tf.expand_dims(
                tf.cast(tf.sequence_mask(self.mask, self.n_frame), tf.float32),
                axis=-1)
            frame_len_mat = tf.expand_dims(
                tf.cast(self.mask, tf.float32), axis=-1)

            # lstm
            with tf.variable_scope("lstm"):
                lstm_cell = tf.nn.rnn_cell.MultiRNNCell(
                    [tf.nn.rnn_cell.LSTMCell(self.n_hidden,
                                             name="lstm_cell",
                                             dtype=tf.float32)
                     for _ in range(self.n_lstm)])
                state = lstm_cell.zero_state(self.batch, dtype=tf.float32)

                # run lstm through time steps manually
                s_attns = []
                t_attns = []
                lstm_outputs = []

                for i in range(self.n_frame):
                    reuse = i != 0

                    xt = tf.reshape(feature[:, i, :, :],
                                    (-1, self.n_dim * self.n_keypoint))

                    # spatio-attention
                    with tf.variable_scope("spatio_attention"):
                        if self.use_spatio_attn:
                            s_attn = attn_dense(xt, state[-1].h,
                                                self.n_keypoint,
                                                name="attention_dense",
                                                reuse=reuse, regularizer=reg3,
                                                rescale=True)
                            s_attn = tf.nn.softmax(s_attn, axis=-1)
                            s_attn = s_attn * self.n_keypoint
                            # record for regularizer
                            s_attns.append(s_attn)
                            s_attn = tf.expand_dims(s_attn, -1)
                        else:
                            s_attn = 1.0

                    # temporal-attention
                    with tf.variable_scope("temporal_attention"):
                        if self.use_temporal_attn:
                            t_attn = attn_dense(xt, state[-1].h, 1,
                                                name="attention_dense",
                                                activation=tf.nn.relu,
                                                reuse=reuse, regularizer=reg3)
                            # record for regularizer
                            t_attns.append(t_attn)
                        else:
                            t_attn = 1.0

                    with tf.variable_scope("spatio_attention"):
                        # add spatio-attention to input sequence
                        xt = s_attn * tf.reshape(xt, (-1, self.n_keypoint,
                                                      self.n_dim))
                        xt = tf.reshape(xt,
                                        (-1, self.n_dim * self.n_keypoint))

                    # run lstm for a single timestep
                    out, state = lstm_cell(xt, state)
                    out = dense(out, self.n_class, name="output_dense",
                                activation=None, reuse=reuse,
                                regularizer=reg3)

                    # add temporal-attention to output
                    with tf.variable_scope("temporal_attention"):
                        out = out * t_attn

                    lstm_outputs.append(out)

            with tf.variable_scope("stack"):
                # stack the tensor
                if self.use_spatio_attn:
                    s_attn_mat = tf.stack(s_attns, axis=1,
                                          name="spatio_attention_matrix")
                else:
                    s_attn_mat = 1.0

                if self.use_temporal_attn:
                    t_attn_mat = tf.stack(t_attns, axis=1,
                                          name="temporal_attention_matrix")
                else:
                    t_attn_mat = 0.0
                out_mat = tf.stack(lstm_outputs, axis=1, name="output_matrix")

            # output
            with tf.variable_scope("output"):
                self.logits = tf.reduce_sum(out_mat * mask_mat,
                                            axis=1) / frame_len_mat
                self.logits = tf.identity(self.logits, "logits")

                self.pred = tf.nn.softmax(self.logits, axis=-1)
                self.pred = tf.identity(self.pred, "predictions")

                correct = tf.equal(tf.argmax(self.pred, -1),
                                   tf.argmax(y_one_hot, -1))
                self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

                add_summary([tf.summary.scalar("accuracy", self.accuracy)],
                            [train_summaries, test_summaries])

            # loss
            with tf.variable_scope("loss"):
                # cross-entropy
                l0 = tf.losses.softmax_cross_entropy(y_one_hot, self.logits)
                tf.identity(l0, "cross_entropy_loss")
                # spatio-attention l2 regularization
                s_reg = tf.reduce_sum(s_attn_mat * mask_mat,
                                      axis=1) / frame_len_mat - 1
                l1 = tf.contrib.layers.apply_regularization(reg1, [s_reg])
                tf.identity(l1, "spatio_loss")
                # temporal-attention l2 regularization
                t_reg = tf.reduce_sum(t_attn_mat * mask_mat,
                                      axis=-1) / tf.math.sqrt(frame_len_mat)
                l2 = tf.contrib.layers.apply_regularization(reg2, [t_reg])
                tf.identity(l2, "temporal_loss")
                # total loss
                loss = tf.losses.get_total_loss()
                self.loss = tf.identity(loss, "total_loss")
                # loss summary
                add_summary(
                    [tf.summary.scalar("loss-total", loss),
                     tf.summary.scalar("loss-spatio-attention", l1),
                     tf.summary.scalar("loss-temporal-attention", l2)],
                    [train_summaries])

            # optimizer
            with tf.variable_scope("optimizer"):
                self.optimizer = tf.train.AdadeltaOptimizer(self.lr)
                self.global_step = tf.get_variable("global_step",
                                                   initializer=0,
                                                   trainable=False)

            # merged summary
            with tf.variable_scope("summary"):
                self.train_summary = tf.summary.merge(train_summaries,
                                                      name="train_summary")
                self.test_summary = tf.summary.merge(test_summaries,
                                                     name="test_summary")

            tf.logging.info("Trainable Vars:")
            self.all_vars = tf.trainable_variables()
            for v in self.all_vars:
                tf.logging.info(v)

    def train(self, train_data, valid_data=None, valid_per=100, save_per=1000,
              freeze_vars=None, restore_from=None, override=True):
        """Train the model

        :param train_data: data generator return x, y, mask for each call
        :param valid_data: data generatpr return x, y, mask for each call
        :param valid_per: a validation step after how many steps
        :param save_per: save the model after how many steps
        :param freeze_vars: a list of keywords to freeze viarables
        :param restore_ckpt: the directory to the saved pretrained
        :param override: whether to override the checkpoint file
        :return:
        """
        ckpt_dir = "./checkpoint/{}".format(self.name)
        if freeze_vars is not None:
            trainable_vars = filter_vars(self.all_vars, freeze_vars)
        else:
            trainable_vars = self.all_vars
        if restore_from is not None:
            for k, v in restore_from.items():
                if bool(v):
                    restore_vars = pick_vars(self.all_vars, v)
                else:
                    restore_vars = self.all_vars
                restore_from[k] = restore_vars
        else:
            restore_from = {}

        tf.logging.info("Vars to be restored:")
        for k, restore_vars in restore_from.items():
            tf.logging.info("From {}:".format(k))
            tf.logging.info([v.name for v in restore_vars])
        tf.logging.info("Vars to be updated:")
        tf.logging.info([v.name for v in trainable_vars])

        gpu_opt = tf.GPUOptions(allow_growth=True,
                                per_process_gpu_memory_fraction=0.7)

        with tf.Session(graph=self.graph,
                        config=tf.ConfigProto(gpu_options=gpu_opt)) as sess:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                op = self.optimizer.minimize(self.loss,
                                             var_list=trainable_vars,
                                             global_step=self.global_step)

            sess.run(tf.global_variables_initializer())
            if len(restore_from) > 0:
                for restore_ckpt, restore_vars in restore_from.items():
                    tf.logging.info("Restoring from {}".format(restore_ckpt))
                    saver = tf.train.Saver(restore_vars)
                    ckpt = tf.train.get_checkpoint_state(restore_ckpt)
                    saver.restore(sess, ckpt.model_checkpoint_path)
            saver = tf.train.Saver()
            tf.logging.info("Please monitor the training process with:\n"
                            "tensorboard --logdir=./summary")

            if tf.gfile.Exists(ckpt_dir) and override:
                tf.gfile.DeleteRecursively(ckpt_dir)

            tf.logging.info("Start training {}".format(self.name))
            for x, y, m in train_data:
                curr_step = sess.run(self.global_step)
                summary, _ = sess.run([self.train_summary, op],
                                      feed_dict={self.x: x,
                                                 self.y: y,
                                                 self.batch: x.shape[0],
                                                 self.mask: m,
                                                 self.training: True})
                self.train_writter.add_summary(summary, curr_step)
                if curr_step % valid_per == 0 and valid_data is not None:
                    _x, _y, _m = valid_data
                    summary, = sess.run([self.test_summary],
                                        feed_dict={self.x: _x,
                                                   self.y: _y,
                                                   self.batch: _x.shape[0],
                                                   self.mask: _m,
                                                   self.training: False})
                    self.test_writter.add_summary(summary, curr_step)
                if curr_step % save_per == 0:
                    saver.save(sess, os.path.join(ckpt_dir,
                                                  "{}.ckpt".format(self.name)),
                               global_step=curr_step)


if __name__ == "__main__":

    config = get_config("./config.ini")

    train_set, valid_set = walk_dataset(config.data.root, config.data.ext,
                                        valid_list=config.train.valid_set)
    train_label = [int(sample.split("/")[-3]) - 1 for sample in train_set]
    valid_label = [int(sample.split("/")[-3]) - 1 for sample in valid_set]
    valid_set, valid_mask = load_batch_parallel(valid_set,
                                                [config.model.n_frame,
                                                 config.model.n_keypoint,
                                                 config.model.n_dim],
                                                as_flow=config.data.as_flow)
    # valid_set = (valid_set - config.data.mean) / config.data.std
    valid_mask = np.sum(valid_mask != -1, axis=-1)

    def training_data(package, max_step=None, max_epoch=None):
        shape = [config.model.n_frame,
                 config.model.n_keypoint,
                 config.model.n_dim]
        for batch in random_batch_generator(package, config.train.batch_size,
                                            max_step=max_step,
                                            max_epoch=max_epoch):
            batch[0], mask = load_batch_parallel(batch[0], shape,
                                                 as_flow=config.data.as_flow)
            # batch[0] = (batch[0] - config.data.mean) / config.data.std
            yield batch[0], batch[1], np.sum(mask != -1, axis=-1)

    # Pretrain the temporal attention
    t0_model = STModel(config.model, 1, use_spatio_attn=False,
                       name="temporal-pretrain")
    data = training_data([train_set, train_label],
                         max_epoch=config.train.n_pretrain_epoch)
    t0_model.train(data, valid_data=[valid_set, valid_label, valid_mask],
                   valid_per=config.train.valid_per_step,
                   save_per=config.train.save_per_step)

    # Train the temporal attention
    t1_model = STModel(config.model, 3, use_spatio_attn=False,
                       name="temporal-train")
    data = training_data([train_set, train_label],
                         max_epoch=config.train.n_pretrain_epoch)
    t1_model.train(data, valid_data=[valid_set, valid_label, valid_mask],
                   valid_per=config.train.valid_per_step,
                   save_per=config.train.save_per_step,
                   freeze_vars=["temporal"],
                   restore_from={"checkpoint/temporal-pretrain":
                                     ["temporal"]})

    # Finetune the temporal attention
    data = training_data([train_set, train_label],
                         max_epoch=config.train.n_finetune_epoch)
    t1_model.train(data, valid_data=[valid_set, valid_label, valid_mask],
                   valid_per=config.train.valid_per_step,
                   save_per=config.train.save_per_step,
                   restore_from={"checkpoint/temporal-train":
                                     ["lstm", "global_step", "batchnorm"]},
                   override=False)

    # Pretrain the spatio attention
    s0_model = STModel(config.model, 1, use_temporal_attn=False,
                       name="spatio-pretrain")
    data = training_data([train_set, train_label],
                         max_epoch=config.train.n_pretrain_epoch)
    s0_model.train(data, valid_data=[valid_set, valid_label, valid_mask],
                   valid_per=config.train.valid_per_step,
                   save_per=config.train.save_per_step)

    # Train the spatio attention
    s1_model = STModel(config.model, 3, use_temporal_attn=False,
                       name="spatio-train")
    data = training_data([train_set, train_label],
                         max_epoch=config.train.n_pretrain_epoch)
    s1_model.train(data, valid_data=[valid_set, valid_label, valid_mask],
                   valid_per=config.train.valid_per_step,
                   save_per=config.train.save_per_step,
                   freeze_vars=["spatio"],
                   restore_from={"checkpoint/spatio-pretrain":
                                     ["spatio"]})

    # Finetune the spatio attention
    data = training_data([train_set, train_label],
                         max_epoch=config.train.n_finetune_epoch)
    s1_model.train(data, valid_data=[valid_set, valid_label, valid_mask],
                   valid_per=config.train.valid_per_step,
                   save_per=config.train.save_per_step,
                   restore_from={"checkpoint/spatio-train":
                                     ["lstm", "global_step", "batchnorm"]},
                   override=False)

    # Train the main network
    st_model = STModel(config.model, 3, name="st-model")
    data = training_data([train_set, train_label],
                         max_epoch=config.train.n_pretrain_epoch)
    st_model.train(data, valid_data=[valid_set, valid_label, valid_mask],
                   valid_per=config.train.valid_per_step,
                   save_per=config.train.save_per_step,
                   freeze_vars=["spatio", "temporal"],
                   restore_from={"checkpoint/spatio-train":
                                     ["spatio"],
                                 "checkpoint/temporal-train":
                                     ["temproal"]})

    # Finetune the main network
    data = training_data([train_set, train_label],
                         max_epoch=config.train.n_finetune_epoch)
    st_model.train(data, valid_data=[valid_set, valid_label, valid_mask],
                   valid_per=config.train.valid_per_step,
                   save_per=config.train.save_per_step,
                   restore_from={"checkpoint/st-model":
                                     ["lstm", "global_step", "batchnorm"]})


