from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops.nn_ops import leaky_relu

try:
    import tfplot
except:
    pass

from ops import conv2d, fc
from util import log

from vqa_util import question2str, answer2str


class Model(object):

    def __init__(self, config,
                 debug_information=False,
                 is_train=True):
        self.debug = debug_information

        self.config = config
        self.batch_size = self.config.batch_size
        self.img_size = self.config.data_info[0]
        self.c_dim = self.config.data_info[2]
        self.q_dim = self.config.data_info[3]
        self.a_dim = self.config.data_info[4]
        self.conv_info = self.config.conv_info

        # create placeholders for the input
        self.img = tf.placeholder(
            name='img', dtype=tf.float32,
            shape=[self.batch_size, self.img_size, self.img_size, self.c_dim],
        )
        self.q = tf.placeholder(
            name='q', dtype=tf.float32, shape=[self.batch_size, self.q_dim],
        )
        self.a = tf.placeholder(
            name='a', dtype=tf.float32, shape=[self.batch_size, self.a_dim],
        )

        self.is_training = tf.placeholder_with_default(bool(is_train), [], name='is_training')

        self.build(is_train=is_train)

    def get_feed_dict(self, batch_chunk, step=None, is_training=None):
        fd = {
            self.img: batch_chunk['img'],  # [B, h, w, c]
            self.q: batch_chunk['q'],  # [B, n]
            self.a: batch_chunk['a'],  # [B, m]
        }
        if is_training is not None:
            fd[self.is_training] = is_training

        return fd

    def build(self, is_train=True):

        n = self.a_dim
        conv_info = self.conv_info

        # build loss and accuracy {{{
        def build_loss(logits, labels):
            # Cross-entropy loss
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            l2_reqularization = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            loss += l2_reqularization
            # Classification accuracy
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return loss, accuracy
        # }}}

        def concat_coor(o, i, d):
            coor = tf.tile(tf.expand_dims(
                [float(int(i / d)) / d, (i % d) / d], axis=0), [self.batch_size, 1])
            o = tf.concat([o, tf.to_float(coor)], axis=1)
            return o

        def g_theta(o_i, o_j, q, scope='g_theta', reuse=True):
            with tf.variable_scope(scope, reuse=reuse) as scope:
                if not reuse: log.warn(scope.name)
                g_1 = fc(tf.concat([o_i, o_j, q], axis=1), 256, name='g_1')
                g_1 = tf.layers.dropout(g_1, rate=0.5, training=is_train)
                g_2 = fc(g_1, 256, name='g_2')
                g_2 = tf.layers.dropout(g_2, rate=0.5, training=is_train)
                g_3 = fc(g_2, 256, name='g_3')
                g_3 = tf.layers.dropout(g_3, rate=0.5, training=is_train)
                g_4 = fc(g_3, 256, name='g_4')
                return g_4

        def attentional_relational_layer(inputs, q, slices):
            input_shape = inputs.get_shape().as_list()

            b, h, w, c = input_shape[0:4]
            print(input_shape)
            reuse = False
            object_combos = []
            attentional_condition_vector = tf.layers.flatten(inputs=inputs)
            #attentional_condition_vector = tf.layers.dense(flattened_inputs, units=c, activation=leaky_relu)
            # attentional_condition_vector = tf.layers.conv2d(inputs=inputs, filters=h*w, kernel_size=[3, 3],
            #                                    strides=(1, 1),
            #                                    padding='SAME', activation=leaky_relu)
            #attentional_condition_vector = tf.layers.flatten(inputs=attentional_condition_vector)
            for i in range(slices):
                attentional_condition_vector_i = tf.concat([attentional_condition_vector,
                                                            tf.expand_dims(self.batch_size * [float(i)], axis=1)],
                                                           axis=1)
                attentional_condition_vector_i = tf.layers.dense(attentional_condition_vector_i, 256,
                                                                 activation=leaky_relu,
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002),
                                              reuse=reuse, name="attentional_features")
                attentional_condition_vector_i = tf.layers.dropout(attentional_condition_vector_i, rate=0.5,
                                                                   training=is_train)
                attentional_condition_vector_i = tf.concat([attentional_condition_vector_i,
                                                            tf.expand_dims(self.batch_size * [float(i)], axis=1)],
                                                           axis=1)
                attention_a = tf.layers.dense(attentional_condition_vector_i, h * w, activation=tf.nn.softmax,
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002),
                                              name="attention_a_{}".format(i))
                attention_a = tf.reshape(attention_a, shape=[b, h, w, 1])

                attention_b = tf.layers.dense(attentional_condition_vector_i, h * w, activation=tf.nn.softmax,
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002),
                                              name="attention_b_{}".format(i))
                attention_b = tf.reshape(attention_b, shape=[b, h, w, 1])

                flattenA = inputs * attention_a
                flattenA = tf.reduce_sum(flattenA, axis=[1, 2])
                flattenB = inputs * attention_b
                flattenB = tf.reduce_sum(flattenB, axis=[1, 2])
                flattenA = tf.layers.flatten(flattenA)
                flattenB = tf.layers.flatten(flattenB)
                object_A = tf.concat([flattenA, tf.expand_dims(self.batch_size * [float(i)], axis=1)], axis=1)
                object_B = tf.concat([flattenB, tf.expand_dims(self.batch_size * [float(i)], axis=1)], axis=1)
                g_i = g_theta(object_A, object_B, q, reuse=reuse)

                reuse = True
                # print(dense_layer.get_shape())
                object_combos.append(g_i)

            all_g = tf.stack(object_combos, axis=0)
            all_g = tf.reduce_mean(all_g, axis=0, name='all_g')
            return all_g

        # Classifier: takes images as input and outputs class label [B, m]
        def CONV(img, q, scope='CONV'):
            with tf.variable_scope(scope) as scope:
                log.warn(scope.name)
                conv_1 = conv2d(img, conv_info[0], is_train, s_h=3, s_w=3, name='conv_1')
                conv_2 = conv2d(conv_1, conv_info[1], is_train, s_h=3, s_w=3, name='conv_2')
                conv_3 = conv2d(conv_2, conv_info[2], is_train, name='conv_3')
                conv_4 = conv2d(conv_3, conv_info[3], is_train, name='conv_4')

                # eq.1 in the paper
                # g_theta = (o_i, o_j, q)
                # conv_4 [B, d, d, k]
                all_g = attentional_relational_layer(inputs=conv_4, q=q, slices=32)
                return all_g

        def f_phi(g, scope='f_phi'):
            with tf.variable_scope(scope) as scope:
                log.warn(scope.name)
                fc_1 = fc(g, 256, name='fc_1')
                fc_2 = fc(fc_1, 256, name='fc_2')
                fc_2 = slim.dropout(fc_2, keep_prob=0.5, is_training=is_train, scope='fc_3/')
                fc_3 = fc(fc_2, n, activation_fn=None, name='fc_3')
                return fc_3

        g = CONV(self.img, self.q, scope='CONV')
        logits = f_phi(g, scope='f_phi')
        self.all_preds = tf.nn.softmax(logits)
        self.loss, self.accuracy = build_loss(logits, self.a)

        # Add summaries
        def draw_iqa(img, q, target_a, pred_a):
            fig, ax = tfplot.subplots(figsize=(6, 6))
            ax.imshow(img)
            ax.set_title(question2str(q))
            ax.set_xlabel(answer2str(target_a)+answer2str(pred_a, 'Predicted'))
            return fig

        try:
            tfplot.summary.plot_many('IQA/',
                                     draw_iqa, [self.img, self.q, self.a, self.all_preds],
                                     max_outputs=4,
                                     collections=["plot_summaries"])
        except:
            pass

        tf.summary.scalar("loss/accuracy", self.accuracy)
        tf.summary.scalar("loss/cross_entropy", self.loss)
        log.warn('Successfully loaded the model.')
