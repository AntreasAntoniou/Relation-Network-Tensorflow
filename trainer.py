from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

from util import log
from pprint import pprint

from input_ops import create_input_ops

import os
import time
import tensorflow as tf
import numpy as np
import tqdm
import matplotlib.pyplot as mplot

class Trainer(object):

    @staticmethod
    def get_model_class(model_name):
        if model_name == 'baseline':
            from model_baseline import Model
        elif model_name == 'relational_network':
            from model_rn import Model
        elif model_name == 'attentional_relational_network':
            from model_attentional_rn import Model
        else:
            raise ValueError(model_name)
        return Model

    def __init__(self,
                 config,
                 dataset_train,
                 dataset_val,
                 dataset_test):
        self.config = config
        hyper_parameter_str = config.dataset_path+'_lr_'+str(config.learning_rate)
        self.train_dir = './train_dir/%s-%s-%s-%s' % (
            config.model,
            config.prefix,
            hyper_parameter_str,
            time.strftime("%Y%m%d-%H%M%S")
        )

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
        log.infov("Train Dir: %s", self.train_dir)

        # --- input ops ---
        self.batch_size = config.batch_size

        _, self.batch_train = create_input_ops(dataset_train, self.batch_size,
                                               is_training=True)

        _, self.batch_val = create_input_ops(dataset_val, self.batch_size,
                                              is_training=False)

        _, self.batch_test = create_input_ops(dataset_test, self.batch_size,
                                              is_training=False)
        self.train_length = len(dataset_train)
        self.val_length = len(dataset_val)
        self.test_length = len(dataset_test)


        # --- create model ---
        Model = self.get_model_class(config.model)
        log.infov("Using Model class : %s", Model)
        self.model = Model(config)

        # --- optimizer ---
        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
        self.learning_rate = config.learning_rate
        if config.lr_weight_decay:
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate,
                global_step=self.global_step,
                decay_steps=10000,
                decay_rate=0.5,
                staircase=True,
                name='decaying_learning_rate'
            )

        self.check_op = tf.no_op()

        self.optimizer = tf.contrib.layers.optimize_loss(
            loss=self.model.loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            optimizer=tf.train.AdamOptimizer,
            clip_gradients=20.0,
            name='optimizer_loss'
        )

        self.summary_op = tf.summary.merge_all()
        try:
            import tfplot
            self.plot_summary_op = tf.summary.merge_all(key='plot_summaries')
        except:
            pass

        self.saver = tf.train.Saver(max_to_keep=5)
        self.best_val_saver = tf.train.Saver(max_to_keep=5)
        self.summary_writer = tf.summary.FileWriter(self.train_dir)

        self.checkpoint_secs = 600  # 10 min

        self.supervisor = tf.train.Supervisor(
            logdir=self.train_dir,
            is_chief=True,
            saver=None,
            summary_op=None,
            summary_writer=self.summary_writer,
            save_summaries_secs=300,
            save_model_secs=self.checkpoint_secs,
            global_step=self.global_step,
        )

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            # intra_op_parallelism_threads=1,
            # inter_op_parallelism_threads=1,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
        )
        self.session = self.supervisor.prepare_or_wait_for_session(config=session_config)

        self.ckpt_path = config.checkpoint
        if self.ckpt_path is not None:
            log.info("Checkpoint path: %s", self.ckpt_path)
            self.saver.restore(self.session, self.ckpt_path)
            log.info("Loaded the pretrain parameters from the provided checkpoint path")

    def train(self):
        log.infov("Training Starts!")
        pprint(self.batch_train)


        step = 0
        output_save_step = 1000
        epoch_train_iter = int(self.train_length/self.batch_size)# * 10
        epoch_val_iter = int(self.val_length/self.batch_size)# * 10
        total_epochs = int(200000 / epoch_train_iter)

        best_val_accuracy = 0.
        with tqdm.tqdm(total=total_epochs) as epoch_bar:

            for e in range(total_epochs):
                train_loss = []
                train_accuracy = []
                val_loss = []
                val_accuracy = []
                total_train_time = []
                with tqdm.tqdm(total=epoch_train_iter) as train_bar:
                    for train_step in range(epoch_train_iter):
                        step, accuracy, summary, loss, step_time = \
                            self.run_single_step(self.batch_train, step=step, is_train=True)
                        step += 1
                        train_loss.append(loss)
                        train_accuracy.append(accuracy)
                        total_train_time.append(step_time)
                        train_bar.update(1)
                        train_bar.set_description("Train loss: {train_loss}, Train accuracy: {train_accuracy},"
                                                  "Train loss mean: {train_loss_mean}, "
                                                  "Train accuracy mean: {train_accuracy_mean}"
                                                  .format(train_loss=loss, train_accuracy=accuracy,
                                                          train_loss_mean=np.mean(train_loss),
                                                          train_accuracy_mean=np.mean(train_accuracy)))

                    train_loss_mean = np.mean(train_loss)
                    train_loss_std = np.std(train_loss)
                    train_accuracy_mean = np.mean(train_accuracy)
                    train_accuracy_std = np.std(train_accuracy)
                    total_train_time = np.sum(total_train_time)

                with tqdm.tqdm(total=epoch_val_iter) as val_bar:
                    for val_iters in range(epoch_val_iter):

                        loss, accuracy = \
                            self.run_test(self.batch_val, is_train=False)
                        val_loss.append(loss)
                        val_accuracy.append(accuracy)

                        val_bar.update(1)
                        val_bar.set_description("Val loss: {val_loss}, Val accuracy: {val_accuracy},"
                                                "Val loss mean: {val_loss_mean}, Val accuracy mean: {val_accuracy_mean}"
                                                .format(val_loss=loss, val_accuracy=loss,
                                                        val_loss_mean=np.mean(val_loss),
                                                        val_accuracy_mean=np.mean(train_accuracy)))

                    val_loss_mean = np.mean(val_loss)
                    val_loss_std = np.std(val_loss)
                    val_accuracy_mean = np.mean(val_accuracy)
                    val_accuracy_std = np.std(val_accuracy)



                if val_accuracy_mean >= best_val_accuracy:
                    best_val_accuracy = val_accuracy_mean
                    val_save_path = self.best_val_saver.save(self.session,
                                                os.path.join(self.train_dir, 'model'),
                                                global_step=step)
                    print("Saved best val model at", val_save_path)

                self.log_step_message(step, train_accuracy_mean, val_loss_mean, val_accuracy_mean, train_loss_mean, total_train_time,
                                      is_train=True)

                self.summary_writer.add_summary(summary, global_step=step)


                log.infov("Saved checkpoint at %d", step)
                save_path = self.saver.save(self.session,
                                            os.path.join(self.train_dir, 'model'),
                                            global_step=step)
                print("Saved current train model at", save_path)
                epoch_bar.update(1)

    def run_single_step(self, batch, step=None, is_train=True):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        fetch = [self.global_step, self.model.accuracy, self.summary_op,
                 self.model.loss, self.check_op, self.optimizer]

        try:
            if step is not None and (step % 100 == 0):
                fetch += [self.plot_summary_op]
        except:
            pass

        fetch_values = self.session.run(
            fetch, feed_dict=self.model.get_feed_dict(batch_chunk, step=step, is_training=True)
        )
        [step, accuracy, summary, loss] = fetch_values[:4]

        try:
            if self.plot_summary_op in fetch:
                summary += fetch_values[-1]
        except:
            pass

        _end_time = time.time()

        return step, accuracy, summary, loss,  (_end_time - _start_time)

    def run_test(self, batch, is_train=False, repeat_times=8):

        batch_chunk = self.session.run(batch)

        loss, accuracy = self.session.run(
            [self.model.loss, self.model.accuracy], feed_dict=self.model.get_feed_dict(batch_chunk,
                                                                                           is_training=False)
        )

        return loss, accuracy

    def log_step_message(self, step, train_accuracy, val_loss, val_accuracy, train_loss, step_time, is_train=True):
        if step_time == 0:
            step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                "Train Loss: {train_loss:.5f} " +
                "Train Accuracy: {train_accuracy:.2f} "
                "Validation Accuracy: {val_accuracy:.2f} " +
                "Validation Loss: {val_loss:.2f} " +
                "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) "
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step=step,
                         train_loss=train_loss,
                         train_accuracy=train_accuracy*100,
                         val_accuracy=val_accuracy*100,
                         val_loss=val_loss,
                         sec_per_batch=step_time,
                         instance_per_sec=8000 / step_time
                         )
               )


def check_data_path(path):
    if os.path.isfile(os.path.join(path, 'data.hy')) \
           and os.path.isfile(os.path.join(path, 'id.txt')):
        return True
    else:
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model', type=str, default='relational_network', choices=['relational_network', 'baseline', "attentional_relational_network"])
    parser.add_argument('--prefix', type=str, default='default')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--dataset_path', type=str, default='Sort-of-CLEVR_default')
    parser.add_argument('--learning_rate', type=float, default=2.5e-4)
    parser.add_argument('--lr_weight_decay', action='store_true', default=False)
    config = parser.parse_args()

    path = os.path.join('./datasets', config.dataset_path)

    if check_data_path(path):
        import sort_of_clevr as dataset
    else:
        raise ValueError(path)

    config.data_info = dataset.get_data_info()
    config.conv_info = dataset.get_conv_info()
    dataset_train, dataset_val, dataset_test = dataset.create_default_splits(path)

    trainer = Trainer(config,
                      dataset_train, dataset_val, dataset_test)

    log.warning("dataset: %s, learning_rate: %f",
                config.dataset_path, config.learning_rate)
    trainer.train()

if __name__ == '__main__':
    main()
