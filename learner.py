from six.moves import range
from six.moves import zip
import os
import pickle
import functools
from absl import flags
import numpy as np
import tensorflow as tf
import data.classification as cls_data
from data.sine_curves import SineCurvesReader
import utils

FLAGS = flags.FLAGS
flags.DEFINE_float("outer_lr", 1e-4, "Outer learning rate (Learning rate for the global optimiser).")
flags.DEFINE_integer("training_batch_size", 12, "Number of tasks in a batch for training.")
flags.DEFINE_integer("eval_batch_size", 100, "Number of tasks in a batch for evaluation.")
flags.DEFINE_boolean("no_early_stopping", False, "Whether to remove early_stopping and "
                        "use the latest model checkpoint.")
flags.DEFINE_boolean("train_on_val", False, "Whether to train on the union of meta-train "
                        "and meta-validation data.")

class CLearner(object):

    def __init__(self, name=""):
        self.name = name
        self.eval_metric_type = 'acc'

    def load_data(self):
        assert FLAGS.dataset_name in ['miniImageNet', 'tieredImageNet'], "Unknown dataset name"
        self.train_data = cls_data.construct_examples_batch(FLAGS.dataset_name, 
                            FLAGS.training_batch_size, "train", train_on_val=FLAGS.train_on_val)
        self.eval_data = cls_data.construct_examples_batch(FLAGS.dataset_name, FLAGS.eval_batch_size, "val")
        self.test_data = cls_data.construct_examples_batch(FLAGS.dataset_name, FLAGS.eval_batch_size, "test")

    def construct_graph(self, model_cls):
        # construct model
        self.model = model_cls()
        # construct loss and accuracy ops
        self.train_loss, self.train_tr_metric, self.train_val_metric = \
                _construct_loss_and_eval_ops_for_classification(self.model, self.train_data, is_training=True)
        self.train_eval_loss, self.train_eval_tr_metric, self.train_eval_val_metric = \
                _construct_loss_and_eval_ops_for_classification(self.model, self.train_data, is_training=False)
        self.eval_loss,  self.eval_tr_metric, self.eval_val_metric = \
                _construct_loss_and_eval_ops_for_classification(self.model, self.eval_data, is_training=False)
        self.test_loss, self.test_tr_metric, self.test_val_metric = \
                _construct_loss_and_eval_ops_for_classification(self.model, self.test_data, is_training=False)
        # construct optimisation ops
        training_variables = tf.compat.v1.trainable_variables()
        training_gradients = tf.gradients(self.train_loss, training_variables)
        training_gradients = utils.clip_gradients(training_gradients, 0, 0) # gradient clipping is not used
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.outer_lr)
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        self.train_op = optimizer.apply_gradients(
                list(zip(training_gradients, training_variables)), self.global_step)

    def train(self, num_steps_limit, checkpoint_steps, checkpoint_path):
        global_step_ev = tf.compat.v1.train.global_step(self.sess, self.global_step)
        best_eval_metric = (0.0 if self.eval_metric_type == 'acc' else 1.e16)
        while global_step_ev <= num_steps_limit:
            if global_step_ev % checkpoint_steps == 0:
                # evaluating model when checkpointing
                eval_tr_metric_ev, eval_val_metric_ev = utils.evaluate_and_average(
                        self.sess, [self.eval_tr_metric, self.eval_val_metric], 10)
                print("[  Step: {1} meta-valid context_{0}: {2:.5f}, "
                        "meta-valid target_{0}: {3:.5f}  ]".format(self.eval_metric_type, 
                        global_step_ev, eval_tr_metric_ev, eval_val_metric_ev))   
                # copy best checkpoints for early stopping
                if self.eval_metric_type == 'acc':
                    if eval_val_metric_ev > best_eval_metric:
                        utils.copy_checkpoint(checkpoint_path, global_step_ev, 
                                eval_val_metric_ev, eval_metric_type=self.eval_metric_type)
                        best_eval_metric = eval_val_metric_ev
                else:
                    if eval_val_metric_ev < best_eval_metric:
                        utils.copy_checkpoint(checkpoint_path, global_step_ev, 
                                eval_val_metric_ev, eval_metric_type=self.eval_metric_type)
                        best_eval_metric = eval_val_metric_ev
                self.visualise(save_name="{0}-{1}".format(self.name, global_step_ev))
            if global_step_ev == num_steps_limit:
                global_step_ev += 1
                continue
            # train step
            _, train_tr_metric_ev, train_val_metric_ev = self.sess.run([self.train_op, self.train_tr_metric, self.train_val_metric])
            global_step_ev = tf.compat.v1.train.global_step(self.sess, self.global_step)



    def evaluate(self, num_examples=10000, eval_set='val'):
        num_estimates = (num_examples // FLAGS.eval_batch_size)
        if eval_set == 'train':
            tr_metric_ev, val_metric_ev = utils.evaluate_and_average(
                    self.sess, [self.train_eval_tr_metric, self.train_eval_val_metric], num_estimates)
        elif eval_set == 'val':
            tr_metric_ev, val_metric_ev = utils.evaluate_and_average(
                    self.sess, [self.eval_tr_metric, self.eval_val_metric], num_estimates)
        elif eval_set == 'test':
            tr_metric_ev, val_metric_ev = utils.evaluate_and_average(
                    self.sess, [self.test_tr_metric, self.test_val_metric], num_estimates)
        early_stopping_step = tf.compat.v1.train.global_step(self.sess, self.global_step) 
        print("[  Evaluation --- context_{0}: {1:.5f}, target_{0}: {2:.5f} @checkpoint step {3} ]".format(self.eval_metric_type, tr_metric_ev, val_metric_ev, early_stopping_step))
        return val_metric_ev, early_stopping_step

    def visualise(self, save_name="test", num_plots=9):
        pass

    def set_session(self, sess):
        self.sess = sess

    def get_session(self):
        return self.sess


def _construct_loss_and_eval_ops_for_classification(inner_model, inputs, is_training):
    call_fn = functools.partial(inner_model.__call__, is_training=is_training)
    per_instance_loss, per_instance_tr_metric, per_instance_val_metric = tf.map_fn(call_fn, inputs, dtype=(tf.float32, tf.float32, tf.float32),
                                                back_prop=is_training)
    loss = tf.reduce_mean(per_instance_loss)
    tr_metric = tf.reduce_mean(per_instance_tr_metric)
    val_metric = tf.reduce_mean(per_instance_val_metric)
    return loss, tr_metric, val_metric


class RLearner(CLearner):

    def __init__(self, name="", result_path="results"):
        super(RLearner, self).__init__(name=name)
        self.eval_metric_type = "mse"
        self.result_path = result_path

    def load_data(self):
        if FLAGS.dataset_name == 'sinusoid':
            self.train_data = SineCurvesReader(batch_size=FLAGS.training_batch_size, 
                    max_num_context=FLAGS.max_num_context).generate_curves()
            self.eval_data = SineCurvesReader(batch_size=FLAGS.eval_batch_size, 
                    max_num_context=FLAGS.max_num_context, testing=True).generate_curves()
            self.test_data = self.eval_data
        else:
            raise NameError("Unknown dataset name")

    def construct_graph(self, model_cls):
        # model
        self.model = model_cls()
        # loss and accuracy
        train_ops = _construct_loss_and_eval_ops_for_regression(self.model, self.train_data, is_training=True)
        self.train_loss, self.train_tr_metric, self.train_val_metric = train_ops[0], train_ops[1], train_ops[2]
        eval_ops = _construct_loss_and_eval_ops_for_regression(self.model, self.eval_data, is_training=False)
        self.eval_loss, self.eval_tr_metric, self.eval_val_metric,  self.eval_tr_input, self.eval_tr_output, \
                self.eval_tr_func, self.eval_val_input, self.eval_val_output, self.eval_val_func, self.eval_val_preds, \
                self.eval_val_sigma = eval_ops
        
        # optimisation
        training_variables = tf.trainable_variables()
        training_gradients = tf.gradients(self.train_loss, training_variables)
        training_gradients = utils.clip_gradients(training_gradients, 0, 0)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.outer_lr)
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        self.train_op = optimizer.apply_gradients(
            list(zip(training_gradients, training_variables)), self.global_step)

    def visualise(self, save_name="", num_plots=9):
        tr_input, tr_output, tr_func, val_input, val_output, val_func, preds, sigma = \
                self.sess.run([self.eval_tr_input, self.eval_tr_output, self.eval_tr_func, 
                self.eval_val_input, self.eval_val_output, self.eval_val_func, self.eval_val_preds, self.eval_val_sigma])
        preds = np.transpose(preds, axes=[1,0,2,3])
        sigma = np.transpose(sigma, axes=[1,0,2,3])
        for idx in range(num_plots):
            utils.plot_iterative_functions(val_input[idx:idx+1], val_output[idx:idx+1], val_func[idx:idx+1], 
                    tr_input[idx:idx+1], tr_output[idx:idx+1], tr_func[idx:idx+1], preds[:, idx:idx+1], 
                    fname=os.path.join(self.result_path, "{0}/iters-{1}-{2}.png".format(self.name, save_name, idx)))
        print("(  Figures saved to {} )".format(os.path.join(self.result_path, self.name)))


def _construct_loss_and_eval_ops_for_regression(inner_model, inputs, is_training):
    call_fn = functools.partial(inner_model.__call__, is_training=is_training)
    per_instance_loss, per_instance_tr_metric, per_instance_val_metric, tr_input, tr_output, tr_func, \
            val_input, val_output, val_func, val_preds, val_sigma = tf.map_fn(call_fn, inputs, \
            dtype=tuple([tf.float32 for i in range(11)]), back_prop=is_training)
    loss = tf.reduce_mean(per_instance_loss)
    tr_metric = tf.reduce_mean(per_instance_tr_metric)
    val_metric = tf.reduce_mean(per_instance_val_metric)
    return loss, tr_metric, val_metric, tr_input, tr_output, tr_func, val_input, val_output, val_func, val_preds, val_sigma