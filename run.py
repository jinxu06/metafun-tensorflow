from six.moves import range
from six.moves import zip
import os
import pickle
import shutil
from absl import flags
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) 
# increase logging verbosity when debugging: 
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG) 
import utils
from model import MetaFunClassifier, MetaFunRegressor
from learner import CLearner, RLearner

FLAGS = flags.FLAGS
# Path to
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Path to restore from and "
                    "save to checkpoints.")
flags.DEFINE_string("result_dir", "results", "Path to results")
# Dataset 
flags.DEFINE_string("dataset_name", "tieredImageNet", "Name of the dataset to "
                        "train on, which will be mapped to data.MetaDataset.")

flags.DEFINE_integer("num_steps_limit", int(1e5), "Number of steps to train for.")
flags.DEFINE_integer("checkpoint_steps", 200, "The frequency, in number of "
                        "steps, of saving the checkpoints.")
# 
flags.DEFINE_string("exp_name", "experiment", "Name of the experiment.")
flags.DEFINE_string("gpus", "0", "Ids of GPUs where the program run.")
flags.DEFINE_boolean("evaluation_mode", False, "Whether to run in an "
                     "evaluation-only mode.")
flags.DEFINE_string("eval_set", "val", "Which set (train/val/test) to evaluate on.")
flags.DEFINE_integer("random_seed", 0, "Global random seed.")


def main(argv):

    del argv  # Unused.  
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpus
    tf.compat.v1.set_random_seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    if FLAGS.model_cls == 'metafun_classifier':
        model_cls = MetaFunClassifier
        task_type = "classification"
    elif FLAGS.model_cls == 'metafun_regressor':
        model_cls = MetaFunRegressor
        task_type = "regression"
    else:
        raise NameError("model cls name not found")

    if task_type == 'classification':
        learner = CLearner(FLAGS.exp_name)
        learner.load_data()
    elif task_type == 'regression':
        learner = RLearner(FLAGS.exp_name)
        learner.load_data()
    else:
        raise NameError("unknown task_type")


    result_dir = os.path.join(FLAGS.result_dir, FLAGS.exp_name)
    learner.construct_graph(model_cls)
    checkpoint_dir = os.path.join(os.path.abspath(FLAGS.checkpoint_dir), FLAGS.exp_name)

    if os.path.exists(checkpoint_dir) and not FLAGS.evaluation_mode:
        # inp = input("{0} already exists. Do you want to overwrite?(Y/N)".format(FLAGS.checkpoint_path))
        # if inp == 'Y':
        #     shutil.rmtree(FLAGS.checkpoint_path)
        shutil.rmtree(checkpoint_dir)

    if FLAGS.evaluation_mode and (not FLAGS.no_early_stopping):
        checkpoint_dir = os.path.join(checkpoint_dir, "best_checkpoint")

    session_config = tf.compat.v1.ConfigProto()
    session_config.gpu_options.allow_growth = True
    with tf.compat.v1.train.MonitoredTrainingSession(
              checkpoint_dir=checkpoint_dir,
              save_summaries_steps=FLAGS.checkpoint_steps,
              log_step_count_steps=FLAGS.checkpoint_steps,
              save_checkpoint_steps=FLAGS.checkpoint_steps,
              config=session_config) as sess:

        learner.set_session(sess)
        if FLAGS.dataset_name == 'sinusoid':
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
        if FLAGS.evaluation_mode:
            if FLAGS.dataset_name == 'sinusoid':
                learner.visualise("{0}-evaluation".format(FLAGS.exp_name), FLAGS.eval_batch_size)
            else:
                ev, early_stopping_step = learner.evaluate(num_examples=10000, eval_set=FLAGS.eval_set)
                if FLAGS.no_early_stopping:
                    with open(os.path.join(checkpoint_dir, "best_checkpoint", "eval_performance"), 'w') as f:
                        f.write("{0},{1}".format(ev, early_stopping_step))
                else:
                    with open(os.path.join(checkpoint_dir, "eval_performance"), 'w') as f:
                        f.write("{0},{1}".format(ev, early_stopping_step))

            
        else:
            learner.train(FLAGS.num_steps_limit, FLAGS.checkpoint_steps, checkpoint_dir)

if __name__ == "__main__":
    tf.compat.v1.app.run()
