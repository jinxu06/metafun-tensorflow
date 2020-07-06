# MIT License

# Copyright (c) 2020 Jin Xu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ============================================================================

# This file includes code from the project github.com/deepmind/leo.

########        Documented Changes        ########

# 1. This file mainly contains functions from github.com/deepmind/leo/utils.py.
# 2. Plotting functions are added here.

########        Original License for LEO        ########

# Copyright 2018 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from six.moves import range
import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_iterative_functions(target_x, target_y, target_f, context_x, context_y, context_f, pred_y, fname=""):

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)

    # ax.plot(target_x[0], target_y[0], 'k:', linewidth=2, label="target")
    ax.plot(target_x[0], target_f[0], color='C3', linestyle='solid', label="target")
    ax.plot(context_x[0], context_y[0], 'k+', markersize=10, label="context")

    colors = ['C0', 'C0', 'C0', 'C0', 'C0', 'C0']
    linestyles = ['dotted', 'dashed', 'dashdot', 'solid', 'solid', 'solid']
    linewidths = [0.8, 0.8, 0.8, 0.8, 1.2, 1.8]
    for iter, p in enumerate(pred_y):
        ax.plot(target_x[0], p[0], label="iter {0}".format(iter), color=colors[iter], 
                linestyle=linestyles[iter], linewidth=linewidths[iter], alpha=1.0)

    ax.legend(loc="lower center", mode="expand", ncol=4, fontsize="x-large")
    plt.yticks([-4, -2, 0, 2, 4], fontsize=16)
    plt.xticks([-6, -4, -2, 0, 2, 4, 6], fontsize=16)
    plt.xlim([-5.5, 5.5])
    plt.ylim([-9.0, 5.5])

    ax.grid(False)
    ax.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(fname)
    plt.close()

def clip_gradients(gradients, gradient_threshold, gradient_norm_threshold):
    """Clips gradients by value and then by norm."""
    if gradient_threshold > 0:
        gradients = [
            tf.clip_by_value(g, -gradient_threshold, gradient_threshold)
            for g in gradients
        ]
    if gradient_norm_threshold > 0:
        gradients = [
            tf.clip_by_norm(g, gradient_norm_threshold) for g in gradients
        ]
    return gradients


def copy_checkpoint(checkpoint_path, global_step, accuracy, eval_metric_type='acc'):
    """Copies the checkpoint to a separate directory."""
    tmp_checkpoint_path = os.path.join(checkpoint_path, "tmp_best_checkpoint")
    best_checkpoint_path = os.path.join(checkpoint_path, "best_checkpoint")
    if _is_previous_accuracy_better(best_checkpoint_path, accuracy, eval_metric_type):
        tf.compat.v1.logging.info("Not copying the checkpoint: there is a better one from "
                        "before a preemption.")
        return

    checkpoint_regex = os.path.join(checkpoint_path,
                                    "model.ckpt-{}.*".format(global_step))
    checkpoint_files = tf.io.gfile.glob(checkpoint_regex)
    graph_file = os.path.join(checkpoint_path, "graph.pbtxt")
    checkpoint_files.append(graph_file)

    _save_files_in_tmp_directory(tmp_checkpoint_path, checkpoint_files, accuracy)

    new_checkpoint_index_file = os.path.join(tmp_checkpoint_path, "checkpoint")
    with tf.io.gfile.GFile(new_checkpoint_index_file, "w") as f:
        f.write("model_checkpoint_path: \"{}/model.ckpt-{}\"\n".format(
            best_checkpoint_path, global_step))

    # We first copy the better checkpoint to a temporary directory, and only
    # when it's created move it to avoid inconsistent state when job is preempted
    # when copying the checkpoint.
    if tf.io.gfile.exists(best_checkpoint_path):
        tf.io.gfile.rmtree(best_checkpoint_path)
    tf.io.gfile.rename(tmp_checkpoint_path, best_checkpoint_path)
    print("(  Copied new best checkpoint with evaluation metric {0:.5f}  )".format(accuracy))


def _save_files_in_tmp_directory(tmp_checkpoint_path, checkpoint_files,
                                 accuracy):
    """Saves the checkpoint files and accuracy in a temporary directory."""

    if tf.io.gfile.exists(tmp_checkpoint_path):
        tf.compat.v1.logging.info("The temporary directory exists, because job was preempted "
                        "before it managed to move it. We're removing it.")
        tf.io.gfile.rmtree(tmp_checkpoint_path)
    tf.io.gfile.mkdir(tmp_checkpoint_path)

    def dump_in_best_checkpoint_path(obj, filename):
        full_path = os.path.join(tmp_checkpoint_path, filename)
        with tf.io.gfile.GFile(full_path, "wb") as f:
            pickle.dump(obj, f)

    for file_ in checkpoint_files:
        just_filename = file_.split("/")[-1]
        tf.io.gfile.copy(
            file_,
            os.path.join(tmp_checkpoint_path, just_filename),
            overwrite=False)

    dump_in_best_checkpoint_path(accuracy, "accuracy")


def _is_previous_accuracy_better(best_checkpoint_path, accuracy, eval_metric_type):
    if not tf.io.gfile.exists(best_checkpoint_path):
        return False

    previous_accuracy_file = os.path.join(best_checkpoint_path, "accuracy")
    with tf.io.gfile.GFile(previous_accuracy_file, "rb") as f:
        previous_accuracy = pickle.load(f)

    if eval_metric_type == 'acc':
        return previous_accuracy > accuracy
    else:
        return previous_accuracy < accuracy


def evaluate_and_average(session, tensors, num_estimates):

    tensor_values_estimates = np.array([session.run(tensors) for _ in range(num_estimates)])
    average_tensor_values = np.mean(tensor_values_estimates, axis=0)
    # average_tensor_value = sum(tensor_value_estimates) / num_estimates
    return tuple(average_tensor_values)