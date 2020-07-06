from six.moves import zip
import math
from absl import flags
import numpy as np
import tensorflow as tf
import sonnet as snt
from tensorflow.contrib.layers import layer_norm
import data.classification as cls_data

FLAGS = flags.FLAGS
# Model Specification
flags.DEFINE_integer("num_iters", 1, "Number of iterations (T).")
flags.DEFINE_integer("dim_reprs", 64, "Dimension of the functional representation outputs (dim(r(x))).")
flags.DEFINE_integer("nn_size", 64, "Size of hidden layers in neural modules.")
flags.DEFINE_integer("nn_layers", 3, "Number of MLP layers in neural modules.")
flags.DEFINE_integer("embedding_layers", 1, "Num of embedding mlp layers.")
flags.DEFINE_float(
    "initial_inner_lr", 1.0, "The initial learning rate for functional updates.")
flags.DEFINE_boolean("use_kernel", False, "If True, use kernel; If False, use attention.")
flags.DEFINE_boolean("use_gradient", False, "If True, use gradient-based local updater; "
                                        "If False, use neural local updater.")
flags.DEFINE_boolean("no_decoder", False, "Whether to remove decoder and directly use the functional "
                                            "representation as the predictor .")
flags.DEFINE_string("initial_state_type", "zero", "Type of initial state (zero/constant/parametric)")
flags.DEFINE_string("attention_type", "dot_product", "Type of attention (only dot_product is supported now)")
flags.DEFINE_string("kernel_type", "se", "Type of kernel functions (se/deep_se)")
flags.DEFINE_boolean("repr_as_inputs", False, "If true, use reprs as inputs to the decoder; "
                                        "If false, use reprs to generate weights of the predictor.")
# Regularisation
flags.DEFINE_float("dropout_rate", 0.0, "Rate of dropout.")
flags.DEFINE_float("l2_penalty_weight", 1e-8, "The weight measuring the "
                   "importance of the l2 regularization in the final loss. See λ₁ "
                   "in LEO paper.")
flags.DEFINE_float("orthogonality_penalty_weight", 1e-3, "The weight measuring "
                   "the importance of the decoder orthogonality regularization "
                   "in the final loss. See λ₂ in LEO paper.")
flags.DEFINE_float("label_smoothing", 0.0, "Label smoothing for classification tasks.")


class MetaFunClassifier(snt.AbstractModule):

    def __init__(self, name="MetaFunClassifier"):
        super(MetaFunClassifier, self).__init__(name=name)
        self._float_dtype = tf.float32
        self._int_dtype = tf.int32
        # Components configurations
        self._use_kernel = FLAGS.use_kernel
        self._use_gradient = FLAGS.use_gradient
        self._attention_type = FLAGS.attention_type
        self._kernel_type = FLAGS.kernel_type
        self._no_decoder = FLAGS.no_decoder
        if self._no_decoder:
            self._dim_reprs = 1
        self._initial_state_type = FLAGS.initial_state_type
        # Architecture configurations
        self._nn_size = FLAGS.nn_size
        self._nn_layers = FLAGS.nn_layers
        self._dim_reprs = FLAGS.dim_reprs
        self._num_iters = FLAGS.num_iters
        self._embedding_layers = FLAGS.embedding_layers
        # Regularisation configurations
        self._l2_penalty_weight = FLAGS.l2_penalty_weight
        self._dropout_rate = FLAGS.dropout_rate
        self._label_smoothing = FLAGS.label_smoothing
        self._orthogonality_penalty_weight = FLAGS.orthogonality_penalty_weight
        # Data configurations
        self._num_classes = FLAGS.num_classes
        self._num_tr_examples_per_class = FLAGS.num_tr_examples_per_class
        self._num_val_examples_per_class = FLAGS.num_val_examples_per_class
        # Other configurations
        self._initial_inner_lr = FLAGS.initial_inner_lr
        self._nonlinearity = tf.nn.relu


    def _build(self, data, is_training=True):
        data = cls_data.ClassificationDescription(*data)
        self.is_training = is_training
        self.embedding_dim = data.tr_input.get_shape()[-1].value
        # initial states
        tr_reprs = self.forward_initialiser(data.tr_input)
        val_reprs = self.forward_initialiser(data.val_input)
        # inner learning rate
        alpha = tf.compat.v1.get_variable("alpha", [1, 1], dtype=self._float_dtype,
                initializer=tf.constant_initializer(self._initial_inner_lr), trainable=True)
        # iterative functional updating
        for k in range(self._num_iters):
            updates = self.forward_local_updater(tr_reprs, data.tr_output, data.tr_input)
            tr_updates = alpha * self.forward_kernel_or_attention(querys=data.tr_input, keys=data.tr_input, values=updates)
            val_updates = alpha * self.forward_kernel_or_attention(querys=data.val_input, keys=data.tr_input, values=updates)
            tr_reprs += tr_updates
            val_reprs += val_updates
        # decode functional representation
        classifier_weights = self.forward_decoder(tr_reprs)
        tr_loss, tr_metric = self.calculate_loss_and_acc(
                data.tr_input, data.tr_output, classifier_weights)
        classifier_weights = self.forward_decoder(val_reprs)
        val_loss, val_metric = self.calculate_loss_and_acc(
                data.val_input, data.val_output, classifier_weights)
        # aggregate loss and metrics in a batch
        batch_tr_loss = tf.reduce_mean(val_loss)
        batch_tr_metric = tf.reduce_mean(tr_metric)
        batch_val_loss = tf.reduce_mean(val_loss)
        batch_val_metric = tf.reduce_mean(val_metric)   
        #
        regularization_penalty = (
           self._l2_regularization + self._decoder_orthogonality_reg)
        return batch_val_loss + regularization_penalty, batch_tr_metric, batch_val_metric

    ### Initialiser r_0(x) ###
    @snt.reuse_variables
    def forward_initialiser(self, x):
        num_points = tf.shape(x)[0]
        if self._initial_state_type == "parametric":
            reprs = self.parametric_initialiser(x)
        elif self._initial_state_type == "constant":
            reprs = self.constant_initialiser(num_points, trainable=True)
        elif self._initial_state_type == 'zero':
            reprs = self.constant_initialiser(num_points, trainable=False)
        else:
            raise NameError("Unknown initial state type")
        tf.compat.v1.logging.info("forwarded {0} initialiser".format(self._initial_state_type))
        return reprs
    # r_0(x) = c
    @snt.reuse_variables
    def constant_initialiser(self, num_points, trainable=False):
        with tf.compat.v1.variable_scope("constant_initialiser"):
            if trainable:
                init = tf.compat.v1.get_variable(
                    "initial_state", [1, self._dim_reprs],
                    dtype=self._float_dtype,
                    initializer=tf.constant_initializer(0.0), trainable=True)
            else:
                init = tf.zeros([1, self._dim_reprs])
            init = tf.tile(init, [num_points, 1])
            init = tf.concat([init for c in range(self._num_classes)], axis=-1)
            return init
    # r_0(x) = MLP(x) 
    @snt.reuse_variables
    def parametric_initialiser(self, x):
        with tf.compat.v1.variable_scope("parametric_initialiser"):
            after_dropout = tf.nn.dropout(x, rate=self.dropout_rate)
            regularizer = tf.contrib.layers.l2_regularizer(self._l2_penalty_weight)
            initializer = tf.initializers.glorot_uniform(dtype=self._float_dtype)
            module = snt.nets.MLP(
                [self._nn_size] * self._nn_layers + [self._dim_reprs],
                activation=self._nonlinearity,
                use_bias=True,
                regularizers={"w": regularizer},
                initializers={"w": initializer},
            )
            outputs = snt.BatchApply(module, n_dims=1)(after_dropout)
            outputs = tf.concat([outputs for c in range(self._num_classes)], axis=-1)
            return outputs

    ### Local Updater u ###
    @snt.reuse_variables
    def forward_local_updater(self, r, y, x=None, iter=""):
        if self._use_gradient:
            updates = self.gradient_local_updater(r=r, y=y, x=x, iter=iter)
            tf.compat.v1.logging.info("forwarded gradient local updater")
        else:
            r_shape = r.shape.as_list()
            r = tf.reshape(r, r_shape[:-1] +[self._num_classes, r_shape[-1]//self._num_classes])
            updates = self.neural_local_updater(r=r, y=y, x=x, iter=iter)
            updates = tf.reshape(updates, shape=r_shape)
            tf.compat.v1.logging.info("forwarded neural local updater")
        return updates
    #
    @snt.reuse_variables
    def neural_local_updater(self, r, y, x=None, iter=""):
        with tf.compat.v1.variable_scope("neural_local_updater{}".format(iter), reuse=tf.compat.v1.AUTO_REUSE):
            y = tf.one_hot(y, self._num_classes)
            y = tf.transpose(y, perm=[0, 2, 1])
            # reprs = tf.nn.dropout(reprs, rate=self.dropout_rate)
            regularizer = tf.contrib.layers.l2_regularizer(self._l2_penalty_weight)
            initializer = tf.initializers.glorot_uniform(dtype=self._float_dtype)
            # MLP m
            module1 = snt.nets.MLP(
                [self._nn_size] * self._nn_layers,
                activation=self._nonlinearity,
                use_bias=True,
                regularizers={"w": regularizer},
                initializers={"w": initializer},
            )
            outputs = snt.BatchApply(module1, n_dims=2)(r)
            agg_outputs = tf.reduce_mean(outputs, axis=-2, keepdims=True)
            outputs = tf.concat([outputs, tf.tile(agg_outputs, [1,self._num_classes,1])], axis=-1)
            # MLP u+
            module2 = snt.nets.MLP(
                [self._nn_size] * self._nn_layers + [self._dim_reprs],
                activation=self._nonlinearity,
                use_bias=True,
                regularizers={"w": regularizer},
                initializers={"w": initializer},
                name="true",
            )
            outputs_t = snt.BatchApply(module2, n_dims=2)(outputs)
            # MLP u-
            module3 = snt.nets.MLP(
                [self._nn_size] * self._nn_layers + [self._dim_reprs],
                activation=self._nonlinearity,
                use_bias=True,
                regularizers={"w": regularizer},
                initializers={"w": initializer},
                name="false",
            )
            outputs_f = snt.BatchApply(module3, n_dims=2)(outputs)
            outputs = outputs_t * y + outputs_f * (1-y)
            return outputs
    # gradient-based local updater, used in ablation study
    @snt.reuse_variables
    def gradient_local_updater(self, r, y, x=None, iter=""):
        with tf.compat.v1.variable_scope("gradient_local_updater{}".format(iter), reuse=tf.compat.v1.AUTO_REUSE):
            lr = tf.compat.v1.get_variable(
                "lr", [1, self._num_classes * self._dim_reprs],
                dtype=self._float_dtype,
                initializer=tf.constant_initializer(1.0), trainable=True)
        classifier_weights = self.forward_decoder(r)
        tr_loss, _= self.calculate_loss_and_acc(
                x, y, classifier_weights)
        batch_tr_loss = tf.reduce_mean(tr_loss)
        loss_grad = tf.gradients(batch_tr_loss, r)[0]
        updates = - lr * loss_grad
        return updates

    ### Kernel and Attention ###
    @snt.reuse_variables
    def forward_kernel_or_attention(self, querys, keys, values, iter=""):
        if self._use_kernel:
            if self._kernel_type == "se":
                rtn_values = self.squared_exponential_kernel(querys, keys, values, iter=iter)
            elif self._kernel_type == 'deep_se':
                rtn_values = self.deep_se_kernel(querys, keys, values, iter=iter)
            else:
                raise NameError("Unknown kernel type")
            tf.compat.v1.logging.info("forwarded {0} kernel".format(self._kernel_type))
        else:
            rtn_values = self.attention_block(querys, keys, values, iter=iter)
            tf.compat.v1.logging.info("forwarded {0} attention".format(self._attention_type))
        return rtn_values

    

    @snt.reuse_variables
    def squared_exponential_kernel(self, querys, keys, values, iter=""):
        num_keys = tf.shape(keys)[0]
        num_querys = tf.shape(querys)[0]
        with tf.compat.v1.variable_scope("squared_exponential_kernel{}".format(iter), reuse=tf.compat.v1.AUTO_REUSE):
            sigma = tf.compat.v1.get_variable("sigma", shape=(), dtype=self._float_dtype, 
                    initializer=tf.constant_initializer(1.0), trainable=True)
            lengthscale = tf.compat.v1.get_variable("lengthscale", shape=(), dtype=self._float_dtype, 
                    initializer=tf.constant_initializer(1.0), trainable=True)
        _keys = tf.tile(tf.expand_dims(keys, axis=1), [1, num_querys, 1])
        _querys = tf.tile(tf.expand_dims(querys, axis=0), [num_keys, 1, 1])
        sq_norm = tf.reduce_sum((_keys - _querys)**2, axis=-1)
        kernel_qk = sigma**2 * tf.exp(- sq_norm / (2.*lengthscale**2))
        k = kernel_qk
        v = tf.einsum('kq,kv->qv', k, values)
        return v

    @snt.reuse_variables
    def deep_se_kernel(self, querys, keys, values, iter=""):
        with tf.compat.v1.variable_scope("deep_se_kernel{}".format(iter), reuse=tf.compat.v1.AUTO_REUSE):
            # deep embedding of keys and querys
            regularizer = tf.contrib.layers.l2_regularizer(self._l2_penalty_weight)
            initializer = tf.initializers.glorot_uniform(dtype=self._float_dtype)
            module = snt.nets.MLP(
                [self.embedding_dim] * self._embedding_layers, 
                activation=self._nonlinearity,
                use_bias=True,
                regularizers={"w": regularizer},
                initializers={"w": initializer},
            )
            keys = snt.BatchApply(module, n_dims=1)(keys)
            querys = snt.BatchApply(module, n_dims=1)(querys)
        num_keys = tf.shape(keys)[0]
        num_querys = tf.shape(querys)[0]
        with tf.compat.v1.variable_scope("deep_se_kernel"):
            sigma = tf.compat.v1.get_variable("sigma", shape=(), dtype=self._float_dtype, 
                    initializer=tf.constant_initializer(1.0), trainable=True)
            lengthscale = tf.compat.v1.get_variable("lengthscale", shape=(), dtype=self._float_dtype, 
                    initializer=tf.constant_initializer(1.0), trainable=True)
        # compute \sum_i k(x, x_i)u_i
        _keys = tf.tile(tf.expand_dims(keys, axis=1), [1, num_querys, 1])
        _querys = tf.tile(tf.expand_dims(querys, axis=0), [num_keys, 1, 1])
        sq_norm = tf.reduce_sum((_keys - _querys)**2, axis=-1)
        kernel_qk = sigma**2 * tf.exp(- sq_norm / (2.*lengthscale**2))
        k = kernel_qk
        v = tf.einsum('kq,kv->qv', k, values)
        return v

    @snt.reuse_variables
    def attention_block(self, querys, keys, values, iter=""):
        config = {
            "rep": "mlp",
            "output_sizes": [self.embedding_dim] * self._embedding_layers,
            "att_type": self._attention_type,
            "normalise": True,
            "scale": 1.0,
            "l2_penalty_weight": self._l2_penalty_weight,
            "nonlinearity": self._nonlinearity,
        }
        with tf.compat.v1.variable_scope("attention_block{}".format(iter), reuse=tf.compat.v1.AUTO_REUSE):
            attention = Attention(config=config)
            v = attention(keys, querys, values)
        return v

    ### Decoder ###
    @snt.reuse_variables
    def forward_decoder(self, cls_reprs):
        if self._no_decoder:
            # use functional representation directly as the predictor, used in ablation study
            tf.compat.v1.logging.info("no decoder used")
            return cls_reprs
        s = cls_reprs.shape.as_list()
        cls_reprs = tf.reshape(cls_reprs, s[:-1]+[self._num_classes, self._dim_reprs])
        weights_dist_params = self.decoder(cls_reprs)
        fan_in = self.embedding_dim
        fan_out = self._num_classes
        stddev_offset = np.sqrt(2. / (fan_out + fan_in))
        classifier_weights = self.sample(weights_dist_params,
                                                  stddev_offset=stddev_offset)
        return classifier_weights
    # this decoder generates weights of softmax
    @snt.reuse_variables
    def decoder(self, inputs):
        with tf.compat.v1.variable_scope("decoder"):
            l2_regularizer = tf.contrib.layers.l2_regularizer(self._l2_penalty_weight)
            orthogonality_reg = get_orthogonality_regularizer(
                self._orthogonality_penalty_weight)
            initializer = tf.initializers.glorot_uniform(dtype=self._float_dtype)
            # 2 * embedding_dim, because we are returning means and variances
            decoder_module = snt.Linear(
                self.embedding_dim * 2,
                use_bias=True,
                regularizers={"w": l2_regularizer},
                initializers={"w": initializer},
            )
            outputs = snt.BatchApply(decoder_module, n_dims=2)(inputs)
            self._orthogonality_reg = orthogonality_reg(decoder_module.w)
            return outputs

    ### Other ###
    @property
    def dropout_rate(self):
        return self._dropout_rate if self.is_training else 0.0

    @property
    def _l2_regularization(self):
        return tf.cast(tf.reduce_sum(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)),
                                dtype=self._float_dtype)

    def loss_fn(self, model_outputs, original_classes):
        original_classes = tf.squeeze(original_classes, axis=-1)
        one_hot_outputs = tf.one_hot(original_classes, depth=self._num_classes)
        return tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=one_hot_outputs, logits=model_outputs, \
                label_smoothing=self._label_smoothing, reduction=tf.compat.v1.losses.Reduction.NONE)

    def predict(self, inputs, weights):
        if self._no_decoder:
            return weights
        after_dropout = tf.nn.dropout(inputs, rate=self.dropout_rate)
        preds = tf.einsum("ik,imk->im", after_dropout, weights)
        return preds

    def calculate_loss_and_acc(self, inputs, true_outputs, classifier_weights):
        model_outputs = self.predict(inputs, classifier_weights)
        model_predictions = tf.argmax(
                    model_outputs, -1, output_type=self._int_dtype)
        accuracy = tf.contrib.metrics.accuracy(model_predictions,
                    tf.squeeze(true_outputs, axis=-1))
        return self.loss_fn(model_outputs, true_outputs), accuracy

    def sample(self, distribution_params, stddev_offset=0.):
        means, unnormalized_stddev = tf.split(distribution_params, 2, axis=-1)
        stddev = tf.exp(unnormalized_stddev)
        stddev -= (1. - stddev_offset)
        stddev = tf.maximum(stddev, 1e-10)
        distribution = tf.distributions.Normal(loc=means, scale=stddev)
        if not self.is_training:
            return means
        samples = distribution.sample()
        return samples

    @property
    def _decoder_orthogonality_reg(self):
        return self._orthogonality_reg


class MetaFunRegressor(snt.AbstractModule):

    def __init__(self, name="MetaFunRegressor"):
        super(MetaFunRegressor, self).__init__(name=name)
        self._float_dtype = tf.float32
        self._int_dtype = tf.int32
        # components configurations
        self._use_kernel = FLAGS.use_kernel
        self._use_gradient = FLAGS.use_gradient
        self._attention_type = FLAGS.attention_type
        self._kernel_type = FLAGS.kernel_type
        self._no_decoder = FLAGS.no_decoder
        if self._no_decoder:
            self._dim_reprs = 1
        self._initial_state_type = FLAGS.initial_state_type
        # neural module configurations
        self._nn_size = FLAGS.nn_size
        self._nn_layers = FLAGS.nn_layers
        self._dim_reprs = FLAGS.dim_reprs
        self._num_iters = FLAGS.num_iters
        self._embedding_layers = FLAGS.embedding_layers
        # regularisation configurations
        self._l2_penalty_weight = FLAGS.l2_penalty_weight
        self._dropout_rate = FLAGS.dropout_rate
        self._orthogonality_penalty_weight = FLAGS.orthogonality_penalty_weight
        #
        self._initial_inner_lr = FLAGS.initial_inner_lr
        self._orthogonality_reg = 0
        self._loss_type = "log_prob" # mse | log_prob
        self._nonlinearity = tf.nn.relu
        self._repr_as_inputs = FLAGS.repr_as_inputs


    def _build(self, data, is_training=True):
        self.is_training = is_training
        self.embedding_dim = data.tr_input.get_shape()[-1].value
        tr_input = data.tr_input
        val_input = data.val_input
        tr_output = data.tr_output
        val_output = data.val_output
        # initial states
        tr_reprs = self.forward_initialiser(tr_input)
        val_reprs = self.forward_initialiser(val_input)
        all_tr_reprs = [tr_reprs]
        all_val_reprs = [val_reprs]
        # inner learning rate
        alpha = tf.compat.v1.get_variable("alpha", [1, 1], dtype=self._float_dtype,
                initializer=tf.constant_initializer(self._initial_inner_lr), trainable=True)
        # iterative functional updating
        for k in range(self._num_iters):
            updates = self.forward_local_updater(r=tr_reprs, y=tr_output, x=tr_input)
            tr_updates = alpha * self.forward_kernel_or_attention(querys=tr_input, keys=tr_input, values=updates)
            val_updates = alpha * self.forward_kernel_or_attention(querys=val_input, keys=tr_input, values=updates)
            tr_reprs += tr_updates
            val_reprs += val_updates
            all_tr_reprs.append(tr_reprs)
            all_val_reprs.append(val_reprs)

        # record predictions at each iteration for visualisation
        all_val_mu = []
        all_val_sigma = []
        output_sizes = [self._nn_size] * (self._nn_layers-1) + [2] # architecture of the predictor
        # decoder r_t(x) at each iteration into the predictor for visualisation
        for k in range(self._num_iters+1):
            weights = self.forward_decoder(all_tr_reprs[k], output_sizes=output_sizes) # generate weights of the predictor
            tr_mu, tr_sigma = self.predict(tr_input, weights, output_sizes=output_sizes) # forward the predictor
            weights = self.forward_decoder(all_val_reprs[k], output_sizes=output_sizes)
            val_mu, val_sigma = self.predict(val_input, weights, output_sizes=output_sizes)
            all_val_mu.append(val_mu)
            all_val_sigma.append(val_sigma)
        # 
        tr_loss, tr_metric = self.calculate_loss_and_metrics(
                tr_output, tr_mu, tr_sigma)
        val_loss, val_metric = self.calculate_loss_and_metrics(
                val_output, val_mu, val_sigma)
        batch_tr_loss = tf.reduce_mean(tr_loss)
        batch_tr_metric = tf.reduce_mean(tr_metric)
        batch_val_loss = tf.reduce_mean(val_loss)
        batch_val_metric = tf.reduce_mean(val_metric)
        #
        all_val_mu = tf.stack(all_val_mu, axis=0)
        all_val_sigma = tf.stack(all_val_sigma, axis=0)
        #
        regularization_penalty = (
           self._l2_regularization + self._decoder_orthogonality_reg)
        return batch_tr_loss + batch_val_loss + regularization_penalty, batch_tr_metric, batch_val_metric, \
                data.tr_input, data.tr_output, data.tr_func, data.val_input, data.val_output, \
                data.val_func, all_val_mu, all_val_sigma


    ### Initialiser r_0(x) ###
    @snt.reuse_variables
    def forward_initialiser(self, x):
        num_points = tf.shape(x)[0]
        if self._initial_state_type == 'zero':
            reprs = self.constant_initialiser(num_points, trainable=False)
        elif self._initial_state_type == "constant":
            reprs = self.constant_initialiser(num_points, trainable=True)
        elif self._initial_state_type == "parametric":
            reprs = self.parametric_initialiser(x)
        else:
            raise NameError("Unknown initial state type")
        tf.compat.v1.logging.info("forwarded {0} initialiser".format(self._initial_state_type))
        return reprs
    # r_0(x) = c
    @snt.reuse_variables
    def constant_initialiser(self, num_points, trainable=False):
        with tf.compat.v1.variable_scope("constant_initialiser"):
            if trainable:
                init = tf.compat.v1.get_variable(
                    "initial_state", [1, self._dim_reprs],
                    dtype=self._float_dtype,
                    initializer=tf.constant_initializer(0.0), trainable=True)
            else:
                init = tf.zeros([1, self._dim_reprs])
            init = tf.tile(init, [num_points, 1])
            return init
    # r_0(x) = MLP(x) 
    @snt.reuse_variables
    def parametric_initialiser(self, x):
        with tf.compat.v1.variable_scope("parametric_initialiser"):
            regularizer = tf.contrib.layers.l2_regularizer(self._l2_penalty_weight)
            initializer = tf.initializers.glorot_uniform(dtype=self._float_dtype)
            module = snt.nets.MLP(
                [self._nn_size] * self._nn_layers + [self._dim_reprs],
                activation=self._nonlinearity,
                use_bias=True,
                regularizers={"w": regularizer},
                initializers={"w": initializer},
            )
            outputs = snt.BatchApply(module, n_dims=1)(x)
            return outputs

    ### Local Updater u ###
    @snt.reuse_variables
    def forward_local_updater(self, r, y, x):
        if self._use_gradient:
            updates = self.gradient_local_updater(r=r, y=y, x=x)
            tf.compat.v1.logging.info("forwarded gradient local updater")
        else:
            updates = self.neural_local_updater(r=r, y=y, x=x)
            tf.compat.v1.logging.info("forwarded neural local updater")
        return updates
    # neural local updater, for regression, we simply concatenate [r, y, x]
    @snt.reuse_variables
    def neural_local_updater(self, r, y, x=None):
        with tf.compat.v1.variable_scope("neural_local_updater"):
            if x is not None:
                reprs = tf.concat([r, y, x], axis=-1)
            else:
                reprs = tf.concat([r, y], axis=-1)
            regularizer = tf.contrib.layers.l2_regularizer(self._l2_penalty_weight)
            initializer = tf.initializers.glorot_uniform(dtype=self._float_dtype)
            module = snt.nets.MLP(
                [self._nn_size] * self._nn_layers + [self._dim_reprs],
                activation=self._nonlinearity,
                use_bias=True,
                regularizers={"w": regularizer},
                initializers={"w": initializer},
            )
            outputs = snt.BatchApply(module, n_dims=1)(reprs)
            return outputs
    # gradient-based local updater, used in ablation study
    @snt.reuse_variables
    def gradient_local_updater(self, r, y, x=None):
        with tf.compat.v1.variable_scope("gradient_local_updater"):
            lr = tf.compat.v1.get_variable(
                "lr", [1, self._dim_reprs],
                dtype=self._float_dtype,
                initializer=tf.constant_initializer(1.0), trainable=True)

        r = tf.stop_gradient(r)
        weights = self.forward_decoder(r)
        tr_mu, tr_sigma = self.predict(x, weights)
        tr_loss, tr_mse = self.calculate_loss_and_metrics(
                tr_mu, tr_sigma, y)

        # self.debug_op = tf.gradients(weights, reprs)[0]
        batch_tr_loss = tf.reduce_mean(tr_loss)
        loss_grad = tf.gradients(batch_tr_loss, r)[0]
        updates = - lr * loss_grad
        return updates

    ### Kernel and Attention ###
    @snt.reuse_variables
    def forward_kernel_or_attention(self, querys, keys, values):
        if self._use_kernel:
            if self._kernel_type == "se":
                rtn_values = self.squared_exponential_kernel(querys, keys, values)
            elif self._kernel_type == 'deep_se':
                rtn_values = self.deep_se_kernel(querys, keys, values)
            else:
                raise NameError("Unknown kernel type")
            tf.compat.v1.logging.info("forwarded {0} kernel".format(self._kernel_type))
        else:
            rtn_values = self.attention_block(querys, keys, values)
            tf.compat.v1.logging.info("forwarded {0} attention".format(self._attention_type))

        return rtn_values
    
    @snt.reuse_variables
    def squared_exponential_kernel(self, querys, keys, values):
        num_keys = tf.shape(keys)[0]
        num_querys = tf.shape(querys)[0]
        with tf.compat.v1.variable_scope("squared_exponential_kernel"):
            sigma = tf.compat.v1.get_variable("sigma", shape=(), dtype=self._float_dtype, 
                    initializer=tf.constant_initializer(1.0), trainable=True)
            lengthscale = tf.compat.v1.get_variable("lengthscale", shape=(), dtype=self._float_dtype, 
                    initializer=tf.constant_initializer(1.0), trainable=True)
        _keys = tf.tile(tf.expand_dims(keys, axis=1), [1, num_querys, 1])
        _querys = tf.tile(tf.expand_dims(querys, axis=0), [num_keys, 1, 1])
        sq_norm = tf.reduce_sum((_keys - _querys)**2, axis=-1)
        kernel_qk = sigma**2 * tf.exp(- sq_norm / (2.*lengthscale**2))
        k = kernel_qk
        v = tf.einsum('kq,kv->qv', k, values)
        return v

    @snt.reuse_variables
    def deep_se_kernel(self, querys, keys, values):
        with tf.compat.v1.variable_scope("deep_se_kernel"):
            regularizer = tf.contrib.layers.l2_regularizer(self._l2_penalty_weight)
            initializer = tf.initializers.glorot_uniform(dtype=self._float_dtype)
            module = snt.nets.MLP(
                [self._nn_size] * self._embedding_layers,
                activation=self._nonlinearity,
                use_bias=True,
                regularizers={"w": regularizer},
                initializers={"w": initializer},
            )
            keys = snt.BatchApply(module, n_dims=1)(keys)
            querys = snt.BatchApply(module, n_dims=1)(querys)
        num_keys = tf.shape(keys)[0]
        num_querys = tf.shape(querys)[0]
        with tf.compat.v1.variable_scope("deep_se_kernel"):
            sigma = tf.compat.v1.get_variable("sigma", shape=(), dtype=self._float_dtype, 
                    initializer=tf.constant_initializer(1.0), trainable=True)
            lengthscale = tf.compat.v1.get_variable("lengthscale", shape=(), dtype=self._float_dtype, 
                    initializer=tf.constant_initializer(1.0), trainable=True)
        # compute \sum_i k(x, x_i)u_i
        _keys = tf.tile(tf.expand_dims(keys, axis=1), [1, num_querys, 1])
        _querys = tf.tile(tf.expand_dims(querys, axis=0), [num_keys, 1, 1])
        sq_norm = tf.reduce_sum((_keys - _querys)**2, axis=-1)
        kernel_qk = sigma**2 * tf.exp(- sq_norm / (2.*lengthscale**2))
        k = kernel_qk
        v = tf.einsum('kq,kv->qv', k, values)
        return v

    @snt.reuse_variables
    def attention_block(self, querys, keys, values):
        config = {
            "rep": "mlp",
            "output_sizes": [self._nn_size] * self._embedding_layers,
            "att_type": self._attention_type,
            "normalise": True,
            "scale": 1.0,
            "l2_penalty_weight": self._l2_penalty_weight,
            "nonlinearity": self._nonlinearity,
        }
        with tf.compat.v1.variable_scope("attention_block"):
            attention = Attention(config=config)
            v = attention(keys, querys, values)
        return v

    @snt.reuse_variables
    def forward_decoder(self, reprs, output_sizes=[40,40,2]):
        if self._no_decoder:
            tf.compat.v1.logging.info("no decoder used")
            return reprs
        else:
            weights_dist_params = self.decoder(reprs, output_sizes=output_sizes)
            stddev_offset = np.sqrt(1. / self._nn_size)
            weights = self.sample(weights_dist_params, stddev_offset=stddev_offset)
            tf.compat.v1.logging.info("forwarded decoder")
            return weights

    @snt.reuse_variables
    def decoder(self, reprs, output_sizes=[40,40,2]):
        with tf.compat.v1.variable_scope("decoder"):
            # _repr_as_inputs = True: Representation is used as inputs to the predictor
            # _repr_as_inputs = False: Representation is used to generate weights of the predictor
            if self._repr_as_inputs:
                return reprs
            regularizer = tf.contrib.layers.l2_regularizer(self._l2_penalty_weight)
            initializer = tf.initializers.glorot_uniform(dtype=self._float_dtype)
            num_layers = len(output_sizes)
            output_sizes = [self.embedding_dim] + output_sizes
            # count number of parameters in the predictor
            num_params = 0
            for i in range(num_layers):
                num_params += (output_sizes[i]+1) * output_sizes[i+1]
            # decode the representation into the weights of the predictor
            module = snt.nets.MLP(
                [self._nn_size] * self._nn_layers + [2 * num_params],
                activation=self._nonlinearity,
                use_bias=True,
                regularizers={"w": regularizer},
                initializers={"w": initializer},
            )
            outputs = snt.BatchApply(module, n_dims=1)(reprs)
            return outputs

    @snt.reuse_variables
    def predict(self, inputs, weights, output_sizes=[40,40,2]):
        # no_decoder = True: functional representation is the predictor itself (unused for regression problems)
        if self._no_decoder:
            if self._dim_reprs == 1: # predictive mean
                return weights, tf.ones_like(weights) * 0.5
            elif self._dim_reprs == 2: # predictive mean and std
                return tf.split(weights, 2, axis=-1)
            else:
                raise Exception("num_reprs must <=2 if no_decoder")
        if self._repr_as_inputs:
            with tf.compat.v1.variable_scope("predict"):
                regularizer = tf.contrib.layers.l2_regularizer(self._l2_penalty_weight)
                initializer = tf.initializers.glorot_uniform(dtype=self._float_dtype)
                outputs = tf.concat([weights, inputs], axis=-1) # weights is actually repr for repr_as_inputs=True
                # construct the predictor conditioned on the repr (weights). 
                # repr is feeded into the network at each layer.
                for i, s in enumerate(output_sizes):
                    module = snt.nets.MLP(
                        [s],
                        use_bias=True,
                        regularizers={"w": regularizer},
                        initializers={"w": initializer},
                    )
                    outputs = snt.BatchApply(module, n_dims=1)(outputs)
                    if i < len(output_sizes)-1:
                        outputs = self._nonlinearity(outputs)
                        outputs = tf.concat([outputs, weights], axis=-1)
                preds = outputs
        else:
            # use the generated weights to construct the predictor
            num_layers = len(output_sizes)
            output_sizes = [self.embedding_dim] + output_sizes
            begin = 0
            preds = inputs
            for i in range(num_layers):
                in_size = output_sizes[i]
                out_size = output_sizes[i+1]
                end = begin + in_size * out_size
                w = tf.reshape(weights[:, begin:end], [-1, in_size, out_size])
                b = tf.reshape(weights[:, end:end+out_size], [-1, out_size])
                begin = end + out_size
                preds = tf.einsum("ik,ikm->im", preds, w) + b
                if i < num_layers - 1:
                    preds = self._nonlinearity(preds)
        # return preds
        mu, log_sigma = tf.split(preds, 2, axis=-1)
        sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)
        return mu, sigma

    def sample(self, distribution_params, stddev_offset=0.):
        # here we consider a deterministic case, but one can try the probabilstic version
        means, unnormalized_stddev = tf.split(distribution_params, 2, axis=-1)
        return means
        ## probabilstic version:
        # stddev = tf.exp(unnormalized_stddev)
        # stddev -= (1. - stddev_offset)
        # stddev = tf.maximum(stddev, 1e-10)
        # distribution = tf.distributions.Normal(loc=means, scale=stddev)
        # if not self.is_training:
        #     return means
        # samples = distribution.sample()
        # return samples


    def loss_fn(self, model_outputs, labels):
        return tf.losses.mean_squared_error(labels=labels, predictions=model_outputs, 
                reduction=tf.compat.v1.losses.Reduction.NONE)

    def calculate_loss_and_metrics(self, target_y, mus, sigmas, coeffs=None):
        if self._loss_type == "mse":
            mu, sigma = mus, sigmas
            mse = self.loss_fn(mu, target_y)
            return mse, mse
        elif self._loss_type == "log_prob":
            mu, sigma = mus, sigmas
            dist = tf.contrib.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
            loss = - dist.log_prob(target_y)
            mse = self.loss_fn(mu, target_y)
            return loss, loss
        else:
            raise NameError("unknown output_dist_type")

    @property
    def _l2_regularization(self):
        return tf.cast(
            tf.reduce_sum(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)),
            dtype=self._float_dtype)

    @property
    def _decoder_orthogonality_reg(self):
        return self._orthogonality_reg

# (Copy from https://github.com/deepmind/leo, see copyright and original license in our LICENSE file.)
def get_orthogonality_regularizer(orthogonality_penalty_weight):
  """Returns the orthogonality regularizer."""
  def orthogonality(weight):
    """Calculates the layer-wise penalty encouraging orthogonality."""
    with tf.name_scope(None, "orthogonality", [weight]) as name:
      w2 = tf.matmul(weight, weight, transpose_b=True)
      wn = tf.norm(weight, ord=2, axis=1, keepdims=True) + 1e-32
      correlation_matrix = w2 / tf.matmul(wn, wn, transpose_b=True)
      matrix_size = correlation_matrix.get_shape().as_list()[0]
      base_dtype = weight.dtype.base_dtype
      identity = tf.eye(matrix_size, dtype=base_dtype)
      weight_corr = tf.reduce_mean(
          tf.math.squared_difference(correlation_matrix, identity))
      return tf.multiply(
          tf.cast(orthogonality_penalty_weight, base_dtype),
          weight_corr,
          name=name)
  return orthogonality


# Attention modules 
# (Adapted from https://github.com/deepmind/neural-processes, see copyright and original license in our LICENSE file.)
def dot_product_attention(q, k, v, normalise):
    """Computes dot product attention.

    Args:
        q: queries. tensor of  shape [B,m,d_k].
        k: keys. tensor of shape [B,n,d_k].
        v: values. tensor of shape [B,n,d_v].
        normalise: Boolean that determines whether weights sum to 1.
    Returns:
        tensor of shape [B,m,d_v].
    """
    d_k = tf.shape(q)[-1]
    scale = tf.sqrt(tf.cast(d_k, tf.float32))
    unnorm_weights = tf.einsum('jk,ik->ij', k, q) / scale # [B,m,n]
    if normalise:
        weight_fn = tf.nn.softmax
    else:
        weight_fn = tf.sigmoid
    weights = weight_fn(unnorm_weights)
    rep = tf.einsum('ik,kj->ij', weights, v)
    return rep

class Attention(snt.AbstractModule):

    def __init__(self, config=None, name="attention"):
        super(Attention, self).__init__(name=name)
        self._float_dtype = tf.float32
        self._int_dtype = tf.int32
        self._rep = config['rep']
        self._output_sizes = config['output_sizes']
        self._att_type = config['att_type']
        self._normalise = config['normalise']
        self._scale = config['scale']
        self._l2_penalty_weight = config['l2_penalty_weight']
        self._nonlinearity = config['nonlinearity']

    def _build(self, x1, x2, r):
        if self._rep == 'identity':
            k, q = (x1, x2)
        elif self._rep == 'mlp':
            # Pass through MLP
            initializer = tf.initializers.glorot_uniform(dtype=self._float_dtype)
            regularizer = tf.contrib.layers.l2_regularizer(self._l2_penalty_weight)
            module = snt.nets.MLP(
                self._output_sizes,
                activation=self._nonlinearity,
                use_bias=True,
                initializers={"w": initializer},
            )
            k = snt.BatchApply(module, n_dims=1)(x1)
            q = snt.BatchApply(module, n_dims=1)(x2)
        else:
            raise NameError("'rep' not among ['identity','mlp']")

        if self._att_type == 'dot_product':
            rep = dot_product_attention(q, k, r, self._normalise)
        else:
            raise NameError(("'att_type' not among ['dot_product']"))
        return rep