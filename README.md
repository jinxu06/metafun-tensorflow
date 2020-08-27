# Meta-Learning with Iterative Functional Updates

This repository contains Tensorflow implementation accompaning the paper, [MetaFun: Meta-Learning with Iterative Functional Updates (Xu et al., ICML 2020)](https://arxiv.org/abs/1912.02738). It includes code for running 1D sinusoid regression, *miniImagenet* and *tieredImageNet* few-shot classification.


## Dependencies

This code requires the following packages:
* Python 3.6
* [TensorFlow v1.15](https://www.tensorflow.org/install/pip) (`pip install tensorflow-gpu==1.15`)
* [DeepMind Sonnet v1.36](https://github.com/deepmind/sonnet/tree/v1.36)  (`pip install dm-sonnet==1.36`)
* [Abseil](https://github.com/abseil/abseil-py) (`pip install absl-py`)
* [Matplotlib](https://matplotlib.org/3.1.1/users/installing.html) (`pip install matploblib`)

Eariler or later versions of these packages may also work, but we haven't upgraded our code to work with *Tensorflow 2* and *Sonnet 2* yet.

## Data

For *miniImageNet* and *tieredImageNet*, we use pretrained [embeddings](http://storage.googleapis.com/leo-embeddings/embeddings.zip) from [LEO](https://github.com/deepmind/leo) as input features. Download and extract the zip file in a preferred data storage location: 
```
wget http://storage.googleapis.com/leo-embeddings/embeddings.zip
unzip embeddings.zip
```

and set `--data_path /path/to/embeddings` when running the code in the command line (see below). 

## Running the code

We provide a few examples for running the code in the command line. Please use `--gpus` to specify which GPU card you want to use, and use `--exp_name` to name the experiment. A new experiment will override checkpoints of previous experiments with the same name.

### Few-shot classification
For new problems, users should 
1. Run experiments with different configurations of hyperparameters on the meta-training set, and compare them on the meta-validation set.
2. Choose the best configuration of hyperparameters, and train the best model on the union of meta-training and meta-validation set, by setting the `--train_on_val` flag.
3. Evaluate the final model on the meta-test set.

To quickly try out the code, below are a few running examples where hyperparameters are directly given. For other problems or settings, please first run randomised hyperparameter search, or refer to the paper appendix (Some results of randomised hyperparameters search are given there).

To train a MetaFun classifier with dot-product attention on tieredImageNet 1-shot problems:
```
python run.py --gpus 0 --exp_name metafun-tiered-1shot-attention --outer_lr 5.55e-05 --dropout_rate 0.123 --nn_size 128 --dim_reprs 128 --nn_layers 2 --num_iters 3 --initial_state_type constant --initial_learning_rate 1.0 --embedding_layers 1 --l2_penalty_weight 1.92e-09 --orthogonality_penalty_weight 1.37e-3 --label_smoothing 0.1 --num_steps_limit 7200 --dataset_name tieredImageNet --num_tr_examples_per_class 1  --checkpoint_dir ./checkpoints --attention_type dot_product --data_path /path/to/embeddings --train_on_val
```
For evaluation on the meta-test set, simply add `--evaluation_mode --eval_set test --no_early_stopping` (It would not make sense to use early stopping here as we also train on the validation set) to the above:
```
python run.py --gpus 0 --exp_name metafun-tiered-1shot-attention --outer_lr 5.55e-05 --dropout_rate 0.123 --nn_size 128 --dim_reprs 128 --nn_layers 2 --num_iters 3 --initial_state_type constant --initial_learning_rate 1.0 --embedding_layers 1 --l2_penalty_weight 1.92e-09 --orthogonality_penalty_weight 1.37e-3 --label_smoothing 0.1 --num_steps_limit 7200 --dataset_name tieredImageNet --num_tr_examples_per_class 1  --checkpoint_dir ./checkpoints --attention_type dot_product --data_path /path/to/embeddings  --train_on_val --evaluation_mode --eval_set test --no_early_stopping

```

Similarly, to train a MetaFun classifier with deep kernels on tieredImageNet 5-shot problems:
```
python run.py --gpus 0 --exp_name metafun-tiered-5shot-kernel --outer_lr 4.5e-05 --dropout_rate 0.148 --nn_size 128 --dim_reprs 128 --nn_layers 3 --num_iters 4 --initial_state_type zero --initial_learning_rate 0.1 --embedding_layers 1 --l2_penalty_weight 6.22e-09 --orthogonality_penalty_weight 7.33e-3 --label_smoothing 0.1 --num_steps_limit 8000 --dataset_name tieredImageNet --num_tr_examples_per_class 5  --checkpoint_dir ./checkpoints --use_kernel --kernel_type deep_se --data_path /path/to/embeddings --train_on_val
```
To evaluate:
```
python run.py --gpus 0 --exp_name metafun-tiered-5shot-kernel --outer_lr 4.5e-05 --dropout_rate 0.148 --nn_size 128 --dim_reprs 128 --nn_layers 3 --num_iters 4 --initial_state_type zero --initial_learning_rate 0.1 --embedding_layers 1 --l2_penalty_weight 6.22e-09 --orthogonality_penalty_weight 7.33e-3 --label_smoothing 0.1 --num_steps_limit 8000 --dataset_name tieredImageNet --num_tr_examples_per_class 5  --checkpoint_dir ./checkpoints --use_kernel --kernel_type deep_se --data_path /path/to/embeddings --train_on_val --evaluation_mode --eval_set test --no_early_stopping
```

### Few-shot regression 
Because the sinusoid regression problem is mainly for qualitative analysis and visualisation, we directly run the following without hyperparameter search:
```
python run.py --gpus 0 --exp_name sinusoid_regression --dataset_name sinusoid --outer_lr 1e-4 --dropout_rate 0.0 --nn_size 128 --dim_reprs 128 --nn_layers 3 --num_iters 5 --initial_state_type zero --initial_learning_rate 0.1 --model_cls metafun_regressor --training_batch_size 16  --eval_batch_size 16  --checkpoint_steps 2000 --l2_penalty_weight 0.0 --max_num_context 10 --num_steps_limit 100000 --checkpoint_dir ./checkpoints --result_dir ./results
```
To evaluate:
```
python run.py --gpus 0 --exp_name sinusoid_regression --dataset_name sinusoid --outer_lr 1e-4 --dropout_rate 0.0 --nn_size 128 --dim_reprs 128 --nn_layers 3 --num_iters 5 --initial_state_type zero --initial_learning_rate 0.1 --model_cls metafun_regressor --training_batch_size 16  --eval_batch_size 16  --checkpoint_steps 2000 --l2_penalty_weight 0.0 --max_num_context 10 --num_steps_limit 100000 --checkpoint_dir ./checkpoints --result_dir ./results --evaluation_mode --eval_set test --no_early_stopping
```
Results for both training and evaluation can be found in `--result_dir`.

To compare to MAML quantitatively (table 1 in the paper) with a similar number of parameters (which we understand may not be a good indicator of model complexity), users can run the following version of our model:
```
python run.py --gpus 0 --exp_name sinusoid_regression --dataset_name sinusoid --outer_lr 1e-4 --dropout_rate 0.0 --nn_size 128 --dim_reprs 128 --nn_layers 3 --num_iters 5 --initial_state_type zero --initial_learning_rate 0.1 --model_cls metafun_regressor --training_batch_size 16  --eval_batch_size 16  --checkpoint_steps 2000 --l2_penalty_weight 0.0 --max_num_context 10 --num_steps_limit 100000 --checkpoint_dir ./checkpoints --result_dir ./results --repr_as_inputs --embedding_layers 2
```
It uses latent representation directly as inputs to the decoder and is not recommended if you do not want to limit the number of parameters.

### Randomised hyperparameter search
[hps.py](https://github.com/jinxu06/metafun-tensorflow/blob/master/hps.py) is an example of script for randomised hyperparameter search. Before running the script, users should change settings at the top of this file, and even change distributions of hyperparameters in *class HPSearch* when necessary. 

To run hyperparameters search for tieredImageNet 5-shot problems on GPU card 0:
```
python hps.py exp_name 0 tieredImageNet 5 run
python hps.py exp_name 0 tieredImageNet 5 compare
python hps.py exp_name 0 tieredImageNet 5 query
```

The `run` step runs lots of experiments with randomly sampled hyperparameters on the meta-training set. Note that it is possible to run the `run` step on multiple GPUs at the same time. The `compare` step compares all these models from the last step on the validation set, choose the best model, and train this best model with different random seeds for 5 times on both meta-training and meta-validation set. The `query` step simply queries mean and standard deviation of test accuracies of these 5 runs.

## Contact

To ask questions about code or report issues, please open an issue on github. For research discussions, please email <jin.xu@stats.ox.ac.uk>.

## Acknowledgements

This repository includes code from two previous projects: [LEO](https://github.com/deepmind/leo) and [Neural Processes](https://github.com/deepmind/neural-processes). The original copyright and licenses are included in [LICENSE](https://github.com/jinxu06/metafun-tensorflow/blob/master/LICENSE).

## Citation

```
@inproceedings{xu2019metafun,
  title = {MetaFun: Meta-Learning with Iterative Functional Updates},
  author = {Xu, Jin and Ton, Jean-Francois and Kim, Hyunjik and Kosiorek, Adam R and Teh, Yee Whye},
  booktitle = {International Conference on Machine Learning (ICML)},
  year = {2020}
}
```



