import os
import sys
import time
import shutil
import pickle
import numpy as np


CHECKPOINT_DIR = "/path/to/checkpoints"
DATA_PATH = "/path/to/embeddings"
NUM_EXP = 40
USE_KERNEL = True

def get_timestamp():
    t = time.time()
    t = int(100 * t)
    return str(t)

class Experiment(object):

    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.result = {}

    def generate_cmds(self, gpu, mode='val', num_steps_limit=0, random_seed=123):
        self.config['dim_reprs'] = self.config['nn_size']
        if mode == 'test':
            self.config['num_steps_limit'] = num_steps_limit
        self.config['checkpoint_dir'] = CHECKPOINT_DIR
        cmds = []
        s = "python run.py"
        s += " --gpus {0}".format(gpu)
        s += " --exp_name {0}".format(self.name)
        for k, v in self.config.items():
            if type(v) == bool:
                if v:
                    s += " --{0}".format(k)
            else:
                s += " --{0} {1}".format(k, v)

        if mode == 'val':
            cmds.append(s + " --random_seed {0}\n".format(random_seed))
            cmds.append(s + " --random_seed {0} --evaluation_mode --eval_set val\n".format(random_seed))
        elif mode == 'test':
            cmds.append(s + " --random_seed {0} --train_on_val\n".format(random_seed))
            cmds.append(s + " --random_seed {0} --train_on_val --evaluation_mode --eval_set test --no_early_stopping\n".format(random_seed))
        else:
            raise NameError("unknown mode: {0}".format(mode))
        return cmds

    def run(self, gpu, mode='val', num_steps_limit=0, random_seed=123):
        cmds = self.generate_cmds(gpu, mode, num_steps_limit, random_seed=random_seed)
        for cmd in cmds:
            os.system(cmd)

    def get_result(self):
        try:
            with open(os.path.join(CHECKPOINT_DIR, self.name, "best_checkpoint", "eval_performance"), 'r') as f:
                m, step = f.read().strip().split(",")
                m = float(m)
                step = int(step)
                return m, step
        except:
            return None

    def delete(self):
        path = os.path.join(CHECKPOINT_DIR, self.name)
        try:
            shutil.rmtree(path)
            print("removed folder "+path)
        except:
            pass

class HPSearch(object):

    def __init__(self, dataset_name, num_tr_examples_per_class, name=""):
        self.dataset_name = dataset_name
        self.num_tr_examples_per_class = num_tr_examples_per_class
        self.name = "hps-{0}-{1}-{2}".format(self.dataset_name, self.num_tr_examples_per_class, name)
        if not os.path.exists(self.name):
            os.makedirs(self.name)
        self.hparams = {
            "outer_lr": HPSampler(lambda s: np.random.uniform(low=1., high=10.), postprocessing=lambda p: p * 1e-5),
            "dropout_rate": HPSampler(lambda s: np.random.uniform(low=0., high=0.5)),
            "nn_size": HPSampler(lambda s: np.random.randint(6, 8), postprocessing=lambda p: np.power(2, p)),
            "nn_layers": HPSampler(lambda s: np.random.randint(2, 4)),
            "num_iters": HPSampler(lambda s: np.random.randint(1, 7)),
            "initial_state_type": HPSampler(lambda s: np.random.choice(3), postprocessing=lambda p: ['zero', 'constant', 'parametric'][p]),
            "initial_inner_lr": HPSampler(lambda s: np.random.choice(3), postprocessing=lambda p: [0.1, 1.0, 10.0][p]),
            "embedding_layers": HPSampler(lambda s: np.random.randint(1, 3)),
            "l2_penalty_weight": HPSampler(lambda s: np.random.uniform(low=-10., high=-8.), postprocessing=lambda p: np.power(10., p)),
            "orthogonality_penalty_weight": HPSampler(lambda s: np.random.uniform(low=-4., high=-2.), postprocessing=lambda p: np.power(10., p)),
            "label_smoothing": HPSampler(lambda s: np.random.randint(0, 3), postprocessing=lambda p: p / 10.),
        }
        # constant hparams
        self.hparams['num_steps_limit'] = 1000
        self.hparams['checkpoint_steps'] = 200
        # if using data augmentation (only for miniImageNet), comment out the line below:
        # self.hparams['embedding_crop'] = 'multiview'
        self.hparams["dataset_name"] = dataset_name
        self.hparams["num_tr_examples_per_class"] = num_tr_examples_per_class
        if USE_KERNEL:
            self.hparams["use_kernel"] = True
            self.hparams["kernel_type"] = 'deep_se'
        else:
            self.hparams["use_kernel"] = False
        # self.hparams["use_gradient"] = True
        # self.hparams["no_decoder"] = True

        if self.hparams["use_kernel"]:
            self.hparams["initial_inner_lr"] = 0.1 # large lr seems to cause instabilit for kernels

        self.hparams["data_path"] = DATA_PATH
        self.experiments = []

    def load_experiments(self):
        self.experiments = []
        _, _, filenames = next(os.walk(self.name))
        for fn in filenames:
            if ".exp" in fn:
                with open(os.path.join(self.name, fn), 'rb') as f:
                    self.experiments += pickle.load(f)

        print("--- loading experiments: ---")
        for experiment in self.experiments:
            print(experiment.name)
        print("----------------------------")

    def gen_config(self):
        config = {}
        for key, value in self.hparams.items():
            if isinstance(value, HPSampler):
                value = value.sample()
            config[key] = value
        return config

    def gen_experiments(self, n):
        ts = get_timestamp()
        for i in range(n):
            config = self.gen_config()
            exp = Experiment(name="model-{0}-{1}".format(ts, i+1), config=config)
            self.experiments.append(exp)
        with open(os.path.join(self.name, ts+".exp"), 'wb') as f:
            pickle.dump(self.experiments, f)
        return self.experiments

    def run(self, gpu):
        for experiment in self.experiments:
            experiment.run(gpu)

    def compare(self):
        d = {}
        best_experiment = None
        best_metric = 0.0
        print("--- compare models: ---")
        for experiment in self.experiments:
            r = experiment.get_result()
            if r is None:
                continue
            d[experiment.name] = r
            print(experiment.name, d[experiment.name][0])
            if d[experiment.name][0] > best_metric:
                best_experiment = experiment
                best_metric = d[experiment.name][0]
        print("best config:", best_experiment.config)
        print("best val performance:", best_metric)
        print("early_stopping at iteration", d[best_experiment.name][1])
        print("----------------------------")
        return best_experiment.config, d[best_experiment.name][1]

    def run_best(self, gpu, config, num_steps_limit, n=5):
        results = []
        for i in range(n):
            exp = Experiment(name="{0}-best-{1}".format(self.name, i), config=config)
            exp.run(gpu, mode='test', num_steps_limit=num_steps_limit, random_seed=i+1)
            #results.append(exp.get_result()[0])
        #print(results)

    def get_best_results(self, n=5):
        results = []
        for i in range(n):
            # exp = Experiment(name="{0}-best-{1}".format(self.name, i), config=config)
            exp = Experiment(name="{0}-best-{1}".format(self.name, i), config={})
            results.append(exp.get_result()[0])
        print(results)
        print(np.mean(results))
        print(np.std(results, ddof=1))

    def delete_all(self):
        all_ts = []
        for experiment in self.experiments:
            experiment.delete()
            ts = experiment.name.split("-")[1]
            if ts not in all_ts:
                all_ts.append(ts)
        for ts in all_ts:
            os.remove(os.path.join(self.name, ts))

class HPSampler(object):

    def __init__(self, generator, state=None, postprocessing=lambda p: p):
        self.generator = generator
        self.state = state
        self.postprocessing = postprocessing

    def sample(self):
        s = self.generator(self.state)
        return self.postprocessing(s)


name = sys.argv[1]
gpu_id = sys.argv[2]
dataset = sys.argv[3]
num_shots = sys.argv[4]
running_mode = sys.argv[5]


h = HPSearch(dataset, num_shots, name=name)
if running_mode == 'run':
    h.gen_experiments(NUM_EXP)
    h.run(gpu_id)
elif running_mode == 'compare':
    h.load_experiments()
    config, num_steps_limit = h.compare()
    h.run_best(gpu_id, config, num_steps_limit)
elif running_mode == 'query':
    h.get_best_results()
elif running_mode == 'delete':
    h.load_experiments()
    h.delete_all()
else:
    raise NameError("unknown mode: {0}".format(running_mode))
