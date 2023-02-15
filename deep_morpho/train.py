import argparse
from importlib import import_module

from deep_morpho import observables as obs
from deep_morpho.experiments import (
    MultiExperiment, Task, load_model_bimonn_classical, ExperimentMorphoBinary
)

parser = argparse.ArgumentParser()
parser.add_argument("--args", default="deep_morpho/saved_args/sandbox/args.py")
path_args_module = parser.parse_known_args()[0].args
path_args_module = path_args_module.replace(".py", "").replace("/", ".")

print(path_args_module)
args_module = import_module(path_args_module)
all_args = args_module.all_args

print(all_args.multi_args[0]["kernel_size"])

all_experiments = MultiExperiment(
    multi_args=all_args.multi_args,
    dest_dir="deep_morpho/results/results_tensorboards",
    stop_on_error=True,
    enforce_experiment_class=True,
)

all_experiments.setup()

with Task("Generating experiments"):
    all_experiments.generate_experiments()

for experiment in all_experiments.experiments:
    experiment.load_model_fn = load_model_bimonn_classical

all_experiments.run_experiments()
