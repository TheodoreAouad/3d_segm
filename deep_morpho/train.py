import argparse
from importlib import import_module

from deep_morpho import observables as obs
from deep_morpho.experiments import (
    MultiExperiment, Task
)

parser = argparse.ArgumentParser()
parser.add_argument("--args", default="deep_morpho/saved_args/sandbox/args.py")
path_args_module = parser.parse_known_args()[0].args
path_args_module = path_args_module.replace(".py", "").replace("/", ".")

print(path_args_module)
args_module = import_module(path_args_module)
all_args = args_module.all_args

print(type(all_args.multi_args[0]["max_epochs"]))

all_experiments = MultiExperiment(
    multi_args=all_args.multi_args,
    dest_dir="deep_morpho/results/results_tensorboards",
    stop_on_error=True,
    enforce_experiment_class=False,
    generate_new_folder=True,
)

all_experiments.setup()

with Task("Generating experiments"):
    all_experiments.generate_experiments()

all_experiments.run_experiments()
