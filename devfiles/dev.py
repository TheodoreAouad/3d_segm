import pathlib

from deep_morpho.saved_args.sandbox.args import all_args
from deep_morpho.experiments.multi_experiment import MultiExperiment
from deep_morpho.models.classical_conv import ConvNetLastLinear



all_args.multi_args[0]["channels"] = [50, 50]
all_args.multi_args[0]["model"] = "BimonnBiselDenseNotBinary"


exp = MultiExperiment(multi_args=all_args.multi_args, dest_dir="todelete")
exp.setup_experiment(all_args.multi_args[0], dest_dir="todelete")
experiment = exp.setup_experiment(all_args.multi_args[0], dest_dir="todelete")

# pathlib.Path('todelete/cifar10/bimonnclassifierlastlinear/version_0/').mkdir(exist_ok=True, parents=True)

experiment.setup()

print("exp channels", experiment.model.model.channels)

model1 = ConvNetLastLinear(kernel_size=5, channels=[50, 50], n_classes=10, input_size=(30, 32, 32), do_maxpool=True)
print("float param", sum([param.numel() for param in model1.parameters() if param.requires_grad]))
print("diff", sum([param.numel() for param in model1.parameters() if param.requires_grad]) + 32 * 50 + 30*50 + 50 + 10)
