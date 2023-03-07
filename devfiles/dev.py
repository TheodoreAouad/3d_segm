import torch.nn as nn
import torch.optim as optim

import deep_morpho.initializer as inits
import deep_morpho.models.bise_base as bise_base
import deep_morpho.models.bise as bise
import deep_morpho.models.lightning_bimonn as lbimonn


n_classes = 10

def print_params(layer):
    print("binary params:", layer.numel_binary())
    print("learnable params:", sum([param.numel() for param in layer.parameters() if param.requires_grad]))


model_args = {
    "kernel_size": (5, 5),
    "channels": [100],
    "atomic_element": 'bisel',
    "input_size": (1, 28, 28),
    "n_classes": n_classes,
    "threshold_mode": {
        "weight": 'softplus',
        "activation": 'tanh',
    },
    "initializer_method": inits.InitBimonnEnum.INPUT_MEAN,
    "initializer_args": {
        "bise_init_method": inits.InitBiseEnum.CUSTOM_CONSTANT_RANDOM_BIAS,
        "lui_init_method": inits.InitBiseEnum.CUSTOM_CONSTANT_CONSTANT_WEIGHTS_RANDOM_BIAS,
        "bise_init_args": {"ub": 1e-2, "max_output_value": 0.95, "p_for_init": "auto"},
        "input_mean": .5,
    },
    "closest_selem_method": bise_base.ClosestSelemEnum.MIN_DIST_DIST_TO_CST,
    "bias_optim_mode": bise_base.BiseBiasOptimEnum.POSITIVE,
    "bias_optim_args": {"offset": 0},
    "weights_optim_mode": bise_base.BiseWeightsOptimEnum.THRESHOLDED,
    "weights_optim_args": {"constant_P": True, "factor": 1},
}


model = lbimonn.LightningBiMoNNClassifierLastLinearNotBinary(
    learning_rate=1e-1,
    loss=nn.BCELoss(),
    optimizer=optim.Adam,
    model_args=model_args,
)
print_params(model.model)

all_modules = 0
for module_ in model.modules():
    if isinstance(module_, bise.BiSEBase):
        all_modules += 1

pass
