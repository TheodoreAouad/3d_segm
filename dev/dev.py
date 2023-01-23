import io
import pathlib

import fsspec
import torch
import torch.nn as nn
import torch.optim as optim

import deep_morpho.initializer as inits
import deep_morpho.models.bise_base as bise_base

import deep_morpho.models.lightning_bimonn as lbimonn


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(device)

n_classes = 10


def print_params(layer):
    print("binary params:", layer.numel_binary())
    print("learnable params:", sum([param.numel() for param in layer.parameters() if param.requires_grad]))


model_args = {
    "kernel_size": (3, 3),
    "channels": [10, 100],
    "atomic_element": 'bisel',
    "input_size": (28, 28),
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


model = lbimonn.LightningBiMoNNClassifierMaxPoolNotBinary(
    learning_rate=1e-1,
    loss=nn.BCELoss(),
    optimizer=optim.Adam,
    model_args=model_args,
)
print_params(model.model)


def atomic_save(checkpoint, filepath: str):
    """Saves a checkpoint atomically, avoiding the creation of incomplete checkpoints.

    Args:
        checkpoint: The object to save.
            Built to be used with the ``dump_checkpoint`` method, but can deal with anything which ``torch.save``
            accepts.
        filepath: The path to which the checkpoint will be saved.
            This points to the file that the checkpoint will be stored in.
    """

    bytesbuffer = io.BytesIO()
    # Can't use the new zipfile serialization for 1.6.0 because there's a bug in
    # torch.hub.load_state_dict_from_url() that prevents it from loading the new files.
    # More details can be found here: https://github.com/pytorch/pytorch/issues/42239
    torch.save(checkpoint, bytesbuffer)
    with fsspec.open(filepath, "wb") as f:
        f.write(bytesbuffer.getvalue())


atomic_save(model.hparams, "todelete/model.ckpt")
pathlib.Path("todelete").mkdir(exist_ok=True, parents=True)
