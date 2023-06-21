import deep_morpho.loss.regularization_dist_activated as rda
import torch.optim as optim
import deep_morpho.models as models
import deep_morpho.binarization.bise_closest_selem as bcs


loss_regu = rda.RegularizationProjActivated()

netl = models.LightningBiMoNNClassifierLastLinearNotBinary(
    model_args={
        "atomic_element": "bisel",
        "kernel_size": 3,
        "channels": [2, 2],
        "n_classes": 10,
        "input_size": (10, 28, 28),
        "closest_selem_handler": bcs.BiseClosestMinDistOnCst,
    },
    learning_rate=.1,
    loss=loss_regu,
    optimizer=optim.Adam,
)

loss_regu.set_model(netl.model)
for m in netl.model.modules():
    if isinstance(m, models.BiSEBase):
        print(m)
