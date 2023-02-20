from .generic_lightning_model import GenericLightningModel

from .classical_conv import (
    ConvNetLastLinear
)


class LightningConvNetLastLinear(GenericLightningModel):
    model_class = ConvNetLastLinear

    @classmethod
    def get_model_from_experiment(cls, experiment: "ExperimentBase") -> GenericLightningModel:
        args = experiment.args
        inpt = experiment.input_sample

        model_args = args.model_args()

        model_args.update({
            "input_size": inpt.shape[1:],
            "n_classes": experiment.trainloader.dataset.n_classes,
        })

        model = cls(
            model_args=model_args,
            learning_rate=args["learning_rate"],
            loss=args["loss"],
            optimizer=args["optimizer"],
            optimizer_args=args["optimizer_args"],
            observables=experiment.observables,
        )
        model.to(experiment.device)
        return model
