from .generic_lightning_model import GenericLightningModel
from .bimonn_axspa import BimonnAxspaResnet, ConvSpalikeMerged, ResnetSpalikeMerged, BimonnAxspaConv


class LightningBimonnAxspaResnet(GenericLightningModel):
    model_class = BimonnAxspaResnet

    def __init__(self, lr_bimonn, lr_classifier, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_bimonn = lr_bimonn
        self.lr_classifier = lr_classifier
        self.learning_rate = lr_bimonn

    def configure_optimizers(self):
        return self.optimizer(
            [{"params": self.model.bimonn.parameters(), "lr": self.lr_bimonn}, {"params": self.model.classification.parameters(), "lr": self.lr_classifier}],
            lr=self.learning_rate, **self.optimizer_args
        )

    @classmethod
    def get_model_from_experiment(cls, experiment: "ExperimentBase") -> GenericLightningModel:
        args = experiment.args
        inpt, segm = experiment.input_sample

        if isinstance(args["initializer_args"], dict):
            args["initializer_args"]["input_mean"] = segm.mean().item()
        elif isinstance(args["initializer_args"], list):
            args["initializer_args"][0]["input_mean"] = segm.mean().item()

        model_args = args.model_args()

        model_args.update({
            "bimonn_channels": args["channels"],
            "bimonn_kernel_size": args["kernel_size"],
            "atomic_element": args["atomic_element"].replace('dual_', ''),
            # "lui_kwargs": {"force_identity": args['force_lui_identity']},
        })

        model = cls(
            model_args=model_args,
            learning_rate=None,
            lr_bimonn=args["lr_bimonn"],
            lr_classifier=args["lr_classifier"],
            loss=args["loss"],
            optimizer=args["optimizer"],
            optimizer_args=args["optimizer_args"],
            observables=experiment.observables,
        )
        model.to(experiment.device)
        return model


class LightningConvSpalikeMerged(GenericLightningModel):
    model_class = ConvSpalikeMerged


class LightningResnetSpalikeMerged(GenericLightningModel):
    model_class = ResnetSpalikeMerged


class LightningBimonnAxspaConv(LightningBimonnAxspaResnet):
    model_class = BimonnAxspaConv
