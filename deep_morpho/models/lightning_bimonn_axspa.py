from .generic_lightning_model import GenericLightningModel
from .bimonn_axspa import BimonnAxspaFromSegm


class LightningBimonnAxspaFromSegm(GenericLightningModel):
    model_class = BimonnAxspaFromSegm

    # def general_step(self, batch, batch_idx, state="",):
    #     (x, segm), y = batch
    #     predictions = self.forward((x, segm))

    #     outputs = self.compute_loss(state=state, ypred=predictions["pred"], ytrue=y)

    #     return outputs, predictions

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
            "lui_kwargs": {"force_identity": args['force_lui_identity']},
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
