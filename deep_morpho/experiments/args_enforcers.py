from abc import ABC, abstractmethod

from torch.nn import CrossEntropyLoss, BCELoss

from general.nn.experiments.experiment_methods import ExperimentMethods


class ArgsEnforcer(ExperimentMethods, ABC):
    @abstractmethod
    def enforce(self, experiment: "ExperimentBase"):
        pass


class ArgsMorpho(ArgsEnforcer):
    def enforce(self, experiment):
        if experiment.args["kernel_size"] == "adapt":
            experiment.args["kernel_size"] = int(max(experiment.args['morp_operation'].max_selem_shape))

        if experiment.args['channels'] == 'adapt':
            experiment.args['channels'] = experiment.args['morp_operation'].in_channels + [experiment.args['morp_operation'].out_channels[-1]]

        if experiment.args["n_atoms"] == 'adapt':
            experiment.args['n_atoms'] = len(experiment.args['morp_operation'])

        experiment.args["model"] = "BiMoNN"


class ArgsNotMorpho(ArgsEnforcer):
    def enforce(self, experiment: "ExperimentBase"):
        experiment.args["morp_operation"] = None


class ArgsDiskorect(ArgsEnforcer):
    def enforce(self, experiment: "ExperimentBase"):
        experiment.args["n_inputs.train"] = experiment.args['n_steps'] * experiment.args['batch_size']
        experiment.args["n_inputs.val"] = experiment.args["batch_size"]
        experiment.args["n_inputs.test"] = experiment.args["batch_size"]

        experiment.args["random_gen_args"] = experiment.args["random_gen_args"].copy()
        # args["random_gen_args"]["border"] = (args["kernel_size"]//2 + 1, args["kernel_size"]//2 + 1)
        experiment.args['random_gen_args']['size'] = experiment.args['random_gen_args']['size'] + (experiment.args["morp_operation"].in_channels[0],)


class ArgsClassification(ArgsEnforcer):
    def enforce(self, experiment: "ExperimentBase"):
        if experiment.args["n_atoms"] == 'adapt':
            experiment.args['n_atoms'] = len(experiment.args['channels']) - 1


class ArgsMnist(ArgsEnforcer):
    def enforce(self, experiment: "ExperimentBase"):
        experiment.args["n_inputs.train"] = 50_000
        experiment.args["n_inputs.val"] = 10_000
        experiment.args["n_inputs.test"] = 10_000


class ArgsCifar(ArgsEnforcer):
    def enforce(self, experiment: "ExperimentBase"):
        import torchvision.transforms as transforms
        from deep_morpho.datasets.cifar_dataset import transform_default

        experiment.args["n_inputs.train"] = 45_000
        experiment.args["n_inputs.val"] = 5_000
        experiment.args["n_inputs.test"] = 10_000
        experiment.args["transform.train"] = transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transform_default,
        ])


class ArgsClassifActivation(ArgsEnforcer):
    def enforce(self, experiment: "ExperimentBase"):
        args = experiment.args
        if "apply_last_activation.net" in args:
            if not args["apply_last_activation"] and args["loss_data_str"] == "BCELoss":
                args.update({
                    "loss_data_str": "CrossEntropyLoss",
                    "loss_data": CrossEntropyLoss(),
                    "loss": {"loss_data": CrossEntropyLoss()}
                })

            elif args["apply_last_activation"] and args["loss_data_str"] == "CrossEntropyLoss":
                args.update({
                    "loss_data_str": "BCELoss",
                    "loss_data": BCELoss(),
                    "loss": {"loss_data": BCELoss()}
                })

# class ArgsMnistClassifChannel(ArgsEnforcer):
#     def enforce(self, experiment: "ExperimentBase"):
#         experiment.args["channels"].insert(0, experiment.args["channel_classif_args"]["levelset_handler_args"]["n_values"])


# class ArgsMnistClassif(ArgsEnforcer):
#     def enforce(self, experiment: "ExperimentBase"):
#         experiment.args["channels"].insert(0, 1)


class ArgsClassifChannel(ArgsEnforcer):
    def enforce(self, experiment: "ExperimentBase"):
        experiment.args['levelset_handler_mode'] = experiment.args['channel_classif_args']['levelset_handler_mode']
        experiment.args['levelset_handler_args'] = experiment.args['channel_classif_args']['levelset_handler_args']
