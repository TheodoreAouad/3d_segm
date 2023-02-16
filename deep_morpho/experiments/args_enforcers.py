from abc import ABC, abstractmethod


from general.nn.experiments.experiment_methods import ExperimentMethods


class ArgsEnforcers(ExperimentMethods, ABC):
    @abstractmethod
    def enforce(self, experiment: "ExperimentBase"):
        pass


class ArgsMorpho(ArgsEnforcers):
    def enforce(self, experiment):
        if experiment.args["kernel_size"] == "adapt":
            experiment.args["kernel_size"] = int(max(experiment.args['morp_operation'].max_selem_shape))

        if experiment.args['channels'] == 'adapt':
            experiment.args['channels'] = experiment.args['morp_operation'].in_channels + [experiment.args['morp_operation'].out_channels[-1]]

        if experiment.args["n_atoms"] == 'adapt':
            experiment.args['n_atoms'] = len(self.args['morp_operation'])


class ArgsDiskorect(ArgsEnforcers):
    def enforce(self, experiment: "ExperimentBase"):
        experiment.args["n_inputs_train"] = experiment.args['n_steps'] * experiment.args['batch_size']
        experiment.args["n_inputs_val"] = experiment.args["batch_size"]
        experiment.args["n_inputs_test"] = experiment.args["batch_size"]

        experiment.args["random_gen_args"] = experiment.args["random_gen_args"].copy()
        # args["random_gen_args"]["border"] = (args["kernel_size"]//2 + 1, args["kernel_size"]//2 + 1)
        experiment.args['random_gen_args']['size'] = experiment.args['random_gen_args']['size'] + (experiment.args["morp_operation"].in_channels[0],)


class ArgsClassification(ArgsEnforcers):
    def enforce(self, experiment: "ExperimentBase"):
        if experiment.args["n_atoms"] == 'adapt':
            experiment.args['n_atoms'] = len(experiment.args['channels']) - 1


class ArgsMnist(ArgsEnforcers):
    def enforce(self, experiment: "ExperimentBase"):
        experiment.args["n_inputs"] = 60_000


class ArgsCifar(ArgsEnforcers):
    def enforce(self, experiment: "ExperimentBase"):
        experiment.args["n_inputs"] = 50_000
        experiment.args["channels"][0] = experiment.args["channel_classif_args"]["levelset_handler_args"]["n_values"] * 3
