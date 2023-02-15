from deep_morpho.datasets import DataModule


def default_load_datamodule_fn(experiment: "ExperimentBase", ) -> DataModule:
    args = experiment.args
    trainloader, valloader, testloader = DataModule.select(args["dataset"]).get_train_val_test_loader(
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        n_inputs_train=args["n_inputs_train"],
        n_inputs_val=args["n_inputs_val"],
        n_inputs_test=args["n_inputs_test"],
        **args.dataset_args(),
    )
    return trainloader, valloader, testloader
 