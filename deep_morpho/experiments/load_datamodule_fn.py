from deep_morpho.datasets import DataModule


def default_load_datamodule_fn(experiment: "ExperimentBase", ) -> DataModule:
    args = experiment.args
    prop_train, prop_val, prop_test = args['train_test_split']
    trainloader, valloader, testloader = DataModule.select(args["dataset"]).get_train_val_test_loader(
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        n_inputs_train=int(prop_train * args['n_inputs']),
        n_inputs_val=int(prop_val * args['n_inputs']),
        n_inputs_test=int(prop_test * args['n_inputs']),
        **args.dataset_args(),
    )
    return trainloader, valloader, testloader
