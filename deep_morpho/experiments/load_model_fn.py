from deep_morpho.models import GenericLightningModel


def default_load_model_fn(experiment: "ExperimentBase",) -> GenericLightningModel:
    args = experiment.args
    model = GenericLightningModel.select(args["model"])(
        model_args=args["model_args"],
        learning_rate=args["learning_rate"],
        loss=args["loss"],
        optimizer=args["optimizer"],
        optimizer_args=args["optimizer_args"],
        reduce_loss_fn=args["reduce_loss_fn"],
        initializer=args["initializer"],
        initializer_args=args["initializer_args"],
    )
    model.to(experiment.device)
    return model


def load_model_bimonn_classical(experiment: "ExperimentBase") -> GenericLightningModel:
    args = experiment.args
    inpt = experiment.input_sample

    if isinstance(args["initializer_args"], dict):
        args["initializer_args"]["input_mean"] = inpt.mean().item()
    elif isinstance(args["initializer_args"], list):
        args["initializer_args"][0]["input_mean"] = inpt.mean().item()

    model_args = args.model_args()

    model_args.update({
        "kernel_size": [args['kernel_size'] for _ in range(args['n_atoms'])],
        "atomic_element": args["atomic_element"].replace('dual_', ''),
        "lui_kwargs": {"force_identity": args['force_lui_identity']},
    })

    model = GenericLightningModel.select(args["model"])(
        model_args=model_args,
        learning_rate=args["learning_rate"],
        loss=args["loss"],
        optimizer=args["optimizer"],
        optimizer_args=args["optimizer_args"],
        observables=experiment.observables,
        # reduce_loss_fn=args["reduce_loss_fn"],
        # initializer=args["initializer"],
        # initializer_args=args["initializer_args"],
    )
    model.to(experiment.device)
    return model
