import pathlib
from os.path import join
from deep_morpho.load_args import load_args


template_path = "deep_morpho/template_args_disko.txt"

selems = ['disk', 'hstick', 'dcross']
tb_path_dict = dict(
    # DISKORECT
    **{str(('diskorect', 'dilation', selem)): f'/hdd/aouadt/these/projets/3d_segm/deep_morpho/results/DGMM_2022/sandbox/0/softplus/diskorect/dilation/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    **{str(('diskorect', 'erosion', selem)): f'/hdd/aouadt/these/projets/3d_segm/deep_morpho/results/DGMM_2022/sandbox/0/softplus/diskorect/erosion/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    **{str(('diskorect', 'opening', selem)): f'/hdd/aouadt/these/projets/3d_segm/deep_morpho/results/DGMM_2022/sandbox/1/softplus/diskorect/opening/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    **{str(('diskorect', 'closing', 'disk')): '/hdd/aouadt/these/projets/3d_segm/deep_morpho/results/DGMM_2022/sandbox/0/softplus/diskorect/closing/version_0'},
    **{str(('diskorect', 'closing', 'hstick')): '/hdd/aouadt/these/projets/3d_segm/deep_morpho/results/DGMM_2022/sandbox/1/softplus/diskorect/closing/version_2'},
    **{str(('diskorect', 'closing', 'dcross')): '/hdd/aouadt/these/projets/3d_segm/deep_morpho/results/DGMM_2022/sandbox/0/softplus/diskorect/closing/version_2'},
    **{str(('diskorect', 'black_tophat', selem)): f'/hdd/aouadt/these/projets/3d_segm/deep_morpho/results/DGMM_2022/sandbox/1/softplus/diskorect/black_tophat/version_{nb}' for selem, nb in zip(selems, [0, 5, 10])},
    **{str(('diskorect', 'white_tophat', selem)): f'/hdd/aouadt/these/projets/3d_segm/deep_morpho/results/DGMM_2022/sandbox/1/softplus/diskorect/white_tophat/version_{nb}' for selem, nb in zip(selems, [18, 19, 2])},
    # MNIST
    **{str(('mnist', 'dilation', selem)): f'/hdd/aouadt/these/projets/3d_segm/deep_morpho/results/DGMM_2022/sandbox/1/softplus/mnist/dilation/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    **{str(('mnist', 'erosion', selem)): f'/hdd/aouadt/these/projets/3d_segm/deep_morpho/results/DGMM_2022/sandbox/1/softplus/mnist/erosion/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    **{str(('mnist', 'opening', selem)): f'/hdd/aouadt/these/projets/3d_segm/deep_morpho/results/DGMM_2022/sandbox/1/softplus/mnist/opening/version_{nb}' for selem, nb in zip(selems, [4, 1, 2])},
    **{str(('mnist', 'closing', selem)): f'/hdd/aouadt/these/projets/3d_segm/deep_morpho/results/DGMM_2022/sandbox/1/softplus/mnist/closing/version_{nb}' for selem, nb in zip(selems, [0, 9, 6])},
    **{str(('mnist', 'black_tophat', selem)): f'/hdd/aouadt/these/projets/3d_segm/deep_morpho/results/DGMM_2022/sandbox/1/softplus/mnist/black_tophat/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    **{str(('mnist', 'white_tophat', selem)): f'/hdd/aouadt/these/projets/3d_segm/deep_morpho/results/DGMM_2022/sandbox/1/softplus/mnist/white_tophat/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    # INVERTED MNIST
    **{str(('inverted_mnist', 'dilation', selem)): f'/hdd/aouadt/these/projets/3d_segm/deep_morpho/results/DGMM_2022/sandbox/1/softplus/inverted_mnist/dilation/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    **{str(('inverted_mnist', 'erosion', selem)): f'/hdd/aouadt/these/projets/3d_segm/deep_morpho/results/DGMM_2022/sandbox/1/softplus/inverted_mnist/erosion/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    **{str(('inverted_mnist', 'opening', selem)): f'/hdd/aouadt/these/projets/3d_segm/deep_morpho/results/DGMM_2022/sandbox/1/softplus/inverted_mnist/opening/version_{nb}' for selem, nb in zip(selems, [3, 1, 4])},
    **{str(('inverted_mnist', 'closing', selem)): f'/hdd/aouadt/these/projets/3d_segm/deep_morpho/results/DGMM_2022/sandbox/1/softplus/inverted_mnist/closing/version_{nb}' for selem, nb in zip(selems, [5, 4, 2])},
    **{str(('inverted_mnist', 'black_tophat', selem)): f'/hdd/aouadt/these/projets/3d_segm/deep_morpho/results/DGMM_2022/sandbox/1/softplus/inverted_mnist/black_tophat/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    **{str(('inverted_mnist', 'white_tophat', selem)): f'/hdd/aouadt/these/projets/3d_segm/deep_morpho/results/DGMM_2022/sandbox/1/softplus/inverted_mnist/white_tophat/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
)

for dataset in ['inverted_mnist', 'mnist', 'diskorect']:
    for operation in ['dilation', 'erosion', 'opening', 'closing', 'white_tophat', 'black_tophat']:
        for selem in ['disk', 'dcross', 'hstick']:
            size = 7
            if selem == "disk":
                size = size // 2
            with open(template_path, "r") as f:
                template = f.read()
            tb_path = tb_path_dict[str((dataset, operation, selem))]

            args = load_args(join(tb_path, "args.yaml"))
            with open(join(tb_path, "seed.txt"), "r") as f:
                seed = f.read()

            template = template.replace("SEED", f"{seed}")
            template = template.replace("OPERATION", f"{operation}")
            template = template.replace("SELEM", f"{selem}")
            template = template.replace("SIZE", f"{size}")
            template = template.replace("DATASET", f"{dataset}")
            template = template.replace("LEARNING_RATE", args['learning_rate'])
            template = template.replace("LOSS", args["loss_data"])
            template = template.replace("OPTIMIZER", args["optimizer"])
            template = template.replace("BATCH", args["batch_size"])
            template = template.replace("IS_INVERTED", str(int(dataset=="inverted_mnist")))
            template = template.replace("N_INPUTS", args['n_inputs'])
            template = template.replace("EPOCHS", args["n_epochs"])
            # template = template.replace("TRAIN_TEST_SPLIT", args["train_test_split"])




            path_dest = join("deep_morpho", "args_folder", dataset, operation, f"{selem}.py")
            pathlib.Path(path_dest).parent.mkdir(exist_ok=True, parents=True)
            with open(path_dest, "w") as f:
                print(template, file=f)
