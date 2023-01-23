TEMPLATE_PATH = "ruche_commands/templates/classif.sh"

with open(TEMPLATE_PATH, "r") as f:
    template = f.read()


for atomic in ["bisel", "dual_bisel"]:
    for dataset in ["classif_mnist", "classif_mnist_channel", "cifar10"]:
        for model in ["maxpool", "linear"]:
            with open(f"ruche_commands/{atomic}/{dataset}_{model}.sh", "w") as f:
                print(template.replace("{ATOMIC}", atomic).replace("{DATASET}", dataset).replace("{MODEL}", model), file=f)
