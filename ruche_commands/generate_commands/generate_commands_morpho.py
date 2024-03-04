TEMPLATE_PATH = "ruche_commands/templates/morpho.sh"

with open(TEMPLATE_PATH, "r") as f:
    template = f.read()


for dataset in ["mnist", "inverted_mnist", "diskorect"]:
    for operation in ["dilation", "erosion", "opening", "closing", "white_tophat", "black_tophat"]:
        with open(f"ruche_commands/recompute_projected/{dataset}_{operation}.sh", "w") as f:
                print(template.replace("{DATASET}", dataset).replace("{OPERATION}", operation), file=f)
