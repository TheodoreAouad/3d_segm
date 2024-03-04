import os

all_paths = []

for dataset in ["mnist", "inverted_mnist", "diskorect"]:
    for operation in ["dilation", "erosion", "opening", "closing", "white_tophat", "black_tophat"]:
        file_path = f"ruche_commands/recompute_projected/{dataset}_{operation}.sh"
        if os.path.exists(file_path):
            all_paths.append(file_path)

with open('todelete.txt', "w") as f:
    print(*all_paths, file=f, sep='\n')