import os
import time



all_paths = []

# for atomic in [
#     'bisel',
#     'dual_bisel',
#     # 'sybisel',
# ]:
#     for dataset in [
#         # 'diskorect',
#         # 'mnist',
#         # 'inverted_mnist',
#         # 'gray_mnist',
#         # 'fashionmnist',
#         "classif_mnist",
#         "classif_mnist_channel",
#         "cifar10",
#     ]:
#         for operation in [
#             # 'erodila',
#             # 'ope',
#             # 'clo',
#             # 'black_tophat',
#             # 'white_tophat'
#             "maxpool",
#             "linear",
#         ]:
#             file_path = f'ruche_commands/{atomic}/{dataset}_{operation}.sh'
#             if os.path.exists(file_path):
#                 all_paths.append(file_path)

for dataset in ["mnist", "inverted_mnist", "diskorect"]:
    for operation in ["dilation", "erosion", "opening", "closing", "white_tophat", "black_tophat"]:
        file_path = f"ruche_commands/recompute_projected/{dataset}_{operation}.sh"
        if os.path.exists(file_path):
            all_paths.append(file_path)


# for atomic in ['dual_bisel']:
#     for dataset in ['inverted_mnist']:
#         for operation in [
#             'erodila',
#             'ope',
#             'clo',
#             # 'black_tophat',
#             'white_tophat'
#         ]:
#             file_path = f'ruche_commands/{atomic}/{dataset}_{operation}.sh'
#             if os.path.exists(file_path):
#                 all_paths.append(file_path)

# for atomic in ['dual_bisel']:
#     for dataset in ['mnist']:
#         for operation in [
#             'erodila',
#             'ope',
#             'clo',
#         ]:
#             file_path = f'ruche_commands/{atomic}/{dataset}_{operation}.sh'
#             if os.path.exists(file_path):
#                 all_paths.append(file_path)

# all_paths.append('ruche_commands/bisel/inverted_mnist_black_tophat.sh')
# all_paths.append('ruche_commands/bisel/fashionmnist_white_tophat.sh')
# all_paths.append('ruche_commands/bisel/mnist_white_tophat.sh')
# all_paths.append('ruche_commands/dual_bisel/fashionmnist_white_tophat.sh')



i = 0
for path in all_paths:
    os.system(f"sbatch {path}")
    print(f"sbatch {path}")
    time.sleep(.5)
    i += 1


print(f"Launched {i} commands.")
