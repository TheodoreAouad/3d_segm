import os
import time


i = 0
for atomic in [
    # 'bisel',
    'dual_bisel',
    # 'sybisel',
]:
    for dataset in [
        'diskorect',
        'mnist',
        'inverted_mnist',
        'gray_mnist',
        'fashionmnist'
    ]:
        for operation in [
            'erodila',
            'ope',
            'clo',
            'black_tophat',
            'white_tophat'
        ]:
            file_path = f'ruche_commands/{atomic}/{dataset}_{operation}.sh'
            if os.path.exists(file_path):
                os.system(f"sbatch {file_path}")
                print(f"sbatch {file_path}")
                time.sleep(.5)
                i += 1

print(f"Launched {i} commands.")
