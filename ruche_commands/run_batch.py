import os


i = 0
for atomic in [
    # 'bisel',
    'dual_bisel',
    'sybisel',
]:
    for dataset in [
        'diskorect',
        'mnist',
        'inverted_mnist',
        'gray_mnist',
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
                i += 1

print(f"Launched {i} commands.")
