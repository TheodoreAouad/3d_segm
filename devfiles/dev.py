rsync -avz -e ssh aouadt-cvn:/hdd/aouadt/these/projets/3d_segm/deep_morpho/results/exp80_mnist_results_df/ ./deep_morpho/results/exp80_mnist_results_df/ --include "*.csv" --exclude "*"