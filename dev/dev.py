from general.utils import load_json, save_json

mtd = load_json('data/deep_morpho/dataset_0/metadata.json')

for key in mtd['seqs'].keys():
    mtd['seqs'][key]['path_target'] += '/images'

save_json(mtd, 'data/deep_morpho/dataset_0/metadata.json')