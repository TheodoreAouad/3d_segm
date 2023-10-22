from deep_morpho.datasets.spalike_generator import SpaLike


gen = SpaLike((256, 256), proba_lesion=.2, max_n_blob=2)

for _ in range(100):
    img, segm = gen.generate_image()
