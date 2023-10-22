from deep_morpho.datasets import DataModule


dataset = DataModule.select("diskorectdataset")(morp_operation=lambda x: x)
dataloader = dataset.get_loader(morp_operation=lambda x: x, batch_size=1)
# dataset = DataModule.select("spalikedataset")(image_size=(256, 256))
# dataloader = dataset.get_loader(image_size=(256, 256), batch_size=1)
next(iter(dataloader))