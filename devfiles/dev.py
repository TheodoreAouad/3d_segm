from importlib import import_module
# from deep_morpho.results.results_tensorboards.Bimonn_exp_77.bimonn_last_linear.7.code.deep_morpho.models import GenericLightningModel
models_old = import_module("deep_morpho.results.results_tensorboards.Bimonn_exp_77.bimonn_last_linear.7.code.deep_morpho.models.lightning_bimonn")
LightningBiMoNNClassifierLastLinearNotBinary = models_old.LightningBiMoNNClassifierLastLinearNotBinary

weights_path = "deep_morpho/results/results_tensorboards/Bimonn_exp_77/bimonn_last_linear/7/mnistclassifchannel/bimonnclassifierlastlinearnotbinary/version_0/best_weights/epoch=26-step=21113.ckpt"

# torch.load(weights_path, map_location="cpu")["state_dict"]["model.classification_layer.bises.weight"]
LightningBiMoNNClassifierLastLinearNotBinary.load_from_checkpoint(weights_path)
