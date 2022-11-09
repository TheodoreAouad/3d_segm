import deep_morpho.initializer as inits
import deep_morpho.models.bise as bise


model = bise.BiSE(
    kernel_size=(7, 7),
    initializer=inits.InitBiseEllipseWeights(init_bias_value=1),
    weights_optim_mode=bise.BiseWeightsOptimEnum.ELLIPSE,
    bias_optim_mode=bise.BiseBiasOptimEnum.POSITIVE,
)
