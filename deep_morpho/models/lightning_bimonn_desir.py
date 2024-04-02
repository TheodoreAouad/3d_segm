from .generic_lightning_model import GenericLightningModel
from .bimonn_desir import ResnetDesirMerged


class LightningResnetDesirMerged(GenericLightningModel):
    model_class = ResnetDesirMerged

