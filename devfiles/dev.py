from deep_morpho.models import GenericLightningModel
from deep_morpho.models import BinaryNN

GenericLightningModel.select("BimonnBiselDenseNotBinary").default_args()
BinaryNN.select("BimonnBiselDenseNotBinary").default_args()