from .dataset import RolloutDataset
from .validators import VALIDATORS

CONSTRUCTOR_REGISTRY = {
    'dataset' : RolloutDataset,
    'validators' : VALIDATORS
}