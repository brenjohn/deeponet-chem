from .dataset import OneStepTracerDataset
from .validators import VALIDATORS

CONSTRUCTOR_REGISTRY = {
    'dataset' : OneStepTracerDataset,
    'validators' : VALIDATORS
}