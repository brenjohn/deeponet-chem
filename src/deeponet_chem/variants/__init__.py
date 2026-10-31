from . import adaptive_step
from . import fixed_step

VARIANT_REGISTRY = {
    'fixed_step'    : fixed_step.CONSTRUCTOR_REGISTRY,
    'adaptive_step' : adaptive_step.CONSTRUCTOR_REGISTRY,
}