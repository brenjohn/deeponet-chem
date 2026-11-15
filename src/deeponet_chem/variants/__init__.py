from . import adaptive_step
from . import fixed_step

VARIANT_FACTORIES = {
    'fixed_step'    : fixed_step.FixedStepFactory,
    'adaptive_step' : adaptive_step.OneStepFactory,
}