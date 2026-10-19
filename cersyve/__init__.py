"""
Cersyve: Control, Environment, Reach Set, Synthesis, and Verification

A framework for learning safe control policies using neural networks with formal verification.
"""

from .utils import (
    uniform,
    collect_data,
    train_dynamics,
    train_constraint,
    forward_with_snr,
    forward_with_apa,
    get_activation_pattern,
    count_activation_pattern,
    load_mlp
)

from .model import (
    create_mlp,
    create_closed_loop_dynamics_model,
    create_linear_closed_loop_model,
    create_box_constraint_model,
    create_value_constraint_model,
    create_value_next_value_model
)

from .buffer import Buffer
from .pretrain import pretrain_value
from .pgd import (
    boundary_guided_search,
    projected_boundary_search,
    filter_counterexample
)
from .enumeration import (
    BinaryDecisionWrapperCBF,
    enumerate_unsafe_regions,
    sample_from_regions,
    get_pure_leaf_rectangles
)
from .finetune import finetune_value_w_MILP, finetune_value_enum_MILP, finetune_value
from .utils import estimate_tfr
from .tolerances import TRAIN_MARGIN, VERIFY_MARGIN


def __getattr__(name):
    # verify_value needs the alpha-beta-CROWN submodule; load it on demand.
    if name == "verify_value":
        from .abcrown import verify_value
        return verify_value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = "1.0.0"
__author__ = ""

__all__ = [
    # Utils
    'uniform',
    'collect_data',
    'train_dynamics',
    'train_constraint',
    'forward_with_snr',
    'forward_with_apa',
    'get_activation_pattern',
    'count_activation_pattern',
    'load_mlp',
    
    # Models
    'create_mlp',
    'create_closed_loop_dynamics_model',
    'create_linear_closed_loop_model',
    'create_box_constraint_model',
    'create_value_constraint_model',
    'create_value_next_value_model',
    
    # Training
    'Buffer',
    'pretrain_value',
    'finetune_value',
    'finetune_value_w_MILP',
    'finetune_value_enum_MILP',
    'estimate_tfr',
    'TRAIN_MARGIN',
    'VERIFY_MARGIN',
    
    # Search
    'boundary_guided_search',
    'projected_boundary_search',
    'filter_counterexample',
    
    # Enumeration
    'BinaryDecisionWrapperCBF',
    'enumerate_unsafe_regions',
    'sample_from_regions',
    'get_pure_leaf_rectangles',

    # Verification
    'verify_value',
]
