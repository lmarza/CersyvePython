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
    count_activation_pattern
)

from .model import (
    create_mlp,
    create_closed_loop_dynamics_model,
    create_value_constraint_model,
    create_value_next_value_model
)

from .buffer import Buffer
from .pretrain import pretrain_value
from .search import (
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
from .finetune import finetune_value, finetune_value_enumeration
from .verify import verify_value

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
    
    # Models
    'create_mlp',
    'create_closed_loop_dynamics_model',
    'create_value_constraint_model',
    'create_value_next_value_model',
    
    # Training
    'Buffer',
    'pretrain_value',
    'finetune_value',
    'finetune_value_enumeration',
    
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
