"""Shared tolerance convention for training and verification.

Training, search, and enumeration operate on a dilated violation set
(violation if h(x) > -TRAIN_MARGIN or V(x') > -TRAIN_MARGIN, given
V(x) <= TRAIN_MARGIN), while the verifier certifies emptiness of the
exact set. TRAIN_MARGIN >= VERIFY_MARGIN guarantees every violation
the verifier can find is also a training counterexample.
"""

TRAIN_MARGIN = 1e-4
VERIFY_MARGIN = 0.0

assert TRAIN_MARGIN >= VERIFY_MARGIN >= 0.0
