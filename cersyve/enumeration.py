"""Enumeration-based counterexample search using Random Forest and hyperrectangles."""

import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import List, Tuple, Dict, Optional


class BinaryDecisionWrapperCBF(nn.Module):
    """
    Binary classification wrapper for Control Barrier Function verification.
    
    Returns:
        1 (safe): V(x) <= 0 AND h(x) <= 0 AND V(f(x)) <= 0
        -1 (unsafe): V(x) <= 0 AND (h(x) > 0 OR V(f(x)) > 0)
    
    This identifies false positives where the value function predicts safety
    but constraints are violated.
    """
    def __init__(
        self, 
        V_model: nn.Module, 
        h_model: nn.Module, 
        f_pi_model: nn.Module,
        tol: float = 0.0
    ):
        """
        Args:
            V_model: Value function model V(x)
            h_model: Constraint model h(x)
            f_pi_model: Closed-loop dynamics model f_pi(x)
            tol: Tolerance for safety checks
        """
        super().__init__()
        self.V_model = V_model
        self.h_model = h_model
        self.f_pi_model = f_pi_model
        self.tol = tol

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for binary classification.
        
        Args:
            x: State tensor [batch_size, x_dim]
            
        Returns:
            Binary classification: 1 (safe) or -1 (unsafe)
        """
        # Compute V(x), h(x), and V(f(x))
        with torch.no_grad():
            V_x = self.V_model(x).squeeze()
            h_x = self.h_model(x).squeeze()
            
            # Compute next state and its value
            x_next = self.f_pi_model(x.T).T
            V_next = self.V_model(x_next).squeeze()
        
        # Safe condition: V(x) <= tol AND h(x) <= tol AND V(f(x)) <= tol
        # We're looking for regions where V(x) <= tol but safety is violated
        
        # Check if V(x) indicates safety
        v_safe = V_x <= self.tol
        
        # Check actual safety: h(x) <= tol AND V(f(x)) <= tol
        h_safe = h_x <= self.tol
        v_next_safe = V_next <= self.tol
        actually_safe = h_safe & v_next_safe
        
        # Return 1 if actually safe when V says safe, -1 if unsafe when V says safe
        # For regions where V > tol, we don't care (mark as safe to exclude from enumeration)
        result = torch.where(
            v_safe & actually_safe,  # V says safe AND actually safe
            torch.tensor(1, dtype=torch.int32),
            torch.where(
                v_safe & ~actually_safe,  # V says safe BUT actually unsafe (FALSE POSITIVE)
                torch.tensor(-1, dtype=torch.int32),
                torch.tensor(1, dtype=torch.int32)  # V says unsafe (don't care)
            )
        )
        
        return result


def get_pure_leaf_rectangles(tree, input_domain: np.ndarray, extract_safe: bool = False) -> List[np.ndarray]:
    """
    Extract hyperrectangles corresponding to pure leaves from a decision tree.
    
    Args:
        tree: Trained decision tree classifier
        input_domain: np.array of shape (n_dims, 2), defining min/max range for each dimension
        extract_safe: If True, extract safe leaves (class 1); if False, extract unsafe leaves (class -1)
        
    Returns:
        List of hyperrectangles defining decision regions for pure leaves of the specified class
    """
    tree_ = tree.tree_
    n_dims = input_domain.shape[0]
    
    # Initialize bounds for root node
    node_bounds = {0: [list(input_domain[i]) for i in range(n_dims)]}
    
    pure_leaf_rects = []
    
    def traverse(node, bounds):
        """Recursively traverse tree and track feature bounds."""
        # Check if leaf node
        if tree_.children_left[node] == -1 and tree_.children_right[node] == -1:
            value = tree_.value[node][0]
            # Check if all samples in this leaf are of the desired class
            if extract_safe:
                # Class index 1 for safe (class 1)
                if value[1] == np.sum(value):  # All samples are class 1 (safe)
                    pure_leaf_rects.append(bounds)
            else:
                # Class index 0 for unsafe (class -1)
                if value[0] == np.sum(value):  # All samples are class -1 (unsafe)
                    pure_leaf_rects.append(bounds)
            return
        
        feature = tree_.feature[node]
        threshold = tree_.threshold[node]
        
        # Copy bounds for left and right children
        left_bounds = [b[:] for b in bounds]
        right_bounds = [b[:] for b in bounds]
        
        # Update bounds based on split
        left_bounds[feature][1] = threshold  # Shrink upper bound
        right_bounds[feature][0] = threshold  # Shrink lower bound
        
        traverse(tree_.children_left[node], left_bounds)
        traverse(tree_.children_right[node], right_bounds)
    
    traverse(0, node_bounds[0])
    return pure_leaf_rects


def exact_remove_contained_regions(
    current_regions: List[np.ndarray], 
    filtered_regions: List[np.ndarray]
) -> List[np.ndarray]:
    """
    Remove regions that are completely contained in other regions.
    
    Args:
        current_regions: New regions to check
        filtered_regions: Existing filtered regions
        
    Returns:
        Combined list with contained regions removed
    """
    if len(filtered_regions) == 0:
        return current_regions.copy()
    
    result = filtered_regions.copy()
    
    for region in current_regions:
        is_contained = False
        for filtered_region in filtered_regions:
            if (np.all(region[:, 0] >= filtered_region[:, 0]) and 
                np.all(region[:, 1] <= filtered_region[:, 1])):
                is_contained = True
                break
        if not is_contained:
            result.append(region)
    
    return result


def is_inside(points: np.ndarray, hyperrectangles: List[np.ndarray]) -> np.ndarray:
    """
    Check if points are inside any of the given hyperrectangles.
    
    Args:
        points: Array of shape (M, n) with M points in n-dimensional space
        hyperrectangles: List of hyperrectangles, each of shape (n, 2)
        
    Returns:
        Boolean array of shape (M,) indicating if each point is inside any rectangle
    """
    if len(hyperrectangles) == 0:
        return np.zeros(len(points), dtype=bool)
    
    points = np.expand_dims(points, axis=1)  # Shape (M, 1, n)
    hyperrectangles = np.array(hyperrectangles)  # Shape (R, n, 2)
    
    mins = hyperrectangles[:, :, 0]  # Shape (R, n)
    maxs = hyperrectangles[:, :, 1]  # Shape (R, n)
    
    inside = np.all((points >= mins) & (points <= maxs), axis=2)  # Shape (M, R)
    return np.any(inside, axis=1)


def enumerate_unsafe_regions(
    wrapper_model: BinaryDecisionWrapperCBF,
    input_domain: np.ndarray,
    n_training_samples: int = 10000,
    n_decision_trees: int = 5,
    max_depth: int = 8,
    heuristic: str = 'gini',
    digits_to_consider: int = 0,
    wilks_filtering: bool = True,
    wilks_samples: int = 3500,
    sample_size_evaluation: int = 10000,
    desired_coverage: float = 99.9,
    enumerate_safe_regions: bool = False,
    verify_safe_regions_with_pgd: bool = False,
    pgd_samples: int = 1000,
    pgd_steps: int = 10,
    pgd_eps: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
    device: str = 'cpu'
) -> Tuple[List[np.ndarray], Dict[str, float]]:
    """
    Enumerate regions where V(x) <= 0 - either safe or unsafe regions.
    
    Uses Random Forest to approximate the decision boundary and extracts
    hyperrectangles corresponding to the specified region type.
    
    Args:
        wrapper_model: Binary classification wrapper for CBF verification
        input_domain: Input domain bounds, shape (n_dims, 2)
        n_training_samples: Number of samples to train RF
        n_decision_trees: Number of trees in RF
        max_depth: Maximum depth of each tree
        heuristic: Split criterion for RF ('gini' or 'entropy')
        digits_to_consider: Decimal places to round samples
        wilks_filtering: Whether to use Wilks filtering for purity check
        wilks_samples: Number of samples for Wilks test
        sample_size_evaluation: Number of samples for coverage evaluation
        desired_coverage: Target coverage percentage of regions
        enumerate_safe_regions: If True, enumerate safe regions; if False, enumerate unsafe regions
        verify_safe_regions_with_pgd: If True and enumerate_safe_regions=True and error=0%, verify with PGD
        pgd_samples: Number of samples for PGD verification
        pgd_steps: Number of PGD steps
        pgd_eps: Step size for PGD
        seed: Random seed
        verbose: Print progress information
        device: Device to run on
        
    Returns:
        regions: List of hyperrectangles representing the specified region type
        stats: Dictionary with statistics (coverage, num_regions, etc.)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    wrapper_model = wrapper_model.to(device)
    wrapper_model.eval()
    
    region_type = "safe" if enumerate_safe_regions else "unsafe"
    
    if verbose:
        print(f"Training random forest to identify {region_type} regions...")
    
    # Sample inputs for training RF
    inputs = np.random.uniform(
        input_domain[:, 0], 
        input_domain[:, 1], 
        size=(n_training_samples, input_domain.shape[0])
    )
    
    if digits_to_consider > 0:
        inputs = np.round(inputs, digits_to_consider)
    
    # Get binary labels from wrapper
    input_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = wrapper_model(input_tensor).cpu().numpy()
    
    # Count samples by type
    n_unsafe = np.sum(outputs == -1)
    n_safe = np.sum(outputs == 1)
    
    if verbose:
        print(f"Training samples: {n_training_samples}")
        print(f"  Unsafe (false positives): {n_unsafe} ({100*n_unsafe/n_training_samples:.2f}%)")
        print(f"  Safe: {n_safe} ({100*n_safe/n_training_samples:.2f}%)")
    
    # Check if we have samples of the requested type
    target_count = n_safe if enumerate_safe_regions else n_unsafe
    if target_count == 0:
        if verbose:
            print(f"No {region_type} regions found!")
        return [], {'coverage': 0.0, 'num_regions': 0, 'error': 0.0}
    
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=n_decision_trees,
        criterion=heuristic,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1
    )
    rf.fit(inputs, outputs)
    
    if verbose:
        print(f"Trained RF with {n_decision_trees} trees, extracting {region_type} regions...")
    
    # Get evaluation samplesF
    eval_inputs = np.random.uniform(
        input_domain[:, 0],
        input_domain[:, 1],
        size=(sample_size_evaluation, input_domain.shape[0])
    )

    if digits_to_consider > 0:
        eval_inputs = np.round(eval_inputs, digits_to_consider)
    
    eval_tensor = torch.tensor(eval_inputs, dtype=torch.float32).to(device)
    with torch.no_grad():
        eval_outputs = wrapper_model(eval_tensor).cpu().numpy()

     # check accuracy of RF on evaluation samples
    print(f"Evaluating RF accuracy on {sample_size_evaluation} samples...")
    eval_rf_outputs = rf.predict(eval_inputs)
    accuracy = np.sum(eval_rf_outputs == eval_outputs) / len(eval_outputs)
    print(f"RF accuracy: {accuracy:.2f}")
    
    n_eval_unsafe = np.sum(eval_outputs == -1)
    n_eval_safe = np.sum(eval_outputs == 1)
    n_eval_target = n_eval_safe if enumerate_safe_regions else n_eval_unsafe
    
    # Extract pure leaves from each tree
    filtered_regions = []
    current_regions = []
    
    for i, tree in enumerate(rf.estimators_):
        rects = get_pure_leaf_rectangles(tree, input_domain, extract_safe=enumerate_safe_regions)
        
        for rect in rects:
            rect_array = np.array(rect)
            
            if digits_to_consider > 0:
                rect_array = np.round(rect_array, digits_to_consider)
            
            # Wilks filtering: verify the region is actually unsafe
            if wilks_filtering:
                random_sample = np.random.uniform(
                    rect_array[:, 0],
                    rect_array[:, 1],
                    size=(wilks_samples, rect_array.shape[0])
                )
                
                if digits_to_consider > 0:
                    random_sample = np.round(random_sample, digits_to_consider)
                
                sample_tensor = torch.tensor(random_sample, dtype=torch.float32).to(device)
                with torch.no_grad():
                    sample_outputs = wrapper_model(sample_tensor).cpu().numpy()
                
                # Only add if all samples are of the target type
                if enumerate_safe_regions:
                    if np.sum(sample_outputs == 1) == wilks_samples:
                        current_regions.append(rect_array)
                else:
                    if np.sum(sample_outputs == -1) == wilks_samples:
                        current_regions.append(rect_array)
            else:
                current_regions.append(rect_array)
        
        # Remove contained regions
        filtered_regions = exact_remove_contained_regions(
            current_regions, 
            filtered_regions
        )
        current_regions = []
        
        # Check coverage
        if len(filtered_regions) > 0:
            target_samples = eval_inputs[eval_outputs == 1] if enumerate_safe_regions else eval_inputs[eval_outputs == -1]
            inside_mask = is_inside(target_samples, filtered_regions)
            coverage = 100 * np.sum(inside_mask) / n_eval_target if n_eval_target > 0 else 0
            
                
            # Compute error: misclassified samples in the enumerated regions
            # For safe regions: % of unsafe samples that fall into safe regions (shouldn't be there)
            # For unsafe regions: % of safe samples that fall into unsafe regions (shouldn't be there)
            if enumerate_safe_regions:
                # Error: unsafe samples in safe regions
                wrong_samples = eval_inputs[eval_outputs == -1]
                if len(wrong_samples) > 0:
                    wrong_inside = is_inside(wrong_samples, filtered_regions)
                    error_rate = 100 * np.sum(wrong_inside) / len(wrong_samples)
                else:
                    error_rate = 0.0
            else:
                # Error: safe samples in unsafe regions
                wrong_samples = eval_inputs[eval_outputs == 1]
                if len(wrong_samples) > 0:
                    wrong_inside = is_inside(wrong_samples, filtered_regions)
                    error_rate = 100 * np.sum(wrong_inside) / len(wrong_samples)
                else:
                    error_rate = 0.0
                
            if verbose and i % 10 == 0:
                print(f"\tTree {i+1}/{n_decision_trees}: "
                      f"{len(filtered_regions)} regions, "
                      f"coverage: {coverage:.2f}%, "
                      f"error rate: {error_rate:.2f}%")
                
            final_coverage = coverage
            if final_coverage >= desired_coverage:
                if verbose:
                    print(f"Reached desired coverage of {desired_coverage}%")
                break

        else:
            final_coverage = 0.0
            error_rate = 0.0
    
        
   
    # PGD verification for safe regions (only if error is 0%)
    pgd_violations_found = 0
    if (enumerate_safe_regions and 
        verify_safe_regions_with_pgd and 
        len(filtered_regions) > 0 and 
        error_rate == 0.0):
        
        if verbose:
            print(f"\n  Error rate is 0%, performing PGD verification on safe regions...")
        
        # Sample points from safe regions
        verification_samples = sample_from_regions(
            filtered_regions, 
            n_samples=pgd_samples,
            digits_to_consider=digits_to_consider
        )
        
        if verification_samples.shape[0] > 0:
            # Import PGD search here to avoid circular import
            from .search import boundary_guided_search
            
            # Extract models from wrapper
            V_model = wrapper_model.V_model
            h_model = wrapper_model.h_model
            f_pi_model = wrapper_model.f_pi_model
            
            # Prepare samples as [dim, N]
            x_samples = verification_samples.T.astype(np.float32)
            x_low = input_domain[:, 0]
            x_high = input_domain[:, 1]
            
            # Run PGD attack
            x_pgd = boundary_guided_search(
                x_samples, x_low, x_high, h_model, V_model, f_pi_model,
                pgd_step=pgd_steps, pgd_eps=pgd_eps, 
                backtrack_step=20, length_discount=0.8,
                bound_guide=True, direct_discount=0.5,
                tol=wrapper_model.tol, device=device
            )
            
            # Check if PGD found violations
            x_pgd_t = torch.from_numpy(x_pgd.astype(np.float32).T).to(device)
            with torch.no_grad():
                v_pgd = V_model(x_pgd_t)
                h_pgd = h_model(x_pgd_t)
                x_next_pgd = f_pi_model(x_pgd_t.T).T
                v_next_pgd = V_model(x_next_pgd)
            
            # Count violations: V <= tol but (h > tol OR V_next > tol)
            v_safe = (v_pgd.squeeze() <= wrapper_model.tol).cpu().numpy()
            h_violated = (h_pgd.squeeze() > wrapper_model.tol).cpu().numpy()
            v_next_violated = (v_next_pgd.squeeze() > wrapper_model.tol).cpu().numpy()
            violations = v_safe & (h_violated | v_next_violated)
            pgd_violations_found = np.sum(violations)
            
            if verbose:
                print(f"  PGD verification: {pgd_violations_found}/{pgd_samples} violations found")
                if pgd_violations_found > 0:
                    print(f"  ⚠ Safe regions may contain {pgd_violations_found} counterexamples!")
    
    stats = {
        'coverage': final_coverage,
        'num_regions': len(filtered_regions),
        'n_unsafe_samples': n_eval_unsafe,
        'n_safe_samples': n_eval_safe,
        'n_eval_samples': sample_size_evaluation,
        'error_rate': error_rate,
        'pgd_violations': pgd_violations_found
    }
    
    if verbose:
        print(f"\nEnumeration complete:")
        print(f"  Found {len(filtered_regions)} {region_type} regions")
        print(f"  Coverage of {region_type} regions: {final_coverage:.2f}%")
        print(f"  Error rate: {error_rate:.2f}%")
        if pgd_violations_found > 0:
            print(f"  PGD violations: {pgd_violations_found}")
    
    return filtered_regions, stats


def sample_from_regions(
    regions: List[np.ndarray],
    n_samples: int,
    digits_to_consider: int = 3
) -> np.ndarray:
    """
    Sample points uniformly from a list of hyperrectangles.
    
    Args:
        regions: List of hyperrectangles, each of shape (n, 2)
        n_samples: Total number of samples to generate
        digits_to_consider: Decimal places to round samples
        
    Returns:
        Array of sampled points, shape (n_samples, n_dims)
    """
    if len(regions) == 0:
        return np.array([])
    
    # Compute volume of each region
    volumes = np.array([np.prod(r[:, 1] - r[:, 0]) for r in regions])
    
    # Sample proportionally to volume
    probabilities = volumes / np.sum(volumes)
    region_indices = np.random.choice(len(regions), size=n_samples, p=probabilities)
    
    samples = []
    for idx in region_indices:
        region = regions[idx]
        sample = np.random.uniform(region[:, 0], region[:, 1])
        samples.append(sample)
    
    samples = np.array(samples)
    
    if digits_to_consider > 0:
        samples = np.round(samples, digits_to_consider)
    
    return samples
