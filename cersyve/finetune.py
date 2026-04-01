"""Fine-tune value function using counterexample-guided search."""

from .milp import milp_verify
from .enumeration import (
    BinaryDecisionWrapperCBF,
    enumerate_unsafe_regions,
    sample_from_regions
)

from collections import deque, defaultdict
from .buffer import Buffer
from .search import boundary_guided_search, projected_boundary_search, filter_counterexample
from .utils import uniform, set_seed
from datetime import datetime
from typing import Optional
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class ScalarTracker:
    def __init__(self, window_size: int = 200, ema_alpha: float = 0.03):
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.windows = defaultdict(lambda: deque(maxlen=window_size))
        self.ema_values = {}

    def update(self, name: str, value: float):
        v = float(value)
        self.windows[name].append(v)

        if name not in self.ema_values:
            self.ema_values[name] = v
        else:
            self.ema_values[name] = (
                self.ema_alpha * v + (1.0 - self.ema_alpha) * self.ema_values[name]
            )

    def mean(self, name: str):
        w = self.windows[name]
        if len(w) == 0:
            return None
        return sum(w) / len(w)

    def std(self, name: str):
        w = self.windows[name]
        if len(w) == 0:
            return None
        m = sum(w) / len(w)
        return (sum((x - m) ** 2 for x in w) / len(w)) ** 0.5

    def ema(self, name: str):
        return self.ema_values.get(name, None)

    def values(self, name: str):
        return list(self.windows[name])


def finetune_value(
    V_model: nn.Module,
    f_pi_model: nn.Module,
    h_model: nn.Module,
    x_low: np.ndarray,
    x_high: np.ndarray,
    lr: float = 1e-4,
    max_iter: int = 100000,
    search_size: int = 1000,
    bnd_ratio: float = 0.1,
    bnd_ratio_avg: float = 0.9,
    min_bnd_ratio: float = 0.01,
    max_bnd_ratio: float = 1.0,
    bnd_eps: float = 0.1,
    search_method: str = "BGB",
    pgd_step: int = 10,
    pgd_eps: float = 0.1,
    backtrack_step: int = 20,
    length_discount: float = 0.8,
    direct_discount: float = 0.5,
    tol: float = 1e-4,
    capacity: int = 10000,
    sample_size: int = 100,
    search_stop: int = 1000,
    replay: int = 1,
    max_skip: int = 1000,
    reg_method: Optional[str] = "ESR",
    esr_max_con: Optional[int] = None,
    eps_h: float = 0.01,
    eps_v: float = 0.01,
    reg_coef: float = 0.1,
    log_dir: Optional[str] = None,
    eval_every: int = 10,
    save_every: int = 1000,
    save_path: Optional[str] = None,
    device: str = "cpu",
    seed: int = 42,
) -> str:
    """
    Fine-tune value function with counterexample-guided search only (no MILP).
    """

    set_seed(seed)

    V_model = V_model.to(device)
    f_pi_model = f_pi_model.to(device)
    h_model = h_model.to(device)

    skipped = 0
    buffer = Buffer(capacity, len(x_low))
    optimizer = torch.optim.Adam(V_model.parameters(), lr=lr)

    if log_dir is None:
        log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(
        log_dir,
        f"finetune_basic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    writer = SummaryWriter(log_path)

    tracker = ScalarTracker(window_size=1000, ema_alpha=0.005)
    histogram_log_freq = 1000
    log_every = 50

    for i in tqdm(range(max_iter), desc="Finetuning value function"):
        searched_boundary_states = 0
        searched_constraint_count = 0
        searched_invariance_count = 0

        # ------------------------------------------------------------
        # 1. Search for counterexamples
        # ------------------------------------------------------------
        if len(buffer) < search_stop:
            n_uniform = max(round(search_size / max(bnd_ratio, 1e-6)), search_size)
            x = uniform(x_low, x_high, n_uniform).astype(np.float32)
            x_t = torch.from_numpy(x).to(device)

            with torch.no_grad():
                v = V_model(x_t.T).T.cpu().numpy()[0, :]

            x_bnd = x[:, np.abs(v) <= bnd_eps]

            if x.shape[1] > 0:
                bnd_ratio = (
                    bnd_ratio_avg * bnd_ratio
                    + (1 - bnd_ratio_avg) * (x_bnd.shape[1] / x.shape[1])
                )
            bnd_ratio = np.clip(bnd_ratio, min_bnd_ratio, max_bnd_ratio)

            searched_boundary_states = x_bnd.shape[1]

            if x_bnd.shape[1] > 0:
                if search_method == "BGB":
                    x_pgd = boundary_guided_search(
                        x_bnd,
                        x_low,
                        x_high,
                        h_model,
                        V_model,
                        f_pi_model,
                        pgd_step=pgd_step,
                        pgd_eps=pgd_eps,
                        backtrack_step=backtrack_step,
                        length_discount=length_discount,
                        bound_guide=True,
                        direct_discount=direct_discount,
                        tol=tol,
                        device=device,
                    )
                elif search_method == "PGD-B":
                    x_pgd = boundary_guided_search(
                        x_bnd,
                        x_low,
                        x_high,
                        h_model,
                        V_model,
                        f_pi_model,
                        pgd_step=pgd_step,
                        pgd_eps=pgd_eps,
                        backtrack_step=backtrack_step,
                        length_discount=length_discount,
                        bound_guide=False,
                        tol=tol,
                        device=device,
                    )
                elif search_method == "PBS":
                    x_pgd = projected_boundary_search(
                        x_bnd,
                        x_low,
                        x_high,
                        h_model,
                        V_model,
                        f_pi_model,
                        tol=tol,
                        device=device,
                    )
                else:
                    raise ValueError(f"Unknown search_method: {search_method}")

                con, inv = filter_counterexample(
                    x_pgd,
                    h_model,
                    V_model,
                    f_pi_model,
                    tol=tol,
                    device=device,
                )

                searched_constraint_count = int(con.sum())
                searched_invariance_count = int(inv.sum())

                ce = con | inv
                if np.any(ce):
                    n_push = min(int(ce.sum()), len(buffer.vacant))
                    if n_push > 0:
                        buffer.push(x_pgd[:, ce][:, :n_push])
            else:
                con = np.zeros(0, dtype=bool)
                inv = np.zeros(0, dtype=bool)

            writer.add_scalar("finetune/searched_boundary_states", searched_boundary_states, i)
            writer.add_scalar("finetune/boundary_state_ratio", bnd_ratio, i)

            writer.add_scalar("finetune/searched_constraint_counterexample", searched_constraint_count, i)
            writer.add_scalar("finetune/searched_invariance_counterexample", searched_invariance_count, i)

            searched_den = max(searched_boundary_states, 1)
            searched_constraint_ratio = searched_constraint_count / searched_den
            searched_invariance_ratio = searched_invariance_count / searched_den

            tracker.update("searched_constraint_ratio", searched_constraint_ratio)
            tracker.update("searched_invariance_ratio", searched_invariance_ratio)

            writer.add_scalar("finetune/searched_constraint_ratio", searched_constraint_ratio, i)
            writer.add_scalar("finetune/searched_invariance_ratio", searched_invariance_ratio, i)

            searched_constraint_ratio_ema = tracker.ema("searched_constraint_ratio")
            if searched_constraint_ratio_ema is not None:
                writer.add_scalar("finetune/searched_constraint_ratio_ema", searched_constraint_ratio_ema, i)

            searched_invariance_ratio_ema = tracker.ema("searched_invariance_ratio")
            if searched_invariance_ratio_ema is not None:
                writer.add_scalar("finetune/searched_invariance_ratio_ema", searched_invariance_ratio_ema, i)

            searched_constraint_ratio_mean = tracker.mean("searched_constraint_ratio")
            if searched_constraint_ratio_mean is not None:
                writer.add_scalar("finetune/searched_constraint_ratio_window_mean", searched_constraint_ratio_mean, i)

            searched_invariance_ratio_mean = tracker.mean("searched_invariance_ratio")
            if searched_invariance_ratio_mean is not None:
                writer.add_scalar("finetune/searched_invariance_ratio_window_mean", searched_invariance_ratio_mean, i)

        # ------------------------------------------------------------
        # 2. Train on counterexamples from buffer
        # ------------------------------------------------------------
        if len(buffer) > 0:
            skipped = 0

            n = min(sample_size, len(buffer))
            x, c = buffer.pop(n)

            con, inv = filter_counterexample(
                x,
                h_model,
                V_model,
                f_pi_model,
                tol=tol,
                device=device,
            )
            x_con, x_inv = x[:, con], x[:, inv]

            c[con | inv] = 0
            c[~con & ~inv] += 1
            push_idx = c < replay
            if np.any(push_idx):
                n_back = min(int(push_idx.sum()), len(buffer.vacant))
                if n_back > 0:
                    buffer.push(x[:, push_idx][:, :n_back], c[push_idx][:n_back])

            n_con = x_con.shape[1]
            n_inv = x_inv.shape[1]

            # --------------------------------------------------------
            # Regularization
            # --------------------------------------------------------
            n_reg = 0
            x_reg = None

            if reg_method is not None and (n_con + n_inv > 0) and \
               (esr_max_con is None or (n_con + n_inv < esr_max_con)):
                if reg_method == "ESR":
                    x_reg_cand = uniform(x_low, x_high, search_size).astype(np.float32)
                    x_reg_t = torch.from_numpy(x_reg_cand).to(device)

                    with torch.no_grad():
                        h_reg = h_model(x_reg_t.T).T.cpu().numpy()[0, :]
                        v_reg = V_model(x_reg_t.T).T.cpu().numpy()[0, :]
                        x_reg_prime = f_pi_model(x_reg_t)
                        v_reg_prime = V_model(x_reg_prime.T).T.cpu().numpy()[0, :]

                    entering = (
                        (h_reg <= 0.0)
                        & (v_reg > 0)
                        & (v_reg <= eps_v)
                        & (v_reg_prime <= 0.0)
                    )
                    x_reg = x_reg_cand[:, entering]
                    n_reg = x_reg.shape[1]

                elif reg_method == "RSR":
                    x_reg = uniform(x_low, x_high, search_size).astype(np.float32)
                    n_reg = x_reg.shape[1]

                else:
                    raise ValueError(f"Unknown reg_method: {reg_method}")

            # --------------------------------------------------------
            # Gradient step
            # --------------------------------------------------------
            optimizer.zero_grad()
            loss = 0.0

            if n_con > 0:
                x_con_t = torch.from_numpy(x_con).to(device)
                con_loss = -V_model(x_con_t.T).T.sum()
                loss += con_loss

            if n_inv > 0:
                x_inv_t = torch.from_numpy(x_inv).to(device)
                v_inv = V_model(x_inv_t.T).T
                x_inv_prime = f_pi_model(x_inv_t)
                v_inv_prime = V_model(x_inv_prime.T).T
                inv_loss = (-v_inv + v_inv_prime).sum()
                loss += inv_loss

            if (n_con + n_inv) > 0:
                loss = loss / (n_con + n_inv)

            if n_reg > 0:
                x_reg_t = torch.from_numpy(x_reg).to(device)
                reg_loss = V_model(x_reg_t.T).T.mean()
                loss = loss + reg_coef * reg_loss

            if isinstance(loss, torch.Tensor):
                loss.backward()
                optimizer.step()

            writer.add_scalar("finetune/sample_size", n, i)

            # ---------------------------
            # Value loss: raw + smooth
            # ---------------------------
            value_loss_scalar = float(loss.item()) if isinstance(loss, torch.Tensor) else 0.0
            tracker.update("value_loss", value_loss_scalar)

            if i % log_every == 0:
                writer.add_scalar("finetune/value_loss", value_loss_scalar, i)

                value_loss_ema = tracker.ema("value_loss")
                if value_loss_ema is not None:
                    writer.add_scalar("finetune/value_loss_ema", value_loss_ema, i)

                value_loss_mean = tracker.mean("value_loss")
                if value_loss_mean is not None:
                    writer.add_scalar("finetune/value_loss_window_mean", value_loss_mean, i)

                value_loss_std = tracker.std("value_loss")
                if value_loss_std is not None:
                    writer.add_scalar("finetune/value_loss_window_std", value_loss_std, i)

            # ---------------------------
            # Sampled counterexamples
            # ---------------------------
            writer.add_scalar("finetune/sampled_constraint_counterexample", n_con, i)
            writer.add_scalar("finetune/sampled_invariance_counterexample", n_inv, i)

            sampled_constraint_ratio = n_con / max(n, 1)
            sampled_invariance_ratio = n_inv / max(n, 1)

            tracker.update("sampled_constraint_ratio", sampled_constraint_ratio)
            tracker.update("sampled_invariance_ratio", sampled_invariance_ratio)

            writer.add_scalar("finetune/sampled_constraint_ratio", sampled_constraint_ratio, i)
            writer.add_scalar("finetune/sampled_invariance_ratio", sampled_invariance_ratio, i)

            sampled_constraint_ratio_ema = tracker.ema("sampled_constraint_ratio")
            if sampled_constraint_ratio_ema is not None:
                writer.add_scalar("finetune/sampled_constraint_ratio_ema", sampled_constraint_ratio_ema, i)

            sampled_invariance_ratio_ema = tracker.ema("sampled_invariance_ratio")
            if sampled_invariance_ratio_ema is not None:
                writer.add_scalar("finetune/sampled_invariance_ratio_ema", sampled_invariance_ratio_ema, i)

            sampled_constraint_ratio_mean = tracker.mean("sampled_constraint_ratio")
            if sampled_constraint_ratio_mean is not None:
                writer.add_scalar("finetune/sampled_constraint_ratio_window_mean", sampled_constraint_ratio_mean, i)

            sampled_invariance_ratio_mean = tracker.mean("sampled_invariance_ratio")
            if sampled_invariance_ratio_mean is not None:
                writer.add_scalar("finetune/sampled_invariance_ratio_window_mean", sampled_invariance_ratio_mean, i)

            # ---------------------------
            # Regularization logging
            # ---------------------------
            if reg_method is not None:
                reg_state_scalar = float(n_reg)
                tracker.update("regularization_state", reg_state_scalar)

                writer.add_scalar("finetune/regularization_state", reg_state_scalar, i)
                writer.add_scalar("finetune/regularization_state_raw", reg_state_scalar, i)

                reg_mean = tracker.mean("regularization_state")
                if reg_mean is not None:
                    writer.add_scalar("finetune/regularization_state_window_mean", reg_mean, i)

                reg_std = tracker.std("regularization_state")
                if reg_std is not None:
                    writer.add_scalar("finetune/regularization_state_window_std", reg_std, i)

                if i % histogram_log_freq == 0:
                    reg_values = tracker.values("regularization_state")
                    if len(reg_values) > 0:
                        writer.add_histogram(
                            "finetune/regularization_state_hist",
                            np.array(reg_values),
                            i
                        )

        else:
            skipped += 1

        # ------------------------------------------------------------
        # 3. Global logging
        # ------------------------------------------------------------
        writer.add_scalar("finetune/total_counterexample", len(buffer), i)
        writer.add_scalar("finetune/skipped_update", skipped, i)

        # ------------------------------------------------------------
        # 4. Evaluation
        # ------------------------------------------------------------
        if i % eval_every == 0:
            x_eval = uniform(x_low, x_high, search_size).astype(np.float32)
            x_eval_t = torch.from_numpy(x_eval).to(device)

            with torch.no_grad():
                v_eval = V_model(x_eval_t.T).T
                fea_rate = (v_eval <= 0).float().mean().item()
                safe_rate = (v_eval <= 0).float().mean().item()

            writer.add_scalar("finetune/predicted_feasible_rate", fea_rate, i)
            writer.add_scalar("finetune/safe_set_size", safe_rate, i)

        # ------------------------------------------------------------
        # 5. Checkpoint
        # ------------------------------------------------------------
        if (i + 1) % save_every == 0:
            ckpt_path = os.path.join(log_path, f"V_finetune_basic_iter{i+1}.pth")
            torch.save(V_model.state_dict(), ckpt_path)

        # ------------------------------------------------------------
        # 6. Early stop
        # ------------------------------------------------------------
        if skipped >= max_skip:
            print(f"\nNo counterexamples found for {max_skip} iterations. Stopping early.")
            break

    # ------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------
    if save_path is None:
        save_path = os.path.join(log_path, "V_finetune_basic.pth")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(V_model.state_dict(), save_path)

    writer.close()

    print(f"\nTraining complete. Model saved to {save_path}")
    return log_path

def make_center_box(x_low: np.ndarray, x_high: np.ndarray, ratio: float):
    """
    Create a box centered in the middle of [x_low, x_high],
    shrunk by 'ratio' in each dimension.
    ratio=1.0 -> full box
    ratio=0.5 -> half-width box
    """
    x_low = x_low.astype(np.float32)
    x_high = x_high.astype(np.float32)

    center = (x_low + x_high) / 2.0
    half = (x_high - x_low) / 2.0
    new_half = half * float(ratio)

    low = center - new_half
    high = center + new_half
    return low.astype(np.float32), high.astype(np.float32)


def finetune_value_w_MILP(
    V_model: nn.Module,
    f_pi_model: nn.Module,
    h_model: nn.Module,
    x_low: np.ndarray,
    x_high: np.ndarray,
    lr: float = 1e-4,
    max_iter: int = 100000,
    search_size: int = 1000,
    bnd_ratio: float = 0.1,
    bnd_ratio_avg: float = 0.9,
    min_bnd_ratio: float = 0.01,
    max_bnd_ratio: float = 1.0,
    bnd_eps: float = 0.1,
    search_method: str = "BGB",
    pgd_step: int = 10,
    pgd_eps: float = 0.1,
    backtrack_step: int = 20,
    length_discount: float = 0.8,
    direct_discount: float = 0.5,
    tol: float = 1e-4,
    capacity: int = 10000,
    sample_size: int = 100,
    search_stop: int = 1000,
    replay: int = 1,
    max_skip: int = 1000,
    reg_method: Optional[str] = "ESR",
    esr_max_con: Optional[int] = None,
    eps_h: float = 0.01,
    eps_v: float = 0.01,
    reg_coef: float = 0.1,
    milp_timeout: int = 300,
    log_dir: Optional[str] = None,
    eval_every: int = 10,
    save_every: int = 1000,
    save_path: Optional[str] = None,
    device: str = "cpu",
    seed: int = 42,
    milp_start_iter: int = 20000,
    milp_cooldown: int = 3000,
    milp_box_shrink: float = 0.35,
    milp_box_grow: float = 1.20,
    milp_box_max: float = 1.0,
    milp_max_inconclusive: int = 6,
):
    """
    Fine-tune value function with MILP formal verification.

    Returns:
        (log_path, tfr, last_iter)
    """
    set_seed(seed)
    V_model = V_model.to(device)
    f_pi_model = f_pi_model.to(device)
    h_model = h_model.to(device)

    skipped = 0
    verified = 0
    formally_verified = False

    milp_inconclusive_count = 0
    milp_box_ratio = float(milp_box_shrink)
    last_milp_iter = -10**9

    buffer = Buffer(capacity, len(x_low))
    optimizer = torch.optim.Adam(V_model.parameters(), lr=lr)

    if log_dir is None:
        log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(
        log_dir,
        f"finetune_milp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(log_path, exist_ok=True)

    writer = SummaryWriter(log_path)

    milp_ckpt_dir = os.path.join(log_path, "milp_checkpoints")
    os.makedirs(milp_ckpt_dir, exist_ok=True)

    last_iter = -1

    for i in tqdm(range(max_iter), desc="Finetuning value function (MILP)"):
        last_iter = i

        # ── Search for counterexamples ──
        if len(buffer) < search_stop:
            n_search = max(1, round(search_size / max(bnd_ratio, 1e-6)))
            x = uniform(x_low, x_high, n_search).astype(np.float32)
            x_t = torch.from_numpy(x).to(device)

            with torch.no_grad():
                v = V_model(x_t.T).T.cpu().numpy()[0, :]

            x_bnd = x[:, (v > -bnd_eps) & (v <= tol)]

            if x.shape[1] > 0:
                bnd_ratio = (
                    bnd_ratio_avg * bnd_ratio
                    + (1 - bnd_ratio_avg) * x_bnd.shape[1] / x.shape[1]
                )
            bnd_ratio = float(np.clip(bnd_ratio, min_bnd_ratio, max_bnd_ratio))

            if x_bnd.shape[1] > 0:
                if search_method == "BGB":
                    x_pgd = boundary_guided_search(
                        x_bnd, x_low, x_high, h_model, V_model, f_pi_model,
                        pgd_step=pgd_step,
                        pgd_eps=pgd_eps,
                        backtrack_step=backtrack_step,
                        length_discount=length_discount,
                        bound_guide=True,
                        direct_discount=direct_discount,
                        tol=tol,
                        device=device,
                    )
                elif search_method == "PGD-B":
                    x_pgd = boundary_guided_search(
                        x_bnd, x_low, x_high, h_model, V_model, f_pi_model,
                        pgd_step=pgd_step,
                        pgd_eps=pgd_eps,
                        backtrack_step=backtrack_step,
                        length_discount=length_discount,
                        bound_guide=False,
                        tol=tol,
                        device=device,
                    )
                elif search_method == "PBS":
                    x_pgd = projected_boundary_search(
                        x_bnd, x_low, x_high, h_model, V_model, f_pi_model,
                        tol=tol,
                        device=device,
                    )
                else:
                    raise ValueError(f"Unknown search_method: {search_method}")

                con, inv = filter_counterexample(
                    x_pgd, h_model, V_model, f_pi_model, tol=tol, device=device
                )
                ce = con | inv

                if np.any(ce):
                    n_push = min(int(ce.sum()), len(buffer.vacant))
                    if n_push > 0:
                        buffer.push(x_pgd[:, ce][:, :n_push])
            else:
                con = np.zeros(0, dtype=bool)
                inv = np.zeros(0, dtype=bool)

            writer.add_scalar("finetune/searched_boundary_states", x_bnd.shape[1], i)
            writer.add_scalar("finetune/boundary_state_ratio", bnd_ratio, i)
            writer.add_scalar("finetune/searched_constraint_counterexample", con.sum(), i)
            writer.add_scalar("finetune/searched_invariance_counterexample", inv.sum(), i)

        # ── Train on counterexamples ──
        if len(buffer) > 0:
            skipped = 0

            n = min(sample_size, len(buffer))
            x, c = buffer.pop(n)

            con, inv = filter_counterexample(
                x, h_model, V_model, f_pi_model, tol=tol, device=device
            )
            x_con, x_inv = x[:, con], x[:, inv]

            c[con | inv] = 0
            c[~con & ~inv] += 1
            push_idx = c < replay
            if np.any(push_idx):
                n_push = min(int(push_idx.sum()), len(buffer.vacant))
                if n_push > 0:
                    buffer.push(x[:, push_idx][:, :n_push], c[push_idx][:n_push])

            n_con, n_inv = x_con.shape[1], x_inv.shape[1]

            # Regularization
            if reg_method is not None and (n_con + n_inv > 0) and \
               (esr_max_con is None or (n_con + n_inv < esr_max_con)):

                if reg_method == "ESR":
                    # Regularization su box centrale ristretto invece che su tutto il dominio
                    x_reg_low, x_reg_high = make_center_box(x_low, x_high, 0.7)
                    x_reg = uniform(x_reg_low, x_reg_high, search_size).astype(np.float32)
                    x_reg_t = torch.from_numpy(x_reg).to(device)

                    with torch.no_grad():
                        h_reg = h_model(x_reg_t.T).T.cpu().numpy()[0, :]
                        v_reg = V_model(x_reg_t.T).T.cpu().numpy()[0, :]
                        x_reg_prime = f_pi_model(x_reg_t)
                        v_reg_prime = V_model(x_reg_prime.T).T.cpu().numpy()[0, :]

                    entering = (
                        (h_reg <= -eps_h)
                        & (v_reg > 0)
                        & (v_reg <= eps_v)
                        & (v_reg_prime <= -eps_v)
                    )
                    x_reg = x_reg[:, entering]
                    n_reg = x_reg.shape[1]

                elif reg_method == "RSR":
                    x_reg_low, x_reg_high = make_center_box(x_low, x_high, 0.7)
                    x_reg = uniform(x_reg_low, x_reg_high, search_size).astype(np.float32)
                    n_reg = x_reg.shape[1]

                else:
                    raise ValueError(f"Unknown reg_method: {reg_method}")
            else:
                n_reg = 0

            # Gradient step
            optimizer.zero_grad()
            loss = 0.0

            if n_con > 0:
                x_con_t = torch.from_numpy(x_con).to(device)
                con_loss = -V_model(x_con_t.T).T.sum()
                loss += con_loss

            if n_inv > 0:
                x_inv_t = torch.from_numpy(x_inv).to(device)
                v_inv = V_model(x_inv_t.T).T
                x_inv_prime = f_pi_model(x_inv_t)
                v_inv_prime = V_model(x_inv_prime.T).T
                inv_loss = (-v_inv + v_inv_prime).sum()
                loss += inv_loss

            if n_con + n_inv > 0:
                loss = loss / (n_con + n_inv)

            if n_reg > 0:
                x_reg_t = torch.from_numpy(x_reg).to(device)
                reg_loss = V_model(x_reg_t.T).T.mean()
                loss = loss + reg_coef * reg_loss

            if isinstance(loss, torch.Tensor):
                loss.backward()
                optimizer.step()

            writer.add_scalar("finetune/sample_size", n, i)
            writer.add_scalar(
                "finetune/value_loss",
                loss.item() if isinstance(loss, torch.Tensor) else 0.0,
                i,
            )
            writer.add_scalar("finetune/sampled_constraint_counterexample", n_con, i)
            writer.add_scalar("finetune/sampled_invariance_counterexample", n_inv, i)
            if reg_method is not None:
                writer.add_scalar("finetune/regularization_state", n_reg, i)
        else:
            skipped += 1

        writer.add_scalar("finetune/total_counterexample", len(buffer), i)
        writer.add_scalar("finetune/skipped_update", skipped, i)
        writer.add_scalar("finetune/verified_times", verified, i)
        writer.add_scalar("finetune/milp_box_ratio_live", milp_box_ratio, i)
        writer.add_scalar("finetune/milp_inconclusive_live", milp_inconclusive_count, i)

        # ── Evaluation ──
        if i % eval_every == 0:
            x_eval = uniform(x_low, x_high, search_size).astype(np.float32)
            x_eval_t = torch.from_numpy(x_eval).to(device)
            with torch.no_grad():
                fea_rate = (V_model(x_eval_t.T).T <= 0).float().mean().item()
            writer.add_scalar("finetune/predicted_feasible_rate", fea_rate, i)

        # Save checkpoint
        if (i + 1) % save_every == 0:
            ckpt_path = os.path.join(log_path, f"V_finetune_milp_iter{i+1}.pth")
            tmp_ckpt_path = ckpt_path + ".tmp"
            torch.save(V_model.state_dict(), tmp_ckpt_path)
            os.replace(tmp_ckpt_path, ckpt_path)

        # ── MILP verification when no counterexamples found ──
        ready_for_milp = (
            i >= milp_start_iter
            and skipped >= max_skip
            and (i - last_milp_iter) >= milp_cooldown
        )

        if ready_for_milp:
            ckpt_path = os.path.join(milp_ckpt_dir, f"before_milp_iter{i+1}.pth")
            tmp_ckpt_path = ckpt_path + ".tmp"
            torch.save(V_model.state_dict(), tmp_ckpt_path)
            os.replace(tmp_ckpt_path, ckpt_path)

            milp_x_low, milp_x_high = make_center_box(x_low, x_high, milp_box_ratio)

            print(f"\n  No counterexamples for {skipped} iterations — running MILP verification...", flush=True)
            print(f"  Iteration: {i}", flush=True)
            print(f"  MILP timeout: {milp_timeout}s", flush=True)
            print(f"  MILP box ratio: {milp_box_ratio:.3f}", flush=True)
            print(f"  MILP box low : {milp_x_low}", flush=True)
            print(f"  MILP box high: {milp_x_high}", flush=True)
            print(f"  Saved checkpoint before MILP to: {ckpt_path}", flush=True)

            last_milp_iter = i

            milp_result, milp_ce = milp_verify(
                V_model,
                h_model,
                f_pi_model,
                milp_x_low,
                milp_x_high,
                timeout=milp_timeout,
                verbose=True,
            )

            print(f"  MILP returned: {milp_result}", flush=True)

            if milp_result is True:
                verified += 1
                writer.add_scalar("finetune/milp_verified", 1, i)
                writer.add_scalar("finetune/milp_box_ratio", milp_box_ratio, i)

                print("  FORMALLY VERIFIED on current MILP box.", flush=True)

                # se siamo arrivati al box completo, fermati
                if milp_box_ratio >= milp_box_max - 1e-12:
                    print("  FULL verification box reached — stopping.", flush=True)
                    formally_verified = True
                    break

                # altrimenti allarga gradualmente il box per verifiche future
                milp_box_ratio = min(milp_box_ratio * milp_box_grow, milp_box_max)
                skipped = 0
                milp_inconclusive_count = 0
                print(
                    f"  Expanding MILP box for future checks to ratio={milp_box_ratio:.3f}",
                    flush=True
                )

            elif milp_result is False:
                n_ce = 0 if milp_ce is None else milp_ce.shape[1]
                print(f"  MILP found {n_ce} counterexample(s) — pushing to buffer.", flush=True)

                if milp_ce is not None and n_ce > 0:
                    n_push = min(n_ce, len(buffer.vacant))
                    if n_push > 0:
                        buffer.push(milp_ce[:, :n_push])

                skipped = 0
                milp_inconclusive_count = 0
                writer.add_scalar("finetune/milp_counterexamples", n_ce, i)
                writer.add_scalar("finetune/milp_box_ratio", milp_box_ratio, i)

            else:
                milp_inconclusive_count += 1
                writer.add_scalar("finetune/milp_inconclusive", 1, i)
                writer.add_scalar("finetune/milp_inconclusive_count", milp_inconclusive_count, i)
                writer.add_scalar("finetune/milp_box_ratio", milp_box_ratio, i)

                print(
                    f"  MILP inconclusive (count={milp_inconclusive_count}/{milp_max_inconclusive}) — continuing training.",
                    flush=True
                )

                # non resettare brutalmente tutto
                skipped = max_skip // 2

                # troppi inconclusive -> restringi di nuovo
                if milp_inconclusive_count >= milp_max_inconclusive:
                    milp_box_ratio = max(milp_box_shrink, milp_box_ratio / milp_box_grow)
                    milp_inconclusive_count = 0
                    print(
                        f"  Too many inconclusive results — shrinking future MILP box to ratio={milp_box_ratio:.3f}",
                        flush=True
                    )

    # Save final model
    if save_path is None:
        save_path = os.path.join(log_path, "V_finetune_milp.pth")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tmp_final_path = save_path + ".tmp"
    torch.save(V_model.state_dict(), tmp_final_path)
    os.replace(tmp_final_path, save_path)

    writer.close()

    if formally_verified:
        print(f"\nTraining complete — model FORMALLY VERIFIED. Saved to {save_path}")
    else:
        print(f"\nTraining complete (max_iter reached or stopped before full-domain proof). Model saved to {save_path}")

    V_model.eval()
    X = uniform(x_low, x_high, 100_000).astype(np.float32)
    X_t = torch.from_numpy(X).to(device)

    with torch.no_grad():
        V = V_model(X_t.T).T.cpu().numpy()[0, :]

        truly_feasible = np.ones(100_000, dtype=bool)
        X_current = X_t.clone()
        for _ in range(100):
            h_current = h_model(X_current.T).T.cpu().numpy()[0, :]
            truly_feasible &= (h_current <= 0)
            X_current = f_pi_model(X_current)

    n_truly_feasible = truly_feasible.sum()
    identified = (V <= 0) & truly_feasible
    n_identified = identified.sum()

    tfr = n_identified / max(n_truly_feasible, 1)
    print("\n")
    return log_path, tfr, last_iter

def finetune_value_enum_MILP(
    V_model: nn.Module,
    f_pi_model: nn.Module,
    h_model: nn.Module,
    x_low: np.ndarray,
    x_high: np.ndarray,
    lr: float = 1e-4,
    max_iter: int = 100000,
    tol: float = 1e-4,
    capacity: int = 10000,
    sample_size: int = 100,
    replay: int = 1,
    max_skip: int = 1000,
    # ── Enumeration parameters ──
    enum_freq: int = 1000,
    enum_n_samples: int = 20000,
    enum_coverage: float = 70.0,
    enum_n_trees: int = 2000,
    enum_max_depth: int = 20,
    enum_n_train: int = 40000,
    refill_threshold: int = 500,
    # ── Neighborhood fallback ──
    fallback_eval_rounds: int = 5,
    fallback_eval_samples: int = 100000,
    fallback_n_neighbors: int = 5000,
    fallback_radius: float = 0.05,
    # ── Regularization ──
    search_size: int = 1000,
    reg_method: Optional[str] = "ESR",
    esr_max_con: Optional[int] = None,
    eps_h: float = 0.01,
    eps_v: float = 0.01,
    reg_coef: float = 0.1,
    # ── MILP ──
    milp_timeout: int = 300,
    # ── Logging / misc ──
    log_dir: Optional[str] = None,
    eval_every: int = 100,
    save_every: int = 5000,
    save_path: Optional[str] = None,
    device: str = 'cpu',
    seed: int = 42
) -> str:
    """
    Fine-tune value function using **enumeration-based search + MILP verification**.

    Instead of PGD adversarial attacks this function uses Random-Forest-based
    enumeration to discover *all* unsafe hyperrectangles
    (regions where ``V(x) <= 0`` but ``h(x) > 0`` or ``V(f_pi(x)) > 0``).
    Counterexamples are then sampled volume-proportionally from those regions
    and pushed into a replay buffer.  The gradient step, ESR/RSR
    regularisation and buffer logic are **identical** to ``finetune_value``.

    **Why this is faster / more robust than PGD-based search:**

    1. *Global coverage* – enumeration finds the full landscape of unsafe
       regions rather than relying on sparse local PGD trajectories.
    2. *Volume-aware sampling* – larger violation regions receive more
       gradient signal, shrinking them efficiently.
    3. *Amortised cost* – regions are re-used for many cheap refills between
       expensive re-enumerations.
    4. *Neighbourhood amplification* – when regions go stale a local
       densification pass finds residual boundary violations that would take
       many PGD rounds to discover.
    5. *Faster MILP convergence* – the global view means fewer training
       iterations before the MILP can certify correctness.

    When sampling finds no more violations for ``max_skip`` iterations the
    algorithm switches to MILP formal verification:

    * **Certified** → training stops.
    * **Counterexample found** → pushed to buffer; training continues.
    * **Inconclusive (timeout)** → training continues.

    Args:
        V_model: Value function model
        f_pi_model: Closed-loop dynamics model
        h_model: Constraint model
        x_low, x_high: State-space bounds (1-D arrays)
        lr: Learning rate
        max_iter: Maximum training iterations
        tol: Tolerance for counterexample detection
        capacity: Replay-buffer capacity
        sample_size: Counterexamples drawn from buffer per gradient step
        replay: Maximum replay count before discarding a sample
        max_skip: Empty iterations before triggering MILP
        enum_freq: Re-enumerate every N iterations
        enum_n_samples: Samples drawn from regions per refill
        enum_coverage: Target RF coverage (%)
        enum_n_trees: Number of RF trees
        enum_max_depth: Maximum RF tree depth
        enum_n_train: RF training-set size
        refill_threshold: Refill from regions when buffer drops below this
        fallback_eval_rounds: Uniform-sampling rounds for neighbourhood fallback
        fallback_eval_samples: Samples per fallback evaluation round
        fallback_n_neighbors: Total neighbourhood samples per fallback
        fallback_radius: Neighbourhood hypercube half-width
        search_size: Sample size for ESR evaluation
        reg_method: Regularisation method ("ESR", "RSR", or None)
        esr_max_con: Maximum counterexamples for ESR
        eps_h: Constraint threshold for ESR
        eps_v: Value threshold for ESR
        reg_coef: Regularisation coefficient
        milp_timeout: Time limit (s) per MILP sub-problem
        log_dir: Directory for TensorBoard logs / checkpoints
        eval_every: Log feasible-rate every N iterations
        save_every: Save a checkpoint every N iterations
        save_path: Final model save path (default: ``log_dir/../V_finetune_enum_milp.pth``)
        device: Torch device
        seed: Random seed

    Returns:
        log_path: Directory containing logs and checkpoints
    """
    set_seed(seed)
    V_model = V_model.to(device)
    f_pi_model = f_pi_model.to(device)
    h_model = h_model.to(device)

    x_dim = len(x_low)
    input_domain = np.stack([x_low, x_high], axis=1)  # (x_dim, 2)

    skipped = 0
    formally_verified = False
    unsafe_regions: list = []
    enum_stale = False          # regions no longer produce violations
    milp_calls = 0              # adaptive skip shrinks after each MILP call
    current_max_skip = max_skip

    buffer = Buffer(capacity, x_dim)
    optimizer = torch.optim.Adam(V_model.parameters(), lr=lr)

    if log_dir is None:
        log_dir = './logs/'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        f"finetune_enum_milp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    writer = SummaryWriter(log_path)

    # ================================================================
    #  Helper: neighbourhood amplification around seed points
    # ================================================================
    def _amplify_seeds(seeds: np.ndarray) -> int:
        """Sample dense neighbourhoods around *seeds* (x_dim, N),
        filter for true violations, push to buffer.

        Returns number of violations pushed.
        """
        n_seeds = seeds.shape[1]
        if n_seeds == 0:
            return 0

        n_per_seed = max(1, fallback_n_neighbors // n_seeds)
        neighbours: list = []
        for j in range(n_seeds):
            centre = seeds[:, j:j + 1]
            noise = np.random.uniform(
                -fallback_radius, fallback_radius,
                size=(x_dim, n_per_seed)
            ).astype(np.float32)
            pts = np.clip(centre + noise, x_low[:, None], x_high[:, None])
            neighbours.append(pts)

        X_nbr = np.concatenate(neighbours, axis=1)
        con, inv = filter_counterexample(
            X_nbr, h_model, V_model, f_pi_model, tol=tol, device=device
        )
        ce_nbr = con | inv

        X_all = np.concatenate([seeds, X_nbr[:, ce_nbr]], axis=1)
        n_push = min(X_all.shape[1], len(buffer.vacant))
        if n_push > 0:
            buffer.push(X_all[:, :n_push])
        return n_push

    def _neighbourhood_fallback() -> int:
        """Large-scale uniform search + neighbourhood densification.

        Returns the number of violations pushed to the buffer.
        """
        all_violations = []
        for _ in range(fallback_eval_rounds):
            X_fb = uniform(x_low, x_high, fallback_eval_samples)
            con, inv = filter_counterexample(
                X_fb, h_model, V_model, f_pi_model, tol=tol, device=device
            )
            ce = con | inv
            if ce.sum() > 0:
                all_violations.append(X_fb[:, ce])

        if len(all_violations) == 0:
            return 0

        seeds = np.concatenate(all_violations, axis=1)
        return _amplify_seeds(seeds)

    # ================================================================
    #  Main training loop
    # ================================================================
    for i in tqdm(range(max_iter), desc="Finetuning (enum+MILP)"):

        # ── 1. Enumeration search (periodic) ──────────────────────
        #   Run at iter 0, then every enum_freq iterations.
        #   Skip only if regions are still productive (not stale).
        need_enum = (i == 0) or (i % enum_freq == 0) if refill_threshold > 0 else False

        if need_enum:
            wrapper = BinaryDecisionWrapperCBF(
                V_model, h_model, f_pi_model, tol=tol
            )
            unsafe_regions, stats = enumerate_unsafe_regions(
                wrapper_model=wrapper,
                input_domain=input_domain,
                n_training_samples=enum_n_train,
                n_decision_trees=enum_n_trees,
                max_depth=enum_max_depth,
                enumerate_safe_regions=False,
                incremental_coverage=True,
                wilks_filtering=False,
                wilks_samples=10,
                sample_size_evaluation=enum_n_train,
                desired_coverage=enum_coverage,
                digits_to_consider=0,
                seed=seed,
                verbose=False,
                device=device
            )
            enum_stale = (len(unsafe_regions) == 0)
            writer.add_scalar('enum/n_regions', len(unsafe_regions), i)
            writer.add_scalar('enum/coverage', stats.get('coverage', 0.0), i)
            if not enum_stale:
                tqdm.write(f"  [iter {i}] Enumeration: {len(unsafe_regions)} unsafe "
                           f"regions, coverage={stats.get('coverage', 0):.1f}%")
            else:
                tqdm.write(f"  [iter {i}] Enumeration: no unsafe regions found.")

        # ── 2. Refill buffer from known regions (cheap) ───────────
        if len(buffer) < refill_threshold and len(unsafe_regions) > 0:
            X_cand = sample_from_regions(
                unsafe_regions, n_samples=enum_n_samples, digits_to_consider=0
            ).T.astype(np.float32)

            con, inv = filter_counterexample(
                X_cand, h_model, V_model, f_pi_model, tol=tol, device=device
            )
            ce = con | inv
            n_found = int(ce.sum())

            if n_found > 0:
                n_push = min(n_found, len(buffer.vacant))
                if n_push > 0:
                    buffer.push(X_cand[:, ce][:, :n_push])
            else:
                # Regions no longer productive
                enum_stale = True
                unsafe_regions = []

            writer.add_scalar('enum/refill_found', n_found, i)

        # ── 3. Neighbourhood fallback (buffer empty, any time) ────
        #   No longer gated on enum_stale — always try when buffer
        #   is empty to avoid dead-zone iterations.
        if len(buffer) == 0:
            n_added = _neighbourhood_fallback()
            if n_added > 0:
                tqdm.write(f"  [iter {i}] Neighbourhood fallback: "
                           f"{n_added} violations → buffer")

        # ── 4. Single gradient step from buffer ───────────────────
        if len(buffer) > 0:
            skipped = 0

            n = min(sample_size, len(buffer))
            x, c = buffer.pop(n)

            # Re-filter (some violations may have been fixed)
            con, inv = filter_counterexample(
                x, h_model, V_model, f_pi_model, tol=tol, device=device
            )
            x_con, x_inv = x[:, con], x[:, inv]

            c[con | inv] = 0
            c[~con & ~inv] += 1
            push_idx = c < replay
            if np.any(push_idx):
                n_back = min(int(push_idx.sum()), len(buffer.vacant))
                if n_back > 0:
                    buffer.push(x[:, push_idx][:, :n_back], c[push_idx][:n_back])

            n_con, n_inv = x_con.shape[1], x_inv.shape[1]

            # ── Regularisation (identical to finetune_value) ──
            n_reg = 0
            x_reg = None
            if reg_method is not None and (n_con + n_inv > 0) and \
               (esr_max_con is None or (n_con + n_inv < esr_max_con)):
                if reg_method == "ESR":
                    x_reg_cand = uniform(x_low, x_high, search_size)
                    x_reg_t = torch.from_numpy(x_reg_cand).to(device)
                    with torch.no_grad():
                        h_reg = h_model(x_reg_t.T).T.cpu().numpy()[0, :]
                        v_reg = V_model(x_reg_t.T).T.cpu().numpy()[0, :]
                        x_reg_prime = f_pi_model(x_reg_t)
                        v_reg_prime = V_model(x_reg_prime.T).T.cpu().numpy()[0, :]
                    entering = (h_reg <= -eps_h) & (v_reg > 0) & \
                               (v_reg <= eps_v) & (v_reg_prime <= -eps_v)
                    x_reg = x_reg_cand[:, entering]
                    n_reg = x_reg.shape[1]
                elif reg_method == "RSR":
                    x_reg = uniform(x_low, x_high, search_size)
                    n_reg = x_reg.shape[1]

            # ── Gradient step ──
            optimizer.zero_grad()
            loss = 0.0

            if n_con > 0:
                x_con_t = torch.from_numpy(x_con).to(device)
                con_loss = -V_model(x_con_t.T).T.sum()
                loss += con_loss

            if n_inv > 0:
                x_inv_t = torch.from_numpy(x_inv).to(device)
                v_inv = V_model(x_inv_t.T).T
                x_inv_prime = f_pi_model(x_inv_t)
                v_inv_prime = V_model(x_inv_prime.T).T
                inv_loss = (-v_inv + v_inv_prime).sum()
                loss += inv_loss

            if n_con + n_inv > 0:
                loss = loss / (n_con + n_inv)

            if n_reg > 0:
                x_reg_t = torch.from_numpy(x_reg).to(device)
                reg_loss = V_model(x_reg_t.T).T.mean()
                loss = loss + reg_coef * reg_loss

            if isinstance(loss, torch.Tensor):
                loss.backward()
                optimizer.step()

            writer.add_scalar('finetune/sample_size', n, i)
            writer.add_scalar('finetune/value_loss',
                              loss.item() if isinstance(loss, torch.Tensor) else 0.0, i)
            writer.add_scalar('finetune/sampled_constraint_counterexample', n_con, i)
            writer.add_scalar('finetune/sampled_invariance_counterexample', n_inv, i)
            if reg_method is not None:
                writer.add_scalar('finetune/regularization_state', n_reg, i)
        else:
            skipped += 1

        writer.add_scalar('finetune/total_counterexample', len(buffer), i)
        writer.add_scalar('finetune/skipped_update', skipped, i)

        # ── 5. Evaluation ─────────────────────────────────────────
        if i % eval_every == 0:
            x_eval = uniform(x_low, x_high, search_size)
            x_eval_t = torch.from_numpy(x_eval).to(device)
            with torch.no_grad():
                fea_rate = (V_model(x_eval_t.T).T <= 0).float().mean().item()
            writer.add_scalar('finetune/predicted_feasible_rate', fea_rate, i)

        # ── 6. Checkpoint ─────────────────────────────────────────
        if (i + 1) % save_every == 0:
            torch.save(
                V_model.state_dict(),
                os.path.join(log_path, f'V_finetune_enum_milp_iter{i+1}.pth')
            )

        # ── 7. MILP verification when search is exhausted ─────────
        #   Uses adaptive skip: halves after each MILP call so the
        #   tail phase (sparse violations) converges faster.
        if skipped >= current_max_skip:
            tqdm.write(f"\n  No counterexamples for {current_max_skip} iterations "
                       f"— running MILP verification...")
            milp_result, milp_ce = milp_verify(
                V_model, h_model, f_pi_model, x_low, x_high,
                timeout=milp_timeout, verbose=True
            )
            milp_calls += 1

            if milp_result is True:
                tqdm.write("  FORMALLY VERIFIED: no violations exist "
                           "(MILP certificate).")
                formally_verified = True
                writer.add_scalar('finetune/milp_verified', 1, i)
                break

            elif milp_result is False:
                # ── Amplify the MILP counterexample ──
                n_ce = milp_ce.shape[1]
                tqdm.write(f"  MILP found {n_ce} counterexample(s) "
                           f"— amplifying with neighbourhood search...")
                n_amplified = _amplify_seeds(milp_ce)
                tqdm.write(f"  Amplified to {n_amplified} violations in buffer.")

                skipped = 0
                # Shrink max_skip: at most half, but never below 50
                current_max_skip = max(50, current_max_skip // 2)
                writer.add_scalar('finetune/milp_counterexamples', n_ce, i)
                writer.add_scalar('finetune/current_max_skip', current_max_skip, i)

            else:
                tqdm.write("  MILP inconclusive (timeout) "
                           "— continuing training.")
                skipped = 0
                current_max_skip = max(50, current_max_skip // 2)
                writer.add_scalar('finetune/milp_inconclusive', 1, i)

    # ── Save final model ──────────────────────────────────────────
    if save_path is None:
        save_path = os.path.join(log_path, 'V_finetune_enum_milp.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(V_model.state_dict(), save_path)
    writer.close()

    if formally_verified:
        print(f"\nTraining complete — model FORMALLY VERIFIED. Saved to {save_path}")
    else:
        print(f"\nTraining complete (max_iter reached). Model saved to {save_path}")


    V_model.eval()
    X = uniform(x_low, x_high, 100_000)
    X_t = torch.from_numpy(X).to(device)  # [x_dim, N]

    with torch.no_grad():
        V = V_model(X_t.T).T.cpu().numpy()[0, :]  # [N]

        # Forward simulate and check constraints along the trajectory
        truly_feasible = np.ones(100_000, dtype=bool)
        X_current = X_t.clone()  # [x_dim, N]
        for step in range(100):
            h_current = h_model(X_current.T).T.cpu().numpy()[0, :]
            truly_feasible &= (h_current <= 0)
            X_current = f_pi_model(X_current)

    n_truly_feasible = truly_feasible.sum()
    identified = (V <= 0) & truly_feasible
    n_identified = identified.sum()
    missed = truly_feasible & (V > 0)  # truly feasible but V says unsafe

    tfr = n_identified / max(n_truly_feasible, 1)
    print("\n\n")
    return log_path, tfr, i
