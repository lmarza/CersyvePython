# CersyvePython

Python implementation of Cersyve, converted from the Julia implementation of the
Intelligent Control Lab (CMU): [Cersyve.jl](https://github.com/intelligent-control-lab/Cersyve.jl/tree/master).

Cersyve learns a dynamics model f(x, u), a constraint model h(x) and a value function V(x)
whose zero sublevel set {V <= 0} is a certified forward-invariant safe set: every state of
the set satisfies h(x) <= 0 and its successor under the closed loop stays in the set.
Fine-tuning repairs the pretrained V with counterexamples until a mixed-integer linear
program (MILP) proves both conditions over the whole state box.

Two fine-tuning methods share one loop, one verifier and one stop rule:

- PGD+MILP: projected-gradient search for counterexamples (cersyve/pgd.py).
- Enum+MILP: a random forest over labelled states enumerates the violating regions, and
  counterexamples are sampled from its unsafe leaves (cersyve/enumeration.py).

A third loop, finetune_value in cersyve/finetune.py, fine-tunes with sampling only and no
verifier; it is the sampling baseline.

## Installation

```bash
conda create -y -n cersyve python=3.11 && conda activate cersyve
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The verifier needs Gurobi. Put the licence file at ~/gurobi.lic or set GRB_LICENSE_FILE.

## Layout

```
cersyve/        pgd.py (PGD search), enumeration.py (random forest enumeration), milp.py
                (reference PuLP/Gurobi verifier milp_verify and the direct Gurobi verifier
                milp_fast_verify), abcrown.py (alpha-beta-CROWN wrapper/alternative), finetune.py
                (the loops, estimate_tfr), model.py, buffer.py, utils.py, pretrain.py, tolerances.py
configs/        _base.py: every loop parameter with its default; <task>.py: the five benchmarks;
                final/: the parameters of every table row
systems/        pendulum, unicycle, cartpole, robot_arm, roboracer
examples/       run_finetune.py (one run), campaign.py (CARBS search), <task>/ data collection
                and pretraining
scripts/        replay_verifier.py (proof seconds), make_spec.py (search space of a campaign)
models/<task>/  V_pretrain.pth, f.pth, h.pth; results/<run>/seed<k>.json (result records with
                the verifier log); checkpoints/ (the certificates)
data/           trajectory data of the learned benchmarks
```

## One run

```bash
python examples/run_finetune.py --task TASK --method {pgd_milp,enum_milp} --seeds 1 0 2 [--tag TAG] [--config JSON] [--override key=value ...]
```

Tasks: pendulum, unicycle, cartpole, robot_arm, roboracer. Defaults are PGD_MILP and
ENUM_MILP of configs/<task>.py; --config applies a JSON of loop parameters, --override single
values. The loop stops at its first proof. The runner then re-checks the certificate with the
reference MILP, estimates the certified true-feasible rate (TFR: certified volume over the
volume of the states that stay safe for 100 steps) and writes
models/<task>/results/<method>[_<tag>]/seed<k>.json with every verifier call (stage, verdict,
seconds) and the certificate to models/<task>/checkpoints/.

A new system needs collect_data, train_dynamics, train_constraint and pretrain as in
examples/pendulum/.

```bash
python examples/run_finetune.py --task pendulum  --method pgd_milp  --seeds 1 0 2 --config configs/final/pendulum_pgd_milp.json
python examples/run_finetune.py --task pendulum  --method enum_milp --seeds 1 0 2 --config configs/final/pendulum_enum_milp.json
```
