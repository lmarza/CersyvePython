# CersyvePython

A Python implementation of the Cersyve framework for learning safe control policies using neural networks with counterexample-guided training.

Converted from the original Julia implementation by Intelligent Control Lab (CMU): [Cersyve.jl](https://github.com/intelligent-control-lab/Cersyve.jl/tree/master)

---

### Overview

Cersyve combines:
- **Data-driven dynamics learning**: Learn system dynamics $f(x,u)$ from trajectory data
- **Constraint learning**: Learn safety constraints $h(x)$
- **Value function learning**: Learn a value function $V(x)$ that certifies safety
- **Counterexample-guided training**: Use adversarial search to find and eliminate safety violations
- **Regularization techniques**: APA (Activation Pattern Alignment) and SNR (Signal-to-Noise Ratio)

New: 
- **Enumeration of unsafe regions**: Replacing the Counterexample-guided training to find and eliminate safety violations with more efficient and guaranteed approach.

### Installation

```bash
# Clone the repository
git clone https://github.com/lmarza/CersyvePython
cd CersyvePython

# Create a conda environment (optional but recommended)
conda create -n cersyve python=3.8
conda activate cersyve

# Install dependencies
pip install -r requirements.txt
```
---

## Quick Start

### Complete Workflow for Double Integrator System

```bash
cd examples/double_integrator/

# Step 1: Collect trajectory data
python collect_data_double_integrator.py

# Step 2: Train dynamics model f(x,u) -> dx
python train_dynamics_double_integrator.py

# Step 3: Train constraint model h(x)
python train_constraint_double_integrator.py

# Step 4: Pretrain value function V(x)
python pretrain_double_integrator.py

# Step 5: Finetune value function using counterexample search
python finetune_double_integrator.py
```

### Visualize Training Progress

```bash
tensorboard --logdir=logs/double_integrator/
```

### Visualize the pre-trained value function and the fine-tuned one

```bash
python visualize_comparison.py
```

### Visualize the pre-trained value function and enumeration of unsafe areas

```bash
python enumerate_and_visualize_safe_regions.py
```
