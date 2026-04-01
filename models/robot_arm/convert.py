import h5py
import numpy as np
import torch
from collections import OrderedDict


def convert_jld2_to_pth(input_file, output_file):
    """Convert a JLD2 Flux.jl Chain state file to a PyTorch state_dict .pth file.

    Julia/Flux stores weight matrices in column-major order (shape: in x out),
    while PyTorch expects (out x in), so every weight matrix is transposed.
    The layer field names '1', '2', '3' in the JLD2 map to PyTorch Sequential
    indices 0, 2, 4 (because ReLU modules occupy the odd positions).
    """
    # JLD2 layer field name -> PyTorch Sequential index
    layer_map = {'1': 0, '2': 2, '3': 4}

    state_dict = OrderedDict()

    with h5py.File(input_file, 'r') as f:
        val = f['state'][()]
        layers = val['layers']

        for jl_name, pt_idx in layer_map.items():
            if jl_name not in layers.dtype.names:
                print(f"Warning: layer '{jl_name}' not found, skipping.")
                continue

            layer = layers[jl_name]
            w_ref = layer['weight']
            b_ref = layer['bias']

            # Dereference HDF5 object references
            w = f[w_ref][()].astype(np.float32)   # shape (in, out) in Julia
            b = f[b_ref][()].astype(np.float32)   # shape (out,)

            # Transpose weight: (in, out) -> (out, in) for PyTorch
            w = w.T

            state_dict[f'{pt_idx}.weight'] = torch.from_numpy(w.copy())
            state_dict[f'{pt_idx}.bias']   = torch.from_numpy(b.copy())

            print(f"Layer '{jl_name}' -> '{pt_idx}.weight' {tuple(w.shape)}, "
                  f"'{pt_idx}.bias' {tuple(b.shape)}")

    if not state_dict:
        print("Warning: no layers were converted.")
        return

    torch.save(state_dict, output_file)
    print(f"\nSaved {len(state_dict) // 2} layers to '{output_file}'")


#convert_jld2_to_pth('V_pretrain.jld2', 'V_pretrain.pth')
#convert_jld2_to_pth('V_finetune.jld2', 'V_finetune.pth')
#convert_jld2_to_pth('f.jld2', 'f.pth')
convert_jld2_to_pth('h.jld2', 'h.pth')
