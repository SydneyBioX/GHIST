import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import pandas as pd
import random


def create_tensors(n, predefined_value, n_ct):
    tensors = [torch.randn(1, n_ct) for _ in range(n)]

    current_avg = torch.mean(torch.cat(tensors, dim=0), dim=0)

    for tensor in tensors:
        tensor += predefined_value - current_avg

    return tensors
    
    
def adjust_pr(
    out_cell_type,
    comp_est,
    comp_out,
    cell_types,
    cell_types_target,
    ignore_idx=None,
    scale=1.0,
    is_invasive=False,
    target_cts=["B", "Myeloid", "Malignant"],
):

    out_cell_type_ct = torch.argmax(out_cell_type, dim=1)
    out_cell_type_ct = list(out_cell_type_ct.detach().cpu().numpy())
    n_cells = len(out_cell_type_ct)

    # find indices of batch where it is a predicted target cell type
    target_idx = [cell_types.index(x) for x in cell_types_target]
    target_pr = [i for i, j in enumerate(out_cell_type_ct) if j in target_idx]

    if ignore_idx is not None:
        target_pr = [x for x in target_pr if x not in ignore_idx]

    n_imm = len(target_pr)

    # no target cell types
    if n_imm < 1:
        return out_cell_type_ct, [], []

    else:

        comp_out = comp_out[target_idx].detach().cpu().numpy()
        if is_invasive:
            comp_out = 0 * comp_out
        comp_out = comp_out * n_cells

        # get comp_est in int
        comp_est = comp_est[target_idx].detach().cpu().numpy()

        # create n random vectors that sum to comp_est
        tensors = create_tensors(n_imm, torch.tensor(comp_est), len(cell_types_target))
        noise = np.stack([tensor.numpy() for tensor in tensors])
        noise = noise[:,-1]        

        comp_est = comp_est * n_cells

        # find position of target CT in the list of cell types of interest
        # and find its corresponding estimated composition
        n_target_cts = len(cell_types_target)
        a = np.zeros(n_target_cts)

        for ct in cell_types_target:
            ct_target_idx = cell_types_target.index(ct)
            if is_invasive and ct not in target_cts:
                a[ct_target_idx] = 0
            else:
                a[ct_target_idx] = (
                    scale * (comp_est[ct_target_idx] - comp_out[ct_target_idx])
                )
                
        a = np.expand_dims(a, 0)
        a = np.tile(a, (n_imm, 1))

        out_cell_type = out_cell_type.detach().cpu().numpy()
        
        out_cell_type_adj = out_cell_type[np.ix_(target_pr, target_idx)] + a * np.abs(noise)
        
        out_cell_type_adj_ct = np.argmax(out_cell_type_adj, 1)
        
        # map back to original cell type idx
        mapping_dict = {
            old_value: new_value
            for old_value, new_value in zip(
                list(range(len(cell_types_target))), target_idx
            )
        }
        mapped_list = [mapping_dict[x] for x in out_cell_type_adj_ct]

        # take argmax and get adjusted cell types
        out_cell_type_final = out_cell_type_ct.copy()
        swapped_idx = []

        swapped_idx_target = []

        target_ct_idx = [cell_types.index(x) for x in target_cts]

        for index, new_value in zip(target_pr, mapped_list):
            out_cell_type_final[index] = new_value
            if new_value != out_cell_type_ct[index]:
                swapped_idx.append(index)

                # for ensuring consistency with expr
                if new_value in target_ct_idx:
                    swapped_idx_target.append(index)
                                
        return out_cell_type_final, swapped_idx, swapped_idx_target