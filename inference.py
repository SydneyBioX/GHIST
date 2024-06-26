import argparse
import logging
import os
import sys
import natsort
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
import glob

from model.adjustments import *
from dataio.dataset_input_inference import DataProcessing
from model.model import Framework
from utils.utils import *

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import csv

from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata


def main(config):
    opts = json_file_to_pyobj(config.config_file)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    device = get_device(opts.model.gpu_ids)

    # Create experiment directories
    make_new = False
    timestamp = get_experiment_id(
        make_new, opts.experiment_dirs.load_dir, config.fold_id
    )
    experiment_path = f"experiments/{timestamp}"
    model_dir = experiment_path + "/" + opts.experiment_dirs.model_dir
    test_output_dir = (
        experiment_path + "/" + opts.experiment_dirs.inference_output_dir + "/"
    )
    os.makedirs(test_output_dir, exist_ok=True)

    # Set up the model
    logging.info("Initialising model")

    classes = opts.data.cell_types
    n_classes = len(classes)
    print(classes)
    print(f"Num cell types {n_classes}")

    gene_names = read_txt(opts.data_sources.fp_genes)

    if config.use_sc:
        df_ref_raw = pd.read_csv(opts.data_sources.fp_sc_ref, index_col=0)

        gene_names = natsort.natsorted(
            list(set(df_ref_raw.columns.tolist()) & set(gene_names))
        )

        df_ref = pd.DataFrame(0, index=df_ref_raw.index, columns=gene_names)
        for col in gene_names:
            df_ref[col] = df_ref_raw[col]

        n_ref = df_ref.shape[0]
        expr_ref = opts.data.expr_scale * df_ref.to_numpy()
        print("SC ref shape ", expr_ref.shape)
        expr_ref_torch = torch.from_numpy(expr_ref).float().to(device)

    else:
        n_ref = 0
        expr_ref_torch = None

    n_genes = len(gene_names)
    print(f"{n_genes} genes")
    
    model = Framework(
        n_classes,
        n_genes,
        opts.model.emb_dim,
        n_ref,
        device,
        config.use_sc
    )

    # Get list of model files
    if config.epoch < 0:
        saved_model_paths = glob.glob(f"{model_dir}/epoch_*.pth")
        saved_model_paths = sorted_alphanumeric(saved_model_paths)
        saved_model_names = [
            (os.path.basename(x)).split(".")[0] for x in saved_model_paths
        ]
        saved_model_epochs = [x.split("_")[1] for x in saved_model_names]
        saved_model_epochs = list(set(saved_model_epochs))
        saved_model_epochs = sorted_alphanumeric(saved_model_epochs)
        if config.epoch == -2:
            saved_model_epochs = np.array(saved_model_epochs, dtype="int")
        elif config.epoch == -1:
            saved_model_epochs = np.array(saved_model_epochs[-1], dtype="int")
            saved_model_epochs = [saved_model_epochs]
    else:
        saved_model_epochs = [config.epoch]

    # Dataloader
    logging.info("Preparing data")
    
    if config.no_normstain:
        normstain = False 
    else:
        normstain = True

    test_dataset = DataProcessing(
        opts.data_sources,
        opts.data,
        gene_names,
        config.fold_id,
        normstain
    )
    dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=opts.training.batch_size,
        shuffle=False,
        num_workers=opts.data.num_workers,
        drop_last=False,
    )

    n_test_examples = len(dataloader)
    logging.info("Total number of patches: %d" % n_test_examples)

    logging.info("Begin prediction")

    with torch.no_grad():

        for epoch_idx, test_epoch in enumerate(saved_model_epochs):

            if config.demo:
                load_path = (
                    experiment_path
                    + "/"
                    + opts.experiment_dirs.model_dir
                    + "/model.pth"
                )
            else:
                load_path = model_dir + "/epoch_%d_model.pth" % (test_epoch)
                        
            # Restore model
            checkpoint = torch.load(load_path)

            model.load_state_dict(checkpoint["model_state_dict"])
            epoch = checkpoint["epoch"]
            print("Predict using " + load_path)

            model.to(device)

            model = model.eval()

            pbar = tqdm(dataloader)

            all_gt = []
            all_pr = []
            all_gt_ct = []
            all_pr_ct = []
            all_pr_adj = []
            all_ids = []
            all_expr = None
            all_expr_gt = None
            all_comp_est = np.zeros(
                (n_test_examples * opts.training.batch_size, n_classes)
            )
            all_comp_est_i = 0
            all_comp_est_by_cell = None
            all_area = None

            epoch_comp_est = np.zeros(n_classes)
            epoch_comp_out = np.zeros(n_classes)

            for (
                batch_nuclei,
                batch_he_img,
                batch_n_cells,
                patch_ids,
            ) in pbar:

                batch_nuclei = batch_nuclei.to(device)
                batch_he_img = batch_he_img.to(device)
                batch_n_cells = batch_n_cells.to(device)
                patch_ids = patch_ids.to(device)

                (
                    out_cell_type,
                    _,
                    _,
                    out_expr,
                    out_expr_immune,
                    out_expr_invasive,
                    _,
                    _,
                    _,
                    _,
                    _,
                    comp_estimated,
                    batch_area,
                    patch_ids_pc,
                ) = model(
                    batch_he_img,
                    batch_nuclei,
                    batch_n_cells,
                    expr_ref_torch,
                    batch_ct=None,
                    batch_expr=None,
                    patch_ids=patch_ids,
                )

                if out_cell_type.shape[0] == 0:
                    continue

                # neighbourhood compositions
                comp_out_raw = torch.nn.functional.softmax(out_cell_type, dim=1)
                comp_out_raw = torch.argmax(comp_out_raw, 1)
                comp_out = torch.nn.functional.one_hot(
                    comp_out_raw, num_classes=n_classes
                )
                comp_out = comp_out.float()
                comp_out = torch.mean(comp_out, 0)

                # adjustments based on predicted composition
                adjusted_out_cell_type = []
                comp_estimated_sum = torch.zeros(n_classes).to(device)

                for i_batch in range(batch_n_cells.shape[0]):
                    n_cells_batch = int(batch_n_cells[i_batch])

                    if n_cells_batch > 0:
                        idx_start = torch.sum(batch_n_cells[:i_batch]).item()
                        idx_end = idx_start + n_cells_batch

                        comp_out_raw_patch = torch.nn.functional.softmax(
                            out_cell_type[idx_start:idx_end, :], dim=1
                        )
                        comp_out_raw_patch = torch.argmax(comp_out_raw_patch, 1)
                        comp_out_patch = torch.nn.functional.one_hot(
                            comp_out_raw_patch, num_classes=n_classes
                        )
                        comp_out_patch = comp_out_patch.float()
                        comp_out_patch = torch.mean(comp_out_patch, 0)

                        # refinements based on predicted compositions

                        # mask highly confident cell predictions
                        pred_logits = F.softmax(
                            out_cell_type[idx_start:idx_end, :], dim=1
                        )

                        ct_index_imm = [opts.data.cell_types.index("T")]
                        high_conf_imm = torch.where(
                            pred_logits[:, ct_index_imm] > opts.data.high_conf_prob
                        )[0]
                        high_conf_imm = high_conf_imm.cpu().numpy().tolist()

                        # cell type refinement

                        (
                            adjusted_out_cell_type_patch_immune,
                            idx_swapped_immune_all,
                            idx_swapped_immune,
                        ) = adjust_pr(
                            out_cell_type[idx_start:idx_end, :],
                            comp_estimated[i_batch, :],
                            comp_out_patch,
                            opts.data.cell_types,
                            ["B", "Myeloid", "T"],
                            ignore_idx=high_conf_imm,
                            scale=config.alpha,
                        )

                        # ensure consistency with expressions
                        if len(idx_swapped_immune) > 0:
                            idx_swapped_immune = [
                                x + idx_start for x in idx_swapped_immune
                            ]
                            out_expr[idx_swapped_immune] = out_expr_immune[
                                idx_swapped_immune
                            ]

                        if config.is_invasive:
                            ct_index_inv = [opts.data.cell_types.index("Malignant")]

                            # malignant
                            (
                                adjusted_out_cell_type_patch_invasive,
                                idx_swapped_invasive_all,
                                idx_swapped_invasive,
                            ) = adjust_pr(
                                out_cell_type[idx_start:idx_end, :],
                                comp_estimated[i_batch, :],
                                comp_out_patch,
                                opts.data.cell_types,
                                ["Malignant", "Epithelial"],
                                ignore_idx=[],
                                scale=10000,
                            )

                            if len(idx_swapped_invasive) > 0:
                                idx_swapped_invasive = [
                                    x + idx_start for x in idx_swapped_invasive
                                ]
                                out_expr[idx_swapped_invasive] = out_expr_invasive[
                                    idx_swapped_invasive
                                ]

                        adjusted_out_cell_type_patch = (
                            adjusted_out_cell_type_patch_immune.copy()
                        )
                        if config.is_invasive:
                            for index in idx_swapped_invasive_all:
                                adjusted_out_cell_type_patch[index] = (
                                    adjusted_out_cell_type_patch_invasive[index]
                                )

                        adjusted_out_cell_type.extend(adjusted_out_cell_type_patch)

                        # neighbourhood compositions
                        comp_estimated_sum += n_cells_batch * comp_estimated[i_batch, :]

                        all_comp_est[all_comp_est_i, :] = (
                            n_cells_batch
                            * comp_estimated[i_batch, :].detach().cpu().numpy()
                        )

                        repeated_by_cells = np.tile(
                            all_comp_est[all_comp_est_i, :], (n_cells_batch, 1)
                        )

                        # also for each corresponding cell
                        if all_comp_est_by_cell is None:
                            all_comp_est_by_cell = repeated_by_cells.copy()
                        else:
                            all_comp_est_by_cell = np.vstack(
                                (all_comp_est_by_cell, repeated_by_cells)
                            )

                        all_comp_est_i += 1

                # save predicted expr
                out_expr = (1 / opts.data.expr_scale) * out_expr.detach().cpu().numpy()

                if all_expr is None:
                    all_expr = out_expr.copy()
                else:
                    all_expr = np.vstack((all_expr, out_expr))

                # nuclei areas
                if all_area is None:
                    all_area = batch_area.detach().cpu().numpy()
                else:
                    all_area = np.hstack((all_area, batch_area.detach().cpu().numpy()))

                # cell IDs
                patch_ids_pc = list(patch_ids_pc.detach().cpu().numpy())
                all_ids.extend(patch_ids_pc)

            # duplicates from overlapped regions - keep the prediction with largest area
            combined = np.vstack((np.array(all_ids), all_area, np.arange(len(all_ids))))
            combined = np.transpose(combined)
            combined_df = pd.DataFrame(
                combined, index=np.arange(len(all_ids)), columns=["id", "area", "index"]
            )
            sorted_df = combined_df.sort_values(by="area", ascending=False)
            # Drop duplicate rows based on the 'id' column, keeping only the first occurrence (largest area)
            unique_df = sorted_df.drop_duplicates(subset="id", keep="first")

            #TODO meets min area
            # unique_df = unique_df[unique_df['area'] >= opts.data.min_nuc_area]

            unique_indices = unique_df["index"].astype(int).tolist()

            unique_indices = sorted(unique_indices)
            all_ids = [all_ids[ui] for ui in unique_indices]
            
            all_expr = all_expr[unique_indices, :]

            # save predicted expr
            df_all_expr = pd.DataFrame(all_expr, index=all_ids, columns=gene_names)
            fp_expr_out = test_output_dir + "epoch_%d_expr.csv" % test_epoch
            df_all_expr.to_csv(fp_expr_out)
            print(f"Saved predicted expressions of {len(unique_indices)} cells to {fp_expr_out}")


logging.info("Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        default="configs/config_demo_inference.json",
        type=str,
        help="config file path",
    )
    parser.add_argument(
        "--epoch",
        default=-2,
        type=int,
        help="test model from this epoch, -1 for last, -2 for all",
    )
    parser.add_argument(
        "--mode",
        default="test",
        type=str,
        help="test or val",
    )
    parser.add_argument(
        "--fold_id",
        default=1,
        type=int,
        help="which cross-validation fold",
    )
    parser.add_argument(
        "--alpha",
        default=2,
        type=float,
        help="hyperparameter for neighbourhood composition refinement",
    )
    parser.add_argument("--demo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_sc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--is_invasive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no_normstain", action=argparse.BooleanOptionalAction, default=False)

    config = parser.parse_args()
    main(config)
