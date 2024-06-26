import argparse
import logging
import os
import sys
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
import natsort

from dataio.dataset_input import DataProcessing
from model.model import Framework
from utils.utils import *


def main(config):
    opts = json_file_to_pyobj(config.config_file)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    device = get_device(opts.model.gpu_ids)

    # Create experiment directories
    if config.demo is True or config.resume_epoch is not None:
        make_new = False
    else:
        make_new = True

    timestamp = get_experiment_id(
        make_new, opts.experiment_dirs.load_dir, config.fold_id
    )
    experiment_path = f"experiments/{timestamp}"
    os.makedirs(experiment_path + "/" + opts.experiment_dirs.model_dir, exist_ok=True)

    # Save copy of current config file
    shutil.copyfile(
        config.config_file, experiment_path + "/" + os.path.basename(config.config_file)
    )

    # Set up the model
    logging.info("Initialising model")

    classes = opts.data.cell_types
    n_classes = len(classes)
    print(classes)
    print(f"Num cell types {n_classes}")

    df_expr = pd.read_csv(opts.data_sources.fp_expr, index_col=0)
    gene_names = df_expr.columns.tolist()[1:]
    n_genes = len(gene_names)
    print(f"{n_genes} genes")

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

    model = Framework(
        n_classes,
        n_genes,
        opts.model.emb_dim,
        n_ref,
        device,
        config.use_sc
    )

    # Dataloader
    logging.info("Preparing data")

    train_dataset = DataProcessing(
        opts.data_sources, opts.data, classes, gene_names, config.fold_id, mode="train"
    )
    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=opts.training.batch_size,
        shuffle=True,
        num_workers=opts.data.num_workers,
        drop_last=True,
    )

    n_train_examples = len(dataloader)
    logging.info("Total number of training batches: %d" % n_train_examples)

    # Optimiser
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opts.training.learning_rate,
        betas=(opts.training.beta1, opts.training.beta2),
        weight_decay=opts.training.weight_decay,
        eps=opts.training.eps,
    )

    global_step = 0

    # Starting epoch
    if config.resume_epoch is not None:
        initial_epoch = config.resume_epoch
    else:
        initial_epoch = 0

    # Restore saved model
    if config.resume_epoch is not None or config.demo:
        logging.info("Resume training")

        if config.demo:
            load_path = (
                experiment_path
                + "/"
                + opts.experiment_dirs.model_dir
                + "/model.pth"
            )
        else:
            load_path = (
                experiment_path
                + "/"
                + opts.experiment_dirs.model_dir
                + "/epoch_%d_model.pth" % (config.resume_epoch)
            )
        checkpoint = torch.load(load_path)

        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint["epoch"]
        print("Loaded " + load_path)

        model.to(device)

        if config.demo:
            load_path = (
                experiment_path
                + "/"
                + opts.experiment_dirs.model_dir
                + "/optim.pth"
            )
        else:
            load_path = (
                experiment_path
                + "/"
                + opts.experiment_dirs.model_dir
                + "/epoch_%d_optim.pth" % (config.resume_epoch)
            )
        checkpoint = torch.load(load_path)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Loaded " + load_path)

    else:
        model.to(device)

    logging.info("Begin training")

    loss_map = nn.CrossEntropyLoss(reduction="mean")
    loss_ct_hist = nn.CrossEntropyLoss(reduction="mean")
    loss_expr_ct = nn.CrossEntropyLoss(reduction="mean")
    loss_expr_ct_embed = nn.CosineEmbeddingLoss(reduction="mean")
    loss_expr = nn.MSELoss(reduction="mean")
    loss_expr_immune = nn.MSELoss(reduction="mean")
    loss_expr_invasive = nn.MSELoss(reduction="mean")
    loss_logits = nn.MSELoss(reduction="mean")
    loss_comp_est = nn.KLDivLoss(reduction="batchmean")
    loss_comp_gt = nn.KLDivLoss(reduction="batchmean")

    for epoch in range(initial_epoch, opts.training.total_epochs):
        print(f"Epoch: {epoch+1}")
        model.train()

        optimizer.param_groups[0]["lr"] = opts.training.learning_rate * (
            1 - epoch / opts.training.total_epochs
        )

        loss_epoch = 0
        loss_epoch_map = 0
        loss_epoch_ct_hist = 0
        loss_epoch_expr_ct = 0
        loss_epoch_expr_ct_embed = 0
        loss_epoch_expr = 0
        loss_epoch_expr_immune = 0
        loss_epoch_expr_invasive = 0
        loss_epoch_logits = 0
        loss_epoch_comp_est = 0
        loss_epoch_comp_gt = 0

        pbar = tqdm(dataloader)
        loss_total = None
        for (
            batch_nuclei,
            batch_type_patch,
            batch_he_img,
            batch_expr,
            batch_n_cells,
            batch_ct,
            _,
        ) in pbar:
            optimizer.zero_grad()

            batch_nuclei = batch_nuclei.to(device)
            batch_type_patch = batch_type_patch.to(device)
            batch_he_img = batch_he_img.to(device)
            batch_expr = batch_expr.to(device)
            batch_n_cells = batch_n_cells.to(device)
            batch_ct = batch_ct.to(device)

            (
                out_cell_type,
                out_map,
                batch_ct_pc,
                out_expr,
                out_expr_immune,
                out_expr_invasive,
                out_cell_type_expr,
                fv_cell_type_expr,
                out_cell_type_gt_expr,
                fv_cell_type_gt_expr,
                batch_expr_pc,
                comp_estimated,
                _,
            ) = model(
                batch_he_img,
                batch_nuclei,
                batch_n_cells,
                expr_ref_torch,
                batch_ct,
                batch_expr,
            )

            if batch_ct_pc.shape[0] == 0:
                continue

            loss_expr_val = loss_expr(out_expr, batch_expr_pc)
            loss_map_val = loss_map(out_map, batch_type_patch)
            loss_ct_hist_val = loss_ct_hist(out_cell_type, batch_ct_pc)
            loss_expr_ct_val = loss_expr_ct(out_cell_type_expr, batch_ct_pc)
            loss_expr_ct_embed_val = 100 * loss_expr_ct_embed(
                fv_cell_type_expr,
                fv_cell_type_gt_expr,
                target=torch.ones(batch_ct_pc.size(0)).to(device),
            )
            loss_logits_val = loss_logits(out_cell_type_expr, out_cell_type_gt_expr)

            imm_ct_idx_1 = classes.index("B")
            imm_ct_idx_2 = classes.index("Myeloid")
            inv_ct_idx = classes.index("Malignant")

            imm_mask = torch.isin(
                batch_ct_pc, torch.tensor([imm_ct_idx_1, imm_ct_idx_2]).to(device)
            )
            imm_idx = torch.where(imm_mask)[0]
            if imm_idx.shape[0] > 0:
                loss_expr_immune_val = (1 / n_classes) * loss_expr_immune(
                    out_expr_immune[imm_idx, :], batch_expr_pc[imm_idx, :]
                )
            else:
                loss_expr_immune_val = torch.tensor(0.0).to(device)

            inv_idx = torch.where(batch_ct_pc == inv_ct_idx)[0]
            if inv_idx.shape[0] > 0:
                loss_expr_invasive_val = (1 / n_classes) * loss_expr_invasive(
                    out_expr_invasive[inv_idx, :], batch_expr_pc[inv_idx, :]
                )
            else:
                loss_expr_invasive_val = torch.tensor(0.0).to(device)

            # composition every patch
            comp_estimated_sum = torch.zeros(n_classes).to(device)
            for i_batch in range(opts.training.batch_size):
                n_cells_batch = int(batch_n_cells[i_batch])

                if n_cells_batch > 0:
                    comp_estimated_sum += n_cells_batch * comp_estimated[i_batch, :]

            comp_estimated_sum = comp_estimated_sum / torch.sum(batch_n_cells)

            # composition losses
            comp_gt = torch.nn.functional.one_hot(batch_ct_pc, num_classes=n_classes)
            comp_gt = comp_gt.float()
            comp_gt = torch.mean(comp_gt, 0)
            comp_out_raw = torch.nn.functional.softmax(out_cell_type, dim=1)
            comp_out_raw = torch.argmax(comp_out_raw, 1)
            comp_out = torch.nn.functional.one_hot(comp_out_raw, num_classes=n_classes)
            comp_out = comp_out.float()
            comp_out = torch.mean(comp_out, 0)

            # KL DIVERGENCE
            kl_eps = 10e-12
            comp_estimated_kl = comp_estimated_sum + kl_eps
            comp_out_kl = comp_out + kl_eps
            comp_gt_kl = comp_gt + kl_eps

            comp_estimated_kl = F.softmax(comp_estimated_kl, dim=0)
            comp_out_kl = F.softmax(comp_out_kl, dim=0)
            comp_gt_kl = F.softmax(comp_gt_kl, dim=0)

            comp_estimated_log = torch.log(comp_estimated_kl)
            comp_out_log = torch.log(comp_out_kl)

            loss_comp_est_val = loss_comp_est(comp_estimated_log, comp_gt_kl)
            loss_comp_gt_val = loss_comp_gt(comp_out_log, comp_gt_kl)

            # sum all losses
            loss = (
                loss_map_val
                + loss_ct_hist_val
                + loss_expr_ct_val
                + loss_expr_val
                + loss_expr_immune_val
                + loss_expr_invasive_val
                + loss_expr_ct_embed_val
                + loss_logits_val
                + loss_comp_est_val
                + loss_comp_gt_val
            )

            loss.backward()

            loss_total = loss.item()

            loss_epoch += loss.mean().item()

            loss_epoch_map += loss_map_val.mean().item()
            loss_epoch_ct_hist += loss_ct_hist_val.mean().item()
            loss_epoch_expr_ct += loss_expr_ct_val.item()
            loss_epoch_expr += loss_expr_val.item()
            loss_epoch_expr_immune += loss_expr_immune_val.item()
            loss_epoch_expr_invasive += loss_expr_invasive_val.item()
            loss_epoch_expr_ct_embed += loss_expr_ct_embed_val.item()
            loss_epoch_logits += loss_logits_val.item()
            loss_epoch_comp_est += loss_comp_est_val.item()
            loss_epoch_comp_gt += loss_comp_gt_val.item()

            pbar.set_description(f"loss: {loss_total:.4f}")

            optimizer.step()

        print(
            "Epoch[{}/{}], Loss:{:.4f}".format(
                epoch + 1, opts.training.total_epochs, loss_epoch
            )
        )
        print(
            "REF_EXPR:{:.4f}, CT:{:.4f}, MAP:{:.4f}, EXPR_CT:{:.4f}, EXPR_IMM:{:.4f}, EXPR_INV:{:.4f}, EXPR_CT_FV:{:.4f}, EXPR_CT_LOGITS:{:.4f}, COMP_EST:{:.4f}, COMP_GT:{:.4f}".format(
                loss_epoch_expr,
                loss_epoch_ct_hist,
                loss_epoch_map,
                loss_epoch_expr_ct,
                loss_epoch_expr_immune,
                loss_epoch_expr_invasive,
                loss_epoch_expr_ct_embed,
                loss_epoch_logits,
                loss_epoch_comp_est,
                loss_epoch_comp_gt,
            )
        )

        # Save model
        if (epoch % opts.save_freqs.model_freq) == 0:
            save_path = f"{experiment_path}/{opts.experiment_dirs.model_dir}/epoch_{epoch+1}_model.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                },
                save_path,
            )
            logging.info("Model saved: %s" % save_path)
            save_path = f"{experiment_path}/{opts.experiment_dirs.model_dir}/epoch_{epoch+1}_optim.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                save_path,
            )
            logging.info("Optimiser saved: %s" % save_path)

        global_step += 1

    logging.info("Training finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        default="configs/config_demo.json",
        type=str,
        help="config file path",
    )
    parser.add_argument(
        "--resume_epoch",
        default=None,
        type=int,
        help="resume training from this epoch, set to None for new training",
    )
    parser.add_argument(
        "--fold_id",
        default=1,
        type=int,
        help="which cross-validation fold",
    )
    parser.add_argument("--demo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_sc", action=argparse.BooleanOptionalAction, default=False)

    config = parser.parse_args()
    main(config)
