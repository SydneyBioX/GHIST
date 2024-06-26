import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import *
from .backbone import *


class Framework(nn.Module):

    def __init__(
        self,
        n_classes,
        n_genes,
        emb_dim,
        n_ref,
        device,
        use_sc,
        in_channels=3,
    ):
        super(Framework, self).__init__()

        self.cnn = Backbone(
            n_channels=in_channels,
            bilinear=True,
            is_deconv=True,
            is_batchnorm=True,
            n_classes=n_classes + 1,
        )

        dim_fv = 384 * 2
        num_heads = 8
        self.hidden_size = emb_dim
        self.device = device
        self.use_sc = use_sc

        self.embed_hist = Embed(dim_fv, self.hidden_size)

        self.estimate_comp = MLPSoftmax(self.hidden_size, self.hidden_size, n_classes)

        if self.use_sc:
            self.mlp_weights = MLPSoftmax(self.hidden_size, self.hidden_size, n_ref)
            self.mlp_weights_immune = MLPSoftmax(self.hidden_size, self.hidden_size, n_ref)
            self.mlp_weights_invasive = MLPSoftmax(self.hidden_size, self.hidden_size, n_ref)
        else:
            self.mlp_offset = MLP(self.hidden_size, self.hidden_size, n_genes)
            self.mlp_offset_immune = MLP(self.hidden_size, self.hidden_size, n_genes)
            self.mlp_offset_invasive = MLP(self.hidden_size, self.hidden_size, n_genes)
    
        self.refine_expr = CrossAttention(
            n_classes, n_genes, self.hidden_size, num_heads=num_heads
        )

        self.refine_expr_immune = CrossAttention(
            n_classes, n_genes, self.hidden_size, num_heads=num_heads
        )

        self.refine_expr_invasive = CrossAttention(
            n_classes, n_genes, self.hidden_size, num_heads=num_heads
        )

        self.mlp_hist = MLP(self.hidden_size, self.hidden_size, n_classes)
        self.mlp_genes = MLP(n_genes, self.hidden_size, n_classes)

    def forward(
        self,
        x_hist,
        nuclei_mask,
        n_cells,
        ref_orig,
        batch_ct=None,
        batch_expr=None,
        patch_ids=None,
        do_st_mlp=True,
    ):
        out_map, hd1, h1 = self.cnn(x_hist)

        # [batch, n_cells, ...] to [batch*n_cells, ...]
        n_cells_total = torch.sum(n_cells)
        all_fv = torch.zeros((n_cells_total, 2 * (hd1.shape[1] + h1.shape[1])))
        all_fv = all_fv.to(self.device)

        all_area = torch.zeros(n_cells_total).to(self.device)

        # patch level features
        patch_area_hd1 = hd1.shape[2] * hd1.shape[3]
        patch_area_h1 = h1.shape[2] * h1.shape[3]
        fv_hd1 = torch.sum(hd1, (2, 3)) / patch_area_hd1
        fv_h1 = torch.sum(h1, (2, 3)) / patch_area_h1

        i_fv = 0
        batch_size = x_hist.shape[0]

        # features per cell/nucleus
        for i_batch in range(batch_size):
            c_mask_all = nuclei_mask[i_batch]
            cids = torch.unique(c_mask_all, sorted=True)[1:]
            for cid in cids:
                c_mask = torch.where(c_mask_all == cid, 1, 0)
                c_area = torch.sum(c_mask)

                # aggregate (mean) of features in nuclei mask -> cell's fv
                c_fv_hd1 = c_mask * hd1[i_batch]
                c_fv_hd1 = torch.sum(c_fv_hd1, (1, 2)) / c_area
                c_fv_h1 = c_mask * h1[i_batch]
                c_fv_h1 = torch.sum(c_fv_h1, (1, 2)) / c_area

                all_fv[i_fv, :] = torch.cat(
                    (c_fv_hd1, c_fv_h1, fv_hd1[i_batch], fv_h1[i_batch])
                )
                all_area[i_fv] = c_area

                i_fv += 1

        # cell level embeddings
        embeddings = self.embed_hist(all_fv)

        # get vectors arranged per cell and patch level embeddings
        for i_batch in range(batch_size):
            n_cells_batch = int(n_cells[i_batch])
            if i_batch == 0:
                if batch_ct is not None:
                    batch_ct_pc = batch_ct[0, :n_cells_batch]
                if batch_expr is not None:
                    batch_expr_pc = batch_expr[0, :n_cells_batch, :]
                if patch_ids is not None:
                    patch_ids_pc = patch_ids[0, :n_cells_batch]

                embeddings_patches = torch.mean(embeddings[:n_cells_batch, :], 0)
                embeddings_patches = embeddings_patches.unsqueeze(0)

            else:
                if batch_ct is not None:
                    batch_ct_pc = torch.cat(
                        (batch_ct_pc, batch_ct[i_batch, :n_cells_batch]), 0
                    )
                if batch_expr is not None:
                    batch_expr_pc = torch.cat(
                        (batch_expr_pc, batch_expr[i_batch, :n_cells_batch, :]), 0
                    )
                if patch_ids is not None:
                    patch_ids_pc = torch.cat(
                        (patch_ids_pc, patch_ids[i_batch, :n_cells_batch]), 0
                    )

                sum_idx_prev = torch.sum(n_cells[:i_batch]).item()

                embeddings_sample = torch.mean(
                    embeddings[sum_idx_prev : sum_idx_prev + n_cells_batch, :], 0
                )
                embeddings_sample = embeddings_sample.unsqueeze(0)
                embeddings_patches = torch.cat(
                    (embeddings_patches, embeddings_sample), 0
                )

        # composition prediction [batch_size, n_classes]
        comp_estimated = self.estimate_comp(embeddings_patches)

        # tile composition to use for expression refinement
        for i_batch in range(n_cells.shape[0]):
            n_cells_batch = int(n_cells[i_batch])

            if n_cells_batch > 0:

                comp_tiled = torch.tile(
                    comp_estimated[i_batch, :], (n_cells_batch, 1)
                )

                if i_batch == 0:
                    comp_tiled_all = comp_tiled.clone()
                else:
                    comp_tiled_all = torch.cat(
                        (comp_tiled_all, comp_tiled), 0
                    )

        # auxiliary cell type classification
        out_cell_type, _ = self.mlp_hist(embeddings)

        m = nn.ReLU()

        if self.use_sc:
                
            # [n_cells_total, n_ref]
            ref_weights = self.mlp_weights(embeddings)
            # [n_cells_total, n_ref, n_genes]
            ref = ref_orig.unsqueeze(0).repeat(n_cells_total, 1, 1)

            ref_weighted = ref_weights.unsqueeze(-1) * ref
            ref_weighted = torch.sum(ref_weighted, 1)

            ref_offsets = self.refine_expr(comp_tiled_all, ref_weighted)

            out_expr = ref_weighted + ref_offsets

            out_expr = m(out_expr)

            # cell type groups

            # related to immune cells

            # [n_cells_total, n_ref]
            ref_weights_immune = self.mlp_weights_immune(embeddings)
            # [n_cells_total, n_ref, n_genes]
            ref_immune = ref_orig.unsqueeze(0).repeat(n_cells_total, 1, 1)

            ref_weighted_immune = ref_weights_immune.unsqueeze(-1) * ref_immune
            ref_weighted_immune = torch.sum(ref_weighted_immune, 1)

            ref_offsets_immune = self.refine_expr_immune(
                comp_tiled_all, ref_weighted_immune
            )
            out_expr_immune = ref_weighted_immune + ref_offsets_immune

            out_expr_immune = m(out_expr_immune)

            # related to invasiveness

            # [n_cells_total, n_ref]
            ref_weights_invasive = self.mlp_weights_invasive(embeddings)
            # [n_cells_total, n_ref, n_genes]
            ref_invasive = ref_orig.unsqueeze(0).repeat(n_cells_total, 1, 1)

            ref_weighted_invasive = ref_weights_invasive.unsqueeze(-1) * ref_invasive
            ref_weighted_invasive = torch.sum(ref_weighted_invasive, 1)

            ref_offsets_invasive = self.refine_expr_invasive(
                comp_tiled_all, ref_weighted_invasive
            )
            out_expr_invasive = ref_weighted_invasive + ref_offsets_invasive

            out_expr_invasive = m(out_expr_invasive)

        else:

            ref_offsets, _ = self.mlp_offset(embeddings)
            ref_offsets = m(ref_offsets)
            ref_offsets = self.refine_expr(comp_tiled_all, ref_offsets)
            out_expr = m(ref_offsets)

            # cell type groups

            # related to immune cells
            ref_offsets_immune, _ = self.mlp_offset_immune(embeddings)
            ref_offsets_immune = m(ref_offsets_immune)
            ref_offsets_immune = self.refine_expr_immune(
                comp_tiled_all, ref_offsets_immune
            )
            out_expr_immune = m(ref_offsets_immune)

            # related to invasiveness
            ref_offsets_invasive, _ = self.mlp_offset_invasive(embeddings)
            ref_offsets_invasive = m(ref_offsets_invasive)
            ref_offsets_invasive = self.refine_expr_invasive(
                comp_tiled_all, ref_offsets_invasive
            )
            out_expr_invasive = m(ref_offsets_invasive)

        # predict cell type based on expressions - ground truth and predicted
        out_cell_type_expr, fv_cell_type_expr = self.mlp_genes(out_expr)

        if do_st_mlp is False:
            out_cell_type_gt_expr = None
            fv_cell_type_gt_expr = None
        elif batch_expr is not None:
            out_cell_type_gt_expr, fv_cell_type_gt_expr = self.mlp_genes(batch_expr_pc)
        else:
            out_cell_type_gt_expr = None
            fv_cell_type_gt_expr = None
            batch_expr_pc = None

        if batch_ct is None:
            batch_ct_pc = None

        if patch_ids is not None:
            return (
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
                all_area,
                patch_ids_pc,
            )
        else:
            return (
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
                all_area,
            )
