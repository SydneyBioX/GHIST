import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import sys
import os
import tifffile
import natsort
import h5py
from tqdm import tqdm
import torchvision
import imageio

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
import torch.nn.functional as F

from .utils import load_image


def load_data_h5(fp):
    h5f = h5py.File(fp, "r")
    nuclei_types_idx = h5f["data"][:]
    nuclei_types_ids = h5f["ids"][:]
    h5f.close()
    return nuclei_types_idx, nuclei_types_ids


def check_path(d):
    if not os.path.exists(d):
        sys.exit("Invalid file path %s" % d)


def get_folds(size, fold=1, folds=5):
    divisions = np.linspace(0, size, folds + 1, dtype=int)
    val_width = int(size / 10)
    # print(divisions)

    ws_test = divisions[fold - 1]
    we_test = divisions[fold]

    wp_test = np.arange(ws_test, we_test)

    if we_test == size:
        wp_val = np.arange(0, val_width)
    else:
        wp_val = np.arange(we_test, we_test + val_width)

    wp_train = np.arange(size)

    mask = np.isin(wp_train, wp_test, invert=True)
    wp_train = wp_train[mask]
    mask = np.isin(wp_train, wp_val, invert=True)
    wp_train = wp_train[mask]

    return wp_train, wp_val, wp_test


def find_patch_coordinates(w1, w2, patch_width=256, overlap=30):
    coordinates = []
    step_size = patch_width - overlap
    current_coord = w1

    while current_coord < w2:
        coordinates.append(min(current_coord, w2 - patch_width))
        current_coord += step_size

    return coordinates


def get_input_data(
    fp_nuc_seg,
    fp_hist,
    fp_expr,
    fp_cell_type,
    fp_nuc_sizes,
    n_classes,
    mode,
    opts_data,
    fold_id,
    hsize,
    wsize,
    overlap,
    gene_names,
):

    # cell gene expressions
    df_expr = pd.read_csv(fp_expr, index_col=0)
    df_expr = df_expr[["cell_id"] + gene_names]

    if ".h5" in fp_cell_type:
        # cell types and cell IDs
        idx_ct, ids_ct = load_data_h5(fp_cell_type)
        ids_ct = ids_ct.astype(np.uint32)

        # Drop IDs with invalid class
        data_ct = np.vstack((idx_ct, ids_ct)).T
        df_ct = pd.DataFrame(data_ct, columns=["ct_idx", "c_ids"], index=range(len(idx_ct)))
        df_ct = df_ct.set_index(["c_ids"])

    elif ".csv" in fp_cell_type:
        df_ct = pd.read_csv(fp_cell_type, index_col="c_ids")
        
    df_ct = df_ct[df_ct["ct_idx"].isin(range(n_classes))]
    df_ct["ct_idx"] = df_ct["ct_idx"] + 1
    print(f"Cell type data shape, {df_ct.shape}")

    nuclei = load_image(fp_nuc_seg)
    hist = load_image(fp_hist)
    
    hist_h = hist.shape[0]
    hist_w = hist.shape[1]

    print(f"Histology image {hist.shape}, Nuclei {nuclei.shape}")

    wp_train, wp_val, wp_test = get_folds(hist_h, fold=fold_id, folds=5)

    # whole_h = hist_h
    whole_w = hist_w

    # intersecting cells in cell type and segmentation
    if mode == "train":
        fp_all_intersect = "cell_ids_train_%d.txt" % (fold_id)
        wp = wp_train
    elif mode == "test":
        fp_all_intersect = "cell_ids_test_%d.txt" % (fold_id)
        wp = wp_test
    else:
        fp_all_intersect = "cell_ids_val_%d.txt" % (fold_id)
        wp = wp_val

    nuclei_fold = nuclei[wp, :]

    ids_seg = np.unique(nuclei_fold)
    ids_seg = ids_seg[ids_seg != 0]

    all_intersect = list(set(ids_seg) & set(list(df_ct.index)))

    # meets min size req
    df_sizes = pd.read_csv(fp_nuc_sizes, index_col=0)
    min_nuc_size = opts_data.min_nuc_area
    if "sizes" in df_sizes.columns.tolist():
        df_sizes = df_sizes[df_sizes["sizes"] >= min_nuc_size]
    else:
        df_sizes = df_sizes[df_sizes["orig_size"] >= min_nuc_size]

    ids_meet_min = df_sizes.index.tolist()

    all_intersect = list(set(all_intersect) & set(list(ids_meet_min)))

    all_intersect = list(set(all_intersect) & set(df_expr["cell_id"].tolist()))

    all_intersect = natsort.natsorted(all_intersect)

    with open(fp_all_intersect, "w") as f:
        for line in all_intersect:
            f.write(f"{line}\n")

    n_cells = len(all_intersect)
    print(f"{n_cells} cells")

    # get expr of the cells in patch
    df_expr = df_expr[df_expr["cell_id"].isin(all_intersect)]
    assert list(df_expr.index) == df_expr["cell_id"].tolist()
    df_expr = df_expr.drop("cell_id", axis=1)
    df_expr = opts_data.expr_scale * np.log1p(df_expr)
    gene_names = df_expr.columns

    # overlapping patches

    # h_starts = list(np.arange(0, whole_h - hsize, hsize - overlap))
    # h_starts.append(whole_h - hsize)
    w_starts = list(np.arange(0, whole_w - wsize, wsize - overlap))
    w_starts.append(whole_w - wsize)

    coord_idx = find_patch_coordinates(0, len(wp), patch_width=hsize, overlap=overlap)
    # print(wp.shape)
    h_starts = wp[coord_idx]
    # print(h_starts.shape)
    print("Patches min/max coords", h_starts.min(), h_starts.max()+hsize)

    print("Getting valid patches")
    coords_starts = [(x, y) for x in h_starts for y in w_starts]
    coords_starts_valid = []

    for hs, ws in tqdm(coords_starts):
        nuclei_p = nuclei[hs : hs + hsize, ws : ws + wsize]

        ids_seg = np.unique(nuclei_p)
        ids_seg = ids_seg[ids_seg != 0]
        valid_ids = list(set(ids_seg) & set(all_intersect))
        invalid_ids = list(set(ids_seg) - set(valid_ids))
        dictionary = dict(zip(invalid_ids, [0] * len(invalid_ids)))
        nuclei_valid = np.copy(nuclei_p)
        for k, v in dictionary.items():
            nuclei_valid[nuclei_p == k] = v

        if np.sum(nuclei_valid) > 0:
            coords_starts_valid.append((hs, ws))

    # n_patches = len(coords_starts_valid)
    # print(f"{n_patches} patches from {len(coords_starts)} are valid")

    df_ct = df_ct.loc[all_intersect, :]

    # Initialize min and max values with the first element of the list
    min_hs, min_ws = coords_starts_valid[0]
    max_hs, max_ws = coords_starts_valid[0]

    # Iterate through the list to find min and max values
    for hs, ws in coords_starts_valid:
        if hs < min_hs:
            min_hs = hs
        if hs > max_hs:
            max_hs = hs
        if ws < min_ws:
            min_ws = ws
        if ws > max_ws:
            max_ws = ws

    # print("Minimum hs:", min_hs)
    # print("Maximum hs:", max_hs)
    # print("Minimum ws:", min_ws)
    # print("Maximum ws:", max_ws)
    # print(np.min(wp), np.max(wp))

    # standardisation of histology
    print("Standardisation")
    fp_norms = "standardisation_hist_fold_%d.npy" % (fold_id)

    if mode == "train":
        if not os.path.exists(fp_norms):
            hist_means = np.zeros(3)
            hist_stds = np.zeros(3)
            for hs, ws in tqdm(coords_starts):

                hist_p = hist[hs : hs + hsize, ws : ws + wsize]

                hist_means += np.mean(hist_p, (0, 1))
                hist_stds += np.std(hist_p, (0, 1))

            hist_means = hist_means / len(coords_starts)
            hist_stds = hist_stds / len(coords_starts)

            norms_hist = np.vstack((hist_means, hist_stds))
            np.save(fp_norms, norms_hist)

    norms_hist = np.load(fp_norms)

    return coords_starts_valid, hist, nuclei, all_intersect, df_ct, df_expr, norms_hist


class DataProcessing(data.Dataset):
    def __init__(
        self,
        opts_data_sources,
        opts_data,
        classes,
        gene_names,
        fold_id=1,
        mode="train",
    ):

        check_path(opts_data_sources.fp_nuc_seg)
        check_path(opts_data_sources.fp_hist)
        check_path(opts_data_sources.fp_expr)
        check_path(opts_data_sources.fp_cell_type)
        check_path(opts_data_sources.fp_nuc_sizes)
        
        self.classes = classes
        self.n_classes = len(self.classes)
        self.mode = mode
        self.fold_id = fold_id

        self.max_n_pp = opts_data.max_n_pp
        self.hsize = opts_data.hsize
        self.wsize = opts_data.wsize

        if mode == "train":
            overlap = 0
        else:
            overlap = opts_data.overlap

        (
            coords_starts_valid,
            self.hist,
            self.nuclei,
            self.all_intersect,
            self.df_ct,
            self.df_expr,
            norms_hist,
        ) = get_input_data(
            opts_data_sources.fp_nuc_seg,
            opts_data_sources.fp_hist,
            opts_data_sources.fp_expr,
            opts_data_sources.fp_cell_type,
            opts_data_sources.fp_nuc_sizes,
            self.n_classes,
            self.mode,
            opts_data,
            fold_id,
            self.hsize,
            self.wsize,
            overlap,
            gene_names,
        )

        self.norms_hist = norms_hist.copy()
        self.coords_starts = coords_starts_valid

        self.n_patches = len(self.coords_starts)

        # Augmentation
        self.tfs = v2.Compose(
            [
                v2.ToImage(), 
                # v2.ToDtype(torch.float32, scale=True),
                v2.RandomHorizontalFlip(0.5),
                v2.RandomVerticalFlip(0.5),
                v2.RandomApply([v2.RandomRotation((90, 90))], p=0.25),
                v2.RandomApply([v2.RandomRotation((180, 180))], p=0.25),
                v2.RandomApply([v2.RandomRotation((270, 270))], p=0.25),
                v2.ToDtype(torch.float32),
            ]
        )

        self.tfs_test = v2.Compose(
            [
                v2.ToImage(), 
                # v2.ToDtype(torch.float32, scale=True),
                v2.ToDtype(torch.float32),
            ]
        )

    def __len__(self):
        "Denotes the total number of samples"
        return self.n_patches

    def __getitem__(self, index):
        "Generates one sample of data"

        hs, ws = self.coords_starts[index]

        nuclei_patch = self.nuclei[hs : hs + self.hsize, ws : ws + self.wsize]
        hist_patch = self.hist[hs : hs + self.hsize, ws : ws + self.wsize]

        ids_seg = np.unique(nuclei_patch)
        ids_seg = ids_seg[ids_seg != 0]

        # make sure cells have valid data
        valid_ids = list(set(ids_seg) & set(self.all_intersect))
        invalid_ids = list(set(ids_seg) - set(valid_ids))
        dictionary = dict(zip(invalid_ids, [0] * len(invalid_ids)))
        nuclei_valid = np.copy(nuclei_patch)
        for k, v in dictionary.items():
            nuclei_valid[nuclei_patch == k] = v

        # map to cell type nuclei map
        dictionary = dict(zip(valid_ids, self.df_ct.loc[valid_ids, "ct_idx"].tolist()))
        types_patch = np.copy(nuclei_valid)
        for k, v in dictionary.items():
            types_patch[nuclei_valid == k] = v

        # standardisation
        means = np.expand_dims(self.norms_hist[0, :], (0, 1))
        stds = np.expand_dims(self.norms_hist[1, :], (0, 1))
        hist_patch = hist_patch - means
        hist_patch = hist_patch / stds

        patch_ids = np.unique(nuclei_valid)
        patch_ids = patch_ids[patch_ids != 0]

        expr = self.df_expr.loc[patch_ids, :].to_numpy()

        n_cells = expr.shape[0]
        max_n_pp = self.max_n_pp
        if n_cells > max_n_pp:
            print("exceeds max_n_pp cells, try increasing the value in config")
        expr_pad = np.zeros((max_n_pp, expr.shape[1]))
        expr_pad[:n_cells, :] = expr.copy()

        # cell type labels (previously added 1 to df_ct such that 0 is bkg)
        gt_types = self.df_ct.loc[patch_ids, "ct_idx"].to_numpy() - 1
        gt_types_pad = np.zeros(max_n_pp)
        gt_types_pad[:n_cells] = gt_types.copy()
        gt_types_torch = torch.from_numpy(gt_types_pad).long()

        # cell IDs in patch
        patch_ids_pad = np.zeros(max_n_pp)
        patch_ids_pad[:n_cells] = patch_ids.copy()
        patch_ids_torch = torch.from_numpy(patch_ids_pad).long()

        # number of cells in patch
        n_cells = np.array([n_cells])
        n_cells_torch = torch.from_numpy(n_cells).long()

        x_input = np.concatenate(
            (
                np.expand_dims(nuclei_valid, -1),
                np.expand_dims(types_patch, -1),
                hist_patch,
            ),
            -1,
        )

        # augmentation
        if self.mode == "train":
            x_input = self.tfs(x_input)
        else:
            x_input = self.tfs_test(x_input)

        nuclei_torch = x_input[0, :, :].type(torch.LongTensor)
        types_patch_torch = x_input[1, :, :].type(torch.LongTensor)
        hist_torch = x_input[2:, :, :]

        expr_torch = torch.from_numpy(expr_pad).float()

        return (
            nuclei_torch,
            types_patch_torch,
            hist_torch,
            expr_torch,
            n_cells_torch,
            gt_types_torch,
            patch_ids_torch,
        )
