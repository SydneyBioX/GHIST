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

import torchstain

from .utils import load_image


def norm_stain(target, to_transform):
    T = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x*255)
    ])

    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    
    normalizer.fit(T(target))

    t_to_transform = T(to_transform)
    norm, H, E = normalizer.normalize(I=t_to_transform, stains=True)

    norm = norm.detach().cpu().numpy()

    norm = norm.astype(np.uint8)

    return norm


def load_data_h5(fp):
    h5f = h5py.File(fp, "r")
    nuclei_types_idx = h5f["data"][:]
    nuclei_types_ids = h5f["ids"][:]
    h5f.close()
    return nuclei_types_idx, nuclei_types_ids


def check_path(d):
    if not os.path.exists(d):
        sys.exit("Invalid file path %s" % d)


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
    opts_data,
    fold_id,
    hsize,
    wsize,
    overlap,
    gene_names,
):

    nuclei = load_image(fp_nuc_seg)
    hist = load_image(fp_hist)

    hist_h = hist.shape[0]
    hist_w = hist.shape[1]

    print(f"Histology image {hist.shape}, Nuclei {nuclei.shape}")

    whole_w = hist_w

    wp = np.arange(0, hist_h)

    nuclei_fold = nuclei[wp, :]

    ids_seg = np.unique(nuclei_fold)
    ids_seg = ids_seg[ids_seg != 0]

    all_intersect = list(ids_seg)

    n_cells = len(all_intersect)
    print(f"{n_cells} cells")

    # overlapping patches
    w_starts = list(np.arange(0, whole_w - wsize, wsize - overlap))
    w_starts.append(whole_w - wsize)

    coord_idx = find_patch_coordinates(0, len(wp), patch_width=hsize, overlap=overlap)
    h_starts = wp[coord_idx]
    print("Patches min/max coords", h_starts.min(), h_starts.max()+hsize)

    print("Getting valid patches")
    coords_starts = [(x, y) for x in h_starts for y in w_starts]
    coords_starts_valid = []

    for hs, ws in tqdm(coords_starts):
        nuclei_p = nuclei[hs : hs + hsize, ws : ws + wsize]

        if np.sum(nuclei_p) > 0:
            coords_starts_valid.append((hs, ws))

    # n_patches = len(coords_starts_valid)
    # print(f"{n_patches} patches from {len(coords_starts)} are valid")

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

    # standardisation of histology
    fp_norms = "standardisation_hist_fold_%d.npy" % (fold_id)

    norms_hist = np.load(fp_norms)

    return coords_starts_valid, hist, nuclei, norms_hist


class DataProcessing(data.Dataset):
    def __init__(
        self,
        opts_data_sources,
        opts_data,
        gene_names,
        fold_id=1,
        normstain=True
    ):

        check_path(opts_data_sources.fp_nuc_seg)
        check_path(opts_data_sources.fp_hist)
        
        self.fold_id = fold_id
        self.normstain = normstain

        self.max_n_pp = opts_data.max_n_pp
        self.hsize = opts_data.hsize
        self.wsize = opts_data.wsize

        overlap = opts_data.overlap

        (
            coords_starts_valid,
            self.hist,
            self.nuclei,
            norms_hist,
        ) = get_input_data(
            opts_data_sources.fp_nuc_seg,
            opts_data_sources.fp_hist,
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

        self.tfs_test = v2.Compose(
            [
                v2.ToImage(), 
                # v2.ToDtype(torch.float32, scale=True),
                v2.ToDtype(torch.float32),
            ]
        )
        
        if self.normstain:
            self.target = load_image(opts_data_sources.fp_stain_target)

    def __len__(self):
        "Denotes the total number of samples"
        return self.n_patches

    def __getitem__(self, index):
        "Generates one sample of data"

        hs, ws = self.coords_starts[index]

        nuclei_patch = self.nuclei[hs : hs + self.hsize, ws : ws + self.wsize]
        hist_patch = self.hist[hs : hs + self.hsize, ws : ws + self.wsize]

        if self.normstain:
            hist_patch = norm_stain(self.target, hist_patch)
            
        ids_seg = np.unique(nuclei_patch)
        ids_seg = ids_seg[ids_seg != 0]

        # standardisation
        means = np.expand_dims(self.norms_hist[0, :], (0, 1))
        stds = np.expand_dims(self.norms_hist[1, :], (0, 1))
        hist_patch = hist_patch - means
        hist_patch = hist_patch / stds

        patch_ids = np.unique(nuclei_patch)
        patch_ids = patch_ids[patch_ids != 0]

        n_cells = len(patch_ids)
        max_n_pp = self.max_n_pp
        if n_cells > max_n_pp:
            print("exceeds max_n_pp cells, try increasing the value in config")

        # cell IDs in patch
        patch_ids_pad = np.zeros(max_n_pp)
        patch_ids_pad[:n_cells] = patch_ids.copy()
        patch_ids_torch = torch.from_numpy(patch_ids_pad).long()

        # number of cells in patch
        n_cells = np.array([n_cells])
        n_cells_torch = torch.from_numpy(n_cells).long()

        x_input = np.concatenate(
            (
                np.expand_dims(nuclei_patch, -1),
                hist_patch,
            ),
            -1,
        )

        x_input = self.tfs_test(x_input)

        nuclei_torch = x_input[0, :, :].type(torch.LongTensor)
        hist_torch = x_input[1:, :, :]

        return (
            nuclei_torch,
            hist_torch,
            n_cells_torch,
            patch_ids_torch,
        )
