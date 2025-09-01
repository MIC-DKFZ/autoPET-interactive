from __future__ import annotations
import multiprocessing
import os
from typing import List
from pathlib import Path
from warnings import warn

import cc3d
import torch
from scipy import ndimage
import edt as fastedt
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, subfiles
from nnunetv2.configuration import default_num_processes
from nnunetv2.training.dataloading.nnInteractive_clicks import PointInteraction_stub
from scipy.ndimage import gaussian_filter

def _convert_to_npy(npz_file: str, unpack_segmentation: bool = True, overwrite_existing: bool = False,
                    verify_npy: bool = False, fail_ctr: int = 0) -> None:
    data_npy = npz_file[:-3] + "npy"
    seg_npy = npz_file[:-4] + "_seg.npy"
    try:
        npz_content = None  # will only be opened on demand

        if overwrite_existing or not isfile(data_npy):
            try:
                npz_content = np.load(npz_file) if npz_content is None else npz_content
            except Exception as e:
                print(f"Unable to open preprocessed file {npz_file}. Rerun nnUNetv2_preprocess!")
                raise e
            np.save(data_npy, npz_content['data'])

        if unpack_segmentation and (overwrite_existing or not isfile(seg_npy)):
            try:
                npz_content = np.load(npz_file) if npz_content is None else npz_content
            except Exception as e:
                print(f"Unable to open preprocessed file {npz_file}. Rerun nnUNetv2_preprocess!")
                raise e
            np.save(npz_file[:-4] + "_seg.npy", npz_content['seg'])

        if verify_npy:
            try:
                np.load(data_npy, mmap_mode='r')
                if isfile(seg_npy):
                    np.load(seg_npy, mmap_mode='r')
            except ValueError:
                os.remove(data_npy)
                os.remove(seg_npy)
                print(f"Error when checking {data_npy} and {seg_npy}, fixing...")
                if fail_ctr < 2:
                    _convert_to_npy(npz_file, unpack_segmentation, overwrite_existing, verify_npy, fail_ctr+1)
                else:
                    raise RuntimeError("Unable to fix unpacking. Please check your system or rerun nnUNetv2_preprocess")

    except KeyboardInterrupt:
        if isfile(data_npy):
            os.remove(data_npy)
        if isfile(seg_npy):
            os.remove(seg_npy)
        raise KeyboardInterrupt


def unpack_dataset(folder: str, unpack_segmentation: bool = True, overwrite_existing: bool = False,
                   num_processes: int = default_num_processes,
                   verify: bool = False):
    """
    all npz files in this folder belong to the dataset, unpack them all
    """
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        npz_files = subfiles(folder, True, None, ".npz", True)
        p.starmap(_convert_to_npy, zip(npz_files,
                                       [unpack_segmentation] * len(npz_files),
                                       [overwrite_existing] * len(npz_files),
                                       [verify] * len(npz_files))
                  )


def preprocess_point(point, data_properties, shape):
        """
        Preprocess the points to map them to the correct coordinate system.
        I.e. from the original image space to the cropped/resized image space.
        """
        point = [float(i) for i in point]
        x, y, z = point
        bbox_used_for_cropping = data_properties["bbox_used_for_cropping"]
        shape_after_cropping_and_before_resampling = data_properties["shape_after_cropping_and_before_resampling"]
        # Adapt the centroid to cropped data
        x = max(0, x - bbox_used_for_cropping[2][0])
        x = min(shape_after_cropping_and_before_resampling[2], x - bbox_used_for_cropping[2][0])
        y = max(0, y - bbox_used_for_cropping[1][0])
        y = min(shape_after_cropping_and_before_resampling[1], y - bbox_used_for_cropping[1][0])
        z = max(0, z - bbox_used_for_cropping[0][0])
        z = min(shape_after_cropping_and_before_resampling[0], z - bbox_used_for_cropping[0][0])

        # Adjust for resampling
        factor = [shape[i] / shape_after_cropping_and_before_resampling[i] for i in range(3)]
        x = np.round(x * factor[2]).astype(np.int16)
        y = np.round(y * factor[1]).astype(np.int16)
        z = np.round(z * factor[0]).astype(np.int16)
        return [z, y, x]


def sparse_to_dense_point_gauss(points: dict[str, np.ndarray], shape: tuple[int, ...], properties: dict, sigma: float = 1.0) -> np.ndarray:
    pos_clicks, neg_clicks = np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
    if len(points) > 0:
        for clck in points:
            coord = clck['point']
            label = clck['name']
            coord = preprocess_point(coord, properties, shape)
            if label == 'tumor':
                pos_clicks[tuple(coord)] = 1.0
            elif label == 'background':
                neg_clicks[tuple(coord)] = 1.0 # self.place_point(coord, neg_clicks, n_clck + 1)
            else:
                raise ValueError(f"Unknown label {label} in click json")
        pos_clicks = gaussian_filter(pos_clicks, sigma=sigma)
        neg_clicks = gaussian_filter(neg_clicks, sigma=sigma)
    return pos_clicks, neg_clicks


def generated_sparse_to_dense_point_gauss(clicks: dict, shape: tuple[int, ...], sigma: float = 1.0) -> np.ndarray:
    pos_clicks, neg_clicks = np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
    if len(clicks["background"]) > 0:
        for clck in clicks["tumor"]:
            pos_clicks[tuple(clck)] = 1.0
        for clck in clicks["background"]:
            neg_clicks[tuple(clck)] = 1.0
        pos_clicks = gaussian_filter(pos_clicks, sigma=sigma)
        neg_clicks = gaussian_filter(neg_clicks, sigma=sigma)
    return pos_clicks, neg_clicks

def sparse_to_dense_point_nnInteractive(points: dict[str, np.ndarray], shape: tuple[int, ...], properties: dict, sigma: float = 1.0) -> np.ndarray:
    pos_clicks, neg_clicks = torch.zeros(shape, dtype=torch.float32), torch.zeros(shape, dtype=torch.float32)
    point_interaction = PointInteraction_stub(point_radius=sigma, use_distance_transform=True)
    if len(points) > 0:
        for clck in points:
            coord = clck['point']
            label = clck['name']
            coord = preprocess_point(coord, properties, shape)
            if label == 'tumor':
                pos_clicks = point_interaction.place_point(coord, pos_clicks, binarize=False)
            elif label == 'background':
                neg_clicks = point_interaction.place_point(coord, neg_clicks, binarize=False)
            else:
                raise ValueError(f"Unknown label {label} in click json")
    return pos_clicks, neg_clicks

def generated_sparse_to_dense_point_nnInteractive(clicks: dict, shape: tuple[int, ...], sigma: float = 1.0) -> np.ndarray:
    pos_clicks, neg_clicks = torch.zeros(shape, dtype=torch.float32), torch.zeros(shape, dtype=torch.float32)
    point_interaction = PointInteraction_stub(point_radius=sigma, use_distance_transform=True)
    if len(clicks["background"]) > 0:
        for clck in clicks["tumor"]:
            pos_clicks = point_interaction.place_point(clck, pos_clicks, binarize=False)
        for clck in clicks["background"]:
            neg_clicks = point_interaction.place_point(clck, neg_clicks, binarize=False)
    return pos_clicks, neg_clicks


### autoPET4 click generation ###

def perturb_click(offset, click, label_im):
    import random
    random_offset = [random.randint(0, int(offset)) for _ in range(3)]
    if int(click[0] + random_offset[0]) < label_im.shape[0] and int(click[1] + random_offset[1]) < label_im.shape[1] and int(int(click[2] + random_offset[2])) < label_im.shape[2]: # click inside the volume
        if label_im[
            int(click[0] + random_offset[0]),
            int(click[1] + random_offset[1]),
            int(click[2] + random_offset[2])
        ]:
            return [
                int(click[0] + random_offset[0]),
                int(click[1] + random_offset[1]),
                int(click[2] + random_offset[2])
            ]
    
    # Fallback to original click if perturbed click is invalid
    return [int(click[0]), int(click[1]), int(click[2])]

def simulate_clicks(input_label, input_pet, fg=True, bg=True, center_offset=None, edge_offset=None, pos_click_budget=10, neg_click_budget=10, use_gpu=True):
    if use_gpu:
        import cupy as cp
        from cucim.core.operations import morphology
        print("[INFO] Using GPU for EDT computation")
    if isinstance(input_label, np.ndarray):
        label_im = input_label
        label_im[label_im < 0] = 0
    else:
        raise ValueError("input_label must be numpy array")    
    clicks = {'tumor':[], 'background': []}

    if np.sum(label_im) == 0:
        pass
    else: 
        ##### Tumor Clicks #####
        connected_components = cc3d.connected_components(label_im, connectivity=26)
        unique_labels = np.unique(connected_components)[1:] # Skip background label 0
        size = min(pos_click_budget, len(unique_labels))
        sampled_labels = np.random.choice(unique_labels, size=size, replace=False)

        # Sample center clicks for 10 (click_budget) random components
        for label in sampled_labels:    
            labeled_mask = connected_components == label
            if use_gpu:
                # Attempt to compute EDT using GPU
                labeled_mask = cp.array(labeled_mask)
                edt = morphology.distance_transform_edt(labeled_mask)
                center = cp.unravel_index(cp.argmax(edt), edt.shape)
            else:
                edt = fastedt.edt(labeled_mask)
                center = np.unravel_index(np.argmax(edt), edt.shape)
            
            if center_offset is not None:
                center = perturb_click(center_offset, center, label_im)
            clicks['tumor'].append([int(center[0]), int(center[1]), int(center[2])])
            assert label_im[int(center[0]), int(center[1]), int(center[2])]
        n_clicks = len(clicks['tumor'])

        # Sample boundary clicks if center clicks were not enough to fill the click budget (n=10)
        counter = 0
        while n_clicks < pos_click_budget:
            for label in sampled_labels: 
                labeled_mask = connected_components == label
                if use_gpu:
                    # Attempt to compute EDT using GPU
                    labeled_mask = cp.array(labeled_mask)
                    edt = morphology.distance_transform_edt(labeled_mask)
                    edt_inverted = (cp.max(edt) - edt) * (edt > 0) 
                    boundary_elements = (edt_inverted == cp.max(edt_inverted)) * (labeled_mask > 0)
                    indices = cp.array(cp.nonzero(boundary_elements)).T.get()  # Shape: (num_true, ndim)
                else:
                    edt = fastedt.edt(labeled_mask)
                    edt_inverted = (np.max(edt) - edt) * (edt > 0) 
                    boundary_elements = (edt_inverted == np.max(edt_inverted)) * (labeled_mask > 0)
                    indices = np.array(np.nonzero(boundary_elements)).T  # Shape: (num_true, ndim)
                
                boundary_click = indices[np.random.choice(indices.shape[0])]
                if edge_offset is not None:
                    boundary_click = perturb_click(edge_offset, boundary_click, label_im)

                clicks['tumor'].append([int(boundary_click[0]), int(boundary_click[1]), int(boundary_click[2])])
                assert label_im[int(boundary_click[0]), int(boundary_click[1]), int(boundary_click[2])]
                n_clicks += 1
                if n_clicks == pos_click_budget:
                    break
            counter += 1
            if counter > 10:
                print("Warning: Unable to sample enough tumor clicks. Please check your label image.")
                break
    if bg:
        ##### Background Clicks #####
        assert isinstance(input_pet, np.ndarray), "input_pet must be numpy array"
        pet_img = input_pet
        non_tumor = pet_img[label_im == 0 & (pet_img != 0)]
        th = np.percentile(non_tumor, 99.75)

        non_tumor_high_uptake = (pet_img >= th) * (label_im == 0) * (pet_img != 0)

        connected_components = cc3d.connected_components(non_tumor_high_uptake, connectivity=26)
        unique_labels = np.unique(connected_components)[1:] # Skip background label 0
        size = min(neg_click_budget, len(unique_labels))
        sampled_labels = np.random.choice(unique_labels, size=size, replace=False)

        # Sample center clicks for 10 components (click_budget)
        for label in sampled_labels:  
            labeled_mask = connected_components == label
            if use_gpu:
                # Attempt to compute EDT using GPU
                labeled_mask = cp.array(labeled_mask)
                edt = morphology.distance_transform_edt(labeled_mask)
                center = cp.unravel_index(cp.argmax(edt), edt.shape)
            else:
                edt = fastedt.edt(labeled_mask)
                center = np.unravel_index(np.argmax(edt), edt.shape)
            
            if center_offset is not None:
                center = perturb_click(center_offset, center, ~label_im.astype(np.bool_))

            clicks['background'].append([int(center[0]), int(center[1]), int(center[2])])
            assert not label_im[int(center[0]), int(center[1]), int(center[2])]
        n_clicks = len(clicks['background'])

        # Sample boundary clicks if center clicks were not enough to fill the click budget (n=10)
        counter = 0
        while n_clicks < neg_click_budget:
            for label in sampled_labels:  # Skip background label 0
                labeled_mask = connected_components == label
                if use_gpu:
                    # Attempt to compute EDT using GPU
                    labeled_mask = cp.array(labeled_mask)
                    edt = morphology.distance_transform_edt(labeled_mask)
                    edt_inverted = (cp.max(edt) - edt) * (edt > 0) 
                    boundary_elements = (edt_inverted == cp.max(edt_inverted)) * (labeled_mask > 0)
                    indices = cp.array(cp.nonzero(boundary_elements)).T.get()  # Shape: (num_true, ndim)
                    if len(indices) == 0:
                        indices = cp.array(cp.nonzero(labeled_mask)).T.get()
                else:
                    edt = fastedt.edt(labeled_mask)
                    edt_inverted = (np.max(edt) - edt) * (edt > 0) 
                    boundary_elements = (edt_inverted == np.max(edt_inverted)) * (labeled_mask > 0)
                    indices = np.array(np.nonzero(boundary_elements)).T  # Shape: (num_true, ndim)
                    if len(indices) == 0:
                        indices = np.array(np.nonzero(labeled_mask)).T
                
                boundary_click = indices[np.random.choice(indices.shape[0])]
                if edge_offset is not None:
                    boundary_click = perturb_click(edge_offset, boundary_click, ~label_im.astype(np.bool_))

                clicks['background'].append([int(boundary_click[0]), int(boundary_click[1]), int(boundary_click[2])])
                assert not label_im[int(boundary_click[0]), int(boundary_click[1]), int(boundary_click[2])]
                n_clicks += 1
                if n_clicks == neg_click_budget:
                    break
            counter += 1
            if counter > 10:
                print("Warning: Unable to sample enough background clicks. Please check your label image.")
                break
        
    return clicks


####### Simulate clicks advanced ########

from scipy.ndimage import binary_erosion


def random_point_within_mask(mask):
    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return None
    idx = np.random.choice(coords.shape[0])
    return tuple(coords[idx])


def perturb_click(offset, center, mask):
    max_attempts = 10
    for _ in range(max_attempts):
        perturbed = np.array(center) + np.random.randint(-offset, offset + 1, size=3)
        perturbed = np.clip(perturbed, 0, np.array(mask.shape) - 1)
        if mask[tuple(perturbed)]:
            return tuple(perturbed)
    return center


def sample_point_within_region(mask, edt_weight=0.7):
    # Mix of center and random sampling
    edt = fastedt.edt(mask.astype(np.uint8))
    edt = edt / (np.max(edt) + 1e-5)
    noise = np.random.rand(*mask.shape)
    score = edt_weight * edt + (1 - edt_weight) * noise
    score *= mask
    idx = np.unravel_index(np.argmax(score), score.shape)
    return idx


def simulate_clicks_advanced(input_label, input_pet, fg=True, bg=True, center_offset=None, edge_offset=None, pos_click_budget=10, neg_click_budget=10, use_gpu=True):
    if isinstance(input_label, np.ndarray):
        label_im = input_label.copy()
        label_im[label_im < 0] = 0
    else:
        raise ValueError("input_label must be numpy array")

    clicks = {'tumor': [], 'background': []}

    if fg and np.sum(label_im) > 0:
        connected_components = cc3d.connected_components(label_im, connectivity=26)
        unique_labels = np.unique(connected_components)[1:]
        size = min(pos_click_budget, len(unique_labels))
        sampled_labels = np.random.choice(unique_labels, size=size, replace=False)

        n_clicks = 0
        for label in sampled_labels:
            mask = connected_components == label
            point = sample_point_within_region(mask, edt_weight=np.random.uniform(0.4, 0.9))
            if center_offset is not None:
                point = perturb_click(center_offset, point, mask)
            clicks['tumor'].append(list(map(int, point)))
            n_clicks += 1

        # Fill remaining clicks with near-boundary points inside the object
        counter = 0
        while n_clicks < pos_click_budget:
            for label in sampled_labels:
                mask = connected_components == label
                edt = fastedt.edt(mask.astype(np.uint8))
                # get boundary by thresholding the EDT at a small value (low percentile)
                soft_boundary_mask = mask & (edt < 3)
                point = sample_point_within_region(soft_boundary_mask, edt_weight=np.random.uniform(0.2, 0.6))
                if edge_offset is not None:
                    point = perturb_click(edge_offset, point, mask)
                clicks['tumor'].append(list(map(int, point)))
                n_clicks += 1
                if n_clicks == pos_click_budget:
                    break
            counter += 1
            if counter > 10:
                print("Warning: Unable to sample enough background clicks. Please check your label image.")
                break

    if bg:
        assert isinstance(input_pet, np.ndarray), "input_pet must be numpy array"
        non_tumor_mask = (label_im == 0)
        th = np.percentile(input_pet[non_tumor_mask], 98)
        high_uptake = (input_pet >= th) * (label_im == 0) * (input_pet > 0)

        if np.all(high_uptake == True) or np.all(high_uptake == False):
            return clicks  # No high uptake regions to sample from
        
        # sample background clicks by using the uptake value as probability
        probability_map = input_pet[high_uptake]
        probability_map /= np.sum(probability_map)  # Normalize to sum to 1
        all_indices = np.argwhere(high_uptake == True)
        sampled_indices = np.random.choice(all_indices.shape[0], size=neg_click_budget, p=probability_map)
        sampled_indices = all_indices[sampled_indices]
        for idx in sampled_indices:
            point = tuple(idx)
            clicks['background'].append(list(map(int, point)))




        # connected_components = cc3d.connected_components(high_uptake, connectivity=26)
        # unique_labels = np.unique(connected_components)[1:]
        # size = min(neg_click_budget, len(unique_labels))
        # sampled_labels = np.random.choice(unique_labels, size=size, replace=False)

        # n_clicks = 0
        # for label in sampled_labels:
        #     mask = connected_components == label
        #     point = sample_point_within_region(mask, edt_weight=np.random.uniform(0.4, 0.9))
        #     if center_offset is not None:
        #         point = perturb_click(center_offset, point, mask)
        #     clicks['background'].append(list(map(int, point)))
        #     n_clicks += 1

        # counter = 0
        # while n_clicks < neg_click_budget:
        #     for label in sampled_labels:
        #         mask = connected_components == label
        #         edt = fastedt.edt(mask.astype(np.uint8))
        #         soft_boundary_mask = mask & (edt < 3)
        #         point = sample_point_within_region(soft_boundary_mask, edt_weight=np.random.uniform(0.2, 0.6))
        #         if edge_offset is not None:
        #             point = perturb_click(edge_offset, point, mask)
        #         clicks['background'].append(list(map(int, point)))
        #         n_clicks += 1
        #         if n_clicks == neg_click_budget:
        #             break
        #     counter += 1
        #     if counter > 10:
        #         print("Warning: Unable to sample enough background clicks. Please check your label image.")
        #         break

    return clicks

