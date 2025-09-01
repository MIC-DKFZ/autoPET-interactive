import argparse
import cc3d
import csv
from functools import partial
import multiprocessing
from multiprocessing.pool import Pool
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def multiprocessing_helper(func, args):
    # Helper function to unpack arguments for multiprocessing
    return func(*args)


def nii2numpy(nii_path):
    # input: path of NIfTI segmentation file, output: corresponding numpy array and voxel_vol in ml
    mask_nii = nib.load(str(nii_path))
    mask = mask_nii.get_fdata()
    pixdim = mask_nii.header['pixdim']   
    voxel_vol = pixdim[1]*pixdim[2]*pixdim[3]/1000
    return mask, voxel_vol


def con_comp(seg_array):
    # input: a binary segmentation array output: an array with seperated (indexed) connected components of the segmentation array
    connectivity = 18
    conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
    return conn_comp


def false_pos_pix(gt, pred):
    if np.sum(pred) == 0:
        return 0

    pred_labels = con_comp(pred)
    overlap = pred_labels * gt
    overlapping_ids = list(np.unique(overlap))
    false_pos_mask = ~np.isin(pred_labels, overlapping_ids)
    return np.count_nonzero(false_pos_mask)


def false_neg_pix(gt, pred):
    if np.sum(gt) == 0:
        return 0

    gt_labels = con_comp(gt)
    overlap = gt_labels * pred
    overlapping_ids = list(np.unique(overlap))
    false_neg_mask = ~np.isin(gt_labels, overlapping_ids)
    return np.count_nonzero(false_neg_mask)


def dice_score(gt, pred):
    # compute foreground Dice coefficient
    if np.sum(gt) == 0:
        return np.nan
    overlap = (gt*pred).sum()
    sum = gt.sum()+pred.sum()
    dice_score = 2*overlap/sum
    return dice_score


def compute_metrics(nii_gt_path, nii_pred_path):
    # main function
    case_name = nii_gt_path.name
    gt_array, voxel_vol = nii2numpy(nii_gt_path)
    pred_array, voxel_vol = nii2numpy(nii_pred_path)

    false_neg_vol = false_neg_pix(gt_array, pred_array)*voxel_vol
    false_pos_vol = false_pos_pix(gt_array, pred_array)*voxel_vol
    dice_sc = dice_score(gt_array, pred_array)

    return case_name, dice_sc, false_pos_vol, false_neg_vol


def get_metrics(gt_folder_path, pred_folder_path, num_proc=8):
    gt_folder_path = Path(gt_folder_path)
    pred_folder_path = Path(pred_folder_path)
    out_file_path = pred_folder_path / 'metrics.csv'
    args = []
    for nii_pred_path in Path(pred_folder_path).glob('*.nii.gz'):
        nii_gt_path = gt_folder_path / nii_pred_path.name
        if not nii_gt_path.exists():
            print(f"Missing gt for: {nii_gt_path.name}")
            continue
        args.append((nii_gt_path, nii_pred_path))
    with Pool(min(num_proc, multiprocessing.cpu_count())) as pool:
        results = list(tqdm(pool.imap(partial(multiprocessing_helper, compute_metrics), args), total=len(args)))

    csv_header = ['gt_name', 'Dice', 'FPvol', 'FNvol']
    with open(out_file_path, "w", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(csv_header)
        
        for result in results:
            case_name, dice_sc, false_pos_vol, false_neg_vol = result
            writer.writerow([
                case_name,
                dice_sc,
                false_pos_vol,
                false_neg_vol
            ])

        mean_dice = np.nanmean([r[1] for r in results])
        mean_fp_vol = np.nanmean([r[2] for r in results])
        mean_fn_vol = np.nanmean([r[3] for r in results])
        writer.writerow([
            'mean',
            mean_dice,
            mean_fp_vol,
            mean_fn_vol
        ])

    return mean_dice, mean_fp_vol, mean_fn_vol

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gt_path", help="Path to the nifti GT labels folder")
    parser.add_argument("-p", "--prediction_path", help="Path to the nifti Pred labels folder")
    parser.add_argument("-np", "--num_proc", help="Number of processes to use", default=8, type=int)
    args = parser.parse_args()
    get_metrics(args.gt_path, args.prediction_path, args.num_proc)
