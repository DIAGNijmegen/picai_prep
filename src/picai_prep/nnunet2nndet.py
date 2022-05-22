#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import json
import os
import traceback
from pathlib import Path
from typing import Dict

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from tqdm import tqdm

from picai_prep.data_utils import (PathLike, atomic_file_copy,
                                   atomic_image_write)


def convert_and_save_label(
    lbl_path_src: PathLike,
    lbl_path_dst: PathLike,
    json_path_dst: PathLike,
):
    if os.path.exists(lbl_path_dst) and os.path.exists(json_path_dst):
        # conversion already done
        print(f"Conversion already done for {lbl_path_src}. Skipping..")
        return

    # placeholder for label info
    label_format: Dict[str, Dict[str, int]] = {
        'instances': {
            # example: str(i+1): 0 for i in range(num_gt_lesions)
        }
    }

    # read label in nnUNet format
    lbl = sitk.ReadImage(str(lbl_path_src))
    lbl_arr = sitk.GetArrayFromImage(lbl)

    # split ground truth lesion/structure annotations as individual 'instances'
    total_num_gt_lesions = 0
    new_labelled_gt = np.zeros_like(lbl_arr)
    for label_value in sorted(np.unique(lbl_arr)):
        # iterate over each lesion/structure type (which all have the same label value for nnUNet)
        if label_value == 0:
            continue

        # split non-overlapping structures
        labelled_gt, num_gt_lesions = ndimage.label(lbl_arr == label_value, np.ones((3, 3, 3)))

        # off-set instance indices with with previously extracted structures
        mask = (labelled_gt > 0)
        labelled_gt[mask] += total_num_gt_lesions

        # add instances to new annotation
        new_labelled_gt[mask] = labelled_gt[mask]

        # add 'metadata' of the extracted instances
        label_format['instances'].update({
            str(i+1): int(label_value - 1)
            for i in range(total_num_gt_lesions, total_num_gt_lesions + num_gt_lesions)
        })

        # count total number of instances
        total_num_gt_lesions += num_gt_lesions

    # convert to SimpleITK
    new_labelled_gt = sitk.GetImageFromArray(new_labelled_gt)
    new_labelled_gt.CopyInformation(lbl)

    # save label in nnDetection format
    atomic_image_write(new_labelled_gt, path=lbl_path_dst)

    # save json description
    with open(json_path_dst, "w") as fp:
        json.dump(label_format, fp)


def nnunet2nndet_dataset_json_conversion(
    path_src: PathLike,
    path_dst: PathLike,
):
    # read dataset.json from nnUNet
    with open(path_src) as fp:
        ds = json.load(fp)

    # ensure dataset config has 'modalities' tag (whereas nnUNet used 'modality')
    if 'modalities' not in ds:
        ds['modalities'] = ds['modality']

    # ensure dataset config has 'dim' tag
    if 'dim' not in ds:
        ds['dim'] = 3

    # change label counting, from 0=background, to 0=first structure of interest
    if "background" in ds["labels"].values():
        # lower label value of each structure type
        labels = [name for name in ds["labels"].values() if name != "background"]
        ds["labels"] = {num: name for num, name in enumerate(labels)}

    # if the target class is not specified, default to 0 (the first structure of interest)
    if "target_class" not in ds:
        ds["target_class"] = 0

    # if no test images are specified, set test_labels to False
    if "test" not in ds or len(ds["test"]) == 0:
        ds["test_labels"] = False

    # save dataset.json in nnDetection format
    with open(path_dst, "w") as fp:
        json.dump(ds, fp, indent=4)


def nnunet2nndet(
    nnunet_raw_data_path: PathLike,
    nndet_raw_data_path: PathLike,
    in_dir_annot: PathLike = "labelsTr",
    out_dir_annot: PathLike = "raw_splitted/labelsTr",
    in_dir_scans: PathLike = "imagesTr",
    out_dir_scans: PathLike = "raw_splitted/imagesTr",
):
    # input verification
    nnunet_raw_data_path = Path(nnunet_raw_data_path)
    nndet_raw_data_path = Path(nndet_raw_data_path)
    assert nnunet_raw_data_path.exists(), f"nnUNet data folder not found at {nnunet_raw_data_path}!"

    # make target folders
    nndet_raw_data_path.mkdir(parents=True, exist_ok=True)
    if out_dir_annot is not None:
        out_dir_annot = nndet_raw_data_path / out_dir_annot
        out_dir_annot.mkdir(parents=True, exist_ok=True)
        print(f"Copying annotations from {nnunet_raw_data_path / in_dir_annot} to {out_dir_annot}")
    if out_dir_scans is not None:
        in_dir_scans = nnunet_raw_data_path / in_dir_scans
        out_dir_scans = nndet_raw_data_path / out_dir_scans
        out_dir_scans.mkdir(parents=True, exist_ok=True)
        print(f"Copying scans from {in_dir_scans} to {out_dir_scans}")

    # copy dataset.json
    path_src = nnunet_raw_data_path / "dataset.json"
    path_dst = nndet_raw_data_path / "dataset.json"
    if os.path.exists(path_dst):
        print("dataset.json already copied. Skipping..")
    else:
        nnunet2nndet_dataset_json_conversion(
            path_src=path_src,
            path_dst=path_dst,
        )

    # convert annotations from nnUNet format to nnDetection format
    if out_dir_annot is not None:
        # collect list of annotation files in nnUNet data folder
        files = os.listdir(nnunet_raw_data_path / in_dir_annot)
        files = [fn for fn in files if ".nii.gz" in fn and "._" not in fn]

        # collect list of case names
        subject_list = [fn.replace(".nii.gz", "") for fn in files]
        subject_list = sorted(subject_list)
        print(f"Found {len(subject_list)} annotations")

        for subject_id in tqdm(subject_list):
            # construct paths
            lbl_path_src = nnunet_raw_data_path / in_dir_annot / f"{subject_id}.nii.gz"
            lbl_path_dst = out_dir_annot / f"{subject_id}.nii.gz"
            json_path_dst = out_dir_annot / f"{subject_id}.json"

            # convert nnUNet label to nnDetection format
            try:
                convert_and_save_label(
                    lbl_path_src=lbl_path_src,
                    lbl_path_dst=lbl_path_dst,
                    json_path_dst=json_path_dst,
                )
            except RuntimeError:
                print(f"Error for {lbl_path_src}:")
                print(traceback.format_exc())

    # copy scans from nnUNet data folder to nnDetection data folder
    if out_dir_scans is not None:
        for fn in os.listdir(in_dir_scans):
            # construct paths
            src_path = in_dir_scans / fn
            dst_path = out_dir_scans / fn

            if os.path.exists(dst_path):
                # conversion already done
                print(f"Conversion already done for {src_path}. Skipping..")
                continue

            # copy file
            atomic_file_copy(
                src_path=src_path,
                dst_path=dst_path,
            )
