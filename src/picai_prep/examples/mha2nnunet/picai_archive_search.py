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
from pathlib import Path
from typing import List

from picai_prep.data_utils import PathLike
from tqdm import tqdm


def generate_mha2nnunet_settings(
    archive_dir: PathLike,
    output_path: PathLike,
    subject_list: List[str],
    task: str = "Task2201_picai_baseline",
    strict: bool = True,
):
    """
    Create mha2nnunet_settings.json for an MHA archive with the following structure:
    /path/to/archive/
    ├── [patient UID]/
        ├── [patient UID]_[study UID]_[modality].mha
        ...

    For each study, the T2-weighted scan (_t2w.mha), apparent diffusion scan (_adc.mha)
    and high b-value scan (_hbv.mha) are collected.

    Parameters:
    - archive_dir: path to MHA archive
    - output_path: path to store MHA->nnUNet settings JSON (parent folder should exist)
    """
    # convert paths
    archive_dir = Path(archive_dir)
    output_path = Path(output_path)

    archive_list = []
    for subject_id in tqdm(subject_list):
        # split subject id
        patient_id, study_id = str(subject_id).split("_")

        # construct path to MRI study
        study_dir = archive_dir / patient_id

        if not study_dir.exists():
            if strict:
                raise FileNotFoundError(f"Folder not found for {subject_id} at {study_dir}!")
            print(f"Folder not found for {subject_id} at {study_dir}! Skipping...")
            continue

        # construct scan paths
        scan_paths = [
            f"{patient_id}/{subject_id}_{modality}.mha"
            for modality in ["t2w", "adc", "hbv"]
        ]
        all_scans_found = all([
            os.path.exists(os.path.join(archive_dir, path))
            for path in scan_paths
        ])

        if all_scans_found:
            # store info for complete studies
            archive_list += [{
                "patient_id": patient_id,
                "study_id": study_id,
                "scan_paths": scan_paths,
            }]
        else:
            if strict:
                raise FileNotFoundError(f"Not all scans found for {subject_id} at {study_dir}!")
            print(f"Not all scans found for {subject_id} at {study_dir}! Skipping...")
            continue

    mha2nnunet_settings = {
        "dataset_json": {
            "task": task,
            "description": "bpMRI scans from PI-CAI dataset to train nnUNet baseline",
            "tensorImageSize": "4D",
            "reference": "",
            "licence": "",
            "release": "1.0",
            "modality": {
                "0": "T2W",
                "1": "CT",
                "2": "HBV"
            },
            "labels": {
                "0": "background",
                "1": "lesion"
            }
        },
        "preprocessing": {
            # optionally, resample and perform centre crop:
            # "matrix_size": [
            #     20,
            #     160,
            #     160
            # ],
            # "spacing": [
            #     3.6,
            #     0.5,
            #     0.5
            # ],
        },
        "archive": archive_list
    }

    if not len(archive_list):
        raise ValueError(f"Did not find any MHA scans in {archive_dir}, aborting.")

    with open(output_path, "w") as fp:
        json.dump(mha2nnunet_settings, fp, indent=4)

    print(f""""
    Saved mha2nnunet_settings to {output_path}, with {len(archive_list)} cases.
    """)
