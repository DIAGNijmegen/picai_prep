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
from tqdm import tqdm

from picai_prep.data_utils import PathLike


def generate_dcm2mha_settings(
    archive_dir: PathLike,
    output_path: PathLike
):
    """
    Create dcm2mha_settings.json for a DICOM archive with the following structure:
    /path/to/archive/
    ├── [patient UID]/
        ├── [study UID]/
            ├── [series UID]/
                ├── slice-1.dcm
                ...
                ├── slice-n.dcm

    Parameters:
    - archive_dir: path to DICOM archive
    - output_path: path to store DICOM->MHA settings JSON to
        (parent folder should exist)
    """
    ignore_files = [
        ".DS_Store",
        "LICENSE",
    ]

    archive_list = []

    # traverse DICOM archive
    for patient_id in tqdm(sorted(os.listdir(archive_dir))):
        # traverse each patient
        if patient_id in ignore_files:
            continue

        patient_dir = os.path.join(archive_dir, patient_id)
        for study_id in sorted(os.listdir(patient_dir)):
            # traverse each study
            if study_id in ignore_files:
                continue

            study_dir = os.path.join(patient_dir, study_id)
            for series_id in sorted(os.listdir(study_dir)):
                # traverse each series
                if series_id in ignore_files:
                    continue

                # construct path to series folder
                path = Path(os.path.join(patient_id, study_id, series_id))

                # store info
                archive_list += [{
                    "patient_id": patient_id,
                    "study_id": study_id,
                    "path": path.as_posix(),
                }]

    archive = {
        "mappings": {
            "t2w": {
                "SeriesDescription": [
                    "t2_tse_tra",
                ]
            },
            "cor": {
                "SeriesDescription": [
                    "t2_tse_cor"
                ]
            },
            "sag": {
                "SeriesDescription": [
                    "t2_tse_sag"
                ]
            },
            "adc": {
                "SeriesDescription": [
                    "ep2d_diff_tra_DYNDIST_MIX_ADC",
                    "ep2d_diff_tra_DYNDIST_ADC",
                ]
            },
            "hbv": {
                "SeriesDescription": [
                    "ep2d_diff_tra_DYNDIST_MIXCALC_BVAL",
                    "ep2d_diff_tra_DYNDISTCALC_BVAL",
                ]
            }
        },
        "archive": archive_list,
    }

    if not len(archive_list):
        raise ValueError("Did not find any DICOM series, aborting.")

    with open(output_path, "w") as fp:
        json.dump(archive, fp, indent=4)

    print(f""""
    Saved dcm2mha_settings to {output_path}, with {len(archive_list)} DICOM series.
    """)


if __name__ == '__main__':
    generate_dcm2mha_settings()
