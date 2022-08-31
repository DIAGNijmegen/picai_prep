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
from typing import Dict

from picai_prep.data_utils import PathLike
from tqdm import tqdm


def generate_dcm2mha_settings(
    archive_dir: PathLike,
    output_path: PathLike,
    mappings: Dict = None,
    **kwargs
):
    """
    Create dcm2mha_settings.json for a DICOM archive assuming the following structure:
    /path/to/archive/
    ├── [patient UID]/
        ├── [study UID]/
            ├── [series UID]/
                ├── slice-1.dcm
                ...
                ├── slice-n.dcm

    Parameters
    ----------
    archive_dir:
        path to DICOM archive
    output_path:
        path to store DICOM->MHA settings JSON to (parent folder should exist)
    mappings:
        mapping defining which series within a case is converted, based on metadata tags

    Other Parameters
    ----------------
    num_threads: int, default: 4
        number of threads to use for multiprocessing
    verify_dicom_filenames: bool, default: True
        explicitly verify dicom filenames as a sanity check
    allow_duplicates: bool, default: False
        when multiple series apply to a mapping, convert all
    """

    archive_list = []
    archive_dir = Path(archive_dir)

    # traverse DICOM archive
    for patient_id in tqdm(sorted(os.listdir(archive_dir))):
        # traverse each patient's studies
        patient_dir: Path = archive_dir / patient_id
        if not patient_dir.is_dir():
            continue

        # collect list of available studies
        for study_id in sorted(os.listdir(patient_dir)):
            # traverse each study's sequences
            study_dir = patient_dir / study_id
            if not study_dir.is_dir():
                continue

            for series_id in sorted(os.listdir(study_dir)):
                # construct path to series folder
                path = Path(patient_id, study_id, series_id)

                if not (study_dir / series_id).is_dir():
                    continue

                # store info
                archive_list += [{
                    "patient_id": patient_id,
                    "study_id": study_id,
                    "path": path.as_posix(),
                }]

    if not mappings:
        mappings = {
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
        }

    archive = {
        "mappings": mappings,
        "archive": archive_list
    }
    if kwargs:
        archive["options"] = kwargs

    if not len(archive_list):
        raise ValueError("Did not find any DICOM series, aborting.")

    with open(output_path, "w") as fp:
        json.dump(archive, fp, indent=4)

    print(f""""
    Saved dcm2mha_settings to {output_path}, with {len(archive_list)} DICOM series.
    """)


if __name__ == '__main__':
    generate_dcm2mha_settings()
