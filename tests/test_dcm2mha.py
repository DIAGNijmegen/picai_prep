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


import os
import shutil
from pathlib import Path

import SimpleITK as sitk
from numpy.testing import assert_allclose
from picai_prep.dcm2mha import (Dicom2MHACase, Dicom2MHAConverter,
                                Dicom2MHASettings)


def test_dcm2mha(
    input_dir: str = "tests/input/dcm/ProstateX",
    output_dir: str = "tests/output/mha/ProstateX",
    output_expected_dir: str = "tests/output-expected/mha/ProstateX",
):
    """
    Convert sample DICOM archive to MHA
    """
    # remove output folder (to prevent skipping the conversion)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # test usage from command line
    archive = Dicom2MHAConverter(
        input_dir=input_dir,
        output_dir=output_dir,
        dcm2mha_settings="tests/output-expected/dcm2mha_settings.json"
    )
    archive.convert()

    # compare output
    for patient_id, subject_id in [
        ("ProstateX-0000", "ProstateX-0000_07-07-2011"),
        ("ProstateX-0001", "ProstateX-0001_07-08-2011"),
    ]:
        for modality in ["t2w", "adc", "hbv", "sag", "cor"]:
            # construct paths to MHA images
            path_out = os.path.join(output_dir, patient_id, f"{subject_id}_{modality}.mha")
            path_out_expected = os.path.join(output_expected_dir, patient_id, f"{subject_id}_{modality}.mha")

            # sanity check: check if outputs exist
            assert os.path.exists(path_out), f"Could not find output file at {path_out}!"
            assert os.path.exists(path_out_expected), f"Could not find output file at {path_out_expected}!"

            # read images
            img = sitk.GetArrayFromImage(sitk.ReadImage(str(path_out)))
            img_expected = sitk.GetArrayFromImage(sitk.ReadImage(str(path_out_expected)))

            # compare images
            assert_allclose(img_expected, img)


def test_resolve_duplicates(
    input_dir: str = "tests/input/dcm/ProstateX",
):
    # setup case with duplicates
    case = Dicom2MHACase(
        input_dir=Path(input_dir),
        patient_id="ProstateX-0001",
        study_id="07-08-2011",
        paths=[
            "ProstateX-0001/07-08-2011/10.000000-t2tsetra-17541",
            "ProstateX-0001/07-08-2011/6.000000-t2tsetra-76610",
        ],
        settings=Dicom2MHASettings(
            mappings={
                "t2w": {
                    "SeriesDescription": [
                        "t2_tse_tra"
                    ]
                },
            }
        )
    )

    # resolve duplicates
    case.extract_metadata()
    case.apply_mappings()
    case.resolve_duplicates()

    # check if duplicates were resolved
    matched_series = [serie for serie in case.valid_series if "t2w" in serie.mappings]
    assert len(matched_series) == 1, 'More than one serie after resolving duplicates!'


def test_value_match_contains(
    input_dir: str = "tests/input/dcm/ProstateX",
    patient_id="ProstateX-0001",
    study_id="07-08-2011",
):
    # setup case
    series_list = os.listdir(os.path.join(input_dir, patient_id, study_id))
    case = Dicom2MHACase(
        input_dir=Path(input_dir),
        patient_id=patient_id,
        study_id=study_id,
        paths=[
            os.path.join(patient_id, study_id, series_id) for series_id in series_list
            if os.path.isdir(os.path.join(input_dir, patient_id, study_id, series_id))
        ],
        settings=Dicom2MHASettings(
            mappings={
                "test": {
                    "SeriesDescription": [
                        ""
                    ]
                },
            },
            values_match_func="lower_strip_contains"
        )
    )

    # resolve duplicates
    case.extract_metadata()
    case.apply_mappings()

    # check if duplicates were resolved
    matched_series = [serie for serie in case.valid_series if "test" in serie.mappings]
    assert len(matched_series) == 11, 'Empty lower_strip_contains should match all series!'


def test_value_match_multiple_keys(
    input_dir: str = "tests/input/dcm/ProstateX",
    patient_id="ProstateX-0001",
    study_id="07-08-2011",
):
    # setup case
    series_list = os.listdir(os.path.join(input_dir, patient_id, study_id))
    case = Dicom2MHACase(
        input_dir=Path(input_dir),
        patient_id=patient_id,
        study_id=study_id,
        paths=[
            os.path.join(patient_id, study_id, series_id) for series_id in series_list
            if os.path.isdir(os.path.join(input_dir, patient_id, study_id, series_id))
        ],
        settings=Dicom2MHASettings(
            mappings={
                "test": {
                    "SeriesDescription": [
                        ""
                    ],
                    "imagetype": [
                        "DERIVED\\PRIMARY\\DIFFUSION"
                    ]
                },
            },
            values_match_func="lower_strip_contains"
        )
    )

    # resolve duplicates
    case.extract_metadata()
    case.apply_mappings()

    # check if duplicates were resolved
    matched_series = [serie for serie in case.valid_series if "test" in serie.mappings]
    assert len(matched_series) == 3, 'Should find three diffusion scans!'
