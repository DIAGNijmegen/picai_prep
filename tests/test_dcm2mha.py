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
import subprocess
import sys
from pathlib import Path

import pytest
import SimpleITK as sitk
from numpy.testing import assert_allclose

from picai_prep.dcm2mha import (Dicom2MHACase, Dicom2MHAConverter,
                                Dicom2MHASettings, DICOMImageReader)
from picai_prep.examples.dcm2mha.sample_archive import \
    generate_dcm2mha_settings


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

    # test usage from Python
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


def test_dcm2mha_commandline(
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
    cmd = [
        "python", "-m", "picai_prep", "dcm2mha",
        "--input", input_dir,
        "--output", output_dir,
        "--json", "tests/output-expected/dcm2mha_settings.json",
        "--verbose", "2",
    ]

    # run command
    subprocess.check_call(cmd, shell=(sys.platform == 'win32'))

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
    case.initialize()
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
    case.initialize()
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
    paths = [
        os.path.join(patient_id, study_id, series_id) for series_id in series_list
        if os.path.isdir(os.path.join(input_dir, patient_id, study_id, series_id))
    ]
    archive = Dicom2MHAConverter(
        input_dir=Path(input_dir),
        output_dir="",
        dcm2mha_settings={
            "archive": [
                {
                    "patient_id": patient_id,
                    "study_id": study_id,
                    "path": path
                }
                for path in paths
            ],
            "mappings": {
                "test": {
                    "SeriesDescription": [
                        ""
                    ],
                    "imagetype": [
                        "DERIVED\\PRIMARY\\DIFFUSION"
                    ]
                },
            },
            "options": {
                "values_match_func": "lower_strip_contains"
            }
        }
    )

    case = archive.cases[0]

    # resolve duplicates
    case.initialize()
    case.extract_metadata()
    case.apply_mappings()

    # check if duplicates were resolved
    matched_series = [serie for serie in case.valid_series if "test" in serie.mappings]
    assert len(matched_series) == 3, 'Should find three diffusion scans!'


@pytest.mark.parametrize("input_dir", [
    "tests/input/dcm/ProstateX/ProstateX-0000/07-07-2011/4.000000-t2tsetra-00702",
    "tests/input/dcm/ProstateX/ProstateX-0000/07-07-2011/7.000000-ep2ddifftraDYNDISTADC-48780",
    "tests/input/dcm/ProstateX/ProstateX-0001/07-08-2011/11.000000-tfl3d PD reftra1.5x1.5t3-77124",
])
def test_image_reader(input_dir: str):
    """
    Verify consistency of reading SimpleITK or pydicom.
    """

    reader = DICOMImageReader(path=input_dir)
    image_sitk = reader._read_image_sitk()
    image_pydicom = reader._read_image_pydicom()

    # compare voxel values
    assert_allclose(sitk.GetArrayFromImage(image_sitk), sitk.GetArrayFromImage(image_pydicom))

    # compare physical metadata
    assert_allclose(image_sitk.GetSpacing(), image_pydicom.GetSpacing(), atol=1e-6)
    assert_allclose(image_sitk.GetOrigin(), image_pydicom.GetOrigin(), atol=1e-6)
    assert_allclose(image_sitk.GetDirection(), image_pydicom.GetDirection(), atol=1e-6)

    # compare metadata
    metadata_sitk = {key: image_sitk.GetMetaData(key) for key in image_sitk.GetMetaDataKeys()}
    metadata_pydicom = {key: image_pydicom.GetMetaData(key) for key in image_pydicom.GetMetaDataKeys()}
    keys = set(metadata_pydicom.keys()) & set(metadata_sitk.keys())
    assert len(keys) > 1, 'No metadata found!'
    assert {k: metadata_sitk[k] for k in keys} == {k: metadata_pydicom[k] for k in keys}


@pytest.mark.xfail
@pytest.mark.parametrize("input_dir", [
    "tests/input/dcm/ProstateX/ProstateX-0001/07-08-2011/1.000000-t2localizer-75055",
])
def test_image_reader_invalid_sequence(input_dir: str):
    test_image_reader(input_dir)


@pytest.mark.parametrize("input_dir", [
    "tests/input/dcm/ProstateX-dicom-zip/ProstateX-0001/07-08-2011/8.000000-ep2ddifftraDYNDISTMIXADC-33954",
    "tests/input/dcm/ProstateX-dicom-zip/ProstateX-0001/07-08-2011/11.000000-tfl3d PD reftra1.5x1.5t3-77124",
])
def test_image_reader_dicom_zip(input_dir: str):
    reader1 = DICOMImageReader(path=input_dir)
    reader2 = DICOMImageReader(path=input_dir)

    # read image, then metadata
    image1 = reader1.image
    metadata1 = reader1.metadata

    # read metadata, then image
    metadata2 = reader2.metadata
    image2 = reader2.image

    # compare voxel values
    assert_allclose(sitk.GetArrayFromImage(image1), sitk.GetArrayFromImage(image2))

    # compare metadata
    keys = set(metadata1.keys()) & set(metadata2.keys())
    assert len(keys) > 1, 'No metadata found!'
    assert {k: metadata1[k] for k in keys} == {k: metadata2[k] for k in keys}


@pytest.mark.xfail
@pytest.mark.parametrize("input_dir", [
    "tests/input/dcm/ProstateX-dicom-zip/ProstateX-0001/07-08-2011/1.000000-t2localizer-75055",
    "tests/input/dcm/ProstateX-dicom-zip/ProstateX-0001/07-08-2011/corrupt-sequence",
])
def test_image_reader_dicom_zip_invalid_sequence(input_dir: str):
    test_image_reader_dicom_zip(input_dir)


@pytest.mark.xfail
def test_image_reader_missing_slice():
    """
    Verify that a missing slice is detected.
    """
    input_dir = "tests/input/dcm/ProstateX-missing-slice/ProstateX-0000/07-07-2011/8.000000-ep2ddifftraDYNDISTCALCBVAL-83202"
    _ = DICOMImageReader(path=input_dir).image


def test_image_reader_missing_slice_okay():
    """
    Verify that an image with a missing slice can be converted.
    """
    input_dir = "tests/input/dcm/ProstateX-missing-slice/ProstateX-0000/07-07-2011/8.000000-ep2ddifftraDYNDISTCALCBVAL-83202"
    image = DICOMImageReader(path=input_dir, verify_dicom_filenames=False).image
    image = sitk.GetArrayFromImage(image)
    assert image.shape == (18, 128, 84)


def test_Dicom2MHAConverter_missing_slice_okay(
    input_dir: str = "tests/input/dcm/ProstateX-missing-slice",
    output_dir: str = "tests/output/mha/ProstateX-missing-slice",
    output_expected_dir: str = "tests/output-expected/mha/ProstateX-missing-slice",
):
    """
    Verify that an image with a missing slice can be converted with Dicom2MHAConverter
    """
    # paths
    dcm2mha_settings_path = Path(output_dir) / "dcm2mha_settings.json"

    # remove output folder (to prevent skipping the conversion)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # generate dcm2mha_settings
    dcm2mha_settings_path.parent.mkdir(parents=True, exist_ok=True)
    generate_dcm2mha_settings(
        archive_dir=input_dir,
        output_path=dcm2mha_settings_path,

    )

    # test usage from Python
    archive = Dicom2MHAConverter(
        input_dir=input_dir,
        output_dir=output_dir,
        dcm2mha_settings=dcm2mha_settings_path
    )
    archive.settings.verify_dicom_filenames = False
    archive.convert()

    # compare output
    for patient_id, subject_id in [
        ("ProstateX-0000", "ProstateX-0000_07-07-2011"),
    ]:
        for modality in ["hbv"]:
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
