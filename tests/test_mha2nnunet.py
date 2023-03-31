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
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import pytest
import SimpleITK as sitk
from numpy.testing import assert_allclose

from picai_prep.data_utils import PathLike
from picai_prep.examples.mha2nnunet import picai_archive
from picai_prep.mha2nnunet.converter import MHA2nnUNetConverter
from picai_prep.mha2nnunet.preprocessing import resample_to_reference_scan


def test_mha2nnunet(
    input_dir: PathLike = "tests/output-expected/mha/ProstateX",
    annotations_dir: PathLike = "tests/input/annotations/ProstateX",
    output_dir: PathLike = "tests/output/nnUNet_raw_data",
    output_expected_dir: PathLike = "tests/output-expected/nnUNet_raw_data",
    settings_path: PathLike = "tests/output-expected/mha2nnunet_settings.json",
    task_name: str = "Task100_test",
    subject_list: Optional[List[str]] = None,
):
    """
    Convert sample MHA archive to nnUNet raw data format
    """
    if subject_list is None:
        subject_list = [
            "ProstateX-0000_07-07-2011",
            "ProstateX-0001_07-08-2011",
        ]

    # convert input paths to Path
    input_dir = Path(input_dir)
    annotations_dir = Path(annotations_dir)
    output_dir = Path(output_dir)
    output_expected_dir = Path(output_expected_dir)
    task_dir = output_dir / task_name

    # remove output folder (to prevent skipping the conversion)
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)

    # convert MHA archive to nnUNet raw data
    archive = MHA2nnUNetConverter(
        output_dir=output_dir.as_posix(),
        scans_dir=input_dir.as_posix(),
        annotations_dir=annotations_dir.as_posix(),
        mha2nnunet_settings=settings_path
    )
    archive.convert()
    archive.create_dataset_json()

    # check dataset.json
    path_out = task_dir / "dataset.json"
    path_out_expected = output_expected_dir / task_name / "dataset.json"
    with open(path_out) as fp1, open(path_out_expected) as fp2:
        assert json.load(fp1) == json.load(fp2)

    # compare output
    for subject_id in subject_list:
        print(f"Checking case {subject_id}...")
        case_origin, case_direction = None, None

        for modality in ["0000", "0001", "0002"]:
            # construct paths to MHA images
            path_out = task_dir / "imagesTr" / f"{subject_id}_{modality}.nii.gz"
            path_out_expected = output_expected_dir / task_name / "imagesTr" / f"{subject_id}_{modality}.nii.gz"

            # sanity check: check if outputs exist
            assert path_out.exists(), f"Could not find output file at {path_out}!"
            assert path_out_expected.exists(), f"Could not find output file at {path_out_expected}!"

            # read images
            img = sitk.ReadImage(str(path_out))
            img_expected = sitk.ReadImage(str(path_out_expected))

            # compare images
            assert_allclose(sitk.GetArrayFromImage(img_expected), sitk.GetArrayFromImage(img))

            # check origin and direction (nnUNet checks this too)
            if case_origin is None:
                case_origin = img.GetOrigin()
            else:
                assert case_origin == img.GetOrigin(), "Origin must match between sequences!"
            if case_direction is None:
                case_direction = img.GetDirection()
            else:
                assert case_direction == img.GetDirection(), "Direction must match between sequences!"

        # check annotation
        # construct paths
        path_out = task_dir / "labelsTr" / f"{subject_id}.nii.gz"
        path_out_expected = output_expected_dir / task_name / "labelsTr" / f"{subject_id}.nii.gz"

        # sanity check: check if outputs exist
        assert path_out.exists(), f"Could not find output file at {path_out}!"
        assert path_out_expected.exists(), f"Could not find output file at {path_out_expected}!"

        # read images
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(path_out)))
        img_expected = sitk.GetArrayFromImage(sitk.ReadImage(str(path_out_expected)))

        # compare images
        assert_allclose(img_expected, img)


def test_mha2nnunet_commandline(
    input_dir: PathLike = "tests/output-expected/mha/ProstateX",
    annotations_dir: PathLike = "tests/input/annotations/ProstateX",
    output_dir: PathLike = "tests/output/nnUNet_raw_data",
    output_expected_dir: PathLike = "tests/output-expected/nnUNet_raw_data",
    settings_path: PathLike = "tests/output-expected/mha2nnunet_settings.json",
    task_name: str = "Task100_test",
    subject_list: Optional[List[str]] = None,
):
    """
    Convert sample MHA archive to nnUNet raw data format
    """
    if subject_list is None:
        subject_list = [
            "ProstateX-0000_07-07-2011",
            "ProstateX-0001_07-08-2011",
        ]

    # convert input paths to Path
    input_dir = Path(input_dir)
    annotations_dir = Path(annotations_dir)
    output_dir = Path(output_dir)
    output_expected_dir = Path(output_expected_dir)
    task_dir = output_dir / task_name

    # remove output folder (to prevent skipping the conversion)
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)

    # test usage from command line
    cmd = [
        "python", "-m", "picai_prep", "mha2nnunet",
        "--input", input_dir.as_posix(),
        "--output", output_dir.as_posix(),
        "--annotations", annotations_dir.as_posix(),
        "--json", settings_path,
        "--verbose", "2",
    ]

    # run command
    subprocess.check_call(cmd, shell=(sys.platform == 'win32'))

    # check dataset.json
    path_out = task_dir / "dataset.json"
    path_out_expected = output_expected_dir / task_name / "dataset.json"
    with open(path_out) as fp1, open(path_out_expected) as fp2:
        assert json.load(fp1) == json.load(fp2)

    # compare output
    for subject_id in subject_list:
        print(f"Checking case {subject_id}...")
        case_origin, case_direction = None, None

        for modality in ["0000", "0001", "0002"]:
            # construct paths to MHA images
            path_out = task_dir / "imagesTr" / f"{subject_id}_{modality}.nii.gz"
            path_out_expected = output_expected_dir / task_name / "imagesTr" / f"{subject_id}_{modality}.nii.gz"

            # sanity check: check if outputs exist
            assert path_out.exists(), f"Could not find output file at {path_out}!"
            assert path_out_expected.exists(), f"Could not find output file at {path_out_expected}!"

            # read images
            img = sitk.ReadImage(str(path_out))
            img_expected = sitk.ReadImage(str(path_out_expected))

            # compare images
            assert_allclose(sitk.GetArrayFromImage(img_expected), sitk.GetArrayFromImage(img))

            # check origin and direction (nnUNet checks this too)
            if case_origin is None:
                case_origin = img.GetOrigin()
            else:
                assert case_origin == img.GetOrigin(), "Origin must match between sequences!"
            if case_direction is None:
                case_direction = img.GetDirection()
            else:
                assert case_direction == img.GetDirection(), "Direction must match between sequences!"

        # check annotation
        # construct paths
        path_out = task_dir / "labelsTr" / f"{subject_id}.nii.gz"
        path_out_expected = output_expected_dir / task_name / "labelsTr" / f"{subject_id}.nii.gz"

        # sanity check: check if outputs exist
        assert path_out.exists(), f"Could not find output file at {path_out}!"
        assert path_out_expected.exists(), f"Could not find output file at {path_out_expected}!"

        # read images
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(path_out)))
        img_expected = sitk.GetArrayFromImage(sitk.ReadImage(str(path_out_expected)))

        # compare images
        assert_allclose(img_expected, img)


def test_mha2nnunet_inference(
    input_dir: PathLike = "tests/output-expected/mha/ProstateX",
    output_dir: PathLike = "tests/output/nnUNet_raw_data",
    output_expected_dir: PathLike = "tests/output-expected/nnUNet_raw_data",
    settings_path: PathLike = "tests/output-expected/mha2nnunet_inference_settings.json",
    task_name: str = "Task100_test",
    subject_list: Optional[List[str]] = None,
):
    """
    Convert sample MHA archive to nnUNet raw data format (images only)
    """
    if subject_list is None:
        subject_list = [
            "ProstateX-0000_07-07-2011",
            "ProstateX-0001_07-08-2011",
        ]

    # convert input paths to Path
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_expected_dir = Path(output_expected_dir)
    task_dir = output_dir / task_name

    # remove output folder (to prevent skipping the conversion)
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)

    # convert MHA archive to nnUNet raw data
    archive = MHA2nnUNetConverter(
        output_dir=output_dir.as_posix(),
        scans_dir=input_dir.as_posix(),
        scans_out_dirname='imagesTs',
        mha2nnunet_settings=settings_path
    )
    archive.convert()

    # for inference, no dataset.json should be created
    path_out = output_dir / task_name / "dataset.json"
    assert not path_out.exists(), "dataset.json should not be generated in inference mode!"

    # compare output
    for subject_id in subject_list:
        for modality in ["0000", "0001", "0002"]:
            # construct paths to MHA images
            path_out = output_dir / task_name / "imagesTs" / f"{subject_id}_{modality}.nii.gz"
            path_out_expected = output_expected_dir / task_name / "imagesTr" / f"{subject_id}_{modality}.nii.gz"

            # sanity check: check if outputs exist
            assert path_out.exists(), f"Could not find output file at {path_out}!"
            assert path_out_expected.exists(), f"Could not find output file at {path_out_expected}!"

            # read images
            img = sitk.GetArrayFromImage(sitk.ReadImage(str(path_out)))
            img_expected = sitk.GetArrayFromImage(sitk.ReadImage(str(path_out_expected)))

            # compare images
            assert_allclose(img_expected, img)

    # assert no annotations were converted
    path_out = output_dir / task_name / "labelsTs" / f"{subject_id}_{modality}.nii.gz"
    assert not os.path.exists(path_out), "Annotation should not be converted for inference mha2nnunet"

    path_out = output_dir / task_name / "labelsTr" / f"{subject_id}_{modality}.nii.gz"
    assert not os.path.exists(path_out), "Annotation should not be converted for inference mha2nnunet"


def test_mha2nnunet_picai(
    input_dir: PathLike = "tests/input/mha/picai",
    annotations_dir: PathLike = "tests/input/annotations/picai/csPCa_lesion_delineations/human_expert/resampled",
    output_dir: PathLike = "tests/output/nnUNet_raw_data",
    output_expected_dir: PathLike = "tests/output-expected/nnUNet_raw_data",
    settings_path: PathLike = "tests/output-expected/mha2nnunet_inference_settings_picai.json",
    task_name: str = "Task2201_picai_baseline",
):
    """
    Test mha2nnunet preprocessing with tricky cases from the PI-CAI: Public Training and Development Dataset.
    These cases are typicaly very off-centre, and contain different field-of-views between sequences.

    Notes:
    - 10032_1000032 has a different field-of-view between axial T2-weighted and diffusion scans.
    - 10059_1000059 has a padded T2-weighted scan, and different field-of-view between sequences.
    - 10699_1000715_0002.nii.gz contains all diffusion scans, which is an accident of the released MHA scan.
    - 10730_1000746 is similar to 10059_1000059, but padded on the other side.
    - 10868_1000884 is pretty normal. T2-weighted has 21 slices, while diffusion has 19, which is representative.
    - except for 10699_1000715_0002, all cases in output-expected are aligned between sequences.
    """
    # remove output folder (to prevent skipping the conversion)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    picai_archive.generate_mha2nnunet_settings(
        archive_dir=input_dir,
        annotations_dir=annotations_dir,
        output_path=settings_path
    )

    test_mha2nnunet(
        input_dir=input_dir,
        annotations_dir=annotations_dir,
        output_dir=output_dir,
        output_expected_dir=output_expected_dir,
        settings_path=settings_path,
        task_name=task_name,
        subject_list=[
            "10032_1000032",
            "10059_1000059",
            "10699_1000715",
            "10730_1000746",
            "10868_1000884",
        ]
    )


@pytest.mark.parametrize("subject_id", [
    "ProstateX-0000_07-07-2011",
    "ProstateX-0001_07-08-2011",
])
def test_resample_to_reference_scan(
    subject_id: str,
    input_dir: PathLike = "tests/output-expected/mha/ProstateX",
    annotations_dir: PathLike = "tests/input/annotations/ProstateX",
    output_expected_dir: PathLike = "tests/output-expected/nnUNet_raw_data",
    task_name: str = "Task100_test",
):
    """
    Test resample_to_reference_scan by comparing the annotation with
    annotation -> nnUNet preprocessed -> resample_to_reference_scan(annotation)
    """
    # read images
    lbl = sitk.ReadImage(os.path.join(annotations_dir, f"{subject_id}.nii.gz"))
    lbl_prep = sitk.ReadImage(os.path.join(output_expected_dir, task_name, "labelsTr", f"{subject_id}.nii.gz"))
    reference_scan_original = sitk.ReadImage(os.path.join(input_dir, subject_id.split("_")[0], f"{subject_id}_t2w.mha"))
    reference_scan_preprocessed = sitk.ReadImage(os.path.join(output_expected_dir, task_name, "imagesTr", f"{subject_id}_0000.nii.gz"))

    # translate preprocessed label back to original scan's physical space
    lbl_back = resample_to_reference_scan(
        image=lbl_prep,
        reference_scan_original=reference_scan_original,
        reference_scan_preprocessed=reference_scan_preprocessed,
        interpolator=sitk.sitkNearestNeighbor,
    )

    # compare lbl->prep->back with original lbl
    assert_allclose(sitk.GetArrayFromImage(lbl_back), sitk.GetArrayFromImage(lbl))


if __name__ == "__main__":
    test_resample_to_reference_scan("ProstateX-0001_07-08-2011")
