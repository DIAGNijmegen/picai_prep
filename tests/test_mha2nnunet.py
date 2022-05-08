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
import json
import shutil
from pathlib import Path
from numpy.testing import assert_allclose
import SimpleITK as sitk

from picai_prep.data_utils import PathLike
from picai_prep.mha2nnunet import MHA2nnUNetConverter


def test_mha2nnunet(
    input_dir: PathLike = "tests/output-expected/mha/ProstateX",
    annotations_dir: PathLike = "tests/input/annotations/ProstateX",
    output_dir: PathLike = "tests/output/nnUNet_raw_data",
    output_expected_dir: PathLike = "tests/output-expected/nnUNet_raw_data/Task100_test",
):
    """
    Convert sample MHA archive to nnUNet raw data format
    """
    # convert input paths to Path
    input_dir = Path(input_dir)
    annotations_dir = Path(annotations_dir)
    output_dir = Path(output_dir)
    output_expected_dir = Path(output_expected_dir)

    # remove output folder (to prevent skipping the conversion)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # convert MHA archive to nnUNet raw data
    archive = MHA2nnUNetConverter(
        input_path=input_dir.as_posix(),
        annotations_path=annotations_dir.as_posix(),
        output_path=output_dir.as_posix(),
        settings_path="tests/output-expected/mha2nnunet_settings.json"
    )
    archive.convert()

    # check dataset.json
    path_out = output_dir / "Task100_test" / "dataset.json"
    path_out_expected = output_expected_dir / "dataset.json"
    with open(path_out) as fp1, open(path_out_expected) as fp2:
        assert json.load(fp1) == json.load(fp2)

    # compare output
    for subject_id in [
        "ProstateX-0000_07-07-2011",
        "ProstateX-0001_07-08-2011",
    ]:
        for modality in ["0000", "0001", "0002"]:
            # construct paths to MHA images
            path_out = output_dir / "Task100_test" / "imagesTr" / f"{subject_id}_{modality}.nii.gz"
            path_out_expected = output_expected_dir / "imagesTr" / f"{subject_id}_{modality}.nii.gz"

            # sanity check: check if outputs exist
            assert path_out.exists(), f"Could not find output file at {path_out}!"
            assert path_out_expected.exists(), f"Could not find output file at {path_out_expected}!"

            # read images
            img = sitk.GetArrayFromImage(sitk.ReadImage(str(path_out)))
            img_expected = sitk.GetArrayFromImage(sitk.ReadImage(str(path_out_expected)))

            # compare images
            assert_allclose(img_expected, img)

        # check annotation
        # construct paths
        path_out = output_dir / "Task100_test" / "labelsTr" / f"{subject_id}.nii.gz"
        path_out_expected = output_expected_dir / "labelsTr" / f"{subject_id}.nii.gz"

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
    output_expected_dir: PathLike = "tests/output-expected/nnUNet_raw_data/Task100_test",
):
    """
    Convert sample MHA archive to nnUNet raw data format (images only)
    """
    # convert input paths to Path
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_expected_dir = Path(output_expected_dir)

    # remove output folder (to prevent skipping the conversion)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # convert MHA archive to nnUNet raw data
    archive = MHA2nnUNetConverter(
        input_path=input_dir.as_posix(),
        output_path=output_dir.as_posix(),
        out_dir_scans="imagesTs",
        settings_path="tests/output-expected/mha2nnunet_inference_settings.json"
    )
    archive.convert()

    # check dataset.json
    path_out = output_dir / "Task100_test" / "dataset.json"
    path_out_expected = output_expected_dir / "dataset.json"
    with open(path_out) as fp1, open(path_out_expected) as fp2:
        assert json.load(fp1) == json.load(fp2)

    # compare output
    for subject_id in [
        "ProstateX-0000_07-07-2011",
        "ProstateX-0001_07-08-2011",
    ]:
        for modality in ["0000", "0001", "0002"]:
            # construct paths to MHA images
            path_out = output_dir / "Task100_test" / "imagesTs" / f"{subject_id}_{modality}.nii.gz"
            path_out_expected = output_expected_dir / "imagesTr" / f"{subject_id}_{modality}.nii.gz"

            # sanity check: check if outputs exist
            assert path_out.exists(), f"Could not find output file at {path_out}!"
            assert path_out_expected.exists(), f"Could not find output file at {path_out_expected}!"

            # read images
            img = sitk.GetArrayFromImage(sitk.ReadImage(str(path_out)))
            img_expected = sitk.GetArrayFromImage(sitk.ReadImage(str(path_out_expected)))

            # compare images
            assert_allclose(img_expected, img)

    # assert no annotations were converted
    path_out = output_dir / "Task100_test" / "labelsTs" / f"{subject_id}_{modality}.nii.gz"
    assert not os.path.exists(path_out), "Annotation should not be converted for inference mha2nnunet"

    path_out = output_dir / "Task100_test" / "labelsTr" / f"{subject_id}_{modality}.nii.gz"
    assert not os.path.exists(path_out), "Annotation should not be converted for inference mha2nnunet"
