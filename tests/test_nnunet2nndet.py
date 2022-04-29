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
import json
from numpy.testing import assert_allclose
from pathlib import Path
import SimpleITK as sitk
from typing import Optional

from picai_prep.nnunet2nndet import nnunet2nndet
from picai_prep.data_utils import PathLike


def test_nnunet2nndet(
    nnunet_raw_data_path: PathLike = "tests/output-expected/nnUNet_raw_data/Task100_test",
    nndet_raw_data_path: PathLike = "tests/output/nnDet_raw_data/Task100_test",
    in_dir_annot: Optional[PathLike] = "labelsTr",
    out_dir_annot: Optional[PathLike] = "raw_splitted/labelsTr",
    in_dir_scans: Optional[PathLike] = "imagesTr",
    out_dir_scans: Optional[PathLike] = "raw_splitted/imagesTr",
    nndet_raw_data_path_expected: PathLike = "tests/output-expected/nnDet_raw_data/Task100_test",
):
    # convert input paths to Path
    nnunet_raw_data_path = Path(nnunet_raw_data_path)
    nndet_raw_data_path = Path(nndet_raw_data_path)
    nndet_raw_data_path_expected = Path(nndet_raw_data_path_expected)

    # remove output folder (to prevent skipping the conversion)
    if os.path.exists(nndet_raw_data_path):
        shutil.rmtree(nndet_raw_data_path)

    # perform conversion from nnUNet to nnDet raw data
    nnunet2nndet(
        nnunet_raw_data_path=nnunet_raw_data_path,
        nndet_raw_data_path=nndet_raw_data_path,
        in_dir_annot=in_dir_annot,
        out_dir_annot=out_dir_annot,
        in_dir_scans=in_dir_scans,
        out_dir_scans=out_dir_scans,
    )

    # check dataset.json
    path_out = nndet_raw_data_path / "dataset.json"
    path_out_expected = nndet_raw_data_path_expected / "dataset.json"
    with open(path_out) as fp1, open(path_out_expected) as fp2:
        assert json.load(fp1) == json.load(fp2)

    # compare output
    for subject_id in [
        "ProstateX-0000_07-07-2011-NA-MR prostaat kanker detectie WDSmc MCAPRODETW-05711",
        "ProstateX-0001_07-08-2011-NA-MR prostaat kanker detectie WDSmc MCAPRODETW-95738",
    ]:
        # compare annotations
        for extension in ["nii.gz", "json"]:
            # construct paths
            path_out = nndet_raw_data_path / out_dir_annot / f"{subject_id}.{extension}"
            path_out_expected = nndet_raw_data_path_expected / out_dir_annot / f"{subject_id}.{extension}"

            # sanity check: check if outputs exist
            assert os.path.exists(path_out), f"Could not find output file at {path_out}!"
            assert os.path.exists(path_out_expected), f"Could not find expected output file at {path_out_expected}!"

            if extension == "nii.gz":
                # read images
                img = sitk.GetArrayFromImage(sitk.ReadImage(str(path_out)))
                img_expected = sitk.GetArrayFromImage(sitk.ReadImage(str(path_out_expected)))

                # compare images
                assert_allclose(img_expected, img)
            elif extension == "json":
                with open(path_out) as fp1, open(path_out_expected) as fp2:
                    json_out = json.load(fp1)
                    json_expected = json.load(fp2)
                    assert json_out == json_expected
            else:
                raise ValueError(f"Unexpected extension: {extension}")

        # compare scans
        for modality in ["0000", "0001", "0002"]:
            # construct paths to MHA images
            path_out = nndet_raw_data_path / out_dir_scans / f"{subject_id}_{modality}.nii.gz"
            path_out_expected = nnunet_raw_data_path / in_dir_scans / f"{subject_id}_{modality}.nii.gz"

            # sanity check: check if outputs exist
            assert os.path.exists(path_out), f"Could not find output file at {path_out}!"
            assert os.path.exists(path_out_expected), f"Could not find expected output file at {path_out_expected}!"

            # read images
            img = sitk.GetArrayFromImage(sitk.ReadImage(str(path_out)))
            img_expected = sitk.GetArrayFromImage(sitk.ReadImage(str(path_out_expected)))

            # compare images
            assert_allclose(img_expected, img)
