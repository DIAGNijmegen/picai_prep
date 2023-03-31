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

import SimpleITK as sitk
from numpy.testing import assert_allclose
from picai_prep import Dicom2DCEConverter


def test_dce_conversion(
    input_dir: str = "tests/input/dcm/ProstateX",
    output_dir: str = "tests/output/mha/ProstateX",
    output_expected_dir: str = "tests/output-expected/mha/ProstateX",
):
    """
    Convert DCE series to single 4D MHA.
    """
    # remove output folder (to prevent skipping the conversion)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # test usage from command line
    archive = Dicom2DCEConverter(
        input_dir=input_dir,
        output_dir=output_dir,
        dcm2dce_settings="tests/output/dcm2mha_settings.json"
    )
    case = archive.cases[-1]
    case.initialize()
    case.extract_metadata()
    case._convert_dce(output_dir)

    # compare output
    for patient_id, subject_id in [
        ("ProstateX-0218", "ProstateX-0218_02-18-2011"),
    ]:
        for modality in ["dce"]:
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
