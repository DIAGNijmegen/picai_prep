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
from numpy.testing import assert_allclose
from pathlib import Path
import SimpleITK as sitk
import pytest

from picai_prep.preprocessing import Sample, PreprocessingSettings
from picai_prep.data_utils import atomic_image_write


@pytest.mark.parametrize("subject_id", [
    "ProstateX-0000_07-07-2011-NA-MR prostaat kanker detectie WDSmc MCAPRODETW-05711",
    "ProstateX-0001_07-08-2011-NA-MR prostaat kanker detectie WDSmc MCAPRODETW-95738",
])
def test_preprocessing_for_nnunet_inference(subject_id):
    """
    Preprocess mpMRI scans (align, resample, crop, ...)
    """
    # setup
    input_dir = Path("tests/output/mha/ProstateX")
    output_dir = Path("tests/output/nnUNet_raw_data-preprocess_mpMRI_study")
    expected_output_dir = Path("tests/output-expected/nnUNet_raw_data/Task100_test/imagesTr")
    spacing = (3.0, 0.5, 0.5)
    matrix_size = (20, 160, 160)
    physical_size = [
        voxel_spacing * size
        for voxel_spacing, size in zip(spacing, matrix_size)
    ]

    # pack info
    patient_id = subject_id.split('_')[0]
    all_scan_properties = [
        {
            'input_path': input_dir / patient_id / f"{subject_id}_t2w.mha",
            'output_path': output_dir / f"{subject_id}_0000.nii.gz",
            'type': 'T2W',
        },
        {
            'input_path': input_dir / patient_id / f"{subject_id}_adc.mha",
            'output_path': output_dir / f"{subject_id}_0001.nii.gz",
            'type': 'ADC',
        },
        {
            'input_path': input_dir / patient_id / f"{subject_id}_hbv.mha",
            'output_path': output_dir / f"{subject_id}_0002.nii.gz",
            'type': 'HBV',
        },
    ]

    # read images
    scans = []
    for scan_properties in all_scan_properties:
        scans += [sitk.ReadImage(str(scan_properties['input_path']))]

    # perform preprocessing
    sample = Sample(
        scans=scans,
        lbl=None,
        settings=PreprocessingSettings(
            physical_size=physical_size,
            spacing=spacing,
        ),
        name=subject_id,
    )
    sample.preprocess()

    assert sample.lbl is None, "Label was created out of thin air!"

    # write images
    for scan, scan_properties in zip(sample.scans, all_scan_properties):
        atomic_image_write(scan, path=scan_properties['output_path'], mkdir=True)

    # verify result
    for modality in ["0000", "0001", "0002"]:
        path_out = output_dir / f"{subject_id}_{modality}.nii.gz"
        path_out_expected = expected_output_dir / f"{subject_id}_{modality}.nii.gz"

        # sanity check: check if outputs exist
        assert os.path.exists(path_out), f"Could not find output file at {path_out}!"
        assert os.path.exists(path_out_expected), f"Could not find output file at {path_out_expected}!"

        # read images
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(path_out)))
        img_expected = sitk.GetArrayFromImage(sitk.ReadImage(str(path_out_expected)))

        # compare images
        assert_allclose(img_expected, img)
