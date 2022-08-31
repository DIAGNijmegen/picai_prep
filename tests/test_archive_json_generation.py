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

from picai_prep.data_utils import PathLike
from picai_prep.examples.dcm2mha.sample_archive import \
    generate_dcm2mha_settings
from picai_prep.examples.mha2nnunet.sample_archive import \
    generate_mha2nnunet_settings
from picai_prep.examples.mha2nnunet.sample_archive_inference import \
    generate_mha2nnunet_settings as generate_mha2nnunet_inference_settings


def test_archive_json(
    archive_dir: PathLike = "tests/input/dcm/ProstateX/",
    output_path: PathLike = "tests/output/dcm2mha_settings.json"
):
    """
    Create archive.json for ProstateX sample
    """
    generate_dcm2mha_settings(
        archive_dir=archive_dir,
        output_path=output_path,
    )

    with open(output_path) as fp:
        archive = json.load(fp)

    # compare with expected output
    with open("tests/output-expected/dcm2mha_settings.json") as fp:
        expected_archive = json.load(fp)

    assert archive == expected_archive, "Archive json specification is not as expected!"


def test_mha2nnunet_settings_generation(
    archive_dir: PathLike = "tests/output-expected/mha/ProstateX/",
    output_path: PathLike = "tests/output/mha2nnunet_settings.json"
):
    """
    Create mha2nnunet_settings.json for ProstateX sample
    """
    generate_mha2nnunet_settings(
        archive_dir=archive_dir,
        output_path=output_path
    )

    with open(output_path) as fp:
        archive = json.load(fp)

    # compare with expected output
    with open("tests/output-expected/mha2nnunet_settings.json") as fp:
        expected_archive = json.load(fp)

    assert archive == expected_archive, "Archive json specification is not as expected!"


def test_mha2nnunet_settings_generation_for_inference(
    archive_dir: PathLike = "tests/output-expected/mha/ProstateX/",
    output_path: PathLike = "tests/output/mha2nnunet_inference_settings.json"
):
    """
    Create mha2nnunet_settings.json for ProstateX sample
    """
    generate_mha2nnunet_inference_settings(
        archive_dir=archive_dir,
        output_path=output_path
    )

    with open(output_path) as fp:
        archive = json.load(fp)

    # compare with expected output
    with open("tests/output-expected/mha2nnunet_inference_settings.json") as fp:
        expected_archive = json.load(fp)

    assert archive == expected_archive, "Archive json specification is not as expected!"
