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


from typing import Tuple

import pydicom
import SimpleITK as sitk

from picai_prep.resources.dcm2mha_schema import dcm2mha_schema
from picai_prep.resources.dicom_tags import dicom_tags
from picai_prep.resources.mha2nnunet_schema import mha2nnunet_schema


def lower_strip(s: str):
    return s.lower().strip()


def plural(num: int, word: str):
    return f"{num} {word}{'' if num == 1 else 's'}"


def get_pydicom_value(data: pydicom.dataset.Dataset, key: str):
    key = '0x' + key.replace('|', '')
    if key in data:
        result = data[key]
        return result.value if not result.is_empty else None
    return None


def make_sitk_readers() -> Tuple[sitk.ImageFileReader, sitk.ImageSeriesReader]:
    """Initialise SimpleITK series and file readers"""
    series_reader = sitk.ImageSeriesReader()
    file_reader = sitk.ImageFileReader()

    for reader in (series_reader, file_reader):
        reader.LoadPrivateTagsOn()

    return file_reader, series_reader


__all__ = [
    # Explicitly expose these functions for easier imports
    "dcm2mha_schema",
    "dicom_tags",
    "mha2nnunet_schema",
]
