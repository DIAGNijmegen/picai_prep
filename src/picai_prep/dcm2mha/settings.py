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
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import SimpleITK as sitk

Metadata = Dict[str, str]
Mapping = Dict[str, List[str]]
Mappings = Dict[str, Mapping]


@dataclass
class Dicom2MHASettings:
    mappings: Dict[str, Dict[str, List[str]]]
    verify_dicom_filenames: bool = True
    allow_duplicates: bool = False
    metadata_match_func: Optional[Callable[[Metadata, Mappings], bool]] = None
    values_match_func: Union[str, Callable[[str, str], bool]] = "lower_strip_equals"
    scan_postprocess_func: Optional[Callable[[sitk.Image], sitk.Image]] = None
    num_threads: int = 4
    verbose: int = 1

    def __post_init__(self):
        # Validate the mappings
        for mapping_name, mapping in self.mappings.items():
            for key, values in mapping.items():
                if not isinstance(values, list):
                    raise ValueError(f'Mapping {mapping_name} has non-list values for key {key}: {values}')
                if not all([isinstance(value, str) for value in values]):
                    raise ValueError(f'Mapping {mapping_name} has non-string value for key {key}: {values}')