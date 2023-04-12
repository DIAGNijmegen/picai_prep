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
from typing import Any, Callable, Dict, Optional

import SimpleITK as sitk

from picai_prep.data_utils import PathLike
from picai_prep.preprocessing import PreprocessingSettings


@dataclass
class MHA2nnUNetSettings:
    dataset_json: Dict[str, Any]
    preprocessing: PreprocessingSettings
    scans_dirname: PathLike = "imagesTr"
    annotation_dirname: PathLike = "labelsTr"
    annotation_preprocess_func: Optional[Callable[[sitk.Image], sitk.Image]] = None
    annotation_postprocess_func: Optional[Callable[[sitk.Image], sitk.Image]] = None
    scan_preprocess_func: Optional[Callable[[sitk.Image], sitk.Image]] = None
    scan_postprocess_func: Optional[Callable[[sitk.Image], sitk.Image]] = None
    num_threads: int = 4
    verbose: int = 1

    @property
    def task_name(self) -> str:
        return self.dataset_json['task']
