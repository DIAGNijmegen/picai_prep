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
from typing import Dict, Union

from picai_prep.converter import Case
from picai_prep.data_utils import PathLike
from picai_prep.dcm2mha.converter import Dicom2MHAConverter
from picai_prep.dcm2dce.case import Dicom2DCECase


class Dicom2DCEConverter(Dicom2MHAConverter):
    def __init__(
        self,
        input_dir: PathLike,
        output_dir: PathLike,
        dcm2dce_settings: Union[PathLike, Dict] = None,
        case_class: Case = Dicom2DCECase,
    ):
        """
        Convert DCE scans from a DICOM Archive to a single 4D MHA scan.
        Experimental.
        """
        super().__init__(
            input_dir=input_dir,
            output_dir=output_dir,
            dcm2mha_settings=dcm2dce_settings,
            case_class=case_class
        )

    def convert(self):
        self._convert(
            title='Dicom2DCE',
            cases=self.cases,
            parameters={
                'output_dir': self.output_dir
            },
            num_threads=self.settings.num_threads,
        )
