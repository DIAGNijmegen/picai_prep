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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import jsonschema
import SimpleITK as sitk

from picai_prep.converter import Converter
from picai_prep.data_utils import PathLike
from picai_prep.utilities import dcm2mha_schema
from picai_prep.dcm2mha.case import Dicom2MHACase

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


class Dicom2MHAConverter(Converter):
    def __init__(
        self,
        input_dir: PathLike,
        output_dir: PathLike,
        dcm2mha_settings: Union[PathLike, Dict] = None,
        case_class: Case = Dicom2MHACase,
    ):
        """
        Convert DICOM Archive to MHA Archive.
        See https://github.com/DIAGNijmegen/picai_prep for additional documentation.

        Parameters
        ----------
        input_dir: PathLike
            path to the DICOM archive. Used as base path for the relative paths of the archive items.
        output_dir: PathLike
            path to store the resulting MHA archive.
        dcm2mha_settings: Union[PathLike, Dict], default: None
            object with mappings, cases and optional parameters. May be a dictionary containing mappings, archive,
            and optionally options, or a path to a JSON file with these elements.
            - mappings: criteria to map DICOM sequences to their MHA counterparts
            - cases: list of DICOM sequences in the DICOM archive. Each case is to be an object with a patient_id,
                study_id and path to DICOM sequence
            - options: (optional)
                - num_threads: number of multithreading threads.
                    Default: 4.
                - verify_dicom_filenames: whether to check if DICOM filenames contain consecutive
                    numbers.
                    Default: True
                - allow_duplicates: whether multiple DICOM series can map to the same MHA postfix.
                    Default: False
                - metadata_match_func: method to match DICOM metadata to MHA sequences. Only use
                    this if you know what you're doing.
                    Default: check if the observed value matches with any of the allowed values.
                - values_match_func: criteria to consider two values a match, when comparing the
                    value from the DICOM metadata against the provided allowed vaues in the mapping.
                    Default: lower_strip_equals (case-insensitive matching of values without trailing or leading spaces)
                - verbose: control logfile verbosity. 0 does not output a logfile,
                    1 logs cases which have critically failed, 2 logs all cases (may lead to
                    very large log files)
                    Default: 1
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.case_class = case_class

        # parse settings
        if isinstance(dcm2mha_settings, (Path, str)):
            with open(dcm2mha_settings) as fp:
                dcm2mha_settings = json.load(fp)

        jsonschema.validate(dcm2mha_settings, dcm2mha_schema, cls=jsonschema.Draft7Validator)
        self.settings = Dicom2MHASettings(
            mappings=dcm2mha_settings['mappings'],
            **dcm2mha_settings.get('options', {})
        )

        self.cases = self._init_cases(dcm2mha_settings['archive'])

        self.initialize_log(self.output_dir, self.settings.verbose)

    def _init_cases(self, archive: List[Dict]) -> List[Case]:
        cases = {}
        for item in archive:
            key = tuple(item[id] for id in ["patient_id", "study_id"])
            cases[key] = cases.get(key, []) + [item['path']]
        return [
            self.case_class(input_dir=self.input_dir, patient_id=patient_id,
                            study_id=study_id, paths=paths, settings=self.settings)
            for (patient_id, study_id), paths in cases.items()
        ]

    def convert(self) -> None:
        self._convert(
            title='Dicom2MHA',
            cases=self.cases,
            parameters={
                'output_dir': self.output_dir
            },
            num_threads=self.settings.num_threads,
        )
