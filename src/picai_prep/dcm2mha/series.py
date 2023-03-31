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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union
from picai_prep.exceptions import (ArchiveItemPathNotFoundError, NoMappingsApplyError)
from picai_prep.imagereader import DICOMImageReader

Metadata = Dict[str, str]
Mapping = Dict[str, List[str]]
Mappings = Dict[str, Mapping]

@dataclass
class Series:
    path: Path
    patient_id: str
    study_id: str

    # image metadata
    filenames: Optional[List[str]] = None
    spacing_inplane: Optional[Sequence[float]] = None
    metadata: Optional[Metadata] = field(default_factory=dict)

    mappings: List[str] = field(default_factory=list)
    error: Optional[Exception] = None
    _log: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.path.exists():
            raise ArchiveItemPathNotFoundError(self.path)
        if not self.path.is_dir():
            raise NotADirectoryError(self.path)

    def extract_metadata(self, verify_dicom_filenames: bool = True) -> None:
        """
        Verify DICOM slices and extract metadata from the last DICOM slice
        """
        # try:
        reader = DICOMImageReader(self.path, verify_dicom_filenames=verify_dicom_filenames)
        self.metadata = reader.metadata
        self.spacing_inplane = self.metadata["spacing_inplane"]
        self.filenames = reader.dicom_slice_paths
        self.write_log('Extracted metadata')

    @staticmethod
    def metadata_matches(
        metadata: Metadata,
        mapping: Mapping,
        values_match_func: Callable[[str, str], bool],
    ) -> bool:
        """
        Determine whether Series' metadata matches the mapping.
        By default, values are trimmed from whitespace and case-insensitively compared.
        """
        for dicom_tag, allowed_values in mapping.items():
            dicom_tag = dicom_tag.lower().strip()
            if dicom_tag not in metadata:
                # metadata does not exist
                return False

            # check if the observed value matches with any of the allowed values
            if not any(values_match_func(needle=value, haystack=metadata[dicom_tag]) for value in allowed_values):
                return False

        return True

    def apply_mappings(
        self,
        mappings: Mappings,
        metadata_match_func: Optional[Callable[[Metadata, Mappings], bool]] = None,
        values_match_func: Optional[Union[Callable[[str, str], bool], str]] = "lower_strip_equals",
    ) -> None:
        """
        Apply mappings to the series
        """
        # resolve metadata match function
        if metadata_match_func is None:
            metadata_match_func = self.metadata_matches

        # resolve value match function
        if isinstance(values_match_func, str):
            variant = values_match_func

            def values_match_func(needle: str, haystack: str) -> bool:
                if "lower" in variant:
                    needle = needle.lower()
                    haystack = haystack.lower()
                if "strip" in variant:
                    needle = needle.strip()
                    haystack = haystack.strip()
                if "equals" in variant:
                    return needle == haystack
                elif "contains" in variant:
                    return needle in haystack
                elif "regex" in variant:
                    return re.search(needle, haystack) is not None
                else:
                    raise ValueError(f'Unknown values match function variant {variant}')

        for name, mapping in mappings.items():
            if metadata_match_func(metadata=self.metadata, mapping=mapping, values_match_func=values_match_func):
                self.mappings.append(name)

        if len(self.mappings) == 0:
            raise NoMappingsApplyError()
        self.write_log(f'Applied mappings [{", ".join(self.mappings)}]')

    def write_log(self, msg: str) -> None:
        self._log.append(msg)

    def compile_log(self) -> str:
        log = [f'\t{item}' for item in self._log]
        return '\n'.join([self.path.as_posix()] + log + [f'\tFATAL: {self.error}\n' if not self.is_valid else ''])

    @property
    def is_valid(self) -> bool:
        return self.error is None

    def __repr__(self) -> str:
        return f"Series({self.path.name})"
