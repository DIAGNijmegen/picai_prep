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
import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

from picai_prep.data_utils import PathLike, atomic_image_write
from picai_prep.exceptions import (CriticalErrorInSiblingError,
                                   MissingDICOMFilesError, NoMappingsApplyError,
                                   UnreadableDICOMError)
from picai_prep.utilities import plural
from picai_prep.dcm2mha.series import Series
from picai_prep.dcm2mha.settings import Dicom2MHASettings
from picai_prep.case import Case
from picai_prep.imagereader import DICOMImageReader

Metadata = Dict[str, str]
Mapping = Dict[str, List[str]]
Mappings = Dict[str, Mapping]

@dataclass
class _Dicom2MHACaseBase:
    input_dir: Path
    paths: List[PathLike]
    settings: Dicom2MHASettings


@dataclass
class Dicom2MHACase(Case, _Dicom2MHACaseBase):
    series: List[Series] = field(default_factory=list)

    def convert_item(self, output_dir: Path) -> None:
        self.initialize()
        self.extract_metadata()
        self.apply_mappings()
        self.resolve_duplicates()
        self.process_and_write(output_dir)

    def initialize(self) -> None:
        self.write_log(f'Importing {plural(len(self.paths), "serie")}')

        full_paths = set()
        for path in self.paths:
            full_path = self.input_dir / path
            serie = Series(full_path, self.patient_id, self.study_id)

            # if we find duplicate paths with the same patient and study id,
            # invalidate this series and continue for logging purposes
            if path in full_paths:
                raise FileExistsError(path)
            full_paths.add(full_path)
            self.write_log(f'\t+ ({len(self.series)}) {full_path}')
            self.series.append(serie)

        if not self.is_valid:
            self.invalidate()

    def extract_metadata(self) -> None:
        self.write_log(f'Extracting metadata from {plural(len(self.valid_series), "serie")}')
        errors = []

        for i, serie in enumerate(self.valid_series):
            try:
                serie.extract_metadata(
                    verify_dicom_filenames=self.settings.verify_dicom_filenames
                )
            except (MissingDICOMFilesError, UnreadableDICOMError) as e:
                serie.error = e
                errors.append(i)

        self.write_log(f'\t({plural(len(errors), "error")}{f" {errors}" if len(errors) > 0 else ""})')

    def apply_mappings(self) -> None:
        self.write_log(f'Applying mappings to {len(self.valid_series)} series')
        errors = []

        for i, serie in enumerate(self.valid_series):
            try:
                serie.apply_mappings(
                    mappings=self.settings.mappings,
                    metadata_match_func=self.settings.metadata_match_func,
                    values_match_func=self.settings.values_match_func,
                )
            except NoMappingsApplyError as e:
                serie.error = e
                errors.append(i)

        self.write_log(f'\t({plural(len(errors), "error")}{f" {errors}" if len(errors) > 0 else ""})')

    def resolve_duplicates(self) -> None:
        self.write_log(f'Resolving duplicates between {plural(len(self.valid_series), "serie")}')

        # define tiebreakers, which should have: name, value_func, pick_largest
        tiebreakers = [
            ('slice count', lambda a: len(a.filenames), True),
            ('image resolution', lambda a: np.prod(a.spacing_inplane), False),
            ('filename', lambda a: str(a.path), False),
        ]

        # create dict collecting all items for each mapping
        matched_series: Dict[str, List[Series]] = {
            mapping: [] for serie in self.valid_series for mapping in serie.mappings
        }
        for serie in self.valid_series:
            for mapping in serie.mappings:
                matched_series[mapping] += [serie]

        # use tiebreakers to select a single item for each mapping
        for mapping, series in matched_series.items():
            if self.settings.allow_duplicates:
                for i, serie in enumerate(series):
                    serie.mappings.remove(mapping)
                    serie.mappings.append(f'{mapping}_{i}')
            else:
                for name, value_func, pick_largest in tiebreakers:
                    if len(series) > 1:
                        serie_value_pairs = [(serie, value_func(serie)) for serie in series]
                        serie_value_pairs.sort(key=lambda a: a[1], reverse=pick_largest)
                        _, best_value = serie_value_pairs[0]

                        for serie, value in serie_value_pairs:
                            if value != best_value:
                                serie.mappings.remove(mapping)
                                serie.write_log(f'Removed by {name} tiebreaker from "{mapping}"')
                                series.remove(serie)

    def process_and_write(self, output_dir: Path) -> None:
        total = sum([len(serie.mappings) for serie in self.valid_series])
        self.write_log(f'Writing {plural(total, "serie")}')
        errors, skips = [], []

        patient_dir = output_dir / self.patient_id
        for i, serie in enumerate(self.valid_series):
            for mapping in serie.mappings:
                mapping_save_name = mapping
                if ":" in mapping_save_name:
                    mapping_save_name = mapping_save_name.split(':')[0]
                dst_path = patient_dir / f"{self.subject_id}_{mapping_save_name}.mha"
                if dst_path.exists():
                    serie.write_log(f'Skipped "{mapping}", already exists: {dst_path}')
                    skips.append(i)
                    continue

                try:
                    image = DICOMImageReader(serie.path, verify_dicom_filenames=self.settings.verify_dicom_filenames).image
                except Exception as e:
                    serie.write_log(
                        f'Skipped "{mapping}", reading DICOM sequence failed, maybe corrupt data? Error: {e}')
                    if self.settings.verbose >= 2:
                        logging.error(traceback.format_exc())
                    else:
                        logging.error(str(e))
                    errors.append(i)
                else:
                    if self.settings.scan_postprocess_func is not None:
                        image = self.settings.scan_postprocess_func(image)
                    try:
                        atomic_image_write(image=image, path=dst_path, mkdir=True)
                    except Exception as e:
                        serie.write_log(f'Skipped "{mapping}", write error: {e}')
                        logging.error(str(e))
                        errors.append(i)
                    else:
                        serie.write_log(f'Wrote image to {dst_path}')

        self.write_log(f'Wrote {total - len(errors) - len(skips)} MHA files to {patient_dir.as_posix()}\n'
                       f'\t({plural(len(errors), "error")}{f" {errors}" if len(errors) > 0 else ""}, '
                       f'{len(skips)} skipped{f" {skips}" if len(skips) > 0 else ""})')

    def invalidate(self, error: Exception = None) -> None:
        if error is None:
            error = CriticalErrorInSiblingError()
        for serie in self.valid_series:
            serie.error = error

    @property
    def subject_id(self) -> str:
        return f"{self.patient_id}_{self.study_id}"

    @property
    def is_valid(self) -> bool:
        return all([serie.is_valid for serie in self.series])

    @property
    def valid_series(self) -> List[Series]:
        return [item for item in self.series if item.is_valid]

    def write_log(self, msg: str) -> None:
        self._log.append(msg)

    def compile_log(self) -> str:
        """For questions: Stan.Noordman@Radboudumc.nl"""
        if self.settings.verbose == 0:
            return

        divider = '=' * 120
        summary = {}
        serie_log = []

        # summarize each serie's log (if any)
        for i, serie in enumerate(self.series):
            serie_log.append(f'({i}) {serie.compile_log()}')
            if serie.error:
                summary[serie.error.__class__.__name__] = summary.get(serie.error.__class__.__name__, []) + [i]

        # these are the errors that are not fatal
        ignored_errors = {e.__name__ for e in [NoMappingsApplyError]}

        # check if we should log any errors
        if len(set(summary.keys()).difference(ignored_errors)) > 0 or self.settings.verbose >= 2:
            # don't worry, this just looks nice in the log
            return '\n'.join([divider,
                              f'CASE {self.patient_id}_{self.study_id}',
                              f'\tPATIENT ID\t{self.patient_id}',
                              f'\tSTUDY ID\t{self.study_id}\n',
                              *self._log,
                              '\nSERIES', divider.replace('=', '-'),
                              'Errors found:',
                              *[f'\t{key}: {value}' for key, value in summary.items()],
                              '', *serie_log, ''])

    def cleanup(self) -> None:
        self.series = None
        super().cleanup()

    def __repr__(self) -> str:
        return f'Case({self.subject_id})'
