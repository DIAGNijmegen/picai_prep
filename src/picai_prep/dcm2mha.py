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
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import jsonschema
import numpy as np
import pydicom.errors
import SimpleITK as sitk

from picai_prep.converter import Case, Converter
from picai_prep.data_utils import PathLike, atomic_image_write
from picai_prep.errors import (ArchiveItemPathNotFoundError,
                               CriticalErrorInSiblingError,
                               MissingDICOMFilesError, NoMappingsApplyError,
                               UnreadableDICOMError)
from picai_prep.utilities import (dcm2mha_schema, dicom_tags,
                                  get_pydicom_value, lower_strip,
                                  make_sitk_readers, plural)

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


@dataclass
class Series:
    path: Path
    patient_id: str
    study_id: str

    # image metadata
    filenames: Optional[List[str]] = None
    spacing: Optional[Sequence[float]] = None
    metadata: Optional[Metadata] = field(default_factory=dict)

    mappings: List[str] = field(default_factory=list)

    error: Optional[Exception] = None

    _log: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.path.exists():
            raise ArchiveItemPathNotFoundError(self.path)
        if not self.path.is_dir():
            raise NotADirectoryError(self.path)

    def verify_dicom_filenames(self, filenames: List[PathLike]) -> bool:
        """Verify DICOM filenames have increasing numbers, with no gaps"""
        vdcms = [d.rsplit('.', 1)[0] for d in filenames]
        vdcms = [int(''.join(c for c in d if c.isdigit())) for d in vdcms]
        missing_slices = False
        for num in range(min(vdcms), max(vdcms) + 1):
            if num not in vdcms:
                missing_slices = True
                break
        if missing_slices:
            raise MissingDICOMFilesError(self.path)
        return True

    def extract_metadata(self, verify_dicom_filenames: bool = True) -> None:
        """
        Verify DICOM slices and extract metadata from the last DICOM slice
        """
        file_reader, series_reader = make_sitk_readers()
        self.filenames = [os.path.basename(dcm) for dcm in series_reader.GetGDCMSeriesFileNames(str(self.path))]

        # verify DICOM files are found
        if len(self.filenames) == 0:
            raise MissingDICOMFilesError(self.path)

        if verify_dicom_filenames:
            # verify DICOM filenames have increasing numbers, with no gaps
            self.verify_dicom_filenames(self.filenames)

        # extract metadata from last DICOM slice
        dicom_slice_path = self.path / self.filenames[-1]

        try:
            file_reader.SetFileName(str(dicom_slice_path))
            file_reader.ReadImageInformation()
            self.spacing = file_reader.GetSpacing()

            for key in file_reader.GetMetaDataKeys():
                # collect all available metadata (with DICOM tags, e.g. 0010|1010, as keys)
                self.metadata[key] = file_reader.GetMetaData(key)
            for name, key in dicom_tags.items():
                # collect metadata with DICOM names, e.g. patientsage, as keys)
                self.metadata[name] = file_reader.GetMetaData(key) if file_reader.HasMetaDataKey(key) else ''
        except Exception as e:
            self.write_log(f"Reading with SimpleITK failed for {self.path} with error: {e}. Attempting with pydicom.")
            try:
                with pydicom.dcmread(dicom_slice_path) as data:
                    self.spacing = data.PixelSpacing
                    for name, key in dicom_tags.items():
                        self.metadata[name] = get_pydicom_value(data, key)
            except pydicom.errors.InvalidDicomError:
                e = UnreadableDICOMError(self.path)
                self.error = e
                logging.error(str(e))

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
            dicom_tag = lower_strip(dicom_tag)
            if dicom_tag not in metadata:
                # metadata does not contain the information we need
                return False

            # check if observed value is in the list of allowed values
            if not any(values_match_func(needle=value, haystack=metadata[dicom_tag]) for value in allowed_values):
                return False

        return True

    def apply_mappings(
        self,
        mappings: Mappings,
        metadata_match_func: Optional[Callable[[Metadata, Mappings], bool]] = None,
        values_match_func: Optional[Callable[[str, str], bool]] = None,
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

    def write_log(self, msg: str):
        self._log.append(msg)

    def compile_log(self):
        log = [f'\t{item}' for item in self._log]
        return '\n'.join([self.path.as_posix()] + log + [f'\tFATAL: {self.error}\n' if not self.is_valid else ''])

    @property
    def is_valid(self):
        return self.error is None

    def __repr__(self):
        return f"Series({self.path.name})"


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

    def initialize(self):
        self.write_log(f'Importing {plural(len(self.paths), "serie")}')

        full_paths = set()
        for path in self.paths:
            full_path = self.input_dir / path
            serie = Series(full_path, self.patient_id, self.study_id)
            try:
                # if we find duplicate paths with the same patient and study id,
                # invalidate this series and continue for logging purposes
                if path in full_paths:
                    raise FileExistsError(path)
                full_paths.add(full_path)
            except Exception as e:
                serie.error = e
                logging.error(str(e))
            finally:
                self.write_log(f'\t+ ({len(self.series)}) {full_path}')
                self.series.append(serie)

        if not self.is_valid:
            self.invalidate()

    def extract_metadata(self):
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

    def apply_mappings(self):
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

    def resolve_duplicates(self):
        self.write_log(f'Resolving duplicates between {plural(len(self.valid_series), "serie")}')

        # define tiebreakers, which should have: name, value_func, pick_largest
        tiebreakers = [
            ('slice count', lambda a: len(a.filenames), True),
            ('image resolution', lambda a: np.prod(a.spacing), False),
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

    def process_and_write(self, output_dir: Path):
        total = sum([len(serie.mappings) for serie in self.valid_series])
        self.write_log(f'Writing {plural(total, "serie")}')
        errors, skips = [], []

        patient_dir = output_dir / self.patient_id
        for i, serie in enumerate(self.valid_series):
            for mapping in serie.mappings:
                dst_path = patient_dir / f"{self.subject_id}_{mapping}.mha"
                if dst_path.exists():
                    serie.write_log(f'Skipped "{mapping}", already exists: {dst_path}')
                    skips.append(i)

                try:
                    image = read_image_series(serie.path)
                except Exception as e:
                    serie.write_log(
                        f'Skipped "{mapping}", reading DICOM sequence failed, maybe corrupt data? Error: {e}')
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

    def invalidate(self):
        for serie in self.valid_series:
            serie.error = CriticalErrorInSiblingError()

    @property
    def subject_id(self) -> str:
        return f"{self.patient_id}_{self.study_id}"

    @property
    def is_valid(self):
        return all([serie.is_valid for serie in self.series])

    @property
    def valid_series(self):
        return [item for item in self.series if item.is_valid]

    def write_log(self, msg: str):
        self._log.append(msg)

    def compile_log(self):
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

    def __repr__(self):
        return f'Case({self.subject_id})'


class Dicom2MHAConverter(Converter):
    def __init__(
        self,
        input_dir: PathLike,
        output_dir: PathLike,
        dcm2mha_settings: Union[PathLike, Dict] = None,
        case_class: Case = Dicom2MHACase,
    ):
        """
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
                    Default: None
                - values_match_func: criteria to consider two values a match, when comparing the
                    value from the DICOM metadata against the provided allowed vaues in the mapping.
                    Default: None
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

    def convert(self):
        self._convert(
            title='Dicom2MHA',
            cases=self.cases,
            parameters={
                'output_dir': self.output_dir
            },
            num_threads=self.settings.num_threads,
        )


def read_image_series(image_series_path: PathLike) -> sitk.Image:
    file_reader, series_reader = make_sitk_readers()
    dicom_slice_paths = series_reader.GetGDCMSeriesFileNames(str(image_series_path))

    try:
        series_reader.SetFileNames(dicom_slice_paths)
        image: sitk.Image = series_reader.Execute()

        file_reader.SetFileName(dicom_slice_paths[-1])
        dicom_slice: sitk.Image = file_reader.Execute()
        for key in dicom_slice.GetMetaDataKeys():
            if len(dicom_slice.GetMetaData(key)) > 0:
                image.SetMetaData(key, dicom_slice.GetMetaData(key))
    except RuntimeError:
        files = [pydicom.dcmread(dcm) for dcm in dicom_slice_paths]

        # skip files with no SliceLocation (eg. scout views)
        slices = filter(lambda a: hasattr(a, 'SliceLocation'), files)
        slices = sorted(slices, key=lambda s: s.SliceLocation)

        # create and fill 3D array
        image = np.zeros([len(slices)] + list(slices[0].pixel_array.shape))
        for i, s in enumerate(slices):
            image[i, :, :] = s.pixel_array

        # convert to SimpleITK
        image: sitk.Image = sitk.GetImageFromArray(image)
        image.SetSpacing(list(slices[0].PixelSpacing) + [slices[0].SliceThickness])

        for key in dicom_tags.values():
            value = get_pydicom_value(files[0], key)
            if value is not None:
                image.SetMetaData(key, value)

    return image
