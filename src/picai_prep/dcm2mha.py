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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import jsonschema
import numpy as np
import pydicom.errors
import SimpleITK as sitk
from tqdm import tqdm

from picai_prep.data_utils import PathLike, atomic_image_write
from picai_prep.utilities import (dcm2mha_schema, dicom_tags,
                                  get_pydicom_value, lower_strip,
                                  make_sitk_readers, metadata_defaults, plural)

Metadata = Dict[str, str]
Mapping = Dict[str, List[str]]
Mappings = Dict[str, Mapping]


class SeriesException(Exception):
    """Base Exception for errors in an item (series within a case)"""

    def __init__(self, message: str):
        super().__init__(message)

    def __str__(self):
        return f'{type(self).__name__}: {", ".join([a for a in self.args])}'


class MissingDICOMFilesError(SeriesException):
    """Exception raised when a DICOM series has missing DICOM slices"""

    def __init__(self, path: PathLike):
        super().__init__(f"Missing DICOM slices detected in {path}")


class NoMappingsApplyError(SeriesException):
    """Exception raised when no mappings apply to the case"""

    def __init__(self):
        super().__init__('None of the provided mappings apply to this item')


class UnreadableDICOMError(SeriesException):
    """Exception raised when a DICOM series could not be loaded"""

    def __init__(self, path: PathLike):
        super().__init__(f'Could not read {path} using either SimpleITK or pydicom')


class ArchiveItemPathNotFoundError(SeriesException):
    """Exception raised when a DICOM series could not be found"""

    def __init__(self, path: PathLike):
        super().__init__(f"Provided archive item path not found ({path})")


@dataclass
class Dicom2MHASettings:
    mappings: Dict[str, Dict[str, List[str]]]
    num_threads: int = 4
    verify_dicom_filenames: bool = True
    allow_duplicates: bool = False
    metadata_match_func: Optional[Callable[[Metadata, Mappings], bool]] = None
    values_match_func: Union[str, Callable[[str, str], bool]] = "lower_strip_equals"

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
    resolution: Optional[float] = None
    metadata: Optional[Metadata] = field(default_factory=dict)

    mappings: List[str] = field(default_factory=list)

    error: Optional[Exception] = None
    _log: List[str] = field(default_factory=list)

    def __repr__(self):
        return self.path.name

    def __post_init__(self):
        if not self.path.exists():
            raise ArchiveItemPathNotFoundError(self.path)
        if not self.path.is_dir():
            raise NotADirectoryError(self.path)

    @property
    def is_valid(self):
        return self.error is None

    def write_log(self, msg: str):
        self._log.append(msg)

    def verify_dicom_filenames(self, filenames: List[PathLike]) -> bool:
        vdcms = [d.rsplit('.', 1)[0] for d in filenames]
        vdcms = [int(''.join(c for c in d if c.isdigit())) for d in vdcms]
        missing_slices = False
        for num in range(min(vdcms), max(vdcms) + 1):
            if num not in vdcms:
                missing_slices = True
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
            self.resolution = np.prod(file_reader.GetSpacing())
            for name, key in dicom_tags.items():
                self.metadata[name] = file_reader.GetMetaData(key) if file_reader.HasMetaDataKey(key) else ''
        except Exception as e:
            self.write_log(f"Reading with SimpleITK failed for {self.path} with error: {e}. Attempting with pydicom.")
            try:
                with pydicom.dcmread(dicom_slice_path) as data:
                    self.resolution = np.prod(data.PixelSpacing)
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


class Case:
    pass


@dataclass
class Dicom2MHACase(Case):
    input_dir: Path
    patient_id: str
    study_id: str
    paths: List[PathLike]
    settings: Dicom2MHASettings

    def __repr__(self):
        return f'Case({self.subject_id})'

    @property
    def valid_series(self):
        return [item for item in self.series if item.is_valid]

    @property
    def subject_id(self):
        return f"{self.patient_id}_{self.study_id}"

    def invalidate(self):
        for serie in self.valid_series:
            serie.error = SeriesException('Invalidated due to critical error in sibling')

    def write_log(self, msg: str):
        self._log.append(msg)

    def compile_log(self):
        divider = '=' * 120
        log = [divider,
               f'CASE {self.subject_id}',
               f'\tPATIENT ID\t{self.patient_id}',
               f'\tSTUDY ID\t{self.study_id}\n']
        log += self._log
        log += ['\nSERIES', divider.replace('=', '-')]
        for i, serie in enumerate(self.series):
            log.append(f'({i}) {serie.path.as_posix()}')
            log.extend([f'\t{item}' for item in serie._log])
            log.append(f'\tFATAL: {serie.error}\n' if not serie.is_valid else '')
        return '\n'.join(log)

    def convert(self, output_dir):
        try:
            self.initialize()
            self.extract_metadata()
            self.apply_mappings()
            self.resolve_duplicates()
            self.process_and_write(output_dir)
        except Exception as e:
            self.invalidate()
            logging.error(str(e))
        finally:
            return self.compile_log()

    def initialize(self):
        self.series: List[Series] = []
        self._log = [f'Importing {plural(len(self.paths), "serie")}']

        full_paths = set()
        for path in self.paths:
            full_path = self.input_dir / path
            serie = Series(full_path, self.patient_id, self.study_id)
            try:
                if path in full_paths:
                    raise FileExistsError(path)
                full_paths.add(full_path)
            except Exception as e:
                serie.error = e
                logging.error(str(e))
            finally:
                self.write_log(f'\t+ ({len(self.series)}) {full_path}')
                self.series.append(serie)

        if not all([serie.is_valid for serie in self.series]):
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
            ('image resolution', lambda a: a.resolution, False),
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
                    serie.write_log(f'Skipped "{mapping}", reading DICOM sequence failed, maybe corrupt data? Error: {e}')
                    logging.error(str(e))
                    errors.append(i)
                else:
                    # temporarily commented out
                    # if self.scan_postprocess_func is not None:
                    #     image = self.scan_postprocess_func(image)
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


class Dicom2MHAConverter:
    def __init__(
        self,
        input_dir: PathLike,
        output_dir: PathLike,
        dcm2mha_settings: Union[PathLike, Dict] = None,
    ):
        """
        Parameters
        ----------
        - input_dir: path to the DICOM archive. Used as base path for the relative paths
            of the archive items.
        - output_dir: path to store the MHA archive.
        - dcm2mha_settings: object with mappings, cases and optional parameters. May be
            a dictionary containing `mappings`, `archive`, and optionally `options`,
            or a path to a JSON file with these elements.
            - mappings: criteria to map DICOM sequences to their MHA counterparts
            - cases: list of DICOM sequences in the DICOM archive. Each case should contain:
                - patient_id: unique patient identifier
                - study_id: unique study identifier
                - path: path to DICOM sequence.
            - options: (optional)
                - num_threads: number of multithreading threads. Default: 4.
                - verify_dicom_filenames: whether to check if DICOM filenames contain consequtive
                    numbers. Default: True
                - allow_duplicates: whether multiple DICOM series can map to the same MHA postfix.
                    Default: False
                - metadata_match_func: method to match DICOM metadata to MHA sequences. Only use
                    this if you know what you're doing.
                - values_match_func: criterium to consider two values a match, when comparing the
                    value from the DICOM metadata against the provided allowed vaues in the mapping.

        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

        logfile = self.output_dir / f'picai_prep_{datetime.now().strftime("%Y%m%d%H%M%S")}.log'
        logging.basicConfig(level=logging.INFO, format='%(message)s', filename=logfile)
        logging.info(f'Output directory set to {self.output_dir.absolute().as_posix()}')
        print(f'Writing log to {logfile.absolute()}')

    def _init_cases(self, archive: List[Dict]) -> List[Dicom2MHACase]:
        cases = {}
        for item in archive:
            key = tuple(item[id] for id in metadata_defaults.keys())  # (patient_id, study_id)
            cases[key] = cases.get(key, []) + [item['path']]

        return [
            Dicom2MHACase(self.input_dir, patient_id, study_id, paths, self.settings)
            for (patient_id, study_id), paths in cases.items()
        ]

    def convert(self):
        start_time = datetime.now()
        logging.info(f'Program started at {start_time.isoformat()}\n')

        with ThreadPoolExecutor(max_workers=self.settings.num_threads) as pool:
            futures = {pool.submit(case.convert, self.output_dir): case for case in self.cases}
            for future in tqdm(as_completed(futures), total=len(self.cases)):
                case_log = future.result()
                logging.info(case_log)

        end_time = datetime.now()
        logging.info(f'Program ended at {end_time.isoformat()}\n\t(runtime {end_time - start_time})')


def read_image_series(image_series_path: PathLike) -> sitk.Image:
    file_reader, series_reader = make_sitk_readers()
    dicom_slice_paths = series_reader.GetGDCMSeriesFileNames(str(image_series_path))

    try:
        series_reader.SetFileNames(dicom_slice_paths)
        image: sitk.Image = series_reader.Execute()

        file_reader.SetFileName(dicom_slice_paths[-1])
        dicom_slice: sitk.Image = file_reader.Execute()
        for key in dicom_tags.values():
            if dicom_slice.HasMetaDataKey(key) and len(dicom_slice.GetMetaData(key)) > 0:
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
