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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import jsonschema
import numpy as np
import pydicom.errors
import SimpleITK as sitk
from tqdm import tqdm

from picai_prep.data_utils import PathLike, atomic_image_write
from picai_prep.utilities import (dcm2mha_schema, dicom_tags,
                                  get_pydicom_value, lower_strip,
                                  make_sitk_readers, metadata_defaults, plural)


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
    mappings: Dict[str, Dict] = field(default_factory=dict)  # TODO: add what kind of Dict inside
    num_threads: int = 4
    verify_dicom_filenames: bool = True
    allow_duplicates: bool = False
    tags: Dict = field(default_factory=dict)  # TODO: add what kind of Dict

    def __post_init__(self):
        for name, mapping in self.mappings.items():
            self.mappings[name] = self._process_mapping(name, mapping)

    def _process_mapping(self, name: str, mapping: Dict) -> Dict:  # TODO: what kind of Dict
        """TODO: add a docstring"""
        map = dict()
        for key, values in mapping.items():
            l_key = lower_strip(key)  # TODO: make lower strip optional
            try:
                # add key to group of metadata we intend to extract from the dicoms
                self.tags[l_key] = dicom_tags[l_key]

                if len(values) == 0 or any(not isinstance(v, str) for v in values):
                    raise ValueError(f"Invalid non-string elements found in {name}/{key} mapping")

                map[l_key] = [lower_strip(v) for v in values]
            except KeyError:
                raise KeyError(f"Invalid key '{l_key}' in '{name}' mapping, see picai_prep/resources/dicom_tags.py for valid keys.")
        return map


@dataclass
class Series:
    path: Path
    patient_id: str
    study_id: str

    # image metadata
    filenames: Optional[List[str]] = None
    resolution: Optional[float] = None
    metadata: Optional[Dict[str, str]] = field(default_factory=dict)

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

    def extract_metadata(self, file_reader: sitk.ImageFileReader, filenames: List[PathLike], tags: Dict[str, str]):
        self.filenames = filenames
        dicom_slice_path = self.path / filenames[-1]

        try:
            file_reader.SetFileName(str(dicom_slice_path))
            file_reader.ReadImageInformation()
            self.resolution = np.prod(file_reader.GetSpacing())
            for name, tag in tags.items():
                # TODO: make lower strip optional
                self.metadata[name] = lower_strip(file_reader.GetMetaData(tag) if file_reader.HasMetaDataKey(tag) else '')
        except Exception as e:
            self.write_log(f"Reading with SimpleITK failed for {self.path} with error: {e}. Attempting with pydicom.")
            try:
                with pydicom.dcmread(dicom_slice_path) as data:
                    self.resolution = np.prod(data.PixelSpacing)
                    for name, id in tags.items():
                        self.metadata[name] = lower_strip(get_pydicom_value(data, id))
            except pydicom.errors.InvalidDicomError:
                e = UnreadableDICOMError(self.path)
                self.error = e
                logging.error(str(e))


class Case:
    pass


@dataclass
class Dicom2MHACase(Case):
    input_dir: Path
    patient_id: str
    study_id: str
    paths: List[PathLike]
    settings: Dicom2MHASettings

    def __post_init__(self):
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

    def __repr__(self):
        return f'Case({self.patient_id}_{self.study_id})'

    @property
    def valid_series(self):
        return [item for item in self.series if item.is_valid]

    def invalidate(self):
        for serie in self.valid_series:
            serie.error = SeriesException('Invalidated due to critical error in sibling')

    def write_log(self, msg: str):
        self._log.append(msg)

    def compile_log(self):
        divider = '=' * 120
        log = [divider,
               f'CASE {self.patient_id}_{self.study_id}',
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
            self.extract_metadata()
            self.apply_mappings()
            self.resolve_duplicates()
            self.process_and_write(output_dir)
        except Exception as e:
            self.invalidate()
            logging.error(str(e))
        finally:
            return self.compile_log()

    def extract_metadata(self):
        vseries = len(self.valid_series)
        self.write_log(f'Extracting metadata from {plural(vseries, "serie")}')
        errors = []

        file_reader, series_reader = make_sitk_readers()
        for i, serie in enumerate(self.valid_series):
            try:
                dicom_filenames = [os.path.basename(dcm) for dcm in series_reader.GetGDCMSeriesFileNames(serie.path.as_posix())]

                # verify DICOM files are found
                if len(dicom_filenames) == 0:
                    raise MissingDICOMFilesError(serie.path)

                if self.settings.verify_dicom_filenames:
                    serie.verify_dicom_filenames(dicom_filenames)

                serie.extract_metadata(file_reader, dicom_filenames, self.settings.tags)
                serie.write_log('Extracted metadata')
            except (MissingDICOMFilesError, UnreadableDICOMError) as e:
                serie.error = e
                errors.append(i)

        self.write_log(f'\t({plural(len(errors), "error")}{f" {errors}" if len(errors) > 0 else ""})')

    def apply_mappings(self):
        vseries = len(self.valid_series)
        self.write_log(f'Applying mappings to {vseries} series')
        errors = []

        for i, serie in enumerate(self.valid_series):
            try:
                for name, mapping in self.settings.mappings.items():
                    for key, values in mapping.items():
                        # TODO: option for mapping matching... [strict, stripped (lowerstrip()), contains, Callable]
                        if any(v == serie.metadata[key] for v in values):
                            serie.mappings.append(name)

                if len(serie.mappings) == 0:
                    raise NoMappingsApplyError()
                serie.write_log(f'Applied mappings [{", ".join(serie.mappings)}]')
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

        dir = output_dir / self.patient_id
        for i, serie in enumerate(self.valid_series):
            for mapping in serie.mappings:
                destination = (dir / '_'.join([self.patient_id, self.study_id, mapping])).with_suffix('.mha')
                if destination.exists():
                    serie.write_log(f'Skipped "{mapping}", already exists: {destination}')
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
                        atomic_image_write(image=image, path=destination, mkdir=True)
                    except Exception as e:
                        serie.write_log(f'Skipped "{mapping}", write error: {e}')
                        logging.error(str(e))
                        errors.append(i)
                    else:
                        serie.write_log(f'Wrote image to {destination}')

        self.write_log(f'Wrote {total - len(errors) - len(skips)} MHA files to {dir.as_posix()}\n'
                       f'\t({plural(len(errors), "error")}{f" {errors}" if len(errors) > 0 else ""}, '
                       f'{len(skips)} skipped{f" {skips}" if len(skips) > 0 else ""})')


class Dicom2MHAConverter:
    def __init__(self, input_dir: PathLike, output_dir: PathLike, dcm2mha_settings: Union[PathLike, Dict]):
        """TODO: add docstring"""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(dcm2mha_settings, (Path, str)):
            with open(dcm2mha_settings) as fp:
                dcm2mha_settings = json.load(fp)

        jsonschema.validate(dcm2mha_settings, dcm2mha_schema, cls=jsonschema.Draft7Validator)

        self.settings = Dicom2MHASettings(dcm2mha_settings['mappings'], **dcm2mha_settings.get('options', {}))
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
