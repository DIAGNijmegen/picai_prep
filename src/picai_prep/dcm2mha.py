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
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import jsonschema
import numpy as np
import pydicom.errors
import SimpleITK as sitk
from tqdm import tqdm

from picai_prep.archive import ArchiveConverter
from picai_prep.data_utils import PathLike, atomic_image_write
from picai_prep.utilities import (dcm2mha_schema, get_pydicom_value,
                                  lower_strip, make_sitk_readers,
                                  metadata_defaults, metadata_dict, plural)


class Dicom2MHAConverter(ArchiveConverter):
    def __init__(
        self,
        input_path: PathLike,
        output_path: PathLike,
        settings_path: PathLike,
        verify_dicom_filenames: bool = True,
        scan_postprocess_func: Optional[Callable[[sitk.Image], sitk.Image]] = None,
        num_threads: int = 4,
        silent=False
    ):
        """
        Converts DICOM files into MHA files with respect to a given mapping

        Parameters:
        - input_path: Root directory for input, e.g. /path/to/archive/
        - output_path: Root directory for output
        - settings_path: Path to JSON mappings file
        - silent: Control level of logging
        """
        super().__init__(
            input_path=input_path,
            output_path=output_path,
            num_threads=num_threads,
            silent=silent
        )

        # store parameters
        self.verify_dicom_filenames = verify_dicom_filenames
        self.scan_postprocess_func = scan_postprocess_func

        # read and verify conversion settings
        with open(settings_path) as fp:
            self.settings = json.load(fp)
        jsonschema.validate(self.settings, dcm2mha_schema, cls=jsonschema.Draft7Validator)

        # collect relevant metadata to extract (predefined StudyInstanceUID and PatientID)
        self.metadata = set()
        self.mappings = dict()
        for name, mapping in self.settings['mappings'].items():
            tech_mapping = dict()
            for key, value in mapping.items():
                try:
                    tech = metadata_dict[lower_strip(key)]
                    self.metadata.add(tech)

                    if len(value) == 0 or any(type(v) is not str for v in value):
                        raise ValueError(f"Non-string elements found in {name}/{key} mapping")

                    tech_mapping[tech] = [lower_strip(v) for v in value]
                except KeyError:
                    print(f"Invalid key '{key}' in '{name}' mapping, see metadata.json for valid keys.")
            self.mappings[name] = tech_mapping

        self.next_history()  # create initial history step with no errors
        self.next_history()  # working history

    def _check_archive_paths(self):
        """
        Check that all input paths are valid
        """
        sources = set()
        for a in tqdm(self.settings['archive'], desc="Checking archive paths"):
            item = {id: a.get(id, None) for id in metadata_defaults.keys()}
            self.items.append(item)
            source = os.path.abspath(os.path.join(self.input_dir, a['path']))
            item['source'] = source

            if not os.path.exists(source):
                item['error'] = (f"Provided archive item path not found ({source})", 'path not found')
            elif not os.path.isdir(source):
                item['error'] = (f"Provided archive item path is not a directory ({source})", 'path not a directory')
            elif source in sources:
                item['error'] = (f"Provided archive item path already exists ({source})", 'path already exists')
            sources.add(source)

        for item in self.items:  # add errors retroactively
            if 'error' in item:
                error, log = item['error']
                item['error'] = error
                self.item_log(item, log)

        # report number of valid items after adding errors
        self.info("Provided dcm2mha archive is valid.", self.get_history_report())

    def _extract_metadata_function(self, item):
        file_reader, series_reader = make_sitk_readers()
        dicom_filenames = [os.path.basename(dcm) for dcm in series_reader.GetGDCMSeriesFileNames(item['source'])]

        # verify DICOM files were found
        if len(dicom_filenames) == 0:
            item['error'] = f"No DICOM data found at {item['source']}"
            self.item_log(item, "missing DICOM data")
            return

        if self.verify_dicom_filenames:
            # verify DICOM filenames have slice numbers that go from a to z
            vdcms = [d.rsplit('.', 1)[0] for d in dicom_filenames]
            vdcms = [int(''.join(c for c in d if c.isdigit())) for d in vdcms]
            missing_slices = False
            for num in range(min(vdcms), max(vdcms) + 1):
                if num not in vdcms:
                    missing_slices = True
            if missing_slices:
                item['error'] = f"Missing DICOM slices detected in {item['source']}"
                self.item_log(item, "missing DICOM slices")
                return

        dicom_slice_path = os.path.join(item['source'], dicom_filenames[-1])
        metadata = dict()

        try:
            # extract metadata
            file_reader.SetFileName(dicom_slice_path)
            file_reader.ReadImageInformation()
            item['resolution'] = np.prod(file_reader.GetSpacing())
            for key in self.metadata:
                metadata[key] = lower_strip(file_reader.GetMetaData(key)) if file_reader.HasMetaDataKey(key) else None

            # extract PatientID and StudyInstanceUID if not set
            for id, value in metadata_defaults.items():
                if item[id] is None:
                    if file_reader.HasMetaDataKey(value['key']):
                        item[id] = lower_strip(file_reader.GetMetaData(value['key']))
                    else:
                        item['error'] = value['error']
                        self.item_log(item, 'metadata key error')
                        continue
        except Exception as e:
            print(f"Reading with SimpleITK failed for {item['source']} with {e}. Trying with pydicom now.")
            try:
                with pydicom.dcmread(dicom_slice_path) as d:
                    # extract metadata
                    item['resolution'] = np.prod(d.PixelSpacing)
                    for key in self.metadata:
                        metadata[key] = lower_strip(get_pydicom_value(d, key))

                    # extract PatientID and StudyInstanceUID if not set
                    for id, value in metadata_defaults.items():
                        if item[id] is None:
                            item[id] = lower_strip(get_pydicom_value(d, value['key']))
                            if item[id] is None:
                                item['error'] = value['error']
                                self.item_log(item, 'metadata key error')
                                continue

            except pydicom.errors.InvalidDicomError:
                item['error'] = 'Could not open using either SimpleITK or pydicom'
                self.item_log(item, 'corrupted DICOM file')
                return

        item['metadata'] = metadata
        item['dcms'] = dicom_filenames

    def _extract_metadata(self):
        """
        Collect all dicom files in each item
        """
        self.info(f"Extracting metadata from {self.valid_items_str()}.")

        total = 0
        with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
            futures = {pool.submit(self._extract_metadata_function, item): item for item in self.valid_items}
            for future in tqdm(as_completed(futures), total=self.num_valid_items):
                item = futures[future]
                try:
                    future.result()
                except Exception as e:
                    item['error'] = f'Unexpected error: {e}'
                    self.item_log(item, 'unexpected error')
                else:
                    total += (len(item['dcms']) if 'dcms' in item else 0)

        self.info(f"Collected {plural(total, 'DICOM file')} from {self.valid_items_str()}.", self.get_history_report())

    @staticmethod
    def maps_to(mapping, metadata):
        """metadata maps to 'mapping' if any value in each key match"""
        for key, values in mapping.items():
            if not any(v == metadata[key] for v in values):
                return False
        return True

    def _apply_mappings(self):
        self.info(f"Applying mappings to {self.valid_items_str()} using extracted metadata.")

        total = 0
        for item in self.valid_items:
            item['mappings'] = []
            for name, mapping in self.mappings.items():
                if self.maps_to(mapping, item['metadata']):
                    item['mappings'].append(name)
                    total += 1

            if len(item['mappings']) == 0:
                item['error'] = 'None of the provided mappings apply to this item'
                self.item_log(item, 'no matching mapping')
                continue

        self.info(f"Mapped {self.valid_items_str()}, totalling {plural(total, 'mapping')}.", self.get_history_report())

    def _resolve_duplicates(self):
        self.info("Setting MHA conversion target paths, then resolving duplicate paths through tiebreaker strategies.")

        items = list(self.valid_items)
        duplicates = dict()

        tiebreakers = [
            {
                'extract': lambda a: len(items[a]['dcms']),
                'reverse': True,
                'error_key': 'tiebreaker (slice count)',
                'error': 'Not selected from duplicate pool due to slice count tiebreaker'
            },
            {
                'extract': lambda a: items[a]['resolution'],
                'reverse': False,
                'error_key': 'tiebreaker (image resolution)',
                'error': 'Not selected from duplicate pool due to image resolution tiebreaker'
            },
        ]

        # set targets, then resolve duplicates
        target_count = 0
        for i, item in enumerate(items):
            targets = ["_".join([item[key] for key in metadata_defaults.keys()] + [map]) for map in item['mappings']]
            item['targets'] = dict()
            for target in targets:
                item['targets'][target] = None  # denotes target error status, None = OK
                duplicates[target] = duplicates.get(target, []) + [i]
                target_count += 1

        # remove targets from duplicates that do not win their tiebreakers
        remove_c = 0
        for target, dups in filter(lambda a: len(a) > 1, duplicates.items()):
            # index, [extractions]
            dups = [(tuple([d] + [tb['extract'](d) for tb in tiebreakers])) for d in dups]
            for i, tb in enumerate(tiebreakers):
                i += 1
                # sort duplicates by 'extract' in 'reverse' order
                dups.sort(key=lambda a: a[i], reverse=tb['reverse'])
                best = dups[0]
                # duplicates which cannot compete with best are removed
                for d in dups:
                    if d[i] != best[i]:
                        dups.remove(d)
                        item = items[d[0]]
                        item['targets'][target] = tb['error']
                        self.item_log(item, tb['error_key'])
                        remove_c += 1
            for d in dups[1:]:
                item = items[d[0]]
                item['targets'][target] = 'Not selected from duplicate pool due to final randomizing tiebreaker'
                self.item_log(item, 'tiebreaker (random)')
                remove_c += 1

        # if target is still None, it is selected. otherwise, the reason for deselection is stated
        self.info(f"Selected {plural(target_count, 'target')}, removed {remove_c} by tiebreaker.", self.get_history_report())

    def _convert_function(self, args) -> bool:
        item, target, destination = args
        # the series is already verified in an earlier step
        try:
            image = read_image_series(item['source'])
        except Exception as e:
            item['targets'][target] = f'Reading DICOM sequence failed, maybe corrupt data? Error: {e}'
            self.item_log(item, 'corrupt data')
            return False

        if self.scan_postprocess_func is not None:
            image = self.scan_postprocess_func(image)

        try:
            atomic_image_write(image=image, path=destination, mkdir=True)
        except Exception as e:
            item['targets'][target] = f'Error during writing: {e}'
            self.item_log(item, 'write error')
            return False
        return True

    def _convert(self):
        convert_count = 0
        for item in self.valid_items:
            for _, status in item['targets'].items():
                convert_count += 1 if status is None else 0

        self.info(f"Converting {convert_count} DICOM series to MHA.")

        # convert all items with valid targets to mha
        success_count, skip_count = 0, 0
        targets = []
        for item in self.valid_items:
            for target, status in item['targets'].items():
                if status is None:  # therefore selected
                    destination = os.path.join(self.output_dir, item['patient_id'], target + '.mha')
                    if os.path.exists(destination):
                        item['targets'][target] = 'Skipped, target output path already exists'
                        self.item_log(item, 'skipped')
                        skip_count += 1
                    else:
                        targets.append((item, target, destination))

        with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
            futures = {pool.submit(self._convert_function, args): args for args in targets}
            for future in tqdm(as_completed(futures), total=len(targets)):
                item, target, _ = futures[future]
                try:
                    success = future.result()
                except Exception as e:
                    item['targets'][target] = f'Unexpected error: {e}'
                    self.item_log(item, 'unexpected error')
                else:
                    success_count += 1 if success else 0

        self.info(f"{success_count} converted successfully.", self.get_history_report())

    def convert(self):
        for step in [
            self._check_archive_paths,
            self._extract_metadata,
            self._apply_mappings,
            self._resolve_duplicates,
            self._convert
        ]:
            if self.has_valid_items or step == self._check_archive_paths:
                step()
                self.next_history()
            else:
                self.info("Aborted conversion, no items to convert.")
                return

        self.complete()

    def item_log_value(self, item):
        lpad = ' ' * (len(str(len(self.items))) + 1)

        if 'error' in item:
            result = f"{lpad}{item['error']}"
        elif 'targets' in item:
            result = []
            for target, status in item['targets'].items():
                result.append(f"{lpad}{target} --> {'OK' if status is None else status}")
            result = '\n'.join(result)
        else:
            return None

        return f"{str(self.items.index(item)).zfill(len(lpad))} {item['patient_id']}/{item['study_id']}\n" \
               f"\t{result}"


def read_image_series(image_series_path):
    file_reader, series_reader = make_sitk_readers()
    dicom_slice_paths = series_reader.GetGDCMSeriesFileNames(image_series_path)

    try:
        series_reader.SetFileNames(dicom_slice_paths)
        image = series_reader.Execute()

        file_reader.SetFileName(dicom_slice_paths[-1])
        dicom_slice = file_reader.Execute()
        for key in metadata_dict.values():
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
        image = sitk.GetImageFromArray(image)
        image.SetSpacing(list(slices[0].PixelSpacing) + [slices[0].SliceThickness])

        for key in metadata_dict.values():
            value = get_pydicom_value(files[0], key)
            if value is not None:
                image.SetMetaData(key, value)

    return image
