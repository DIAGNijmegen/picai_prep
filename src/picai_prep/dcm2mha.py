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

import jsonschema
import numpy as np
import pydicom.errors
import SimpleITK as sitk
from tqdm import tqdm

from picai_prep.archive import ArchiveConverter
from picai_prep.data_utils import PathLike, atomic_image_write
from picai_prep.utilities import (dcm2mha_schema, get_pydicom_value,
                                  lower_strip, metadata_defaults,
                                  metadata_dict, plural)

isr = sitk.ImageSeriesReader()
isr.LoadPrivateTagsOn()
ifr = sitk.ImageFileReader()
ifr.LoadPrivateTagsOn()


class Dicom2MHAConverter(ArchiveConverter):
    def __init__(self, input_path: PathLike, output_path: PathLike, settings_path: PathLike, silent=False):
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
            silent=silent
        )

        # read and verify conversion settings
        with open(settings_path) as fp:
            settings = json.load(fp)
        jsonschema.validate(settings, dcm2mha_schema, cls=jsonschema.Draft7Validator)

        # collect relevant metadata to extract (predefined StudyInstanceUID and PatientID)
        self.metadata = set()

        self.mappings = dict()
        for name, mapping in settings['mappings'].items():
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

        sources = set()
        for a in settings['archive']:
            item = {id: a.get(id, None) for id in metadata_defaults.keys()}
            self.items.append(item)
            source = a['path'] if os.path.isabs(a['path']) else os.path.abspath(os.path.join(input_path, a['path']))
            item['source'] = source

            if not os.path.exists(source):
                item['error'] = (f"Provided archive item path not found ({a['path']})", 'path not found')
            elif not os.path.isdir(source):
                item['error'] = (f"Provided archive item path is not a directory ({a['path']})", 'path not a directory')
            elif source in sources:
                item['error'] = (f"Provided archive item path already exists ({a['path']})", 'path already exists')
            sources.add(source)

        self.next_history()  # create initial history step with no errors
        self.next_history()  # this history
        for item in self.items:  # add errors retroactively
            if 'error' in item:
                error, log = item['error']
                item['error'] = error
                self.item_log(item, log)
        self.info("Provided dcm2mha archive is valid.", self.get_history_report())  # report number of valid items after adding errors
        self.next_history()  # next history

    def _extract_metadata(self):
        """
        Collect all dicom files in each item
        """
        self.info(f"Extracting metadata from {self.valid_items_str()}.")

        total = 0
        for item in tqdm(self.valid_items):
            dcms = [os.path.basename(dcm) for dcm in isr.GetGDCMSeriesFileNames(item['source'])]

            # verify DICOMS go from a to z
            if len(dcms) == 0:
                item['error'] = 'No DICOM data found'
                self.item_log(item, 'missing DICOM data')
                continue

            vdcms = [d.rsplit('.', 1)[0] for d in dcms]
            vdcms = [int(''.join(c for c in d if c.isdigit())) for d in vdcms]
            missing_slices = False
            for num in range(min(vdcms), max(vdcms) + 1):
                if num not in vdcms:
                    missing_slices = True
            if missing_slices:
                item['error'] = 'Missing DICOM slices detected'
                self.item_log(item, 'missing DICOM slices')
                continue

            dcm = os.path.join(item['source'], dcms[-1])
            metadata = dict()

            try:
                # extract metadata
                ifr.SetFileName(dcm)
                ifr.Execute()
                item['resolution'] = np.prod(ifr.GetSpacing())
                for key in self.metadata:
                    metadata[key] = lower_strip(ifr.GetMetaData(key)) if ifr.HasMetaDataKey(key) else None

                # extract PatientID and StudyInstanceUID if not set
                for id, value in metadata_defaults.items():
                    if item[id] is None:
                        if ifr.HasMetaDataKey(value['key']):
                            item[id] = lower_strip(ifr.GetMetaData(value['key']))
                        else:
                            item['error'] = value['error']
                            self.item_log(item, 'metadata key error')
                            continue
            except Exception:
                try:
                    with pydicom.dcmread(dcm) as d:
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
                    continue

            total += len(dcms)
            item['metadata'] = metadata
            item['dcms'] = dcms

        self.info(f"Collected {plural(total, 'DICOM file')} from {self.valid_items_str()}.", self.get_history_report())

    def _apply_mappings(self):
        self.info(f"Applying mappings to {self.valid_items_str()} using extracted metadata.")

        # metadata maps to 'mapping' if any value in each key match
        def maps_to(mapping, metadata):
            for key, values in mapping.items():
                if not any(v in metadata[key] for v in values):
                    return False
            return True

        total = 0
        for item in self.valid_items:
            item['mappings'] = []
            for name, mapping in self.mappings.items():
                if maps_to(mapping, item['metadata']):
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

    def convert(self, resolve_duplicates: bool = True):
        for step in [self._extract_metadata, self._apply_mappings] + [self._resolve_duplicates] if resolve_duplicates else []:
            if self.has_valid_items:
                step()
                self.next_history()
            else:
                self.info("Aborted conversion, no items to convert.")
                return

        convert_count = 0
        for item in self.valid_items:
            for _, status in item['targets'].items():
                convert_count += 1 if status is None else 0

        self.info(f"Converting {convert_count} DICOM series to MHA.")

        # convert all items with valid targets to mha
        success_count = 0
        skip_count = 0
        for item in tqdm(self.valid_items):
            for target, status in item['targets'].items():
                if status is None:  # therefore selected
                    destination = os.path.join(self.output_dir, item['patient_id'], target + '.mha')
                    if os.path.exists(destination):
                        item['targets'][target] = 'Skipped, target output path already exists'
                        self.item_log(item, 'skipped')
                        skip_count += 1
                        continue

                    # the series is already verified in an earlier step
                    try:
                        volume = image_series_to_volume(item['source'])
                    except Exception:
                        item['targets'][target] = 'Conversion to volume failed, likely corrupt data'
                        self.item_log(item, 'corrupt data')
                        continue

                    try:
                        atomic_image_write(image=volume, path=destination, mkdir=True)
                    except Exception as e:
                        item['targets'][target] = str(e)
                        self.item_log(item, 'unknown error')
                        continue

                    success_count += 1

        self.info(f"{success_count} converted successfully.", self.get_history_report())
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


def image_series_to_volume(image_series_path):
    dcms = isr.GetGDCMSeriesFileNames(image_series_path)

    try:
        isr.SetFileNames(dcms)
        volume = isr.Execute()

        ifr.SetFileName(dcms[-1])
        specimen = ifr.Execute()
        for key in metadata_dict.values():
            if specimen.HasMetaDataKey(key) and len(specimen.GetMetaData(key)) > 0:
                volume.SetMetaData(key, specimen.GetMetaData(key))
    except Exception:
        files = [pydicom.dcmread(dcm) for dcm in dcms]

        # skip files with no SliceLocation (eg. scout views)
        slices = filter(lambda a: hasattr(a, 'SliceLocation'), files)
        slices = sorted(slices, key=lambda s: s.SliceLocation)

        # create and fill 3D array
        volume = np.zeros([len(slices)] + list(slices[0].pixel_array.shape))
        for i, s in enumerate(slices):
            volume[i, :, :] = s.pixel_array

        # convert to SimpleITK
        volume = sitk.GetImageFromArray(volume)
        volume.SetSpacing(list(slices[0].PixelSpacing) + [slices[0].SliceThickness])

        for key in metadata_dict.values():
            value = get_pydicom_value(files[0], key)
            if value is not None:
                volume.SetMetaData(key, value)

    return volume
