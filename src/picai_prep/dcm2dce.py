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
from typing import Dict, List

import SimpleITK as sitk

from picai_prep.data_utils import PathLike, atomic_image_write
from picai_prep.dcm2mha import (Dicom2MHACase, Dicom2MHAConverter, Series,
                                read_image_series)
from picai_prep.errors import DCESeriesNotFoundError

Metadata = Dict[str, str]
Mapping = Dict[str, List[str]]
Mappings = Dict[str, Mapping]


@dataclass
class Dicom2DCECase(Dicom2MHACase):
    series: List[Series] = field(default_factory=list)

    def convert_item(self, output_dir):
        self.initialize()
        self.extract_metadata()
        self._convert_dce(output_dir)

    def _convert_dce(
        self,
        output_dir: PathLike,
        DCE_prefixes: List[str] = None,
        return_image: bool = False,
    ):
        if DCE_prefixes is None:
            DCE_prefixes = [
                # the tags below look like DCE tags, but this is unconfirmed!
                'Perfusie_t1_twist_tra_TTC',
                'Perfusie_t1_twist_tra_TT',
                'Twist_dynamic_Wip576_TT',
                'Perfusie_t1_twist_tra_4mm_TTC',
                'Twist_dynamic_Wip576_pros_TT',
                'Perfusie_t1_twist_tra_3,3mm_TTC',
                'Perfusie_t1_twist_tra_3.3_TTC',
            ]
        if not isinstance(DCE_prefixes, list):
            raise ValueError("DCE_prefixes must be a list")

        # paths
        patient_dir = Path(output_dir) / self.patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)

        # check if file already exists (for joined MHA, i.e., 4D)
        if not return_image:
            dst_path = patient_dir / f"{self.subject_id}_dce.mha"
            if dst_path.exists():
                # TODO: log case is skipped
                return

        # placeholders
        dce_scan_time_map: Dict[str, Path] = {}
        dce_scans: List[sitk.Image] = []

        # collect scan time for all of DCE's T1 scans
        for serie in self.valid_series:
            # Get scan time
            timepoint = None
            for prefix in DCE_prefixes:
                match = re.match(fr'{prefix}?=(?P<time>.+)s', serie.metadata['seriesdescription'])
                if match is not None:
                    timepoint = match.group('time')
                    dce_scan_time_map[timepoint] = serie.path
                    if self.settings.verbose >= 2:
                        print(f"Got {timepoint} from {serie.metadata['seriesdescription']}")
                    break

            if timepoint is None:
                for prefix in DCE_prefixes:
                    if prefix in serie.metadata['seriesdescription']:
                        # try to get scan time from Acquisition Time
                        timepoint = serie.metadata['acquisitiontime']
                        dce_scan_time_map[timepoint] = serie.path
                        if self.settings.verbose >= 2:
                            print(f"Got {timepoint} from {serie.metadata['seriesdescription']} ({serie.path})")
                        break

        # Sort scan-time dictionary
        times = dce_scan_time_map.keys()
        times = sorted(times, key=float)

        if self.settings.verbose >= 2:
            print(f'Sorted times: {times}')

        if len(times) <= 1:
            raise DCESeriesNotFoundError(self.subject_id)

        # Collect all DCE scans in chronological order
        for i, timepoint in enumerate(times):
            ser_dir = dce_scan_time_map[timepoint]
            if self.settings.verbose >= 2:
                print(f"[{i+1}/{len(times)}]: Reading scan at {timepoint}s from {ser_dir}")

            # Collect T1 image of ordered time points
            image = read_image_series(ser_dir)
            dce_scans.append(image)

        joined_images: sitk.Image = sitk.JoinSeries(dce_scans)

        # Copy over metadata to joined image
        img = dce_scans[0]
        for key in img.GetMetaDataKeys():
            joined_images.SetMetaData(key, img.GetMetaData(key))

        # Add metadata for the time of the scans
        scan_times = ",".join(times)
        joined_images.SetMetaData("DCE_SCAN_TIMES", scan_times)

        if return_image:
            return joined_images

        # construct target filename and save to file
        atomic_image_write(joined_images, dst_path)


class Dicom2DCEConverter(Dicom2MHAConverter):
    def _init_cases(self, archive: List[Dict]) -> List[Dicom2DCECase]:
        cases = {}
        for item in archive:
            key = tuple(item[id] for id in ["patient_id", "study_id"])
            cases[key] = cases.get(key, []) + [item['path']]
        return [
            Dicom2DCECase(input_dir=self.input_dir, patient_id=patient_id,
                          study_id=study_id, paths=paths, settings=self.settings)
            for (patient_id, study_id), paths in cases.items()
        ]

    def convert(self):
        self._convert(
            title='Dicom2DCE',
            cases=self.cases,
            parameters={
                'output_dir': self.output_dir
            },
            num_threads=self.settings.num_threads,
        )
