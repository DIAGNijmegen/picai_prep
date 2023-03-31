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
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import SimpleITK as sitk

from picai_prep.converter import Case
from picai_prep.data_utils import atomic_image_write
from picai_prep.preprocessing import Sample
from picai_prep.utilities import plural
from picai_prep.mha2nnunet.settings import MHA2nnUNetSettings


@dataclass
class _MHA2nnUNetCaseBase:
    scans_dir: Path
    annotations_dir: Path
    scan_paths: List[Path]
    settings: MHA2nnUNetSettings


@dataclass
class MHA2nnUNetCase(Case, _MHA2nnUNetCaseBase):
    annotation_path: Optional[Path] = None
    verified_scan_paths: List[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.annotations_dir and self.annotation_path:
            self.annotation_path = self.annotations_dir / self.annotation_path

    def convert_item(self, scans_out_dir: Path, annotations_out_dir: Path) -> None:
        self.initialize(scans_out_dir, annotations_out_dir)
        if self.skip_conversion:
            return
        self.process_and_write(scans_out_dir, annotations_out_dir)

    def initialize(self, scans_out_dir: Path, annotations_out_dir: Path):
        # check if all output paths exist
        if self.output_files_exist(scans_out_dir=scans_out_dir, annotations_out_dir=annotations_out_dir):
            self.skip_conversion = True
            return

        self.write_log(f'Importing {plural(len(self.scan_paths), "scan")}')

        missing_paths = []
        for scan_path in self.scan_paths:
            # check (relative) path of input scans
            path = self.scans_dir / scan_path
            if not path.exists():
                missing_paths.append(path)
                continue

            self.verified_scan_paths.append(path)
            self.write_log(f'\t+ ({len(self.verified_scan_paths)}) {path}')

        if len(missing_paths) > 0:
            raise FileNotFoundError(','.join([str(p) for p in missing_paths]))

        if self.annotation_path:
            self.write_log('Importing annotation')
            if not self.annotation_path.exists():
                raise FileNotFoundError(self.annotation_path)

            self.write_log(f'\t+ {self.annotation_path}')

    def output_files_exist(self, scans_out_dir: Path, annotations_out_dir: Path) -> bool:
        """
        Check whether all preprocessed scans (and annotation) already exist.
        """
        destination_paths = [
            scans_out_dir / f"{self.subject_id}_{i:04d}.nii.gz"
            for i in range(len(self.scan_paths))
        ]
        if self.annotation_path:
            destination_paths.append(annotations_out_dir / f"{self.subject_id}.nii.gz")
        if all(path.exists() for path in destination_paths):
            return True

    def process_and_write(self, scans_out_dir: Path, annotations_out_dir: Path) -> None:
        """
        Read scans (and annotation), perform preprocessing and write to disk.
        """
        self.write_log(
            f'Writing {plural(len(self.verified_scan_paths), "scan")}'
            + ' including annotation' if self.annotation_path else ''
        )

        scans = [sitk.ReadImage(path.as_posix()) for path in self.verified_scan_paths]
        lbl = sitk.ReadImage(self.annotation_path.as_posix()) if self.annotation_path else None

        # set up Sample
        sample = Sample(
            scans=scans,
            lbl=lbl,
            settings=self.settings.preprocessing,
            lbl_preprocess_func=self.settings.annotation_preprocess_func,
            lbl_postprocess_func=self.settings.annotation_postprocess_func,
            scan_preprocess_func=self.settings.scan_preprocess_func,
            scan_postprocess_func=self.settings.scan_postprocess_func,
            name=self.subject_id
        )

        # perform preprocessing
        sample.preprocess()

        # write images
        for i, scan in enumerate(sample.scans):
            destination_path = scans_out_dir / f"{self.subject_id}_{i:04d}.nii.gz"
            atomic_image_write(scan, path=destination_path, mkdir=True)
            self.write_log(f'Wrote image to {destination_path}')

        if lbl:
            destination_path = annotations_out_dir / f"{self.subject_id}.nii.gz"
            atomic_image_write(sample.lbl, path=annotations_out_dir / f"{self.subject_id}.nii.gz", mkdir=True)
            self.write_log(f'Wrote annotation to {destination_path}')

    def compile_log(self) -> Optional[str]:
        if self.settings.verbose == 0:
            # logging is disabled
            return None

        if self.skip_conversion and self.settings.verbose >= 2:
            return f"Skipping {self.subject_id}, already converted."

        if self.is_valid:
            if self.settings.verbose >= 2:
                # conversion was successful and verbose logging is enabled
                return '\n'.join(['=' * 120,
                                  f'CASE {self.subject_id}',
                                  f'\tPATIENT ID\t{self.patient_id}',
                                  f'\tSTUDY ID\t{self.study_id}\n',
                                  *self._log])
            else:
                # conversion was successful and short logging is enabled
                return '\n'.join([f'CASE {self.subject_id} successfully converted'])
        else:
            # conversion failed, log everything
            return '\n'.join(['=' * 120,
                              f'CASE {self.subject_id}',
                              f'\tPATIENT ID\t{self.patient_id}',
                              f'\tSTUDY ID\t{self.study_id}\n',
                              *self._log,
                              'Error:',
                              self.error_trace])
