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
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import jsonschema
import SimpleITK as sitk

from picai_prep.converter import Case, Converter
from picai_prep.data_utils import PathLike, atomic_image_write
from picai_prep.preprocessing import PreprocessingSettings, Sample
from picai_prep.utilities import mha2nnunet_schema, plural


@dataclass
class MHA2nnUNetSettings:
    dataset_json: Dict[str, Any]
    preprocessing: PreprocessingSettings
    scans_dirname: PathLike = "imagesTr"
    annotation_dirname: PathLike = "labelsTr"
    annotation_preprocess_func: Optional[Callable[[sitk.Image], sitk.Image]] = None
    annotation_postprocess_func: Optional[Callable[[sitk.Image], sitk.Image]] = None
    scan_preprocess_func: Optional[Callable[[sitk.Image], sitk.Image]] = None
    scan_postprocess_func: Optional[Callable[[sitk.Image], sitk.Image]] = None
    num_threads: int = 4
    verbose: int = 1

    @property
    def task_name(self) -> str:
        return self.dataset_json['task']


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

    def __post_init__(self):
        if self.annotations_dir and self.annotation_path:
            self.annotation_path = self.annotations_dir / self.annotation_path

    def convert_item(self, scans_out_dir: Path, annotations_out_dir: Path) -> None:
        self.initialize()
        self.process_and_write(scans_out_dir, annotations_out_dir)

    def initialize(self):
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

    def process_and_write(self, scans_out_dir: Path, annotations_out_dir: Path):
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

    def compile_log(self):
        if self.settings.verbose == 0:
            # logging is disabled
            return None

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

        if not self.is_valid:
            # conversion was failed, log everything
            return '\n'.join(['=' * 120,
                              f'CASE {self.subject_id}',
                              f'\tPATIENT ID\t{self.patient_id}',
                              f'\tSTUDY ID\t{self.study_id}\n',
                              *self._log,
                              'Error:',
                              traceback.format_exc()])


class MHA2nnUNetConverter(Converter):
    def __init__(
        self,
        output_dir: PathLike,
        scans_dir: PathLike,
        mha2nnunet_settings: Union[PathLike, Dict],
        scans_out_dirname: str = 'imagesTr',
        annotations_dir: Optional[PathLike] = None,
        annotations_out_dirname: Optional[str] = 'labelsTr',
    ):
        """
        Convert an MHA Archive to an nnUNet Raw Data Archive.
        See https://github.com/DIAGNijmegen/picai_prep for additional documentation.

        Parameters
        ----------
        output_dir: PathLike
            path to store the resulting nnUNet archive.
        scans_dir: PathLike
            directory name to store scans in, relative to `output_dir`.
        scans_out_dirname: str, default: 'imagesTr'
            dirname to store scan output, will be a direct descendant of `output_dir`.
        annotations_dir: PathLike
            path to the annotations archive. Used as base path for the relative paths of the archive items.
        annotations_out_dirname: str, default: 'labelsTr'
            dirname to store annotation output, will be a direct descendant of `output_dir`.
        mha2nnunet_settings: Union[PathLike, Dict]
            object with cases, nnUNet-shaped dataset.json and optional parameters.
            May be a dictionary containing mappings, dataset.json, and optionally options,
            or a path to a JSON file with these elements.
            - dataset_json: see nnU-Net's dataset conversion on details for the dataset.json file:
                https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md
            - cases: list of objects. Each case should be an object with a patient_id,
                study_id, relative paths to scans and optionally a path to an annotation
            - options: (optional)
                - num_threads: number of multithreading threads.
                    Default: 4.
                - verbose: control logfile verbosity. 0 does not output a logfile,
                    1 logs cases which have critically failed, 2 logs all cases (may lead to
                    very large log files)
                    Default: 1
        """
        if isinstance(mha2nnunet_settings, (Path, str)):
            with open(mha2nnunet_settings) as fp:
                mha2nnunet_settings = json.load(fp)

        # validate and parse settings
        jsonschema.validate(mha2nnunet_settings, mha2nnunet_schema, cls=jsonschema.Draft7Validator)
        self.settings = MHA2nnUNetSettings(
            dataset_json=mha2nnunet_settings['dataset_json'],
            preprocessing=PreprocessingSettings(**mha2nnunet_settings['preprocessing']),
            **mha2nnunet_settings.get('options', {})
        )

        # set up paths and create output directory
        self.scans_dir = Path(scans_dir)
        self.annotations_dir = Path(annotations_dir) if annotations_dir else None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.scans_out_dir = output_dir / self.settings.task_name / scans_out_dirname
        self.annotations_out_dir = output_dir / self.settings.task_name / annotations_out_dirname if annotations_dir else None

        # initialize cases to convert
        self.cases = self._init_cases(mha2nnunet_settings['archive'])

        # set up logfile
        self.initialize_log(output_dir, self.settings.verbose)

    def _init_cases(self, archive: List[Dict[str, Any]]) -> List[MHA2nnUNetCase]:
        return [
            MHA2nnUNetCase(scans_dir=self.scans_dir, annotations_dir=self.annotations_dir, settings=self.settings, **kwargs)
            for kwargs in archive
        ]

    def convert(self):
        self._convert(
            title='MHA2nnUNet',
            cases=self.cases,
            parameters={
                'scans_out_dir': self.scans_out_dir,
                'annotations_out_dir': self.annotations_out_dir
            },
            num_threads=self.settings.num_threads,
        )

    def _prepare_dataset_paths(self):
        """Prepare paths to scans and annotation in nnU-Net dataset.json format"""
        return [
            {
                "image": f"./{self.scans_out_dir.name}/{case.subject_id}.nii.gz",
                "label": f"./{self.annotations_out_dir.name}/{case.subject_id}.nii.gz"
            }
            for case in self.valid_cases
        ]

    def create_dataset_json(self, path: PathLike = 'dataset.json', is_testset: bool = False) -> Dict:
        """
        Create dataset.json for nnUNet raw data archive.

        Parameters
        ----------
        path: PathLike, default: nnU-Net raw data archive folder / task / dataset.json
            path to save dataset info to. If None, will not output a file.
        is_testset: bool, default: False
            whether this conversation was a test set or a training set.

        Returns
        -------
        dataset : dict
            contents of dataset.json
        """
        if path is None:
            return

        dataset_path = self.scans_out_dir.parent / path
        if dataset_path.exists():
            logging.info(f"Dataset info already exists at {dataset_path}, saving to ...-conflict.json")
            dataset_path = dataset_path.with_stem(dataset_path.stem + "-conflict")

        logging.info(f'Saving dataset info to {dataset_path}')

        # use contents of archive->dataset_json as starting point
        dataset_settings = self.settings.dataset_json
        if 'name' not in dataset_settings:
            dataset_settings['name'] = '_'.join(self.settings.task_name.split('_')[1:])

        if is_testset:
            dataset_settings["numTest"] = len(self.cases)
            dataset_settings["test"] = self._prepare_dataset_paths()
            if "numTraining" not in dataset_settings:
                dataset_settings["numTraining"] = 0
                dataset_settings["training"] = []
        else:
            dataset_settings['numTraining'] = len(self.cases)
            dataset_settings["training"] = self._prepare_dataset_paths()
            if "numTest" not in dataset_settings:
                dataset_settings["numTest"] = 0
                dataset_settings["test"] = []

        with open(dataset_path, 'w') as fp:
            json.dump(dataset_settings, fp, indent=4)

        return dataset_settings

    @property
    def valid_cases(self):
        return [case for case in self.cases if case.is_valid]
