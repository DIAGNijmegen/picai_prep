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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import jsonschema
import SimpleITK as sitk

from picai_prep.converter import Case, Converter
from picai_prep.data_utils import PathLike, atomic_image_write
from picai_prep.preprocessing import PreprocessingSettings, Sample
from picai_prep.utilities import mha2nnunet_schema, plural


@dataclass
class MHA2nnUNetSettings:
    dataset_json: dict
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
    scans: List[Path] = field(default_factory=list)
    annotation: Path = None

    def compile_log(self):
        if self.settings.verbose == 0:
            return None

        if self.is_valid or self.settings.verbose >= 2:
            return '\n'.join(['=' * 120,
                              f'CASE {self.subject_id}',
                              f'\tPATIENT ID\t{self.patient_id}',
                              f'\tSTUDY ID\t{self.study_id}\n',
                              *self._log])

    def _convert(self, *args):
        scans_out_dir, annotations_out_dir = args
        self.initialize()
        self.process_and_write(scans_out_dir, annotations_out_dir)

    def initialize(self):
        self.write_log(f'Importing {plural(len(self.scan_paths), "scans")}')

        missing_paths = []
        for scan_path in self.scan_paths:
            # check (relative) path of input scans
            path = self.scans_dir / scan_path
            if not path.exists():
                missing_paths.append(path)
                continue

            self.scans.append(path)
            self.write_log(f'\t+ ({len(self.scans)}) {path}')

        if len(missing_paths) > 0:
            raise FileNotFoundError(','.join([str(p) for p in missing_paths]))

        if self.annotations_dir:
            self.write_log('Importing annotation')
            self.annotation = self.annotations_dir / self.annotation_path
            if not self.annotation.exists():
                raise FileNotFoundError(self.annotation)

            self.write_log(f'\t+ {self.annotation}')

    def process_and_write(self, scans_out_dir: Path, annotations_out_dir: Path):
        self.write_log(f'Writing {plural(len(self.scans), "scan")}' + ' including annotation' if self.annotation else '')

        sitk_scans = [sitk.ReadImage(scan.as_posix()) for scan in self.scans]
        sitk_annotation = sitk.ReadImage(self.annotation.as_posix()) if self.annotation else None

        # set up Sample
        sample = Sample(
            scans=sitk_scans,
            lbl=sitk_annotation,
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
            destination = scans_out_dir / f"{self.subject_id}_{i:04d}.nii.gz"
            atomic_image_write(scan, path=destination, mkdir=True)
            self.write_log(f'Wrote image to {destination}')

        if sitk_annotation:
            destination = annotations_out_dir / f"{self.subject_id}.nii.gz"
            atomic_image_write(sample.lbl, path=annotations_out_dir / f"{self.subject_id}.nii.gz", mkdir=True)
            self.write_log(f'Wrote annotation to {destination}')


class MHA2nnUNetConverter(Converter):
    def __init__(
        self,
        output_dir: PathLike,
        scans_dir: PathLike,
        scans_out_dirname: str = 'imagesTr',
        annotations_dir: Optional[PathLike] = None,
        annotations_out_dirname: Optional[str] = 'labelsTr',
        mha2nnunet_settings: Union[PathLike, Dict] = None
    ):
        """
        Parameters
        ----------
        output_dir: PathLike
            path to store the resulting nnUNet archive.
        scans_dir: PathLike
            path to the scan archive. Used as base path for the relative paths of the archive items.
        scans_out_dirname: str, default: 'imagesTr'
            dirname to store scan output, will be a direct descendant of `output_dir`.
        annotations_dir: PathLike
            path to the annotations archive. Used as base path for the relative paths of the archive items.
        annotations_out_dirname: str, default: 'labelsTr'
            dirname to store annotation output, will be a direct descendant of `output_dir`.
        mha2nnunet_settings: Union[PathLike, Dict]
            object with cases, nnUNet-shaped dataset.json and optional parameters.
            May be a dictionary containing mappings, dataset.json, and optionally options, or a path to a JSON file with these elements.
            - dataset_json: see nnUNet description of a valid dataset.json object.
            - cases: list of objects. Each case is to be an object with a patient_id,
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

        jsonschema.validate(mha2nnunet_settings, mha2nnunet_schema, cls=jsonschema.Draft7Validator)
        self.settings = MHA2nnUNetSettings(
            dataset_json=mha2nnunet_settings['dataset_json'],
            preprocessing=PreprocessingSettings(**mha2nnunet_settings['preprocessing']),
            **mha2nnunet_settings.get('options', {})
        )

        self.scans_dir = Path(scans_dir)
        self.annotations_dir = Path(annotations_dir) if annotations_dir else None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.scans_out_dir = output_dir / self.settings.task_name / scans_out_dirname
        self.annotations_out_dir = output_dir / self.settings.task_name / annotations_out_dirname if annotations_dir else None

        self.cases = self._init_cases(mha2nnunet_settings['archive'])

        self.initialize_log(output_dir, self.settings.verbose)

    def _init_cases(self, archive: List[Dict]) -> List[MHA2nnUNetCase]:
        return [MHA2nnUNetCase(scans_dir=self.scans_dir, annotations_dir=self.annotations_dir, settings=self.settings, **kwargs) for kwargs in archive]

    def convert(self):
        self._convert('MHA2nnUNet', self.settings.num_threads, self.cases, (self.scans_out_dir, self.annotations_out_dir))

    def create_dataset_json(self, path: PathLike = '.', is_testset: bool = False, merge: 'MHA2nnUNetConverter' = None) -> Dict:
        """
        Create dataset.json for nnUNet raw data archive.

        Parameters
        ----------
        path: PathLike, default: '.'
            dir path to save ./dataset.json to. If None, will not output a file, if '.', will output to `output_dir`.
        is_testset: bool, default: False
            whether this conversation was a test set or a training set
        merge: MHA2nnUNetConverter, default: None
            merge the contents of this converter with another. If `is_testset` is True, the other converter is
            assumed to be a training set, and vice versa. All other values in dataset.json are taken from this converter.

        Returns
        -------
        dataset : dict
            contents of dataset.json
        """
        # use contents of archive->dataset_json as starting point
        dataset = self.settings.dataset_json
        if 'name' not in dataset:
            dataset['name'] = '_'.join(self.settings.task_name.split('_')[1:])

        testset = self if is_testset else merge
        trainingset = self if not is_testset else merge

        dataset['numTraining'] = len(trainingset.cases) if trainingset else 0
        dataset['numTest'] = len(testset.cases) if testset else 0

        for set, key in [(trainingset, 'training'), (testset, 'test')]:
            dataset[key] = [
                {
                    "image": f"./{set.scans_out_dir.name}/{case.subject_id}.nii.gz",
                    "label": f"./{set.annotations_out_dir.name}/{case.subject_id}.nii.gz"
                }
                for case in set.cases
            ] if set else []

        if path:
            dataset_fn = self.scans_out_dir.parent / 'dataset.json' if path == '.' else Path(path / 'dataset.json')
            logging.info(f'Saving dataset info to {dataset_fn}')
            with open(dataset_fn, 'w') as fp:
                json.dump(dataset, fp, indent=4)

        return dataset
