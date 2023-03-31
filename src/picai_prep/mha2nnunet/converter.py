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
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jsonschema

from picai_prep.converter import Converter
from picai_prep.data_utils import PathLike
from picai_prep.mha2nnunet.preprocessing import PreprocessingSettings
from picai_prep.utilities import mha2nnunet_schema
from picai_prep.mha2nnunet.case import MHA2nnUNetCase
from picai_prep.mha2nnunet.settings import MHA2nnUNetSettings


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
