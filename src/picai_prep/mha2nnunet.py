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


import dataclasses
import json
import logging
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jsonschema
import SimpleITK as sitk
from tqdm import tqdm

from picai_prep.archive import ArchiveConverter
from picai_prep.data_utils import PathLike, atomic_image_write
from picai_prep.preprocessing import PreprocessingSettings, Sample
from picai_prep.utilities import mha2nnunet_schema, plural

@dataclass
class ConversionItem():
    """Class to hold conversion information"""
    input_dir: PathLike
    annotations_dir: Optional[PathLike]
    patient_id: str
    study_id: str
    scan_paths: List[PathLike]
    out_dir_scans: PathLike = "imagesTr"
    out_dir_annot: PathLike = "labelsTr"
    annotation_path: Optional[PathLike] = None
    all_scan_properties: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    label_properties: Dict[str, Any] = dataclasses.field(default_factory=dict)
    error: Optional[Tuple[str, str]] = None
    skip: bool = False

    def __post_init__(self):
        # convert input paths
        self.input_dir = Path(self.input_dir)
        if self.annotations_dir is not None:
            self.annotations_dir = Path(self.annotations_dir)
        self.scan_paths = [Path(path) for path in self.scan_paths]
        self.out_dir_scans = Path(self.out_dir_scans)
        self.out_dir_annot = Path(self.out_dir_annot)

        output_paths = []
        # check if all scans exist and append to conversion plan
        for i, scan_path in enumerate(self.scan_paths):
            # check (relative) path of input scans
            path = self.input_dir / scan_path
            if not os.path.exists(path):
                self.error = (f"Scan not found at {path}", 'scan not found')
                continue

            output_path = self.out_dir_scans / f"{self.subject_id}_{i:04d}.nii.gz"
            self.all_scan_properties += [{
                'input_path': path,
                'output_path': output_path,
            }]
            output_paths += [output_path]

        # check if annotation exists and append to conversion plan
        if self.annotation_path is not None:
            if self.annotations_dir is None:
                self.annotations_dir = self.input_dir

            # check (relative) path of input scans
            path = self.annotations_dir / self.annotation_path
            if not os.path.exists(path):
                self.error = (f"Annotation not found at {path}", 'annotation not found')

            output_path = self.out_dir_annot / f"{self.subject_id}.nii.gz"
            self.label_properties = {
                'input_path': path,
                'output_path': output_path,
            }
            output_paths += [output_path]

        # check if all files already exist
        if all([os.path.exists(path) for path in output_paths]):
            self.skip = True

    @property
    def subject_id(self):
        return f"{self.patient_id}_{self.study_id}"



@dataclass
class MHA2nnUNetSettings:
    dataset_json: dict
    preprocessing: PreprocessingSettings
    num_threads: int = 4


class Dicom2MHAConverter:
    def __init__(
        self,
        input_dir: PathLike,
        output_dir: PathLike,
        mha2nnunet_settings: Union[PathLike, Dict] = None,
    ):
        """
        Parameters
        ----------
        WIP
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # parse settings
        if isinstance(mha2nnunet_settings, (Path, str)):
            with open(mha2nnunet_settings) as fp:
                mha2nnunet_settings = json.load(fp)

        jsonschema.validate(mha2nnunet_settings, mha2nnunet_schema, cls=jsonschema.Draft7Validator)
        self.settings = MHA2nnUNetSettings(
            dataset_json=mha2nnunet_settings['dataset_json'],
            preprocessing=PreprocessingSettings(**mha2nnunet_settings['preprocessing']),
            **mha2nnunet_settings.get('options', {})
        )

        self.cases = self._init_cases(mha2nnunet_settings['archive'])

        logfile = self.output_dir / f'picai_prep_{datetime.now().strftime("%Y%m%d%H%M%S")}.log'
        logging.basicConfig(level=logging.INFO, format='%(message)s', filename=logfile)
        logging.info(f'Output directory set to {self.output_dir.absolute().as_posix()}')
        print(f'Writing log to {logfile.absolute()}')

    # def _init_cases(self, archive: List[Dict]) -> List[Dicom2MHACase]:
    #     cases = {}
    #     for item in archive:
    #         key = tuple(item[id] for id in metadata_defaults.keys())  # (patient_id, study_id)
    #         cases[key] = cases.get(key, []) + [item['path']]
    #
    #     return [
    #         Dicom2MHACase(self.input_dir, patient_id, study_id, paths, self.settings)
    #         for (patient_id, study_id), paths in cases.items()
    #     ]

    def convert(self):
        start_time = datetime.now()
        logging.info(f'MHA2nnUNet conversion started at {start_time.isoformat()}\n')

        with ThreadPoolExecutor(max_workers=self.settings.num_threads) as pool:
            futures = {pool.submit(case.convert, self.output_dir): case for case in self.cases}
            for future in tqdm(as_completed(futures), total=len(self.cases)):
                case_log = future.result()
                logging.info(case_log)

        end_time = datetime.now()
        logging.info(f'MHA2nnUNet conversion ended at {end_time.isoformat()}\n\t(runtime {end_time - start_time})')


class MHA2nnUNetConverter(ArchiveConverter):
    def __init__(
        self,
        input_path: PathLike,
        output_path: PathLike,
        settings_path: PathLike,
        annotations_path: Optional[PathLike] = None,
        lbl_preprocess_func: Optional[Callable[[sitk.Image], sitk.Image]] = None,
        lbl_postprocess_func: Optional[Callable[[sitk.Image], sitk.Image]] = None,
        scan_preprocess_func: Optional[Callable[[sitk.Image], sitk.Image]] = None,
        scan_postprocess_func: Optional[Callable[[sitk.Image], sitk.Image]] = None,
        out_dir_scans: PathLike = "imagesTr",
        out_dir_annot: PathLike = "labelsTr",
        num_threads: int = 4,
        silent: bool = False,
    ):
        """
        Converts MHA archive into nnUNet raw data format, as specified by the conversion json.

        Parameters:
        - input_path: Root directory for input, e.g. /path/to/archive/
        - output_path: Root directory for output
        - archive: JSON mappings file
                   Verifies whether archive.json returns a valid object prior to initialization
        - silent: control verbosity of conversion process
        """
        super().__init__(
            input_path=input_path,
            output_path=output_path,
            num_threads=num_threads,
            silent=silent
        )
        if annotations_path is not None:
            annotations_path = Path(annotations_path)
        self.create_dataset_json = True

        # read and verify conversion settings
        with open(settings_path) as fp:
            settings = json.load(fp)
        jsonschema.validate(settings, mha2nnunet_schema, cls=jsonschema.Draft7Validator)

        # store conversion settings
        self.preprocessing_settings = PreprocessingSettings(**settings['preprocessing'])
        self.items = settings['archive']
        self.dataset_json = settings['dataset_json']
        self.conversion_plan: List[ConversionItem] = []
        self.out_dir_scans = out_dir_scans
        self.out_dir_annot = out_dir_annot
        self.annotations_path = annotations_path
        self.lbl_preprocess_func = lbl_preprocess_func
        self.lbl_postprocess_func = lbl_postprocess_func
        self.scan_preprocess_func = scan_preprocess_func
        self.scan_postprocess_func = scan_postprocess_func

        self.next_history()  # create initial history step
        self.info("Provided mha2nnunet archive is valid.", self.get_history_report())  # report number of items
        self.next_history()  # next history

    def _plan(self):
        """
        Build full conversion plan
        """
        # parse parameters
        self.info(f"Starting preprocessing script for {self.task}.\n"
                  + f"\tReading scans from {self.input_dir}\n"
                  + (f"\tReading annotations from {self.annotations_path}\n" if self.annotations_path else "")
                  + f"\nCreating preprocessing plan with {self.valid_items_str()}.")

        total_scans, total_lbl = 0, 0

        for item in self.items:
            # parse conversion info and store result
            conv_item = ConversionItem(
                input_dir=self.input_dir,
                annotations_dir=self.annotations_path,
                out_dir_scans=self.output_dir / self.task / self.out_dir_scans,
                out_dir_annot=self.output_dir / self.task / self.out_dir_annot,
                **item
            )
            item['conversion_item'] = conv_item
            if conv_item.error is not None:
                message, error_type = conv_item.error
                item['error'] = message
                self.item_log(item, error_type)
            else:
                total_scans += len(item['scan_paths'])
                total_lbl += 1 if 'annotation_path' in item else 0

        self.info(f"Preprocessing plan completed, with {self.valid_items_str()} containing a total of "
                  f"{plural(total_scans, 'scan')} and {plural(total_lbl, 'label')}.",
                  self.get_history_report())

    def _convert_item(self, item) -> bool:
        conv_item = item['conversion_item']
        if conv_item.skip:
            # skip, already converted
            return False

        # read images
        scans = []
        for scan_properties in conv_item.all_scan_properties:
            scans += [sitk.ReadImage(str(scan_properties['input_path']))]

        # read label
        lbl = None
        if conv_item.label_properties:
            lbl = sitk.ReadImage(str(conv_item.label_properties['input_path']))

        # set up Sample
        sample = Sample(
            scans=scans,
            lbl=lbl,
            settings=self.preprocessing_settings,
            lbl_preprocess_func=self.lbl_preprocess_func,
            lbl_postprocess_func=self.lbl_postprocess_func,
            scan_preprocess_func=self.scan_preprocess_func,
            scan_postprocess_func=self.scan_postprocess_func,
            name=conv_item.subject_id
        )

        # perform preprocessing
        sample.preprocess()

        # write images
        for scan, scan_properties in zip(sample.scans, conv_item.all_scan_properties):
            atomic_image_write(scan, path=scan_properties['output_path'], mkdir=True)

        if sample.lbl:
            atomic_image_write(sample.lbl, path=conv_item.label_properties['output_path'], mkdir=True)

        return True

    def _write(self):
        """
        Preprocess studies for nnUNet

        Requires:
        - all_scan_properties: list of scan properties:
        [
            {
                'input_path': path to input scan (T2/ADC/high b-value/etc.),
                'output_path': path to store preprocessed scan,
            }
        ]

        - label_properties: label properties:
        {
            'input_path': path to label,
            'output_path': path to store preprocessed label,
        }
        """
        successes, errors = 0, 0

        with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
            futures = {pool.submit(self._convert_item, item): item for item in self.valid_items}
            for future in tqdm(as_completed(futures), total=self.num_valid_items):
                item = futures[future]
                try:
                    success = future.result()
                except Exception as e:
                    item['error'] = f'Unexpected error: {e}'
                    self.item_log(item, 'unexpected error')
                    errors += 1
                else:
                    successes += 1 if success else 0

        skipped = self.num_valid_items - successes - errors
        self.info(f"Processed {plural(self.num_valid_items, 'item')}, "
                  f"with {plural(skipped, 'case')} skipped and {plural(errors, 'error')}.")

    def generate_json(self):
        """
        Create dataset.json for nnUNet raw data archive
        """
        if self.create_dataset_json:
            # use contents of archive->dataset_json as starting point
            json_dict = dict(self.dataset_json)
            if 'name' not in json_dict:
                json_dict['name'] = self.name
            json_dict['numTraining'] = self.num_valid_items
            json_dict['numTest'] = 0  # not implemented currently
            json_dict['training'] = [
                {
                    "image": f"./imagesTr/{item['conversion_item'].subject_id}.nii.gz",
                    "label": f"./labelsTr/{item['conversion_item'].subject_id}.nii.gz"
                }
                for item in self.valid_items
            ]
            json_dict['test'] = []

            dataset_fn = self.output_dir / self.task / "dataset.json"
            if not os.path.exists(dataset_fn):
                self.info(f"Saving dataset info to {dataset_fn}")
                with open(dataset_fn, 'w') as fp:
                    json.dump(json_dict, fp, indent=4)

    def convert(self):
        """
        Perform conversion steps:
        1. Gather metadata
        2. Preprocess and write scans
        3. Generate dataset.json
        """
        for step in [self._plan, self._write]:
            if self.has_valid_items:
                step()
                self.next_history()
            else:
                self.info("Aborted conversion, no items to convert.")

        if len(self.items) == self.num_valid_items:
            self.generate_json()
        else:
            self.info(f"Did not generate dataset.json, as only {self.num_valid_items}/{len(self.items)} items are converted successfully.")

        self.complete()

    def item_log_value(self, item: Dict[str, str]):
        lpad = ' ' * (len(str(len(self.items))) + 1)

        if 'error' in item:
            result = f"{lpad}{item['error']}"
        elif 'skip' in item:
            result = f"{lpad}{item['skip']}"
        else:
            return None

        return f"{str(self.items.index(item)).zfill(len(lpad))} {item['patient_id']}/{item['study_id']}\n" \
               f"\t{result}"

    @property
    def task(self):
        return self.dataset_json['task']

    @property
    def name(self):
        if 'name' in self.dataset_json:
            return self.dataset_json['name']
        else:
            # grab e.g. Prostate_mpMRI_csPCa from Task101_Prostate_mpMRI_csPCa
            return "_".join(self.task.split('_')[1:])
