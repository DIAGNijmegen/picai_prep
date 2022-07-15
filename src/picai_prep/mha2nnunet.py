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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jsonschema
import SimpleITK as sitk
from tqdm import tqdm

from picai_prep.archive import ArchiveConverter
from picai_prep.data_utils import PathLike, atomic_image_write
from picai_prep.preprocessing import PreprocessingSettings, Sample
from picai_prep.utilities import mha2nnunet_schema, plural
from picai_prep.converter import ConverterException

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
    scans_dirname: PathLike = "imagesTr"
    annotation_dirname: PathLike = "labelsTr"
    lbl_preprocess_func: Optional[Callable[[sitk.Image], sitk.Image]] = None
    lbl_postprocess_func: Optional[Callable[[sitk.Image], sitk.Image]] = None
    scan_preprocess_func: Optional[Callable[[sitk.Image], sitk.Image]] = None
    scan_postprocess_func: Optional[Callable[[sitk.Image], sitk.Image]] = None
    num_threads: int = 4
    verbose: int = 1

    @property
    def task_name(self) -> str:
        return self.dataset_json['task']


@dataclass
class MHA2nnUNetCase():
    input_dir: Path
    annotations_dir: Path
    patient_id: str
    study_id: str
    scan_paths: List[Path]
    annotation_path: Optional[Path]
    settings: MHA2nnUNetSettings

    error: Optional[Exception] = None

    _log: List[str] = field(default_factory=list)

    def __repr__(self):
        return f'Case({self.patient_id}_{self.study_id})'

    def invalidate(self, reason: str):
        self.error = ConverterException(f'Invalidated: {reason}')

    @property
    def is_valid(self):
        return self.error is None

    def write_log(self, msg: str):
        self._log.append(msg)

    def compile_log(self):
        return ''

    def convert(self, *args):
        scans_out_dir, annotations_out_dir = args
        try:
            self.initialize()
        except Exception as e:
            self.invalidate(str(e))
            logging.error(str(e))
        finally:
            return self.compile_log()

    def initialize(self):
            self._log = [f'Importing {plural(len(self.scan_paths), "scans")}']

            self.scans, missing_paths = [], []
            # check if all scans exist and append to conversion plan
            # except Exception as e:
            #     serie.error = e
            #     logging.error(str(e))
            # finally:
            #     self.write_log(f'\t+ ({len(self.series)}) {full_path}')
            #     self.series.append(serie)

            try:
                for i, scan_path in enumerate(self.scan_paths):
                    # check (relative) path of input scans
                    path = self.input_dir / scan_path
                    if not path.exists():
                        missing_paths.append(path)
                        continue

                    self.scans.append(path)
                    self.write_log(f'\t+ ({len(self.scans)}) {path}')

                if len(missing_paths) > 0:
                    raise FileNotFoundError(','.join([str(p) for p in missing_paths]))
            except Exception as e:
                self.error = e
                logging.error(str(e))

            if self.annotations_dir:
                self.write_log(f'Importing annotation')
                self.annotation = self.input_dir / self.annotation_path
                if not self.annotation.exists():
                    raise FileNotFoundError(self.annotation)

                self.write_log(f'\t+ {self.annotation}')

            if all([os.path.exists(path) for path in output_paths]):
                self.invalidate('all files already converted')
        # self._log = [f'Importing {plural(len(self.paths), "serie")}']
        #
        # full_paths = set()
        # for path in self.paths:
        #     full_path = self.input_dir / path
        #     serie = Series(full_path, self.patient_id, self.study_id)
        #     try:
        #         if path in full_paths:
        #             raise FileExistsError(path)
        #         full_paths.add(full_path)
        #     except Exception as e:
        #         serie.error = e
        #         logging.error(str(e))
        #     finally:
        #         self.write_log(f'\t+ ({len(self.series)}) {full_path}')
        #         self.series.append(serie)
        #
        # if not all([serie.is_valid for serie in self.series]):
        #     self.invalidate()

    # def __post_init__(self):
    #     output_paths = []
    #     # check if all scans exist and append to conversion plan
    #     for i, scan_path in enumerate(self.scan_paths):
    #         # check (relative) path of input scans
    #         path = self.input_dir / scan_path
    #         if not os.path.exists(path):
    #             self.error = (f"Scan not found at {path}", 'scan not found')
    #             continue
    #
    #         output_path = self.out_dir_scans / f"{self.subject_id}_{i:04d}.nii.gz"
    #         self.all_scan_properties += [{
    #             'input_path': path,
    #             'output_path': output_path,
    #         }]
    #         output_paths += [output_path]
    #
    #     # check if annotation exists and append to conversion plan
    #     if self.annotation_path is not None:
    #         if self.annotations_dir is None:
    #             self.annotations_dir = self.input_dir
    #
    #         # check (relative) path of input scans
    #         path = self.annotations_dir / self.annotation_path
    #         if not os.path.exists(path):
    #             self.error = (f"Annotation not found at {path}", 'annotation not found')
    #
    #         output_path = self.out_dir_annot / f"{self.subject_id}.nii.gz"
    #         self.label_properties = {
    #             'input_path': path,
    #             'output_path': output_path,
    #         }
    #         output_paths += [output_path]
    #
    #     # check if all files already exist
    #     if all([os.path.exists(path) for path in output_paths]):
    #         self.skip = True


class MHA2nnUNetConverter:
    def __init__(
        self,
        output_dir: PathLike,
        scans_dir: PathLike,
        scans_out_dirname: str = None,
        annotations_dir: Optional[PathLike] = None,
        annotations_out_dirname: Optional[str] = None,
        mha2nnunet_settings: Union[PathLike, Dict] = None
    ):
        """
        Parameters
        ----------
        WIP
        """
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

        self.scans_dir = Path(scans_dir)
        self.annotations_dir = Path(annotations_dir) if annotations_dir else None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.scans_out_dir = output_dir / self.settings.task_name / scans_out_dirname
        self.annotations_out_dir = output_dir / self.settings.task_name / annotations_out_dirname if annotations_dir else None

        self.cases = self._init_cases(mha2nnunet_settings['archive'])

        logfile = output_dir / f'picai_prep_{datetime.now().strftime("%Y%m%d%H%M%S")}.log'
        logging.basicConfig(level=logging.INFO, format='%(message)s', filename=logfile)
        logging.info(f'Output directory set to {output_dir.absolute().as_posix()}')
        print(f'Writing log to {logfile.absolute()}')

    def _init_cases(self, archive: List[Dict]) -> List[MHA2nnUNetCase]:
        return [MHA2nnUNetCase(self.scans_dir, self.annotations_dir, **kwargs, settings=self.settings) for kwargs in archive]

    def convert(self):
        start_time = datetime.now()
        logging.info(f'MHA2nnUNet conversion started at {start_time.isoformat()}\n')

        with ThreadPoolExecutor(max_workers=self.settings.num_threads) as pool:
            futures = {pool.submit(case.convert, (self.scans_out_dir, self.annotations_out_dir)): case for case in self.cases}
            for future in tqdm(as_completed(futures), total=len(self.cases)):
                case_log = future.result()
                logging.info(case_log)

        end_time = datetime.now()
        logging.info(f'MHA2nnUNet conversion ended at {end_time.isoformat()}\n\t(runtime {end_time - start_time})')


class MHA2nnUNetaConverter(ArchiveConverter):
    def __init__(
        self,
        input_path: PathLike,
        output_path: PathLike,
        settings_path: PathLike,
        annotations_path: Optional[PathLike] = None,

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
        # self.lbl_preprocess_func = lbl_preprocess_func
        # self.lbl_postprocess_func = lbl_postprocess_func
        # self.scan_preprocess_func = scan_preprocess_func
        # self.scan_postprocess_func = scan_postprocess_func

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
