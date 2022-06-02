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


import datetime
import logging
import os
import json
import jsonschema
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union, Set, Tuple, List

from picai_prep.data_utils import PathLike
from picai_prep.utilities import plural, metadata_defaults, lower_strip


class Case:
    pass
    # path: PathLike
    # patient_id: Optional[str] = None
    # study_id: Optional[str] = None
    # input_dir: Optional[PathLike] = None
    # skip: Optional[bool] = None
    #
    # # image metadata
    # resolution: Optional[List[float]] = None
    # metadata: Optional[Dict[str, str]] = None
    #
    # # dcm2mha specific attributes
    # dicom_paths: Optional[List[PathLike]] = None
    # mappings: Optional[List[str]] = None
    # targets: List[PathLike] = None  # TODO: fix this default value/type
@dataclass
class Dicom2MHAItem:
    patient_id: str
    study_id: str
    path: Path

    def __post_init__(self):
        self.resolution = None
        self.metadata = None

class Dicom2MHACase(Case):
    def __init__(self):
        self.items = {}

    def add(self, input_dir: Path, patient_id: str, study_id: str, path: PathLike):
        source = input_dir / path

        if not source.exists():
            item['error'] = (f"Provided archive item path not found ({source})", 'path not found')
        elif not os.path.isdir(source):
            item['error'] = (f"Provided archive item path is not a directory ({source})", 'path not a directory')
        elif source in sources:
            item['error'] = (f"Provided archive item path already exists ({source})", 'path already exists')
        sources.add(source)


class Dicom2MHAConverter:
    def __init__(self, input_dir: PathLike, output_dir: PathLike, dcm2mha_settings: Union[PathLike, Dict]):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        if isinstance(dcm2mha_settings, PathLike):
            with open(dcm2mha_settings) as fp:
                settings = json.load(fp)
        else:
            settings = dcm2mha_settings

        from picai_prep.utilities import dcm2mha_schema
        jsonschema.validate(settings, dcm2mha_schema, cls=jsonschema.Draft7Validator)

        self.options = settings.get('options', {})
        self.mappings, self.tags = self._dicom_names_to_tags(settings.get('mappings', {}))
        self.cases = self._define_cases(settings.get('archive', {}))


    @staticmethod
    def _define_cases(archive: List[Dict]) -> List[Dicom2MHACase]:
        cases = {}
        for item in archive:



    @staticmethod
    def _dicom_names_to_tags(mappings: Dict) -> Tuple[Dict, Set]:
        from picai_prep.utilities import dicom_tags

        tags = set()
        for name, mapping in mappings.items():
            map = dict()
            for key, value in mapping.items():
                try:
                    metadata = dicom_tags[lower_strip(key)]
                    tags.add(metadata)

                    if len(value) == 0 or any(type(v) is not str for v in value):
                        raise ValueError(f"Non-string elements found in {name}/{key} mapping")

                    map[metadata] = [lower_strip(v) for v in value]
                except KeyError:
                    raise KeyError(f"Invalid key '{key}' in '{name}' mapping, see metadata.json for valid keys.")
            mappings[name] = map
        return mappings, tags



class ArchiveConverter(ABC):
    def __init__(
        self,
        input_path: PathLike,
        output_path: PathLike,
        num_threads: int = 4,
        silent=False
    ):
        super().__init__()
        self.items = []
        self._history = None

        self.input_dir = Path(os.path.abspath(input_path))
        self.output_dir = Path(os.path.abspath(output_path))
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.num_threads = num_threads
        self.silent = silent
        self._start_time = datetime.datetime.now()

        logfile = f'picai_prep_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.log'
        logging.basicConfig(filemode='w', level=logging.INFO, format='%(message)s',
                            filename=self.output_dir / logfile)

        self.info(f'Program started at {self._start_time.isoformat()}', force=True)
        self.info(f"Output directory set to {self.output_dir.absolute().as_posix()}, writing log to {self.output_dir / logfile}")

    @abstractmethod
    def convert(self):
        raise NotImplementedError()

    @property
    def valid_items(self) -> filter:
        return filter(lambda a: a.get('error', None) is None, self.items)

    @property
    def has_valid_items(self):
        return self.num_valid_items > 0

    @property
    def num_valid_items(self):
        return len(list(self.valid_items))

    def valid_items_str(self, syntax: str = 'item') -> str:
        return plural(self._history.num_items, syntax)

    def item_log(self, item, msg: str):
        self._history.add(msg)
        if item:
            ie = self.item_log_value(item)
            if ie:
                logging.error('\n\t' + ie)

    @abstractmethod
    def item_log_value(self, item: Dict[str, str]):
        raise NotImplementedError()

    def info(self, *msg, force=False):
        msg = ' '.join([str(m) for m in msg if m])
        logging.info('\n' + msg)
        if not self.silent or force:
            print(msg, '\n')

    def complete(self):
        end_time = datetime.datetime.now()
        program_duration = end_time - self._start_time
        self.info(f'Program ended at {end_time.isoformat()} (took {program_duration})', force=True)

    def next_history(self):
        self._history = History(len(list(self.items))) if self._history is None else self._history.next()

    def get_history_report(self):
        return self._history.report()


class History:
    def __init__(self, num_items: int, past: Optional["History"] = None):
        """
        Parameters:
        - num_items: number of items remaining in the preprocessing pipeline
        - past: History object of previous step in the preprocessing pipeline

        The ledger holds the number of errors/messages of a specific type,
        which allows to show aggregates to the user.
        """
        self.num_items = num_items
        self.ledger = dict()
        self.past = past

    def report(self):
        report = []
        if self.past is not None:
            prev, now = self.past.num_items, self.num_items
            report += [f"{prev - now} ignored. ({prev} -> {now})"] if prev - now > 0 else [' ']

        report += [f"\t{item_error}: {count}" for item_error, count in self.ledger.items()]
        return '\n'.join(report)

    def add(self, key: str):
        self.ledger[key] = self.ledger.get(key, 0) + 1
        self.num_items -= 1

    def next(self):
        return History(self.num_items, self)
