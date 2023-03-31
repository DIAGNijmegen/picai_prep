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
import gc
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Case(ABC):
    patient_id: str
    study_id: str
    error: Optional[Exception] = None
    skip_conversion: bool = False
    _log: List[str] = field(default_factory=list)

    @abstractmethod
    def convert_item(self, **kwargs):
        """"
        Execute conversion process, please implement this.
        """
        raise NotImplementedError()

    def convert(self, **kwargs):
        """"
        Execute conversion process, while handling errors.
        Please override the `convert_item` method to implement the conversion process.
        """
        try:
            self.convert_item(**kwargs)
        except Exception as e:
            self.invalidate(e)
        self.compile_log()

    @property
    def subject_id(self):
        return f'{self.patient_id}_{self.study_id}'

    @property
    def is_valid(self):
        return self.error is None

    def invalidate(self, error: Exception):
        self.error = error
        self.error_trace = traceback.format_exc()

    def write_log(self, msg: str):
        self._log.append(msg)

    @abstractmethod
    def compile_log(self):
        raise NotImplementedError()

    def cleanup(self):
        self._log = None
        gc.collect()

    def __repr__(self):
        return f'Case({self.subject_id})'
