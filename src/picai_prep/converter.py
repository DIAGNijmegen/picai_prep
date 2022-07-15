import logging
from datetime import datetime
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from picai_prep.data_utils import PathLike


class ConverterException(Exception):
    """Base Exception for errors in an item (series within a case)"""

    def __init__(self, message: str):
        super().__init__(message)

    def __str__(self):
        return f'{type(self).__name__}: {", ".join([a for a in self.args])}'


class ArchiveItemPathNotFoundError(ConverterException):
    """Exception raised when a archive path could not be found"""

    def __init__(self, path: PathLike):
        super().__init__(f"Provided archive item path not found ({path})")


@dataclass
class Case:
    patient_id: str
    study_id: str

    error: Optional[Exception] = None

    _log: List[str] = field(default_factory=list)

    def __repr__(self):
        return f'Case({self.patient_id}_{self.study_id})'

    @property
    def subject_id(self):
        return f'{self.patient_id}_{self.study_id}'

    def invalidate(self, error: Exception):
        self.error = error

    @property
    def is_valid(self):
        return self.error is None

    def write_log(self, msg: str):
        self._log.append(msg)

    @abstractmethod
    def compile_log(self):
        pass

    def convert(self, *args):
        try:
            self._convert(*args)
        except Exception as e:
            self.invalidate(e)
        finally:
            return self.compile_log()

    @abstractmethod
    def _convert(self, *args):
        pass


class Converter:
    @staticmethod
    def initialize_log(output_dir: Path, verbose: int):
        if verbose > 0:
            logfile = output_dir / f'picai_prep_{datetime.now().strftime("%Y%m%d%H%M%S")}.log'
            logging.basicConfig(level=logging.INFO, format='%(message)s', filename=logfile)
            logging.info(f'Output directory set to {output_dir.absolute().as_posix()}')
            print(f'Writing log to {logfile.absolute()}')
        else:
            logging.disable(logging.INFO)

    @staticmethod
    def _convert(title: str, num_threads: int, cases: list, parameters: tuple):
        start_time = datetime.now()
        logging.info(f'{title} conversion started at {start_time.isoformat()}\n')

        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = {pool.submit(case.convert, *parameters): case for case in cases}
            for future in tqdm(as_completed(futures), total=len(cases)):
                case_log = future.result()
                if case_log:
                    logging.info(case_log)

        end_time = datetime.now()
        logging.info(f'{title} conversion ended at {end_time.isoformat()}\n\t(runtime {end_time - start_time})')