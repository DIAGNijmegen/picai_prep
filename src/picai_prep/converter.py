import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm


@dataclass
class Case(ABC):
    patient_id: str
    study_id: str
    error: Optional[Exception] = None
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
        finally:
            return self.compile_log()

    @property
    def subject_id(self):
        return f'{self.patient_id}_{self.study_id}'

    @property
    def is_valid(self):
        return self.error is None

    def invalidate(self, error: Exception):
        self.error = error

    def write_log(self, msg: str):
        self._log.append(msg)

    @abstractmethod
    def compile_log(self):
        raise NotImplementedError()

    def __repr__(self):
        return f'Case({self.subject_id})'

    def __del__(self):
        print("Case deleted from memory")


class Converter:
    @staticmethod
    def initialize_log(output_dir: Path, verbose: int):
        if verbose >= 1:
            logfile = output_dir / f'picai_prep_{datetime.now().strftime("%Y%m%d%H%M%S")}.log'
            print(f'Writing log to {logfile.absolute()}')
            logging.basicConfig(level=logging.INFO, format='%(message)s', filename=logfile)
            logging.info(f'Output directory set to {output_dir.absolute().as_posix()}')
        else:
            logging.disable(logging.INFO)

    @staticmethod
    def _convert(title: str, cases: List[Case], parameters: Dict[str, Any], num_threads: int = 4):
        start_time = datetime.now()
        logging.info(f'{title} conversion started at {start_time.isoformat()}\n')

        if False:
            with ThreadPoolExecutor(max_workers=num_threads) as pool:
                futures = {pool.submit(case.convert, **parameters): case for case in cases}
                for future in tqdm(as_completed(futures), total=len(cases)):
                    case_log = future.result()
                    if case_log:
                        logging.info(case_log)

                    # remove case from memory
                    case = futures[future]
                    del futures[future]
                    del case
        else:
            for case in tqdm(cases):
                case_log = case.convert(**parameters)
                if case_log:
                    logging.info(case_log)
                del case

        end_time = datetime.now()
        logging.info(f'{title} conversion ended at {end_time.isoformat()}\n\t(runtime {end_time - start_time})')
