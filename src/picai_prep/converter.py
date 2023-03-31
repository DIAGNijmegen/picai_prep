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
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from picai_prep.case import Case


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
        num_cases_skipped = 0

        if num_threads >= 2:
            with ThreadPoolExecutor(max_workers=num_threads) as pool:
                futures = {pool.submit(case.convert, **parameters): case for case in cases}
                for future in tqdm(as_completed(futures), total=len(cases)):
                    case = futures[future]
                    case_log = future.result()
                    if case_log:
                        logging.info(case_log)
                    if case.skip_conversion:
                        num_cases_skipped += 1
                    case.cleanup()
        else:
            for case in tqdm(cases):
                case_log = case.convert(**parameters)
                if case_log:
                    logging.info(case_log)
                if case.skip_conversion:
                    num_cases_skipped += 1
                case.cleanup()

        logging.info(f'Skipped conversion of {num_cases_skipped}')
        end_time = datetime.now()
        logging.info(f'{title} conversion ended at {end_time.isoformat()}\n\t(runtime {end_time - start_time})')
