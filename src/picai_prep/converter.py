from dataclasses import dataclass
from pathlib import Path

from picai_prep.utilities import dcm2mha_schema, dicom_tags, get_pydicom_value, lower_strip, make_sitk_readers, plural
from picai_prep.data_utils import PathLike, atomic_image_write

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
class Settings:
    num_threads: int = 4
    verbose: int = 1



@dataclass
class Case:
    patient_id: str
    study_id: str

    def __repr__(self):
        return f'Case({self.patient_id}_{self.study_id})'

    @property
    def subject_id(self):
        return f'{self.patient_id}_{self.study_id}'

