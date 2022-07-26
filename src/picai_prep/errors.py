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


class CriticalErrorInSiblingError(ConverterException):
    """Exception raised when a critical error in a sibling item occurs"""

    def __init__(self):
        super().__init__("Critical error in sibling item")


class MissingDICOMFilesError(ConverterException):
    """Exception raised when a DICOM series has missing DICOM slices"""

    def __init__(self, path: PathLike):
        super().__init__(f"Missing DICOM slices detected in {path}")


class NoMappingsApplyError(ConverterException):
    """Exception raised when no mappings apply to the case"""

    def __init__(self):
        super().__init__('None of the provided mappings apply to this item')


class UnreadableDICOMError(ConverterException):
    """Exception raised when a DICOM series could not be loaded"""

    def __init__(self, path: PathLike):
        super().__init__(f'Could not read {path} using either SimpleITK or pydicom')


class DCESeriesNotFoundError(ConverterException):
    """Exception raised when DCE series could not be found"""

    def __init__(self, subject_id: str):
        super().__init__(f"DCE series not found for {subject_id}")
