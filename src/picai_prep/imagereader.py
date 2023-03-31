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
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pydicom
import pydicom.errors
import SimpleITK as sitk

from picai_prep.data_utils import PathLike
from picai_prep.exceptions import (MissingDICOMFilesError, UnreadableDICOMError)
from picai_prep.utilities import dicom_tags


class DICOMImageReader:
    """
    Read folder containing DICOM slices (possibly enclosed in a 'dicom.zip' file).
    If both DICOM slices and a 'dicom.zip' file are present, the dicom.zip used.

    Parameters
    ----------
    image_series_path: PathLike
        path to the folder containing the DICOM slices. The folder should contain the DICOM slices,
        or a zip file named "dicom.zip" containing the DICOM slices.

    Usage
    -----
    >>> image = DICOMImageReader('path/to/dicom/folder').image

    or

    >>> reader = DICOMImageReader('path/to/dicom/folder')
    >>> image = reader.image
    >>> metadata = reader.metadata
    """

    def __init__(self, path: PathLike, verify_dicom_filenames: bool = True):
        self.path = Path(path)
        self.verify_dicom_filenames = verify_dicom_filenames
        self._image = None
        self._metadata = None
        self.dicom_slice_paths: Optional[List[str]] = None

        self.series_reader = sitk.ImageSeriesReader()
        if (self.path / "dicom.zip").exists():
            self.path = self.path / "dicom.zip"
            with zipfile.ZipFile(self.path, "r") as zf:
                self.dicom_slice_paths = [
                    self.path / name
                    for name in zf.namelist()
                    if name.endswith(".dcm")
                ]
            if self.verify_dicom_filenames:
                self._verify_dicom_filenames()
        else:
            self._set_dicom_list()

    @property
    def image(self) -> sitk.Image:
        if self._image is None:
            self._image = self._read_image()
        return self._image

    @property
    def metadata(self) -> Dict[str, str]:
        if self._metadata is None:
            self._metadata = self._read_metadata()
        return self._metadata

    def _read_image(self, path: Optional[PathLike] = None) -> sitk.Image:
        if path is None:
            path = self.path
        path = Path(path)

        if path.name == "dicom.zip":
            return self._read_image_dicom_zip(path=path)
        try:
            return self._read_image_sitk(path=path)
        except RuntimeError:
            # try again with pydicom
            return self._read_image_pydicom(path=path)

    @staticmethod
    def _filter_localizer_slices(dicom_slice_paths: List[str]) -> List[str]:
        """
        Filter out localizer slices (slices with ImageType == LOCALIZER).
        WARNING: this is slow and a heuristic that may not work for all datasets.
        """
        filtered_dicom_slice_paths = []
        for path in dicom_slice_paths:
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(path))
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()
            image_type = reader.GetMetaData("0008|0008")
            if "LOCALIZER" not in image_type.upper():
                filtered_dicom_slice_paths.append(path)
        return filtered_dicom_slice_paths

    def _set_dicom_list(self, path: Optional[PathLike] = None) -> None:
        """
        Set the list of paths to the DICOM slices.

        Parameters
        ----------
        path: PathLike
            path to the folder containing the DICOM slices. The folder should contain the DICOM slices,
            or a zip file named "dicom.zip" containing the DICOM slices.
            default: self.path
        """
        if path is None:
            path = self.path

        self.dicom_slice_paths = self.series_reader.GetGDCMSeriesFileNames(str(path))

        # verify DICOM files are found
        if len(self.dicom_slice_paths) == 0:
            raise MissingDICOMFilesError(self.path)

        if self.verify_dicom_filenames:
            self._verify_dicom_filenames()

    def _read_image_sitk(
        self,
        path: Optional[PathLike] = None,
    ) -> sitk.Image:
        """
        Read image using SimpleITK.

        Parameters
        ----------
        path: PathLike
            path to the folder containing the DICOM slices. The folder should contain the DICOM slices,
            or a zip file named "dicom.zip" containing the DICOM slices.
            default: self.path

        Returns
        -------
        image: SimpleITK.Image
        """
        if path is not None:
            self.path = path
            self._set_dicom_list(path=path)

        # read DICOM sequence
        try:
            self.series_reader.SetFileNames(self.dicom_slice_paths)
            image: sitk.Image = self.series_reader.Execute()
        except RuntimeError:
            # try again while removing localizer slices
            self.dicom_slice_paths = self._filter_localizer_slices(self.dicom_slice_paths)
            self.series_reader.SetFileNames(self.dicom_slice_paths)
            image: sitk.Image = self.series_reader.Execute()

        # read metadata from the last DICOM slice
        reader = sitk.ImageFileReader()
        reader.SetFileName(self.dicom_slice_paths[0])
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()

        # set metadata
        metadata = {key: reader.GetMetaData(key).strip() for key in reader.GetMetaDataKeys()}
        for key, value in metadata.items():
            if len(value) > 0:
                image.SetMetaData(key, value)

        return image

    def _read_image_pydicom(self, path: Optional[PathLike] = None) -> sitk.Image:
        """
        Read image using pydicom. Warning: experimental! This function has limited capabilities.

        Parameters
        ----------
        path: PathLike
            path to the folder containing the DICOM slices. The folder should contain the DICOM slices,
            or a zip file named "dicom.zip" containing the DICOM slices.
            default: self.path

        Returns
        -------
        image: SimpleITK.Image
        """
        if path is not None:
            self.path = path
            self._set_dicom_list(path=path)

        files = [pydicom.dcmread(dcm) for dcm in self.dicom_slice_paths]

        # skip files with no SliceLocation (eg. scout views)
        slices = filter(lambda a: hasattr(a, 'SliceLocation'), files)
        slices = sorted(slices, key=lambda s: s.SliceLocation)

        # create and fill 3D array
        image = np.zeros([len(slices)] + list(slices[0].pixel_array.shape))
        for i, s in enumerate(slices):
            image[i, :, :] = s.pixel_array

        # convert to SimpleITK
        image: sitk.Image = sitk.GetImageFromArray(image)
        ref = slices[0]  # corresponds to the same slice as with SimpleITK
        image.SetSpacing(list(ref.PixelSpacing) + [ref.SliceThickness])
        image.SetOrigin(ref.ImagePositionPatient)
        image.SetDirection(self.get_orientation_tuple_sitk(ref))

        for key in ref.keys():
            # collect all available metadata (with DICOM tags, e.g. 0010|1010, as keys)
            value = self.get_pydicom_value(ref, key)
            if value is not None:
                key = str(key).replace(", ", "|").replace("(", "").replace(")", "")
                image.SetMetaData(key, value)

        return image

    def _read_image_dicom_zip(self, path: PathLike) -> sitk.Image:
        """
        Read image from dicom.zip file using SimpleITK.
        """
        with zipfile.ZipFile(path) as zf:
            if not zf.namelist():
                raise RuntimeError('dicom.zip is empty')

            with tempfile.TemporaryDirectory() as tempdir:
                zf.extractall(tempdir)
                return self._read_image(tempdir)

    def _read_metadata(self) -> Dict[str, str]:
        if self._image is not None:
            return self._collect_metadata_sitk(self._image)

        if self.path.name == "dicom.zip":
            # read metadata from dicom.zip with pydicom
            with zipfile.ZipFile(self.path) as zf:
                if not zf.namelist():
                    raise RuntimeError('dicom.zip is empty')

                with tempfile.TemporaryDirectory() as tempdir:
                    targetpath = zf.extract(member=zf.namelist()[-1], path=tempdir)
                    return self._read_metadata_from_file(targetpath)

        # extract metadata from first/last(?) DICOM slice
        dicom_slice_path = self.dicom_slice_paths[0]
        return self._read_metadata_from_file(dicom_slice_path)

    def _read_metadata_from_file(self, path: PathLike) -> Dict[str, str]:
        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(path))
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()
            return self._collect_metadata_sitk(reader)
        except Exception:
            try:
                with pydicom.dcmread(path, stop_before_pixels=True) as ds:
                    return self._collect_metadata_pydicom(ds)
            except pydicom.errors.InvalidDicomError:
                raise UnreadableDICOMError(path)

    def _collect_metadata_sitk(self, ref: Union[sitk.Image, sitk.ImageFileReader]) -> Dict[str, str]:
        metadata = {}
        for key in ref.GetMetaDataKeys():
            # collect all available metadata (with DICOM tags, e.g. 0010|1010, as keys)
            metadata[key] = ref.GetMetaData(key).strip()
        for name, key in dicom_tags.items():
            # collect metadata with DICOM names, e.g. patientsage, as keys)
            metadata[name] = ref.GetMetaData(key).strip() if ref.HasMetaDataKey(key) else ''

        metadata["spacing_inplane"] = ref.GetSpacing()[0:2]
        metadata["image_direction"] = ref.GetDirection()
        return metadata

    def _collect_metadata_pydicom(self, ds: "pydicom.dataset.Dataset") -> Dict[str, str]:
        metadata = {}
        for key in ds.keys():
            # collect all available metadata (with DICOM tags, e.g. 0010|1010, as keys)
            value = self.get_pydicom_value(ds, key)
            if value is not None:
                key = str(key).replace(", ", "|").replace("(", "").replace(")", "")
                metadata[key] = value
        for key in dicom_tags.values():
            # collect metadata with DICOM names, e.g. patientsage, as keys)
            value = self.get_pydicom_value(ds, key)
            metadata[key] = value if value is not None else ''

        metadata["spacing_inplane"] = ds.PixelSpacing[0:2]
        return metadata

    @staticmethod
    def get_pydicom_value(ds: pydicom.dataset.Dataset, key: "Union[pydicom.tag.BaseTag, str]") -> str:
        if isinstance(key, str):
            key = '0x' + key.replace('|', '')
        if key in ds:
            result = ds[key]
            if result.is_empty:
                return ''
            result = result.value
            if isinstance(result, (list, pydicom.multival.MultiValue)):
                result = "\\".join([str(v) for v in result])
            return str(result)
        return ''

    @staticmethod
    def get_orientation_matrix(ds: pydicom.FileDataset) -> np.ndarray:
        x, y = np.array(list(map(float, ds.ImageOrientationPatient))).reshape(2, 3)
        return np.stack([x, y, np.cross(x, y)])

    def get_orientation_tuple_sitk(self, ds: pydicom.FileDataset) -> Tuple:
        return tuple(self.get_orientation_matrix(ds).transpose().flatten())

    def _verify_dicom_filenames(self, filenames: Optional[List[PathLike]] = None) -> bool:
        """
        Verify DICOM filenames have increasing numbers, with no gaps

        Common prefixes are removed from the filenames before checking the numbers,
        this allows to verify filenames like "1.2.86.1.dcm", ..., "1.2.86.12.dcm".
        """
        if filenames is None:
            filenames = [os.path.basename(dcm) for dcm in self.dicom_slice_paths]

        # remove common prefixes
        common_prefix = os.path.commonprefix(filenames)
        if common_prefix:
            filenames = [fn.replace(common_prefix, "") for fn in filenames]
        common_postfix = os.path.commonprefix([fn[::-1] for fn in filenames])[::-1]
        if common_postfix:
            filenames = [fn.replace(common_postfix, "") for fn in filenames]

        # extract numbers from filenames
        filename_digits = [(''.join(c for c in str(fn) if c.isdigit())) for fn in filenames]
        filename_digits = [int(d) for d in filename_digits if d]
        if len(filename_digits) < 2:
            # either no numbers in the filenames, or only one file
            return True

        missing_slices = False
        for num in range(min(filename_digits), max(filename_digits) + 1):
            if num not in filename_digits:
                missing_slices = True
                break
        if missing_slices:
            raise MissingDICOMFilesError(self.path)
        return True

    def __repr__(self) -> str:
        return f'DICOMImageReader({self.path})'
