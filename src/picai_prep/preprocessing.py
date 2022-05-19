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


import SimpleITK as sitk
import numpy as np
from numpy.testing import assert_allclose
from dataclasses import dataclass
from scipy import ndimage

from typing import List, Callable, Optional, Union, Any, Iterable
try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass


@dataclass
class PreprocessingSettings():
    """
    Preprocessing settings
    - matrix_size: number of voxels output volume (z, y, x)
    - spacing: output voxel spacing in mm (z, y, x)
    - physical_size: size in mm/voxel of the target volume (z, y, x)
    - align_segmentation: whether to align the scans using the centroid of the provided segmentation
    """
    matrix_size: Optional[Iterable[int]] = None
    spacing: Optional[Iterable[float]] = None
    physical_size: Optional[Iterable[float]] = None
    align_segmentation: Optional[sitk.Image] = None

    def __post_init__(self):
        if self.physical_size is None and self.spacing is not None and self.matrix_size is not None:
            # calculate physical size
            self.physical_size = [
                voxel_spacing * num_voxels
                for voxel_spacing, num_voxels in zip(
                    self.spacing,
                    self.matrix_size
                )
            ]

        if self.spacing is None and self.physical_size is not None and self.matrix_size is not None:
            # calculate spacing
            self.spacing = [
                size / num_voxels
                for size, num_voxels in zip(
                    self.physical_size,
                    self.matrix_size
                )
            ]

        if self.align_segmentation is not None:
            raise NotImplementedError("Alignment of scans based on segmentation is not implemented yet.")


def resample_img(
    image: sitk.Image,
    out_spacing: Iterable[float] = (2.0, 2.0, 2.0),
    out_size: Optional[Iterable[int]] = None,
    is_label: bool = False,
    pad_value: Optional[Union[float, int]] = 0.,
) -> sitk.Image:
    """
    Resample images to target resolution spacing
    Ref: SimpleITK
    """
    # get original spacing and size
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # convert our z, y, x convention to SimpleITK's convention
    out_spacing = list(out_spacing)[::-1]

    if out_size is None:
        # calculate output size in voxels
        out_size = [
            int(np.round(
                size * (spacing_in / spacing_out)
            ))
            for size, spacing_in, spacing_out in zip(original_size, original_spacing, out_spacing)
        ]

    # determine pad value
    if pad_value is None:
        pad_value = image.GetPixelIDValue()

    # set up resampler
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(list(out_spacing))
    resample.SetSize(out_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    # perform resampling
    image = resample.Execute(image)

    return image


def crop_or_pad(
    image: "Union[sitk.Image, npt.NDArray[Any]]",
    size: Iterable[int] = (64, 64, 64)
) -> "Union[sitk.Image, npt.NDArray[Any]]":
    """
    Resize image by cropping and/or padding
    """
    # input conversion and verification
    if isinstance(image, sitk.Image):
        shape = image.GetSize()
        size = list(size)[::-1]
    else:
        assert isinstance(image, (np.ndarray, np.generic))
        shape = image.shape
        size = list(size)
    rank = len(size)
    assert rank <= len(shape) <= rank + 1, \
        f"Example size doesn't fit image size. Got shape={shape}, output size={size}"

    # set identity operations for cropping and padding
    padding = [[0, 0] for _ in range(rank)]
    slicer = [slice(None) for _ in range(rank)]

    # for each dimension, determine process (cropping or padding)
    for i in range(rank):
        if shape[i] < size[i]:
            padding[i][0] = (size[i] - shape[i]) // 2
            padding[i][1] = size[i] - shape[i] - padding[i][0]
        else:
            # create slicer object to crop image
            idx_start = int(np.floor((shape[i] - size[i]) / 2.))
            idx_end = idx_start + size[i]
            slicer[i] = slice(idx_start, idx_end)

    # crop and/or pad image
    if isinstance(image, sitk.Image):
        pad_filter = sitk.ConstantPadImageFilter()
        pad_filter.SetPadLowerBound([pad[0] for pad in padding])
        pad_filter.SetPadUpperBound([pad[1] for pad in padding])
        return pad_filter.Execute(image[tuple(slicer)])
    else:
        return np.pad(image[tuple(slicer)], padding)


@dataclass
class Sample:
    scans: List[sitk.Image]
    lbl: Optional[sitk.Image]
    name: Optional[str]
    settings: PreprocessingSettings
    lbl_preprocess_func: Optional[Callable[[sitk.Image], sitk.Image]] = None
    lbl_postprocess_func: Optional[Callable[[sitk.Image], sitk.Image]] = None
    scan_preprocess_func: Optional[Callable[[sitk.Image], sitk.Image]] = None
    scan_postprocess_func: Optional[Callable[[sitk.Image], sitk.Image]] = None
    num_gt_lesions: Optional[int] = None

    def __post_init__(self):
        if self.lbl is not None:
            # keep track of connected components
            lbl = sitk.GetArrayFromImage(self.lbl)
            _, num_gt_lesions = ndimage.label(lbl, structure=np.ones((3, 3, 3)))
            self.num_gt_lesions = num_gt_lesions

    def resample_to_first_scan(self):
        """Resample scans and label to the first scan"""
        # set up resampler to resolution, field of view, etc. of first scan
        resampler = sitk.ResampleImageFilter()  # default linear
        resampler.SetReferenceImage(self.scans[0])
        resampler.SetInterpolator(sitk.sitkBSpline)

        # resample other images
        self.scans[1:] = [resampler.Execute(scan) for scan in self.scans[1:]]

        # resample annotation
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        if self.lbl is not None:
            self.lbl = resampler.Execute(self.lbl)

    def resample_spacing(self, spacing: Optional[Iterable[float]] = None):
        """Resample scans and label to the target spacing"""
        if spacing is None:
            assert self.settings.spacing is not None
            spacing = self.settings.spacing

        # resample scans to target resolution
        self.scans = [
            resample_img(scan, out_spacing=spacing, is_label=False)
            for scan in self.scans
        ]

        # resample annotation to target resolution
        if self.lbl is not None:
            self.lbl = resample_img(self.lbl, out_spacing=spacing, is_label=True)

    def centre_crop(self):
        """Centre crop scans and label"""
        self.scans = [
            crop_or_pad(scan, size=self.settings.matrix_size)
            for scan in self.scans
        ]

        if self.lbl is not None:
            self.lbl = crop_or_pad(self.lbl, size=self.settings.matrix_size)

    def align_physical_metadata(self, check_almost_equal=True):
        """Align the origin and direction of each scan, and label"""
        case_origin, case_direction, case_spacing = None, None, None
        for img in self.scans:
            # copy metadata of first scan (nnUNet and nnDetection require this to match exactly)
            if case_origin is None:
                case_origin = img.GetOrigin()
                case_direction = img.GetDirection()
                case_spacing = img.GetSpacing()
            else:
                if check_almost_equal:
                    # check if current scan's metadata is almost equal to the first scan
                    assert_allclose(img.GetOrigin(), case_origin)
                    assert_allclose(img.GetDirection(), case_direction)
                    assert_allclose(img.GetSpacing(), case_spacing)

                # copy over first scan's metadata to current scan
                img.SetOrigin(case_origin)
                img.SetDirection(case_direction)
                img.SetSpacing(case_spacing)

        if self.lbl is not None:
            assert case_origin is not None and case_direction is not None and case_spacing is not None
            self.lbl.SetOrigin(case_origin)
            self.lbl.SetDirection(case_direction)
            self.lbl.SetSpacing(case_spacing)

    def preprocess(self):
        """Perform all preprocessing steps"""
        # user-defined preprocessing steps
        if self.lbl is not None and self.lbl_preprocess_func:
            # apply label transformation
            self.lbl = self.lbl_preprocess_func(self.lbl)
        if self.scan_preprocess_func:
            # apply scan transformation
            self.scans = [self.scan_preprocess_func(scan) for scan in self.scans]

        if self.settings.spacing is not None:
            # resample scans and label to specified spacing
            self.resample_spacing()

        if self.settings.matrix_size is not None:
            # perform centre crop
            self.centre_crop()

        # resample scans and label to first scan's spacing, field-of-view, etc.
        self.resample_to_first_scan()

        # copy physical metadata to align subvoxel differences between sequences
        self.align_physical_metadata()

        if self.lbl is not None:
            # check connected components of annotation
            lbl = sitk.GetArrayFromImage(self.lbl)
            _, num_gt_lesions = ndimage.label(lbl, structure=np.ones((3, 3, 3)))
            assert self.num_gt_lesions == num_gt_lesions, \
                f"Label has changed due to resampling/other errors for {self.name}! " \
                + f"Have {self.num_gt_lesions} -> {num_gt_lesions} isolated ground truth lesions"

        # user-defined postprocessing steps
        if self.lbl is not None and self.lbl_postprocess_func:
            # apply label transformation
            self.lbl = self.lbl_postprocess_func(self.lbl)
        if self.scan_postprocess_func:
            # apply scan transformation
            self.scans = [self.scan_postprocess_func(scan) for scan in self.scans]


def translate_pred_to_reference_scan(
    pred: "npt.NDArray[Any]",
    reference_scan: sitk.Image,
    out_spacing: Iterable[float] = (0.5, 0.5, 3.6),
    is_label: bool = False
) -> sitk.Image:
    """
    Translate prediction back to physical space of input T2 scan
    This function performs the reverse operation of the preprocess_study function
    - pred: softmax / binary prediction
    - reference_scan: SimpleITK image to which the prediction should be resampled and resized
    - out_spacing: spacing to which the reference scan is resampled during preprocessing
    """
    reference_scan_resampled = resample_img(reference_scan, out_spacing=out_spacing, is_label=False, pad_value=0)

    # pad softmax prediction to physical size of resampled reference scan (with inverted order of image sizes)
    pred = crop_or_pad(pred, size=list(reference_scan_resampled.GetSize())[::-1])
    pred_itk = sitk.GetImageFromArray(pred)

    # set the physical properties of the predictions
    pred_itk.CopyInformation(reference_scan_resampled)

    # resample predictions to spacing of original reference scan
    pred_itk_resampled = resample_img(pred_itk, out_spacing=reference_scan.GetSpacing(), is_label=is_label, pad_value=0)
    return pred_itk_resampled
