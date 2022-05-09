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
from dataclasses import dataclass
from scipy import ndimage

from typing import List, Tuple, Callable, Optional, Union, Any, Iterable, cast
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
    - align_physical_space: whether to align sequences to eachother, based on metadata
    - crop_to_first_physical_centre: whether to crop to physical centre of first sequence,
        or to the new centre after aligning sequences
    - align_segmentation: whether to align the scans using the centroid of the provided segmentation
    """
    matrix_size: Iterable[int] = (20, 160, 160)
    spacing: Optional[Iterable[float]] = None
    physical_size: Optional[Iterable[float]] = None
    align_physical_space: bool = False
    crop_to_first_physical_centre: bool = False
    align_segmentation: Optional[sitk.Image] = None

    def __post_init__(self):
        if self.physical_size is None:
            assert self.spacing, "Need either physical_size or spacing"
            # calculate physical size
            self.physical_size = [
                voxel_spacing * num_voxels
                for voxel_spacing, num_voxels in zip(
                    self.spacing,
                    self.matrix_size
                )
            ]

        if self.spacing is None:
            assert self.physical_size, "Need either physical_size or spacing"
            # calculate spacing
            self.spacing = [
                size / num_voxels
                for size, num_voxels in zip(
                    self.physical_size,
                    self.matrix_size
                )
            ]

    @property
    def _spacing(self) -> Iterable[float]:
        return cast(Iterable[float], self.spacing)

    @property
    def _physical_size(self) -> Iterable[float]:
        return cast(Iterable[float], self.physical_size)


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


def get_overlap_start_indices(img_main: sitk.Image, img_secondary: sitk.Image):
    # convert start index from main image to secondary image
    point_secondary = img_secondary.TransformIndexToPhysicalPoint((0, 0, 0))
    index_main = img_main.TransformPhysicalPointToContinuousIndex(point_secondary)

    # clip index
    index_main = np.clip(index_main, a_min=0, a_max=None)

    # convert main index back to secondary image
    point_main = img_main.TransformContinuousIndexToPhysicalPoint(index_main)
    index_secondary = img_secondary.TransformPhysicalPointToContinuousIndex(point_main)

    # round secondary index up (round to 5 decimals for e.g. 18.999999999999996)
    index_secondary = np.ceil(np.round(index_secondary, decimals=5))

    # convert secondary index once again to main image
    point_secondary = img_secondary.TransformContinuousIndexToPhysicalPoint(index_secondary)
    index_main = img_main.TransformPhysicalPointToIndex(point_secondary)

    # convert and return result
    return np.array(index_secondary).astype(int), np.array(index_main).astype(int)


def get_overlap_end_indices(img_main: sitk.Image, img_secondary: sitk.Image):
    # convert end index from secondary image to primary image
    point_secondary = img_secondary.TransformIndexToPhysicalPoint(img_secondary.GetSize())
    index_main = img_main.TransformPhysicalPointToContinuousIndex(point_secondary)

    # clip index
    index_main = [min(sz, i) for (i, sz) in zip(index_main, img_main.GetSize())]

    # convert primary index back to secondary image
    point_main = img_main.TransformContinuousIndexToPhysicalPoint(index_main)
    index_secondary = img_secondary.TransformPhysicalPointToContinuousIndex(point_main)

    # round secondary index down (round to 5 decimals for e.g. 18.999999999999996)
    index_secondary = np.floor(np.round(index_secondary, decimals=5))

    # convert secondary index once again to primary image
    point_secondary = img_secondary.TransformContinuousIndexToPhysicalPoint(index_secondary)
    index_main = img_main.TransformPhysicalPointToIndex(point_secondary)

    # convert and return result
    return np.array(index_secondary).astype(int), np.array(index_main).astype(int)


def crop_to_common_physical_space(
    img_main: sitk.Image,
    img_sec: sitk.Image
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Crop SimpleITK images to the largest shared physical volume
    """
    # determine crop indices
    idx_start_sec, idx_start_main = get_overlap_start_indices(img_main, img_sec)
    idx_end_sec, idx_end_main = get_overlap_end_indices(img_main, img_sec)

    # check extracted indices
    assert ((idx_end_sec - idx_start_sec) > np.array(img_sec.GetSize()) / 2).all(), \
        "Found unrealistically little overlap when aligning scans, aborting."
    assert ((idx_end_main - idx_start_main) > np.array(img_main.GetSize()) / 2).all(), \
        "Found unrealistically little overlap when aligning scans, aborting."

    # apply crop
    slices = [slice(idx_start, idx_end) for (idx_start, idx_end) in zip(idx_start_main, idx_end_main)]
    img_main = img_main[slices]

    slices = [slice(idx_start, idx_end) for (idx_start, idx_end) in zip(idx_start_sec, idx_end_sec)]
    img_sec = img_sec[slices]

    return img_main, img_sec


def get_physical_centre(image: sitk.Image):
    return image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize()) / 2.0)


@dataclass
class Sample:
    scans: List[sitk.Image]
    lbl: Optional[sitk.Image]
    name: Optional[str]
    settings: PreprocessingSettings
    lbl_transformation: Optional[Callable[[sitk.Image], sitk.Image]] = None
    num_gt_lesions: Optional[int] = None

    def __post_init__(self):
        # determine main centre
        self.main_centre = get_physical_centre(self.scans[0])

        if self.lbl is not None:
            # keep track of connected components
            lbl = sitk.GetArrayFromImage(self.lbl)
            _, num_gt_lesions = ndimage.label(lbl, structure=np.ones((3, 3, 3)))
            self.num_gt_lesions = num_gt_lesions

    def crop_to_common_physical_space(self):
        """
        Align physical centre of the first scan (e.g., T2W) with subsequent scans (e.g., ADC, high b-value)
        """
        main_centre = get_physical_centre(self.scans[0])

        should_align_scans = False
        for scan in self.scans[1:]:
            secondary_centre = get_physical_centre(scan)

            # calculate distance from center of first scan (e.g., T2W) to center of secondary scan (e.g., ADC, high b-value)
            distance = np.sqrt(np.sum((np.array(main_centre) - np.array(secondary_centre))**2))

            # if difference in center coordinates is more than 2mm, align the scans
            if distance > 2:
                print(f"Aligning scans with distance of {distance:.1f} mm between centers for {self.name}.")
                should_align_scans = True

        if should_align_scans:
            for i, main_scan in enumerate(self.scans):
                for j, secondary_scan in enumerate(self.scans):
                    if i == j:
                        continue

                    # align scans
                    img_main, img_sec = crop_to_common_physical_space(main_scan, secondary_scan)
                    self.scans[i] = img_main
                    self.scans[j] = img_sec

    def resample(self):
        """Resample scans and label"""
        self.scans = [
            resample_img(scan, out_spacing=self.settings._spacing, is_label=False)
            for scan in self.scans
        ]

        if self.lbl is not None:
            self.lbl = resample_img(self.lbl, out_spacing=self.settings._spacing, is_label=True)

    def centre_crop(self):
        """Centre crop scans and label"""
        self.scans = [
            crop_or_pad(scan, size=self.settings.matrix_size)
            for scan in self.scans
        ]

        if self.lbl is not None:
            self.lbl = crop_or_pad(self.lbl, size=self.settings.matrix_size)

    def preprocess(self):
        """Perform all preprocessing steps"""
        if self.lbl is not None and self.lbl_transformation:
            # apply label transformation
            self.lbl = self.lbl_transformation(self.lbl)

        if self.settings.align_physical_space:
            # align sequences based on metadata
            self.crop_to_common_physical_space()

        # resample scans and label
        self.resample()

        # perform centre crop
        self.centre_crop()

        if self.lbl is not None:
            # check connected components of annotation
            lbl = sitk.GetArrayFromImage(self.lbl)
            _, num_gt_lesions = ndimage.label(lbl, structure=np.ones((3, 3, 3)))
            assert self.num_gt_lesions == num_gt_lesions, \
                f"Label has changed due to resampling/other errors for {self.name}! " \
                + f"Have {self.num_gt_lesions} -> {num_gt_lesions} isolated ground truth lesions"


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
