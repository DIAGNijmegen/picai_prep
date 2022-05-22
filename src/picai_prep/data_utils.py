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
import shutil
from pathlib import Path
from typing import Union

import SimpleITK as sitk

PathLike = Union[str, Path]


def atomic_image_write(
    image: sitk.Image,
    path: PathLike,
    backup_existing_file: bool = False,
    compress: bool = True,
    mkdir: bool = False
):
    """
    Safely write image to disk, by:
    1. Writing the image to a temporary file: path/to/.tmp.[filename]
    2. IF writing succeeded:
    2a. (optional) rename existing file to path/to/backup.[filename]
    2b. rename file to target name, which is an atomic operation
    This way, no partially written files can exist at the target path (except for temporary files)
    """
    path = Path(path)

    if mkdir:
        os.makedirs(path.parent, exist_ok=True)

    # save image to temporary file
    path_tmp = path.with_name(f".tmp.{path.name}")
    sitk.WriteImage(image, path_tmp.as_posix(), useCompression=compress)

    # backup existing file?
    if backup_existing_file and path.exists():
        dst_path_bak = path.with_name(f"backup.{path.name}")
        if dst_path_bak.exists():
            raise FileExistsError(f"Existing backup file found at {dst_path_bak}.")
        os.rename(path, dst_path_bak)

    # rename temporary file
    os.rename(path_tmp, path)


def atomic_file_copy(
    src_path: PathLike,
    dst_path: PathLike,
    backup_existing_file: bool = False,
    mkdir: bool = False
):
    """
    Safely copy file, by:
    1. Writing the file to a temporary file: path/to/.tmp.[filename]
    2. IF writing succeeded:
    2a. (optional) backup existing file to path/to/backup.[filename]
    2b. rename file to target name, which is an atomic operation
    This way, no partially written files can exist at the target path (except for temporary files)
    """
    dst_path = Path(dst_path)

    if mkdir:
        os.makedirs(dst_path.parent, exist_ok=True)

    # copy file to temporary location
    dst_path_tmp = dst_path.with_name(f".tmp.{dst_path.name}")
    try:
        shutil.copy(src_path, dst_path_tmp)
    except PermissionError:
        shutil.copyfile(src_path, dst_path_tmp)

    # backup existing file?
    if backup_existing_file and dst_path.exists():
        dst_path_bak = dst_path.with_name(f"backup.{dst_path.name}")
        if dst_path_bak.exists():
            raise FileExistsError(f"Existing backup file found at {dst_path_bak}.")
        os.rename(dst_path, dst_path_bak)

    # rename temporary file
    os.rename(dst_path_tmp, dst_path)
