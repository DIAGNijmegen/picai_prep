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


import argparse
import sys

from picai_prep import Dicom2MHAConverter, MHA2nnUNetConverter, nnunet2nndet
from picai_prep.examples import dcm2mha, mha2nnunet


def run_dcm2mha(args):
    """Wrapper for DICOM → MHA conversion"""
    archive = Dicom2MHAConverter(
        input_dir=args.input,
        output_dir=args.output,
        dcm2mha_settings=args.json,
    )
    archive.convert()


def generate_dcm2mha_settings(args):
    """Wrapper to generate dcm2mha settings"""
    func = {
        "sample_archive": dcm2mha.sample_archive.generate_dcm2mha_settings,
    }[args.structure]

    kwargs = {
        "archive_dir": args.input,
        "output_path": args.json,
    }

    # generate settings
    func(**kwargs)


def run_mha2nnunet(args):
    """Wrapper for MHA → nnUNet conversion"""
    annotations_path = args.annotations if args.annotations else args.input

    archive = MHA2nnUNetConverter(
        input_path=args.input,
        output_path=args.output,
        annotations_path=annotations_path,
        settings_path=args.json,
        out_dir_scans=args.out_dir_scans,
        out_dir_annot=args.out_dir_annot,
        silent=args.silent
    )
    archive.convert()


def generate_mha2nnunet_settings(args):
    """Wrapper to generate mha2nnunet settings"""
    func = {
        "picai_archive": mha2nnunet.picai_archive.generate_mha2nnunet_settings,
        "picai_archive_inference": mha2nnunet.picai_archive_inference.generate_mha2nnunet_settings,
        "sample_archive": mha2nnunet.sample_archive.generate_mha2nnunet_settings,
        "sample_archive_inference": mha2nnunet.sample_archive_inference.generate_mha2nnunet_settings
    }[args.structure]

    kwargs = {
        "archive_dir": args.input,
        "output_path": args.json,
    }

    if args.annotations is not None:
        kwargs["annotations_dir"] = args.annotations

    # generate settings
    func(**kwargs)


def run_nnunet2nndet(args):
    """Wrapper for nnUNet → nnDetection conversion"""
    nnunet2nndet(
        nnunet_raw_data_path=args.input,
        nndet_raw_data_path=args.output,
    )


# Set up command line arguments
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

# Argument parser for DICOM → MHA
dcm = subparsers.add_parser('dcm2mha')
dcm.add_argument("-j", "--json", type=str, required=True,
                 help="Path to JSON mappings file")
dcm.add_argument("-i", "--input", type=str, required=True,
                 help="Root directory for input, e.g. /path/to/archive/")
dcm.add_argument("-o", "--output", type=str, required=True,
                 help="Root directory for output")
dcm.add_argument("-s", "--silent", action='store_true', required=False,
                 help="Mute log messages")
dcm.set_defaults(func=run_dcm2mha)


# Argument parser for dcm2mha_settings
mha = subparsers.add_parser('dcm2mha_settings')
mha.add_argument("-s", "--structure", type=str, required=True,
                 help="Structure of DICOM Archive")
mha.add_argument("-i", "--input", type=str, required=True,
                 help="Path to DICOM Archive")
mha.add_argument("-j", "--json", type=str, required=True,
                 help="Path to store dcm2mha_settings.json")
mha.set_defaults(func=generate_dcm2mha_settings)


# Argument parser for MHA → nnUNet
mha = subparsers.add_parser('mha2nnunet')
mha.add_argument("-j", "--json", type=str, required=True,
                 help="Path to JSON mappings file")
mha.add_argument("-i", "--input", type=str, required=True,
                 help="Path to PICAI .mha data")
mha.add_argument("-a", "--annotations", type=str, required=False,
                 help="Path to PICAI annotations (defaults to --input)")
mha.add_argument("-o", "--output", type=str, required=True,
                 help="Root directory for output")
mha.add_argument("--out_dir_scans", type=str, default="imagesTr",
                 help="Folder for scans (relative to root directory)")
mha.add_argument("--out_dir_annot", type=str, default="labelsTr",
                 help="Folder for annotations (relative to root directory)")
mha.add_argument("-s", "--silent", action='store_true', required=False,
                 help="Mute log messages")
mha.set_defaults(func=run_mha2nnunet)


# Argument parser for mha2nnunet_settings
mha = subparsers.add_parser('mha2nnunet_settings')
mha.add_argument("-s", "--structure", type=str, required=True,
                 help="Structure of MHA Archive")
mha.add_argument("-i", "--input", type=str, required=True,
                 help="Path to MHA Archive")
mha.add_argument("-a", "--annotations", type=str, required=False,
                 help="Path to PICAI annotations (defaults to --input)")
mha.add_argument("-j", "--json", type=str, required=True,
                 help="Path to store mha2nnunet_settings.json")
mha.set_defaults(func=generate_mha2nnunet_settings)


# Argument parser for nnUNet → nnDetection
nnunet2nndet_parser = subparsers.add_parser('nnunet2nndet')
nnunet2nndet_parser.add_argument("-i", "--input", type=str, required=True,
                                 help="Path to nnUNet raw data folder")
nnunet2nndet_parser.add_argument("-o", "--output", type=str, required=True,
                                 help="Path to nnDet raw data folder")
nnunet2nndet_parser.set_defaults(func=run_nnunet2nndet)


if __name__ == '__main__':
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)
