![Tests](https://github.com/DIAGNijmegen/picai_prep/actions/workflows/tests.yml/badge.svg)

# Preprocessing scripts for 3D medical images

This repository contains code to process 3D medical images, geared towards prostate cancer detection in MRI. This repository contains the official preprocessing pipeline of the [Prostate Imaging: Cancer AI (PI-CAI)](https://pi-cai.grand-challenge.org/) Grand Challenge.

Supported conversions:
- DICOM archive → MHA archive
- MHA archive → nnUNet raw data archive
- nnUNet raw data archive → nnDetection raw data archive

The MHA → nnUNet conversion includes resampling sequences to a shared voxel spacing, taking a centre crop, and optionally aligning sequences based on the scans's metadata. [This function](src/picai_prep/preprocessing.py#L462) can also be used independently.

## Installation
`picai_prep` is pip-installable:

`pip install git+https://github.com/DIAGNijmegen/picai_prep`

## Usage
The preprocessing pipeline consists of four independent stages: DICOM → MHA → nnUNet → nnDetection. These three conversion steps can be performed independently. See below for documentation on each step.

### DICOM → MHA
The conversion to MHA is controlled through a configuration file which lists all DICOM sequences. This configuration file specifies how different sequences should be selected from the available DICOM sequences. An excerpt of the format is given below:

```json
"mappings": {
    "t2w": {
        "SeriesDescription": ["t2_tse_tra"]
    },
},
"archive": [
    {
        "patient_id": "ProstateX-0000",
        "study_id": "07-07-2011-NA-MR prostaat kanker detectie WDSmc MCAPRODETW-05711",
        "path": "ProstateX-0000/07-07-2011-NA-MR prostaat kanker detectie WDSmc MCAPRODETW-05711/3.000000-t2tsesag-87368"
    },
]
```

The full configuration file can be found [here](tests/output-expected/dcm2mha_settings.json). This configuration file can be generated, as demonstrated [here](src/picai_prep/examples/dcm2mha/sample_archive.py).

Using this configuration file, the DICOM → MHA conversion can be performed from Python:

```python
from picai_prep import Dicom2MHAConverter

archive = Dicom2MHAConverter(
    input_path="/input/path/to/dicom/archive",
    output_path="/output/path/to/mha/archive",
    settings_path="/path/to/dcm2mha_settings.json",
)
archive.convert()
```

Or from the command line:

```bash
python -m picai_prep dcm2mha --input /input/path/to/dicom/archive --output /output/path/to/mha/archive --json /path/to/dcm2mha_settings.json
```

### MHA → nnUNet
The conversion to nnUNet is controlled through a configuration file which lists all input sequences (and optionally, annotations). This configuration file specifies which sequences should be selected from the available (MHA) sequences. An excerpt of the format is given below:

```json
"dataset_json": {
    "task": "Task100_test",
    ...
},
"preprocessing": {
    "matrix_size": [20, 160, 160],
    "spacing": [3.0, 0.5, 0.5]
},
"archive": [
    {
        "patient_id": "ProstateX-0000",
        "study_id": "07-07-2011-NA-MR prostaat kanker detectie WDSmc MCAPRODETW-05711",
        "scan_paths": [
            "ProstateX-0000/ProstateX-0000_07-07-2011-NA-MR prostaat kanker detectie WDSmc MCAPRODETW-05711_t2w.mha",
            "ProstateX-0000/ProstateX-0000_07-07-2011-NA-MR prostaat kanker detectie WDSmc MCAPRODETW-05711_adc.mha",
            "ProstateX-0000/ProstateX-0000_07-07-2011-NA-MR prostaat kanker detectie WDSmc MCAPRODETW-05711_hbv.mha"
        ],
        "annotation_path": "ProstateX-0000_07-07-2011-NA-MR prostaat kanker detectie WDSmc MCAPRODETW-05711.nii.gz"
    },
]
```

The full configuration file can be found [here](tests/output-expected/mha2nnunet_settings.json). This configuration file can be generated, as demonstrated [here](src/picai_prep/examples/mha2nnunet/sample_archive.py).

Using this configuration file, the MHA → nnUNet conversion can be performed from Python:

```python
from picai_prep import MHA2nnUNetConverter

archive = MHA2nnUNetConverter(
    input_path="/input/path/to/mha/archive",
    annotations_path="/input/path/to/annotations",  # defaults to input_path
    output_path="/output/path/to/nnUNet_raw_data",
    settings_path="/path/to/mha2nnunet_settings.json",
)
archive.convert()
```

Or from the command line:

```bash
python -m picai_prep mha2nnunet --input /input/path/to/mha/archive --annotations /input/path/to/annotations --output /output/path/to/nnUNet_raw_data --json /path/to/mha2nnunet_settings.json
```

### nnUNet → nnDetection
For specific applications, the nnUNet and nnDetection raw data formats can be converted to eachother. The conversion from nnDetection to nnUNet can always be performed, see [nnDetection's documentation](https://github.com/MIC-DKFZ/nnDetection#nnu-net-for-detection). The conversion from nnUNet to nnDetection requires instances to be non-touching, such that they can be correctly identified as individual objects. If this assumption holds, raw data archive can be converted from nnUNet to nnDetection using Python:

```python
from picai_prep import nnunet2nndet

nnunet2nndet(
    nnunet_raw_data_path="/input/path/to/nnUNet_raw_data/Task100_test",
    nndet_raw_data_path="/output/path/to/nnDet_raw_data/Task100_test",
)
```

Or from the command line:

```bash
python -m picai_prep nnunet2nndet --input /input/path/to/nnUNet_raw_data/Task100_test --output /output/path/to/nnDet_raw_data/Task100_test
```
