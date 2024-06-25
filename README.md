# Preprocessing Utilities for 3D Medical Image Archives
![Tests](https://github.com/DIAGNijmegen/picai_prep/actions/workflows/tests.yml/badge.svg)

This repository contains standardized functions to process 3D medical images and image archives —with its processing strategy being geared towards clinically significant prostate cancer (csPCa) detection in MRI. It is used for the official preprocessing pipeline of the [PI-CAI challenge](https://pi-cai.grand-challenge.org/).

## Supported Conversions
- [`DICOM Archive` → `MHA Archive`][dcm2mha]
- [`MHA Archive` → `nnU-Net Raw Data Archive`][mha2nnunet]
- [`nnU-Net Raw Data Archive` → `nnDetection Raw Data Archive`][nnunet2nndet]

Note: the [`MHA Archive` → `nnU-Net Raw Data Archive`][mha2nnunet] conversion includes resampling sequences to a shared voxel spacing (per sample). Optionally, this step can resample all samples to a uniform voxel spacing and/or take a centre crop.

## Installation
`picai_prep` is pip-installable:

`pip install picai_prep`

## Usage
Our preprocessing pipeline consists of four independent stages: [`DICOM Archive`][dicom-archive] → [`MHA Archive`][mha-archive] → [`nnU-Net Raw Data Archive`][nnunet-archive] → [`nnDetection Raw Data Archive`][nndetection-archive]. All three conversion steps between these four stages can be performed independently. See below for documentation on each step.


### DICOM Archive → MHA Archive
Conversion from [`DICOM Archive`][dicom-archive] → [`MHA Archive`][mha-archive] is controlled through a configuration file, which lists all DICOM sequences. This configuration file specifies how different sequences should be selected from the available DICOM sequences. An excerpt of the format is given below:

```json
"mappings": {
    "t2w": {
        "SeriesDescription": ["t2_tse_tra"]
    },
},
"options": {
	"num_threads": 4,
	"verify_dicom_filenames": True,
	"allow_duplicates": False
},
"archive": [
    {
        "patient_id": "ProstateX-0000",
        "study_id": "07-07-2011-NA-MR prostaat kanker detectie WDSmc MCAPRODETW-05711",
        "path": "ProstateX-0000/07-07-2011-NA-MR prostaat kanker detectie WDSmc MCAPRODETW-05711/3.000000-t2tsesag-87368"
    },
]
```

This excerpt includes all available options:

```
num_threads: int, default: 4
	number of threads to use for multiprocessing
verify_dicom_filenames: bool, default: True
	explicitly verify dicom filenames as a sanity check
allow_duplicates: bool, default: False
	when multiple series apply to a mapping, convert all
```

Full configuration file for this excerpt, can be found [here](tests/output-expected/dcm2mha_settings.json). It can also be generated, as follows:

```python
from picai_prep.examples.dcm2mha.sample_archive import generate_dcm2mha_settings

generate_dcm2mha_settings(
    archive_dir="/path/to/picai_public_images/",
    output_path="/path/to/picai_public_images/dcm2mha_settings.json"
)
```

Using this configuration file, the [`DICOM Archive`][dicom-archive] → [`MHA Archive`][mha-archive] conversion can be performed using Python:

```python
from picai_prep import Dicom2MHAConverter

archive = Dicom2MHAConverter(
    input_dir="/input/path/to/dicom/archive",
    output_dir="/output/path/to/mha/archive",
    dcm2mha_settings="/path/to/dcm2mha_settings.json",
)
archive.convert()
```

Or from the command line:

```bash
python -m picai_prep dcm2mha --input /input/path/to/dicom/archive --output /output/path/to/mha/archive --json /path/to/dcm2mha_settings.json
```

For more examples of `DICOM Archive` structures, see [examples/dcm2mha/](src/picai_prep/examples/dcm2mha/).

#
### MHA Archive → nnU-Net Raw Data Archive
Conversion from the [`MHA Archive`][mha-archive] format to the [`nnU-Net Raw Data Archive`][nnunet-archive] format is controlled through a configuration file, which lists all input sequences (and optionally, annotations). This configuration file specifies which sequences should be selected from the available (MHA) sequences. An excerpt of the format is given below:

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
        "study_id": "07-07-2011",
        "scan_paths": [
            "ProstateX-0000/ProstateX-0000_07-07-2011_t2w.mha",
            "ProstateX-0000/ProstateX-0000_07-07-2011_adc.mha",
            "ProstateX-0000/ProstateX-0000_07-07-20111_hbv.mha"
        ],
        "annotation_path": "ProstateX-0000_07-07-2011.nii.gz"
    },
]
```

Full configuration file for this except, can be found [here](tests/output-expected/mha2nnunet_settings.json). It can also be generated, as follows:

```bash
python -m picai_prep mha2nnunet_settings --structure picai_archive --input /input/images/ --annotations /input/labels/csPCa_lesion_delineations/human_expert/resampled --json /workdir/mha2nnunet_settings.json
```

Or from Python:

```python
from picai_prep.examples.mha2nnunet.picai_archive import generate_mha2nnunet_settings

generate_mha2nnunet_settings(
    archive_dir="/input/images/",
    annotations_dir="/input/labels/csPCa_lesion_delineations/human_expert/resampled
    output_path="/workdir/mha2nnunet_settings.json",
)
```

The `--annotations` (command line) or `annotations_dir` (Python) parameter will check if the annotation is present in the specified folder. If not, the item will be skipped.

Using this configuration file, the `MHA Archive` → `nnU-Net Raw Data Archive` conversion can be performed using Python:

```python
from picai_prep import MHA2nnUNetConverter

archive = MHA2nnUNetConverter(
    scans_dir="/input/path/to/mha/archive",
    annotations_dir="/input/path/to/annotations",  # defaults to input_path
    output_dir="/output/path/to/nnUNet_raw_data",
    mha2nnunet_settings="/path/to/mha2nnunet_settings.json",
)
archive.convert()
archive.create_dataset_json()
```

Or from the command line:

```bash
python -m picai_prep mha2nnunet --input /input/path/to/mha/archive --annotations /input/path/to/annotations --output /output/path/to/nnUNet_raw_data --json /path/to/mha2nnunet_settings.json
```

Or using a Docker container:

```bash
docker run -v /path/to/picai_data:/input \
           -v /path/to/nnUNet_raw_data:/output/ \
           picai_nnunet python -m picai_prep mha2nnunet \
           --input /input/images \
           --annotations /input/labels/csPCa_lesion_delineations/human_expert/resampled \
           --output /output/nnUNet_raw_data \
           --json /input/mha2nnunet_settings.json
```
For more examples of `MHA Archive` structures, see [examples/mha2nnunet/](src/picai_prep/examples/mha2nnunet/).

# 

### nnU-Net Raw Data Archive → nnDetection Raw Data Archive
For certain applications, the nnU-Net and nnDetection raw data archive formats can be converted to each other. Conversion from the nnDetection to nnU-Net structure can always be performed (see [nnDetection's documentation](https://github.com/MIC-DKFZ/nnDetection#nnu-net-for-detection)). However, conversion from nnU-Net to nnDetection structure requires object instances to be non-connected and non-overlapping, such that they can be correctly identified as separate, individual objects. If this assumption holds true, [`nnU-Net Raw Data Archive`][nnunet-archive] → [`nnDetection Raw Data Archive`][nndetection-archive] conversion can be performed using Python:

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

#

### What is a 'DICOM Archive'?
With a DICOM archive we mean a dataset that comprises the scans as DICOM (.dcm) files, such as the [ProstateX dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=23691656). Typically, such an archive is structured in the following way:

```
/path/to/archive/
├── [patient UID]/
    ├── [study UID]/
        ├── [series UID]/
            ├── slice-1.dcm
            ...
            ├── slice-n.dcm
```

In a DICOM archive multiple sequences (such as axial T2-weighted scans) can exist, and each patient can have multiple studies. A single study can even have multiple instances of the same sequence, for example a repeated transversal T2-weighted scan when the first scan experienced motion blur artefacts.

#

### What is an 'MHA Archive'?
With an MHA archive we mean a dataset that comprises the scans as MHA (.mha) files, such as the [PI-CAI dataset](https://zenodo.org/record/6517398#.YnU5uhNBwUE). In case of the PI-CAI Challenge: Public Training and Development Dataset, the archive is structured in the following way (after extracting the zips):

```
/path/to/archive/
├── [patient UID]/
    ├── [patient UID]_[study UID]_[modality].mha
    ...
```

For the PI-CAI dataset, the available modalities are `t2w` (axial T2-weighted scan), `adc` (apparent diffusion coefficient map), `hbv` (calculated high b-value scan), `sag` (sagittal T2-weighted scan) and `cor` (coronal T2-weighted scan).

# 
## Reference
If you are using this codebase or some part of it, please cite the following article:

__[Saha A, Bosma JS, Twilt JJ, et al. Artificial intelligence and radiologists in prostate cancer detection on MRI (PI-CAI): an international, paired, non-inferiority, confirmatory study. Lancet Oncol 2024; 25: 879–887](https://www.thelancet.com/journals/lanonc/article/PIIS1470-2045(24)00220-1/fulltext)__  

**BibTeX:**
```
@ARTICLE{SahaBosmaTwilt2024,
  title = {Artificial intelligence and radiologists in prostate cancer detection on MRI (PI-CAI): an international, paired, non-inferiority, confirmatory study},
  journal = {The Lancet Oncology},
  year = {2024},
  issn = {1470-2045},
  volume={25},
  number={7},
  pages={879--887},
  doi = {https://doi.org/10.1016/S1470-2045(24)00220-1},
  author = {Anindo Saha and Joeran S Bosma and Jasper J Twilt and Bram {van Ginneken} and Anders Bjartell and Anwar R Padhani and David Bonekamp and Geert Villeirs and Georg Salomon and Gianluca Giannarini and Jayashree Kalpathy-Cramer and Jelle Barentsz and Klaus H Maier-Hein and Mirabela Rusu and Olivier Rouvière and Roderick {van den Bergh} and Valeria Panebianco and Veeru Kasivisvanathan and Nancy A Obuchowski and Derya Yakar and Mattijs Elschot and Jeroen Veltman and Jurgen J Fütterer and Constant R. Noordman and Ivan Slootweg and Christian Roest and Stefan J. Fransen and Mohammed R.S. Sunoqrot and Tone F. Bathen and Dennis Rouw and Jos Immerzeel and Jeroen Geerdink and Chris {van Run} and Miriam Groeneveld and James Meakin and Ahmet Karagöz and Alexandre Bône and Alexandre Routier and Arnaud Marcoux and Clément Abi-Nader and Cynthia Xinran Li and Dagan Feng and Deniz Alis and Ercan Karaarslan and Euijoon Ahn and François Nicolas and Geoffrey A. Sonn and Indrani Bhattacharya and Jinman Kim and Jun Shi and Hassan Jahanandish and Hong An and Hongyu Kan and Ilkay Oksuz and Liang Qiao and Marc-Michel Rohé and Mert Yergin and Mohamed Khadra and Mustafa E. Şeker and Mustafa S. Kartal and Noëlie Debs and Richard E. Fan and Sara Saunders and Simon J.C. Soerensen and Stefania Moroianu and Sulaiman Vesal and Yuan Yuan and Afsoun Malakoti-Fard and Agnė Mačiūnien and Akira Kawashima and Ana M.M. de M.G. {de Sousa Machadov} and Ana Sofia L. Moreira and Andrea Ponsiglione and Annelies Rappaport and Arnaldo Stanzione and Arturas Ciuvasovas and Baris Turkbey and Bart {de Keyzer} and Bodil G. Pedersen and Bram Eijlers and Christine Chen and Ciabattoni Riccardo and Deniz Alis and Ewout F.W. {Courrech Staal} and Fredrik Jäderling and Fredrik Langkilde and Giacomo Aringhieri and Giorgio Brembilla and Hannah Son and Hans Vanderlelij and Henricus P.J. Raat and Ingrida Pikūnienė and Iva Macova and Ivo Schoots and Iztok Caglic and Jeries P. Zawaideh and Jonas Wallström and Leonardo K. Bittencourt and Misbah Khurram and Moon H. Choi and Naoki Takahashi and Nelly Tan and Paolo N. Franco and Patricia A. Gutierrez and Per Erik Thimansson and Pieter Hanus and Philippe Puech and Philipp R. Rau and Pieter {de Visschere} and Ramette Guillaume and Renato Cuocolo and Ricardo O. Falcão and Rogier S.A. {van Stiphout} and Rossano Girometti and Ruta Briediene and Rūta Grigienė and Samuel Gitau and Samuel Withey and Sangeet Ghai and Tobias Penzkofer and Tristan Barrett and Varaha S. Tammisetti and Vibeke B. Løgager and Vladimír Černý and Wulphert Venderink and Yan M. Law and Young J. Lee and Maarten {de Rooij} and Henkjan Huisman},
}
```

## Managed By
Diagnostic Image Analysis Group,
Radboud University Medical Center,
Nijmegen, The Netherlands

## Contact Information
- Joeran Bosma: Joeran.Bosma@radboudumc.nl
- Stan Noordman: Stan.Noordman@radboudumc.nl
- Anindo Saha: Anindya.Shaha@radboudumc.nl
- Henkjan Huisman: Henkjan.Huisman@radboudumc.nl

[dicom-archive]: #what-is-a-dicom-archive
[mha-archive]: #what-is-an-mha-archive
[nnunet-archive]: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md
[nndetection-archive]: https://github.com/MIC-DKFZ/nnDetection/#adding-new-data-sets
[dcm2mha]: #dicom-archive--mha-archive
[mha2nnunet]: #mha-archive--nnu-net-raw-data-archive
[nnunet2nndet]: #nnu-net-raw-data-archive--nndetection-raw-data-archive
