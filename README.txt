## DICOM → NIfTI + Defacing Pipeline (mideface + ANTsPy)
====================================================

This project converts MRI DICOM series into compressed NIfTI (.nii.gz) using dcm2niix, then performs
defacing using FreeSurfer's "mideface" on a chosen T1 series. The resulting T1 face mask is rigidly
propagated to other structural sequences using ANTsPy registration, and those sequences are defaced
by zeroing voxels under the propagated mask.

Outputs are written per session, and a JSON report (metadata + conversion/skip/fail status) is saved
inside each session's output folder.


## What this pipeline does
-----------------------
For each session folder (DICOM input):
1) Discover DICOM series folders (walks the directory tree).
2) Pre-filter out series below MIN_SLICES.
3) Choose the "best" T1 candidate by scoring SeriesDescription/ProtocolName + slice count.
4) Convert all kept series using dcm2niix (compressed NIfTI).
5) Keep only usable NIfTIs after MIN_VOL_DIM checks.
6) Run mideface on selected T1 to produce:
   - Defaced T1 NIfTI
   - face mask (NIfTI)
7) For other sequences:
   - If the sequence matches defacing rules (logic_config), rigidly register to T1 and propagate mask.
   - Deface by applying the propagated mask.
   - Otherwise copy the original converted volume to final output.
8) Clean up temporary folders and keep only the final "defaced/" directory.


## Repository structure
-------------------------------
.
├── main.py
├── paths_config.py          # YOU EDIT ONLY session_dir + output_dir paths here
├── logic_config.py          # Defacing and T1-identification rules (usually no change)
├── requirements.txt
└── README.txt


## Key external dependencies (non-Python)
--------------------------------------
This project requires TWO external tools installed on your system:

(1) dcm2niix
    - Used to convert DICOM → NIfTI.

(2) FreeSurfer (for "mideface" and optionally "mri_convert")
    - Used to generate a defaced T1 and a face mask.

IMPORTANT:
Before you run anything, verify that FreeSurfer's mideface is installed and callable:

    mideface --help

If that command fails, fix your FreeSurfer installation / PATH setup before proceeding.

If mideface outputs .mgz files on your system, the script may need FreeSurfer's mri_convert.
Verify it too:

    mri_convert --help


Configuration (paths_config.py)
-------------------------------
You ONLY need to change the following in paths_config.py:

- SESSION_DIR:   path to your root input directory
- OUTPUT_DIR:    path where results will be written

Nothing else is required to be changed for a typical run.

Notes:
- The pipeline scans SESSION_DIR for session folders (immediate subdirectories).
- It also supports the case where SESSION_DIR itself contains DICOMs directly.
- Each session is processed into: OUTPUT_DIR/<session_name>/defaced/


Logic rules (logic_config.py)
-----------------------------
We have set up parameters in logic_config.py based on our local test cases:
- T1 candidate identification keywords and priorities
- Defacing inclusion/exclusion lists (structural_keywords, skip_keywords)
- Minimum matrix size rules

Depending on:
- your scanner naming conventions (SeriesDescription / ProtocolName),
- your data characteristics,
- and your machine / environment variables,
you may need to adjust these rules slightly for best performance.

If the wrong series gets picked as T1, the first thing to tune is:
- t1_identification.keywords
- t1_identification.priority_keywords


Installation
------------
1) Create and activate a Python environment (recommended):

    python -m venv .venv
    source .venv/bin/activate        (macOS/Linux)
    .venv\Scripts\activate           (Windows)

2) Install Python requirements:

    pip install -r requirements.txt

3) Install external tools:
   - Install dcm2niix and ensure it is on PATH (or set its path in paths_config.py if you use explicit paths).
   - Install FreeSurfer and ensure mideface is available:

        mideface --help

   - If needed, ensure mri_convert is available:

        mri_convert --help


Running the pipeline
--------------------
Basic run (using defaults from paths_config.py):

    python main.py

Override input/output on the command line:

    python main.py --session_dir /path/to/sessions --outdir /path/to/output


Input assumptions
-----------------
- SESSION_DIR can either:
  (A) contain session subfolders (recommended), or
  (B) be a single session folder containing DICOMs directly.

- The script detects DICOM files using pydicom header parsing.


Outputs
-------
For each session:
OUTPUT_DIR/<session_name>/defaced/

Contains:
- Defaced T1: <t1name>_defaced.nii.gz
- T1 face mask: <t1stem>_deface_mask.nii.gz
- For other sequences:
  - Defaced: <seqname>_defaced.nii.gz   (if defacing criteria matched)
  - Or original converted NIfTI copied as-is (if not a defacing candidate)
- Registration transforms (for defaced non-T1 sequences):
  - <seqstem>_to_t1.mat
- Session report JSON:
  - <REPORT_FILENAME from paths_config> (saved inside defaced/)

Temporary folders are cleaned:
- _converted_tmp is removed
- mideface temp folder is removed
- Only "defaced/" remains per session output directory


Troubleshooting
---------------
1) "mideface: command not found"
   - FreeSurfer is not installed or not on PATH.
   - Fix FreeSurfer setup, then re-check:
        mideface --help

2) mideface produced MGZ but conversion fails
   - Ensure mri_convert is installed and paths_config.MRI_CONVERT_PATH is valid.
   - Check:
        mri_convert --help

3) dcm2niix fails
   - Confirm your dcm2niix installation.
   - If you use explicit path configuration, ensure paths_config.DCM2NIIX_PATH points to a valid executable.

4) No T1 candidate identified
   - Your naming conventions differ.
   - Update logic_config.py t1_identification keywords/priority_keywords.

5) Output contains only copied (non-defaced) sequences
   - Those sequences likely did not match defacing_logic rules:
     - structural_keywords
     - skip_keywords
     - min_matrix_size