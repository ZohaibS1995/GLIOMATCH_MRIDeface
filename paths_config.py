# paths_config.py

"""
System Configuration: Paths and Execution Settings.
Edit this file to change input/output folders and tool locations.
"""

# =============================================================================
# DIRECTORIES & FILES
# =============================================================================

# The root folder containing the DICOM files you want to convert.
SESSION_DIR = r"/home/zohaib/Downloads/pipeline_test/"

# Where the output NIfTI files and logs will be saved.
OUTPUT_DIR = r"/home/zohaib/gliomatch/gliomatch_deface/output_nifti_defaced_v2"

# The name (ONLY) of the summary JSON file generated per session.
# This will be saved into: <OUTPUT_DIR>/<session_name>/defaced/<REPORT_FILENAME>
REPORT_FILENAME = "session_report.json"

# =============================================================================
# EXTERNAL TOOLS
# =============================================================================

# Path to the dcm2niix executable.
DCM2NIIX_PATH = "dcm2niix"

# Path to the mideface executable (FreeSurfer)
MIDEFACE_PATH = "mideface"

# Path to mri_convert (FreeSurfer)
MRI_CONVERT_PATH = "mri_convert"

# =============================================================================
# PROCESSING THRESHOLDS
# =============================================================================

# Minimum number of slices required to process a series.
MIN_SLICES = 10

# Minimum dimension size for NIfTI output (files smaller than this are deleted).
MIN_VOL_DIM = 10

# Set to True to skip the defacing step entirely (conversion only).
SKIP_DEFACE = False
