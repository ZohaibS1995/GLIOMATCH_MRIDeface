# paths_config.py

"""
System Configuration: Paths and Execution Settings.
Edit this file to change input/output folders and tool locations.
"""

from pathlib import Path

# =============================================================================
# DIRECTORIES & FILES
# =============================================================================

# The root folder containing the DICOM files you want to convert.
SESSION_DIR = r"/home/zohaib/Downloads/pipeline_test/"

# Where the output NIfTI files and logs will be saved.
OUTPUT_DIR = r"/home/zohaib/docker_gliomatch/GLIOMATCH_MRIDeface/output_nifti_defaced_v2"

# The name (ONLY) of the summary JSON file generated per session.
# This will be saved into: <OUTPUT_DIR>/<session_name>/defaced/<REPORT_FILENAME>
REPORT_FILENAME = "session_report.json"

# =============================================================================
# EXTERNAL TOOLS
# =============================================================================

# Path to the dcm2niix executable (host).
DCM2NIIX_PATH = "dcm2niix"

# Host paths for FreeSurfer tools (kept for backward compatibility).
# When USE_DOCKER_FOR_MIDEFACE=True these are not used.
MIDEFACE_PATH = "mideface"
MRI_CONVERT_PATH = "mri_convert"

# =============================================================================
# DOCKER SETTINGS (NEW)
# =============================================================================

# If True, mideface + mri_convert run via Docker, not host FreeSurfer.
USE_DOCKER_FOR_MIDEFACE = True

# Docker binary (usually "docker").
DOCKER_BIN = "docker"

# Docker image that contains FreeSurfer + mideface.
# Build/tag it using the Dockerfile below, e.g.: gliomatch-mideface:7.4.1
FREESURFER_DOCKER_IMAGE = "gliomatch-mideface:7.4.1"

# Path on the HOST to your FreeSurfer license file.
# If you prefer, you can set environment variable FS_LICENSE to point to the license
# and that will take precedence over this value.
FREESURFER_LICENSE_PATH = "license.txt"

# Run container as the host UID:GID to avoid root-owned outputs on the host.
DOCKER_RUN_AS_HOST_USER = True

# =============================================================================
# PROCESSING THRESHOLDS
# =============================================================================

# Minimum number of slices required to process a series.
MIN_SLICES = 10

# Minimum dimension size for NIfTI output (files smaller than this are deleted).
MIN_VOL_DIM = 10

# Set to True to skip the defacing step entirely (conversion only).
SKIP_DEFACE = False
