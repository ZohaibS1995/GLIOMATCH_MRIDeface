import os
import argparse
import json
import subprocess
from pathlib import Path
import shutil
from typing import Tuple, Dict

# --- IMPORT CONFIGURATIONS ---
# Ensure these modules exist in your environment
import paths_config
import logic_config

import pydicom
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence
import nibabel as nib
import numpy as np
import ants  # ANTsPy


# --- DICOM Helpers ---
def is_dicom_file(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        return True
    except Exception:
        return False


def find_series_dirs(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        dicom_files = [p / f for f in filenames]
        dicom_files = [f for f in dicom_files if is_dicom_file(f)]
        if dicom_files:
            yield p, dicom_files


def compute_slice_count(dicom_files):
    iop_positions = []
    slice_locations = []
    have_ipp = False
    have_sl = False
    multiframe_total = 0

    for f in dicom_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:
            continue

        nf = getattr(ds, "NumberOfFrames", None)
        if nf and int(nf) > 1:
            multiframe_total += int(nf)
            continue

        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp and len(ipp) == 3:
            have_ipp = True
            ipp_key = tuple(round(float(x), 1) for x in ipp)
            iop_positions.append(ipp_key)
            continue

        sl = getattr(ds, "SliceLocation", None)
        if sl is not None:
            have_sl = True
            slice_locations.append(round(float(sl), 1))
            continue

    if multiframe_total > 0:
        return multiframe_total
    if have_ipp:
        return len(set(iop_positions))
    if have_sl:
        return len(set(slice_locations))
    return 1


def guess_orientation_from_iop(iop):
    if not iop or len(iop) != 6:
        return ""
    rx, ry, rz, cx, cy, cz = [float(v) for v in iop]
    nx = ry * cz - rz * cy
    ny = rz * cx - rx * cz
    nz = rx * cy - ry * cx
    ax, ay, az = abs(nx), abs(ny), abs(nz)
    if az >= ax and az >= ay:
        return "axial"
    elif ay >= ax and ay >= az:
        return "coronal"
    else:
        return "sagittal"


def extract_metadata(dcm: pydicom.dataset.FileDataset, series_path: Path,
                     num_instances: int, computed_slice_count: int):
    def g(tag, default=None):
        return getattr(dcm, tag, default)

    iop = g("ImageOrientationPatient", [])
    orientation = guess_orientation_from_iop(iop)

    return {
        "SeriesPath": str(series_path),
        "PatientID": g("PatientID", ""),
        "StudyInstanceUID": g("StudyInstanceUID", ""),
        "SeriesInstanceUID": g("SeriesInstanceUID", ""),
        "SeriesNumber": g("SeriesNumber", ""),
        "SeriesDescription": g("SeriesDescription", ""),
        "ProtocolName": g("ProtocolName", ""),
        "Modality": g("Modality", ""),
        "Manufacturer": g("Manufacturer", ""),
        "ManufacturerModelName": g("ManufacturerModelName", ""),
        "MagneticFieldStrength": g("MagneticFieldStrength", ""),
        "RepetitionTime": g("RepetitionTime", ""),
        "EchoTime": g("EchoTime", ""),
        "InversionTime": g("InversionTime", ""),
        "FlipAngle": g("FlipAngle", ""),
        "PixelSpacing": g("PixelSpacing", ""),
        "SliceThickness": g("SliceThickness", ""),
        "SpacingBetweenSlices": g("SpacingBetweenSlices", ""),
        "Rows": g("Rows", ""),
        "Columns": g("Columns", ""),
        "PatientPosition": g("PatientPosition", ""),
        "ImageOrientationPatient": list(iop) if iop else "",
        "ImagePositionPatient": g("ImagePositionPatient", ""),
        "ScanningSequence": g("ScanningSequence", ""),
        "SequenceVariant": g("SequenceVariant", ""),
        "ScanOptions": g("ScanOptions", ""),
        "MRAcquisitionType": g("MRAcquisitionType", ""),
        "ImageType": g("ImageType", ""),
        "DerivedOrientation": orientation,
        "SeriesDate": g("SeriesDate", ""),
        "AcquisitionDate": g("AcquisitionDate", ""),
        "StudyDate": g("StudyDate", ""),
        "AcquisitionDateTime": g("AcquisitionDateTime", ""),
        "AcquisitionTime": g("AcquisitionTime", ""),
        "StudyTime": g("StudyTime", ""),
        "BodyPartExamined": g("BodyPartExamined", ""),
        "num_instances": num_instances,
        "computed_slice_count": computed_slice_count,
    }


def score_t1_candidate(meta: dict) -> int:
    """Scores a series to find the best T1 for mideface input."""
    desc = (meta.get("SeriesDescription") or "").lower()
    prot = (meta.get("ProtocolName") or "").lower()
    n = meta.get("computed_slice_count") or 0

    conf = logic_config.LOGIC.get("t1_identification", {})
    keywords = [k.lower() for k in conf.get("keywords", [])]
    priority = [k.lower() for k in conf.get("priority_keywords", [])]

    score = 0
    full_str = f"{desc} {prot}"

    if any(k in full_str for k in keywords):
        score += 50
    if any(k in full_str for k in priority):
        score += 20
    score += min(n, 300) // 5
    if "head" in full_str or "brain" in full_str:
        score += 5
    return score


def is_candidate_for_defacing(meta: dict) -> bool:
    """Determines if a sequence should be defaced based on config rules."""
    desc = (meta.get("SeriesDescription") or "").lower()
    prot = (meta.get("ProtocolName") or "").lower()
    full_str = f"{desc} {prot}"

    conf = logic_config.LOGIC.get("defacing_logic", {})
    skip_keywords = [k.lower() for k in conf.get("skip_keywords", [])]
    struct_keywords = [k.lower() for k in conf.get("structural_keywords", [])]
    min_mtx = conf.get("min_matrix_size", 128)

    # 1. Check BLACKLIST
    if any(k in full_str for k in skip_keywords):
        return False

    # 2. Check WHITELIST (Structural types)
    is_structural = any(k in full_str for k in struct_keywords)

    # 3. Check Resolution
    try:
        rows = int(meta.get("Rows") or 0)
        cols = int(meta.get("Columns") or 0)
        is_decent_res = (rows >= min_mtx and cols >= min_mtx)
    except (ValueError, TypeError):
        is_decent_res = False

    return is_structural and is_decent_res


def run_dcm2niix(series_dir: Path, output_dir: Path, outname_prefix: str):
    tool = paths_config.DCM2NIIX_PATH
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        tool,
        "-b", "n",
        "-z", "y",
        "-o", str(output_dir),
        "-f", outname_prefix,
        str(series_dir)
    ]
    subprocess.run(cmd, check=True)


def run_mideface(
        input_nifti: Path,
        output_dir: Path,
) -> Tuple[Path, Path]:
    tmp_dir = output_dir / "_mideface_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    mideface_tool = paths_config.MIDEFACE_PATH
    mri_convert_tool = paths_config.MRI_CONVERT_PATH

    cmd = [
        mideface_tool,
        "--i", str(input_nifti),
        "--odir", str(tmp_dir),
    ]

    print(f"[INFO] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    nifti_candidates = list(tmp_dir.glob("*.nii")) + list(tmp_dir.glob("*.nii.gz"))
    mgz_candidates = list(tmp_dir.glob("*.mgz"))

    def pick_preferred(paths):
        if not paths:
            return None
        preferred = [
            p for p in paths
            if any(tok in p.name.lower() for tok in ["defac", "deface", "mideface"])
        ]
        if preferred:
            return preferred[0]
        return max(paths, key=lambda p: p.stat().st_size)

    # defaced image
    if nifti_candidates:
        chosen = pick_preferred(nifti_candidates)
        final_defaced_path = output_dir / f"{input_nifti.stem}_defaced{''.join(chosen.suffixes)}"
        shutil.move(str(chosen), str(final_defaced_path))
    elif mgz_candidates:
        chosen = pick_preferred(mgz_candidates)
        final_defaced_path = output_dir / f"{input_nifti.stem}_defaced.nii.gz"
        if not mri_convert_tool:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError("mideface produced defaced .mgz but mri_convert_path is empty in config.")
        convert_cmd = [
            mri_convert_tool,
            str(chosen),
            str(final_defaced_path),
        ]
        print(f"[INFO] Converting MGZ to NIfTI: {' '.join(convert_cmd)}")
        subprocess.run(convert_cmd, check=True)
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"mideface produced no defaced image in {tmp_dir}")

    # face mask
    final_mask_path = output_dir / f"{input_nifti.stem}_deface_mask.nii.gz"
    face_mask_nii = None
    for cand in ["face.mask.nii.gz", "face.mask.nii"]:
        p = tmp_dir / cand
        if p.exists():
            face_mask_nii = p
            break

    if face_mask_nii is not None:
        shutil.move(str(face_mask_nii), str(final_mask_path))
    else:
        face_mask_mgz = tmp_dir / "face.mask.mgz"
        if face_mask_mgz.exists():
            if not mri_convert_tool:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise RuntimeError("mideface produced face.mask.mgz but mri_convert_path is empty in config.")
            convert_cmd = [
                mri_convert_tool,
                str(face_mask_mgz),
                str(final_mask_path),
            ]
            print(f"[INFO] Converting face.mask.mgz to NIfTI: {' '.join(convert_cmd)}")
            subprocess.run(convert_cmd, check=True)
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError("Expected 'face.mask.mgz' (or NIfTI) from mideface but did not find it.")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return final_defaced_path, final_mask_path


def deface_moving_via_t1_mask(
        moving_img: Path,
        t1_img: Path,
        t1_mask: Path,
        defaced_output: Path,
        transform_output: Path = None,
):
    """
    Registers 'moving_img' (e.g. T2/FLAIR) to 't1_img' using RIGID registration.
    Then applies the INVERSE transform to 't1_mask' to bring the face mask
    onto the moving image, and zeros out the face.
    """
    print(f"[INFO] Registering (Rigid) {moving_img.name} -> {t1_img.name}")

    fixed = ants.image_read(str(t1_img))
    moving = ants.image_read(str(moving_img))
    mask_fixed = ants.image_read(str(t1_mask))

    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="Rigid"
    )

    print(f"[INFO] Propagating mask to {moving_img.name} space...")

    mask_in_moving = ants.apply_transforms(
        fixed=moving,
        moving=mask_fixed,
        transformlist=reg["invtransforms"],
        interpolator="nearestNeighbor"
    )

    mov_arr = moving.numpy()
    mask_arr = mask_in_moving.numpy()
    mov_arr[mask_arr > 0.5] = 0

    defaced = ants.from_numpy(
        mov_arr,
        origin=moving.origin,
        spacing=moving.spacing,
        direction=moving.direction
    )

    print(f"[INFO] Saving defaced image to {defaced_output}")
    ants.image_write(defaced, str(defaced_output))

    if transform_output is not None and reg["fwdtransforms"]:
        trf_src = Path(reg["fwdtransforms"][0])
        if trf_src.exists():
            shutil.copy2(trf_src, transform_output)
            print(f"[INFO] Saved Rigid transform (Moving->T1) to {transform_output}")

    return defaced_output


def make_jsonable(obj):
    if isinstance(obj, dict):
        return {k: make_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_jsonable(v) for v in obj]
    elif isinstance(obj, MultiValue):
        return [make_jsonable(v) for v in obj]
    elif isinstance(obj, Sequence):
        return [make_jsonable(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def cleanup_outdir_keep_defaced_only(outdir: Path, defaced_dir: Path):
    """Deletes everything in outdir except defaced_dir."""
    for item in outdir.iterdir():
        if item.resolve() == defaced_dir.resolve():
            continue
        if item.is_file() or item.is_symlink():
            try:
                item.unlink()
            except Exception:
                pass
        elif item.is_dir():
            shutil.rmtree(item, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="DICOM -> NIfTI with configured defacing pipeline."
    )
    parser.add_argument("--session_dir", default=paths_config.SESSION_DIR,
                        help="session root folder")
    parser.add_argument("--outdir", default=paths_config.OUTPUT_DIR,
                        help="output directory")

    args = parser.parse_args()

    session_dir = Path(args.session_dir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INIT] Session Dir: {session_dir}")
    print(f"[INIT] Output Dir: {outdir}")

    # NEW: final folder (always) + temp conversion folder
    defaced_dir = outdir / "defaced"
    defaced_dir.mkdir(exist_ok=True)
    conversion_dir = outdir / "_converted_tmp"
    conversion_dir.mkdir(exist_ok=True)

    all_metadata = []
    kept_metadata = []
    session_report = []

    # 1. Discover
    print("[INFO] Discovering DICOM series...")
    for series_dir, dicom_files in find_series_dirs(session_dir):
        file_count = len(dicom_files)
        slice_count = compute_slice_count(dicom_files)
        try:
            first_dcm = pydicom.dcmread(str(dicom_files[0]), stop_before_pixels=True, force=True)
        except Exception as e:
            session_report.append({
                "status": "failed",
                "params": {
                    "SeriesPath": str(series_dir),
                    "reason": f"could not read dicom: {e}",
                },
            })
            continue

        meta = extract_metadata(first_dcm, series_dir, num_instances=file_count, computed_slice_count=slice_count)
        all_metadata.append(meta)

    # 2. Filter
    min_slices = paths_config.MIN_SLICES
    print(f"[INFO] Pre-filtering series (min_slices={min_slices})...")
    for meta in all_metadata:
        if (meta.get("computed_slice_count") or 0) >= min_slices:
            kept_metadata.append(meta)
        else:
            m = dict(meta)
            m["reason"] = f"fewer than min_slices ({min_slices})"
            session_report.append({"status": "skipped", "params": m})

    # 3. Pick T1
    print("[INFO] Identifying T1 candidate...")
    t1_candidate_uid = None
    best_score = -1
    for meta in kept_metadata:
        s = score_t1_candidate(meta)
        if s > best_score:
            best_score = s
            t1_candidate_uid = meta.get("SeriesInstanceUID")

    # 4. Convert ALL (into conversion_dir)
    print("[INFO] Converting DICOM to NIfTI...")
    t1_nifti_path = None
    all_nifti_files = []
    min_vol_dim = paths_config.MIN_VOL_DIM

    for i, meta in enumerate(kept_metadata, start=1):
        series_dir = Path(meta["SeriesPath"])
        sd = meta.get("SeriesDescription") or meta.get("ProtocolName") or f"series{i}"
        sd = sd.replace(" ", "_").replace("/", "_")
        series_num = meta.get("SeriesNumber") or i
        outname = f"{int(series_num):03d}_{sd}" if str(series_num).isdigit() else f"{series_num}_{sd}"
        is_t1 = (meta.get("SeriesInstanceUID") == t1_candidate_uid)

        before = set(conversion_dir.glob("*.nii")) | set(conversion_dir.glob("*.nii.gz"))
        try:
            run_dcm2niix(series_dir, conversion_dir, outname)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] dcm2niix failed for {series_dir}: {e}")
            m = dict(meta)
            m["outname_prefix"] = outname
            m["is_t1_candidate"] = is_t1
            m["reason"] = f"dcm2niix failed: {e}"
            session_report.append({"status": "failed", "params": m})
            continue

        after = set(conversion_dir.glob("*.nii")) | set(conversion_dir.glob("*.nii.gz"))
        new_niftis = list(after - before)
        good_niftis = []
        unreadable_niftis = []

        for nf in new_niftis:
            try:
                img = nib.load(str(nf))
                shape = img.shape
            except Exception:
                unreadable_niftis.append(nf.name)
                continue

            if not any(dim < min_vol_dim for dim in shape):
                good_niftis.append({"file": nf.name, "path": nf})

        print(f"[INFO] Converted {series_dir} -> {[g['file'] for g in good_niftis]}")

        m = dict(meta)
        m["outname_prefix"] = outname
        m["is_t1_candidate"] = is_t1
        m["produced_nifti_files"] = [p.name for p in new_niftis]
        m["kept_nifti_files"] = [g["file"] for g in good_niftis]
        m["unreadable_nifti_files"] = unreadable_niftis
        if good_niftis:
            session_report.append({"status": "converted", "params": m})
        else:
            m["reason"] = f"conversion produced no usable NIfTI after MIN_VOL_DIM filtering (MIN_VOL_DIM={min_vol_dim})"
            session_report.append({"status": "skipped", "params": m})

        for gn in good_niftis:
            all_nifti_files.append({
                "path": gn["path"],
                "name": gn["file"],
                "is_t1": is_t1,
                "meta": meta,
            })
            if is_t1 and not t1_nifti_path:
                t1_nifti_path = gn["path"]

    # 5. Report -> save INSIDE defaced_dir so outdir can be cleaned to only defaced/
    report_file = paths_config.REPORT_FILENAME
    report_path = defaced_dir / report_file
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(make_jsonable(session_report), f, indent=2)
    print(f"\n[INFO] Saved session report (metadata) to: {report_path}")

    if t1_nifti_path:
        print(f"[INFO] Selected T1 Candidate: {t1_nifti_path.name}")
    else:
        print("[WARN] No T1 image candidate identified.")

    # 6. Deface
    skip_deface = paths_config.SKIP_DEFACE or False

    try:
        if not skip_deface and t1_nifti_path:
            print("\n" + "=" * 60)
            print("[INFO] Starting mideface workflow...")
            print("=" * 60)

            t1_defaced_path, t1_mask_path = run_mideface(t1_nifti_path, defaced_dir)

            print("\n[INFO] Processing other sequences...")
            for nifti_info in all_nifti_files:
                if nifti_info["is_t1"]:
                    continue

                source_path = nifti_info["path"]

                if is_candidate_for_defacing(nifti_info["meta"]):
                    print(f"\n[INFO] Defacing sequence: {source_path.name}")
                    defaced_out = defaced_dir / f"{source_path.stem}_defaced.nii.gz"
                    transform_out = defaced_dir / f"{source_path.stem}_to_t1.mat"
                    try:
                        deface_moving_via_t1_mask(
                            source_path, t1_nifti_path, t1_mask_path, defaced_out, transform_out
                        )
                    except Exception as e:
                        print(f"[ERROR] Failed: {e}")
                else:
                    print(f"\n[INFO] Copying original (no deface needed): {source_path.name}")
                    shutil.copy2(str(source_path), str(defaced_dir / source_path.name))

            print("\n[INFO] Workflow complete!")

        elif skip_deface:
            print("\n[INFO] Skipping defacing (configured in paths_config).")
            # Put *converted* outputs into defaced_dir as the final output location
            for nifti_info in all_nifti_files:
                src = nifti_info["path"]
                dst = defaced_dir / src.name
                if src.exists() and not dst.exists():
                    shutil.move(str(src), str(dst))

        else:
            print("\n[WARN] Defacing not run (no T1 candidate). Putting converted outputs into defaced/ anyway.")
            for nifti_info in all_nifti_files:
                src = nifti_info["path"]
                dst = defaced_dir / src.name
                if src.exists() and not dst.exists():
                    shutil.move(str(src), str(dst))

    finally:
        # NEW: remove the conversion temp dir (so only defaced/ remains)
        shutil.rmtree(conversion_dir, ignore_errors=True)

        # NEW: ensure outdir contains ONLY defaced/
        cleanup_outdir_keep_defaced_only(outdir, defaced_dir)

        print(f"\n[INFO] Cleanup done. Only remaining folder: {defaced_dir}")


if __name__ == "__main__":
    main()
