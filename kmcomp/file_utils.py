"""File utility functions for kmcomp."""

from multiprocessing import Pool
from pathlib import Path
from time import time
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from kmcomp.exceptions import Exceptions

SUPPORTED_FORMATS = [
    ".png",
    ".webp",
    ".jpg",
    ".jpeg",
]


class FileInfo(TypedDict):
    """For annotation."""

    file_path: Path
    img_shape: tuple[int, int]
    img_df: pd.DataFrame
    disk_size: int | float
    status: str


def _read_dir(path: Path | str) -> list[Path]:
    """Reads the target directory and returns a list of image paths."""
    if isinstance(path, str):
        path = Path(path)
    path = path.resolve()

    if not path.exists():
        raise Exceptions.PathDoesNotExistError(path)

    if not path.is_dir():
        raise Exceptions.NotDirectoryError(path)

    img_paths = []
    for ext in SUPPORTED_FORMATS:
        for file_path in path.glob(f"*{ext}"):
            img_paths.append(file_path)

    if not img_paths:
        raise Exceptions.NoSupportedImageFilesError(path)

    return img_paths


def _read_single_img(_path: Path) -> dict[str, FileInfo] | dict[str, Any]:
    """Read a single image and return as a dictionary."""

    file_disk_size = 0
    try:
        file_disk_size = _path.stat().st_size
    except FileNotFoundError:
        print(f"File not found at {_path.name}. Skipping.")
        return {_path.stem: {"status": "not_found"}}

    res = {
        "file_path": _path,
        "img_shape": tuple(),
        "img_df": pd.DataFrame(),
        "disk_size": file_disk_size,
        "status": "success",
    }

    try:
        img = plt.imread(_path)
    except Exception as e:
        print(f"Warning: Could not read image file {_path.name}. Error: {e}")
        return {
            _path.stem: {"status": "read_failed", "disk_size": file_disk_size}
        }

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    elif img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif not (img.ndim == 3 and img.shape[2] == 3):
        raise Exceptions.UnsupportedImageTypeError(
            path=_path,
            message=f"Unsupported image format with {img.ndim} dimensions and "
            f"{img.shape[2] if img.ndim == 3 else 'N/A'} channels.",
        )

    if img.dtype in [np.uint8, np.uint16, np.uint32]:
        max_val = np.iinfo(img.dtype).max if img.dtype != np.float32 else 1.0
        img_normalized = img.astype(np.float32) / max_val
    else:
        img_normalized = img.astype(np.float32)

    original_shape = img.shape
    img_flat = img_normalized.reshape(original_shape[0] * original_shape[1], 3)

    res["img_shape"] = original_shape
    res["img_df"] = pd.DataFrame(img_flat, columns=["r", "g", "b"])

    return {_path.stem: res}


def _read_images(paths: list[Path]) -> dict[str, FileInfo]:
    """Reads images within path and returns a dictionary of DataFrame of each
    image."""

    start = time()

    with Pool() as pool:
        file_dicts_list = pool.map(_read_single_img, paths)

    dfs = {}
    total_disk_size_bytes = 0
    total_memory_size_bytes = 0

    for file_dict in file_dicts_list:
        key = list(file_dict.keys())[0]
        data = file_dict[key]

        if "disk_size" in data:
            total_disk_size_bytes += data["disk_size"]

        if data.get("status") == "success":
            total_memory_size_bytes += (
                data["img_df"].memory_usage(deep=False).sum()
            )
            dfs[key] = data

    end = time()

    disk_size_mb = f"{total_disk_size_bytes / 1024**2:.2f} MB"
    mem_size_mb = total_memory_size_bytes / 1024**2
    if mem_size_mb >= 1000:
        mem_size_mb = f"{total_memory_size_bytes / 1024**3:.2f} GB"
    else:
        mem_size_mb = f"{mem_size_mb:.2f} MB"

    print("\n--- Image Reading Statistics ---")
    print(f"Total file reads: {len(dfs)}")
    print(f"Time elapsed: {end - start:.2f} seconds.")
    print(f"Total disk size of all image files: {disk_size_mb}")
    print(f"Total in-memory size (DataFrames): {mem_size_mb}")
    print("------- End of read call -------\n")

    return dfs


def save_image(
    compressed_img_uint8: np.ndarray,
    original_path: Path,
    output_spec: Path | str | bool,
    k_clusters: int,
    original_size_kb: float,
):
    """Saves a single compressed image and prints statistics."""

    input_path_obj = Path(original_path)
    output_file = None

    if isinstance(output_spec, bool) and output_spec:
        output_file = input_path_obj.parent / (
            f"compressed_k-{k_clusters}-" + input_path_obj.name
        )
    elif isinstance(output_spec, (str, Path)):
        output_path_obj = Path(output_spec)
        if output_path_obj.is_dir():
            output_file = output_path_obj / (
                f"compressed_k-{k_clusters}-" + input_path_obj.name
            )
        else:
            output_file = output_path_obj

    if output_file is None:
        output_file = input_path_obj.parent / (
            f"compressed_k-{k_clusters}-" + input_path_obj.name
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        compressed_img = Image.fromarray(compressed_img_uint8)
        compressed_img.save(output_file)
        compressed_img_size = output_file.stat().st_size / 1024
        compression_ratio = 1 - (compressed_img_size / original_size_kb)
        print(f"Compressed image with k={k_clusters} saved to {output_file}")
        print(
            f"Original size: {original_size_kb:.2f} KB, "
            f"Compressed size: {compressed_img_size:.2f} KB, "
            f"Compression ratio: {compression_ratio:.2%}"
        )
    except Exception as e:
        print(f"Error saving image {output_file}: {e}")


def read_file(path: Path | str) -> dict[str, FileInfo]:
    """Reads a single image and returns a dictionary of metadata and image
    dataframe."""
    if isinstance(path, str):
        path = Path(path)
    path = path.resolve()

    if not path.exists():
        raise Exceptions.PathDoesNotExistError(path)

    if not path.is_file():
        raise Exceptions.NotFileError(path)

    if path.suffix not in SUPPORTED_FORMATS:
        raise Exceptions.UnsupportedFileTypeError(path)

    return _read_single_img(path)


def read_dir(path: Path | str) -> dict[str, FileInfo]:
    """Reads a directory and returns a dictionary of metadata and image
    dataframes."""
    _paths = _read_dir(path)
    return _read_images(_paths)


def read_files(paths: list[Path | str]) -> dict[str, FileInfo]:
    """Reads a list of image paths and returns a dictionary of metadata and
    image
    dataframes."""
    _paths = []
    for p in paths:
        if isinstance(p, str):
            p = Path(p)
        if not isinstance(p, Path):
            print(
                f"Warning: Skipping invalid item in list (not Path or str): {p}"
            )
            continue

        p = p.resolve()
        if not p.exists():
            print(f"Warning: File not found, skipping: {p}")
            continue
        if not p.is_file():
            print(f"Warning: Path is not a file, skipping: {p}")
            continue
        if p.suffix not in SUPPORTED_FORMATS:
            print(f"Warning: Unsupported file type, skipping: {p}")
            continue
        _paths.append(p)

    return _read_images(_paths)


def save_images():
    """Docstring (Not Implemented - main loop calls save_image)"""
