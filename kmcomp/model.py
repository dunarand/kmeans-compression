"""Main module for K-Means image compression."""

from multiprocessing import Pool
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans

from kmcomp.file_utils import read_dir, read_file, read_files, save_image


def _find_elbow(n_neighbors, clusters, kms):
    """
    Find the elbow point of K-Means cluster.
    Returns the chosen k and the corresponding KMeans model.
    """
    inertias = [km.inertia_ for km in kms]
    kl = KneeLocator(
        x=clusters, y=inertias, curve="convex", direction="decreasing"
    )
    _k = kl.elbow

    if _k is None or _k == 1:
        chosen_k = n_neighbors
        chosen_km = kms[-1]
    else:
        chosen_k = _k
        # clusters list starts at 2 (index 0).
        # So, k=2 is at index 0, k=3 is at index 1.
        # The model for chosen_k is at index (chosen_k - 2).
        chosen_km = kms[chosen_k - 2]
    return chosen_k, chosen_km


def _process_single_image_compression(
    file_data: dict,
    output_path: Path | str | bool,
    n_neighbors: int,
    force: bool,
    save: bool,
) -> tuple[np.ndarray | None, Path, int] | None:
    """
    Performs K-Means compression on a single image.
    This function is designed to be run in parallel by multiprocessing.Pool.

    Returns:
        If save=False: (compressed_img_uint8, original_file_path, k_clusters)
        for plotting.
        If save=True: None (saving is handled as a side effect).
    """

    file_name = file_data.get("file_path").stem
    print(f"[{file_name}] Starting K-Means processing.")

    img_flat_df = file_data.get("img_df")
    img_shape = file_data.get("img_shape")
    disk_size_bytes = file_data.get("disk_size")
    original_file_path = file_data.get("file_path")
    img_size_kb = (disk_size_bytes / 1024) if disk_size_bytes is not None else 0

    if (
        img_flat_df is None
        or img_shape is None
        or not disk_size_bytes
        or not original_file_path
    ):
        print(f"[{file_name}] Skipping: Incomplete file data.")
        return None

    start_time = time()
    km = None
    k_clusters = 0

    if force:
        print(f"[{file_name}] Forcing k={n_neighbors} clusters.")
        km = KMeans(n_clusters=n_neighbors, n_init="auto", random_state=42)
        km.fit(img_flat_df)
        k_clusters = n_neighbors
    else:
        clusters = list(range(2, n_neighbors + 1))
        kms = []
        for n in clusters:
            _km = KMeans(n_clusters=n, n_init="auto", random_state=42)
            _km.fit(img_flat_df)
            kms.append(_km)

        k_clusters, km = _find_elbow(n_neighbors, clusters, kms)

    end_time = time()
    dur = end_time - start_time
    print(
        f"[{file_name}] K-Means clustering completed in {dur:.2f} seconds. "
        f"Using k={k_clusters}"
    )

    centroids = km.cluster_centers_
    labels = km.labels_

    compressed_data_flat = centroids[labels]
    compressed_img_float = compressed_data_flat.reshape(img_shape)

    compressed_img_float = np.clip(compressed_img_float, 0, 1)
    compressed_img_uint8 = (compressed_img_float * 255).astype(np.uint8)

    if save:
        save_image(
            compressed_img_uint8=compressed_img_uint8,
            original_path=original_file_path,
            output_spec=output_path,
            k_clusters=k_clusters,
            original_size_kb=img_size_kb,
        )
        return None
    else:
        return compressed_img_uint8, original_file_path, k_clusters


def main(
    input_path: Path | str | list[Path | str],
    output_path: Path | str | bool = True,
    n_neighbors: int = 12,
    force: bool = False,
    save: bool = False,
) -> None:
    """Main function for K-Means image compression."""

    img_dicts = {}
    try:
        if isinstance(input_path, (Path, str)):
            input_path = Path(input_path)
            if input_path.is_dir():
                img_dicts = read_dir(input_path)
            elif input_path.is_file():
                img_dicts = read_file(input_path)
            else:
                raise FileNotFoundError(f"Input path not found: {input_path}")
        elif isinstance(input_path, list):
            img_dicts = read_files(input_path)
    except Exception as e:
        print(f"Error loading image files: {e}")
        return

    if not img_dicts:
        print("No image files found or loaded.")
        return

    args_list = []
    for file_data in img_dicts.values():
        args_list.append((file_data, output_path, n_neighbors, force, save))

    plot_results = []

    if len(img_dicts) > 1:
        print(
            f"\n-- Starting Parallel Compression of {len(img_dicts)} Images --"
        )

        with Pool() as pool:
            results = pool.starmap(_process_single_image_compression, args_list)

        plot_results = [res for res in results if res is not None]
        print("--- Parallel Compression Complete ---")

    else:
        print("\n--- Starting Serial Compression of 1 Image ---")
        result = _process_single_image_compression(*args_list[0])
        if result is not None:
            plot_results.append(result)

    if not save:
        for (
            compressed_img_uint8,
            original_file_path,
            k_clusters,
        ) in plot_results:
            plt.figure()
            plt.imshow(compressed_img_uint8)
            plt.title(
                f"{original_file_path.name}\nCompressed with k={k_clusters}"
            )
            plt.axis("off")

    if not save and len(plot_results) > 0:
        print("\nDisplaying compressed images... Close plot windows to exit.")
        plt.show()
