# K-Means Compression

Compresses images with k-means algorithm.

## Requirements

- Python >= 3.9

## Installation

Clone the repository

```bash
cd ~
git clone https://github.com/dunarand/kmeans-compression
cd kmeans-compression
```

Create a python virtual environment

```bash
python3 -m venv .venv
```

Activate the Python virtual environment. On Linux/macOS, run

```bash
source ./venv/bin/activate
```

On Windows,

```pwsh
.\.venv\Scripts\Activate.ps1
```

Then, install kmeans-compression as a package

```bash
pip3 install -e .
```

## Usage

```
usage: kmcomp [-h] [-i INPUT_PATH] [-o [OUTPUT_PATH]] [-n N_NEIGHBORS] [-f] [-s]

K-Means Compression CLI Tool

options:
  -h, --help            show this help message and exit
  -i INPUT_PATH, --input INPUT_PATH
                        Path to the input image.
  -o [OUTPUT_PATH], --output [OUTPUT_PATH]
                        Path to the output file.
  -n N_NEIGHBORS, --n-neighbors N_NEIGHBORS
                        Maximum number of clusters to use.
  -f, --force           Force the use of n_neighbors.
  -s, --save            Save the compressed image.

```
