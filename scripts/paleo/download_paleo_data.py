#!/usr/bin/env python
"""Download pre-built palegoESM paleoclimate boundary-condition NetCDF files.

Checks ``/home/guido/paleo_data/palegoESM`` (or ``--dest-dir``) first; if a file
is missing, downloads it from the ``guidov/palegoESM`` GitHub Release assets
(or an override URL via ``PALEGOESM_DATA_BASE_URL`` / Zenodo).

Usage::

    python scripts/paleo/download_paleo_data.py --dest-dir data/paleo
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import urllib.request

DEFAULT_RELEASE_TAG = "v0.1.0-data"
DEFAULT_BASE_URL = os.environ.get(
    "PALEGOESM_DATA_BASE_URL",
    f"https://github.com/guidov/palegoESM/releases/download/{DEFAULT_RELEASE_TAG}",
)

PALEO_FILES = (
    "topo_lgm_ice6gc_0p25deg.nc",
    "topo_pi_1850_0p25deg.nc",
    "surfdata_fv1_lgm_ice6gc.nc",
)


def fetch_paleo_files(dest_dir: Path, base_url: str = DEFAULT_BASE_URL) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    local_cache = Path("/home/guido/paleo_data/palegoESM")
    downloaded: list[Path] = []

    for fname in PALEO_FILES:
        target = dest_dir / fname
        if target.exists():
            print(f"[skip] {target} already exists ({target.stat().st_size / 1e6:.2f} MB)")
            downloaded.append(target)
            continue

        # If running on a machine with /home/guido/paleo_data/palegoESM, symlink or copy
        local_src = local_cache / fname
        if local_src.exists() and local_src.resolve() != target.resolve():
            os.symlink(local_src, target)
            print(f"[link] Linked {target} -> {local_src}")
            downloaded.append(target)
            continue

        url = f"{base_url.rstrip('/')}/{fname}"
        print(f"[download] Fetching {url} -> {target} ...")
        urllib.request.urlretrieve(url, target)
        print(f"[done] Saved {target} ({target.stat().st_size / 1e6:.2f} MB)")
        downloaded.append(target)

    return downloaded


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path("/home/guido/paleo_data/palegoESM"),
        help="Destination directory for paleoclimate NetCDF files",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="Base URL for GitHub Release or Zenodo record",
    )
    args = parser.parse_args(argv)
    fetch_paleo_files(args.dest_dir, base_url=args.base_url)


if __name__ == "__main__":
    main()
