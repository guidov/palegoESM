#!/usr/bin/env python
"""Download pre-built palegoESM paleoclimate boundary-condition NetCDF files.

Downloads from the public Hugging Face dataset repository ``guidov/palegoESM-data``
(https://huggingface.co/datasets/guidov/palegoESM-data), or links from local
``/home/guido/paleo_data/palegoESM`` if present on the machine.

Usage::

    python scripts/paleo/download_paleo_data.py --dest-dir data/paleo
    python scripts/paleo/download_paleo_data.py --include-source-reconstructions
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import urllib.request

HF_REPO_ID = os.environ.get("PALEGOESM_HF_REPO", "guidov/palegoESM-data")
DEFAULT_BASE_URL = os.environ.get(
    "PALEGOESM_DATA_BASE_URL",
    f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main",
)
GITHUB_FALLBACK_URL = "https://github.com/guidov/palegoESM/releases/download/v0.1.0-data"

PALEGOESM_READY_FILES = (
    "palegoESM/topo_pi_1850_0p25deg.nc",
    "palegoESM/topo_lgm_ice6gc_0p25deg.nc",
    "palegoESM/topo_lgm_ice6ga_0p25deg.nc",
    "palegoESM/topo_lgm_ice7gna_0p25deg.nc",
    "palegoESM/topo_lgm_ice5g_0p25deg.nc",
    "palegoESM/surfdata_fv1_lgm_ice6gc.nc",
)

SOURCE_RECONSTRUCTION_FILES = (
    "ice_sheets_and_topo/etopo_10min_topo_v2.nc",
    "ice_sheets_and_topo/i6gcocnx1_21k-0k+etopo_10min_topo_v2_kevin.nc",
    "ice_sheets_and_topo/i6gaocnx1_21k-0k+etopo_10min_topo_v2.nc",
    "ice_sheets_and_topo/i7gnaocnx1_21k-0k+etopo_10min_topo_v2_kevin.nc",
    "ice_sheets_and_topo/i5gocnx1_21k-0k+etopo_10min_topo_v2.nc",
    "ice_sheets_and_topo/orog_b_icetm_10min.0_StoN.nc",
    "ice_sheets_and_topo/orog_b_icetm_10min.21_StoN.nc",
    "ice_sheets_and_topo/I6_Interim_VM5a_orog_b_icetm_0.5.21k_shift.nc",
    "ice_sheets_and_topo/ice5g_v1.2_21.0k_10min.nc",
)


def fetch_paleo_files(
    dest_dir: Path,
    *,
    base_url: str = DEFAULT_BASE_URL,
    include_source: bool = False,
) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    local_root = Path("/home/guido/paleo_data")
    rel_paths = list(PALEGOESM_READY_FILES)
    if include_source:
        rel_paths.extend(SOURCE_RECONSTRUCTION_FILES)

    downloaded: list[Path] = []
    for rel in rel_paths:
        target = dest_dir / Path(rel).name if not include_source else dest_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            print(f"[skip] {target} already exists ({target.stat().st_size / 1e6:.2f} MB)")
            downloaded.append(target)
            continue

        local_src = local_root / rel
        if local_src.exists() and local_src.resolve() != target.resolve():
            os.symlink(local_src.resolve(), target)
            print(f"[link] Linked {target} -> {local_src.resolve()}")
            downloaded.append(target)
            continue

        url = f"{base_url.rstrip('/')}/{rel}"
        print(f"[download] Fetching {url} -> {target} ...")
        try:
            urllib.request.urlretrieve(url, target)
        except Exception:
            fallback_url = f"{GITHUB_FALLBACK_URL}/{Path(rel).name}"
            print(f"[fallback] Trying GitHub Release {fallback_url} -> {target} ...")
            urllib.request.urlretrieve(fallback_url, target)
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
        help="Base URL for Hugging Face Dataset resolve endpoint",
    )
    parser.add_argument(
        "--include-source-reconstructions",
        action="store_true",
        help="Also download raw 10-minute ICE-5G / ICE-6G / ICE-7G / ETOPO source files",
    )
    args = parser.parse_args(argv)
    fetch_paleo_files(
        args.dest_dir,
        base_url=args.base_url,
        include_source=args.include_source_reconstructions,
    )


if __name__ == "__main__":
    main()
