#!/usr/bin/env python
"""Adapt a CLM surface dataset (surfdata) for paleoclimate ice sheets & sea level.

Ports Guido Vettoretti's CESM1 ``lnd_s3/mkraw_s3c`` pipeline
(``create_mksrf_glacier.py``, ``create_mksrf_pft.py``, ``create_mksrf_soilcol.py``
in ``/home/guido/cesm1_paleo_bc/cesm1/lnd_s3/cesmi6gafv1/mkraw_s3c/``) to modify
a CLM ``surfdata`` file directly for :class:`legoesm.land.clm_surface_map.CLMSurfaceParamProvider`:

1. **Glacier imprint (`PCT_GLACIER`)**:
   Where the paleoclimate ice-sheet reconstruction has ``ice_mask > 0`` (e.g.
   Laurentide, Fennoscandian, Cordilleran, Patagonian ice sheets), sets
   ``PCT_GLACIER = 100%`` (or the fractional ice coverage) and sets natural PFTs
   on that cell to 100% bare ground (PFT 0), zeroing vegetated PFTs 1..16.
2. **Newly exposed continental shelves (e.g. Sunda, Sahul, Beringia at LGM)**:
   Where paleoclimate sea-level lowering exposes new land (``paleo_land_mask == 1``
   where the modern ``surfdata`` had ocean / zero land cover), extrapolates
   ``PCT_NAT_PFT`` (or ``PCT_PFT``) and soil texture (``PCT_SAND``, ``PCT_CLAY``)
   from the nearest coastal land cells using periodic longitude-wrapped nearest-
   neighbor filling (vectorized KD-tree equivalent of ``nnapply`` in
   ``create_mksrf_pft.py``).
3. **PFT normalization**:
   Ensures ``PCT_NAT_PFT`` (or ``PCT_PFT``) sums to exactly 100% over all land cells,
   with any residual assigned to PFT 0 (bare ground).

Usage::

    python scripts/paleo/build_paleo_clm_surfdata.py \\
        --base-surfdata <modern_surfdata.nc> \\
        --paleo-topo /home/guido/paleo_data/palegoESM/topo_lgm_ice6gc_0p25deg.nc \\
        --out /home/guido/paleo_data/palegoESM/surfdata_lgm_ice6gc.nc
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import xarray as xr


def _lon_lat_to_unit_sphere(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Convert (lon, lat) [deg] to 3-D unit-sphere coordinates for periodic KD-tree search."""
    lat_r = np.deg2rad(lat_deg)
    lon_r = np.deg2rad(lon_deg)
    cos_lat = np.cos(lat_r)
    return np.column_stack([cos_lat * np.cos(lon_r), cos_lat * np.sin(lon_r), np.sin(lat_r)])


def adapt_clm_surfdata_to_paleo(
    ds_surf: xr.Dataset,
    ds_paleo_topo: xr.Dataset,
) -> xr.Dataset:
    """Return a copy of ``ds_surf`` updated with paleo land mask, ice sheets, and shelf PFTs."""
    ds = ds_surf.copy(deep=True)

    # Extract target CLM 1-D lat/lon coordinates from LATIXY / LONGXY
    lat_1d = np.asarray(ds["LATIXY"].values[:, 0], dtype=np.float64)
    lon_1d = np.asarray(ds["LONGXY"].values[0, :], dtype=np.float64) % 360.0

    # Area/bilinearly coarsen paleo land_mask and ice_mask onto the CLM surfdata grid
    paleo_land = (
        ds_paleo_topo["land_mask"]
        .interp(lat=lat_1d, lon=lon_1d, method="linear")
        .values
    )
    paleo_ice = (
        ds_paleo_topo["ice_mask"]
        .interp(lat=lat_1d, lon=lon_1d, method="linear")
        .values
    )
    paleo_land = np.clip(np.nan_to_num(paleo_land, nan=0.0), 0.0, 1.0)
    paleo_ice = np.clip(np.nan_to_num(paleo_ice, nan=0.0), 0.0, 1.0)

    is_paleo_land = paleo_land > 0.1
    is_glaciated = paleo_ice > 0.5

    # Identify the PFT variable name (CLM5: PCT_NAT_PFT; CLM4/CESM1: PCT_PFT)
    pft_var = "PCT_NAT_PFT" if "PCT_NAT_PFT" in ds.data_vars else "PCT_PFT"
    pft = np.asarray(ds[pft_var].values, dtype=np.float64).copy()  # (n_pft, nlat, nlon)

    # Identify existing valid modern land cells (where PFTs sum > 0 and not 100% glacier)
    old_glacier = (
        np.asarray(ds["PCT_GLACIER"].values, dtype=np.float64)
        if "PCT_GLACIER" in ds.data_vars
        else np.zeros_like(paleo_land)
    )
    old_pft_sum = pft.sum(axis=0)
    valid_donor = (old_pft_sum > 50.0) & (old_glacier < 90.0) & (~is_glaciated)

    # Nearest-neighbor extrapolation onto newly exposed continental shelves
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    donor_xyz = _lon_lat_to_unit_sphere(lon_2d[valid_donor], lat_2d[valid_donor])
    tree = cKDTree(donor_xyz)

    new_shelf = is_paleo_land & (~is_glaciated) & (old_pft_sum < 10.0)
    if np.any(new_shelf):
        shelf_xyz = _lon_lat_to_unit_sphere(lon_2d[new_shelf], lat_2d[new_shelf])
        _, nn_idx = tree.query(shelf_xyz, k=1)
        donor_y, donor_x = np.where(valid_donor)
        src_y = donor_y[nn_idx]
        src_x = donor_x[nn_idx]
        shelf_y, shelf_x = np.where(new_shelf)

        pft[:, shelf_y, shelf_x] = pft[:, src_y, src_x]

        for soil_var in ("PCT_SAND", "PCT_CLAY"):
            if soil_var in ds.data_vars:
                arr = np.asarray(ds[soil_var].values, dtype=np.float64).copy()
                arr[:, shelf_y, shelf_x] = arr[:, src_y, src_x]
                ds[soil_var].values = arr.astype(ds[soil_var].dtype)

    # Apply LGM / paleo glacier mask: 100% bare ground (PFT 0) on glaciated cells
    pft[0, is_glaciated] = 100.0
    pft[1:, is_glaciated] = 0.0

    # Zero out PFTs on paleo ocean cells (set PFT 0 = 100% if CLM requires 100% everywhere)
    pft[0, ~is_paleo_land] = 100.0
    pft[1:, ~is_paleo_land] = 0.0

    # Normalize PFTs on land so sum == 100.0 everywhere
    total = pft.sum(axis=0)
    over = total > 100.0
    if np.any(over):
        pft[:, over] = pft[:, over] * (100.0 / total[over])[None, :]
    total = pft.sum(axis=0)
    under = total < 100.0
    if np.any(under):
        pft[0, under] += 100.0 - total[under]

    ds[pft_var].values = pft.astype(ds[pft_var].dtype)

    # Update PCT_GLACIER, PCT_CROP, PCT_NATVEG, PFTDATA_MASK, LANDFRAC_PFT if present
    if "PCT_GLACIER" in ds.data_vars:
        new_glacier = np.where(is_glaciated, 100.0, np.where(new_shelf, 0.0, old_glacier))
        new_glacier = np.where(is_paleo_land, new_glacier, 0.0)
        ds["PCT_GLACIER"].values = new_glacier.astype(ds["PCT_GLACIER"].dtype)

    if "PCT_CROP" in ds.data_vars:
        # No industrial agriculture in paleoclimate
        ds["PCT_CROP"].values = np.zeros_like(ds["PCT_CROP"].values)

    if "PCT_NATVEG" in ds.data_vars and "PCT_GLACIER" in ds.data_vars:
        glac = np.asarray(ds["PCT_GLACIER"].values, dtype=np.float64)
        lake = (
            np.asarray(ds["PCT_LAKE"].values, dtype=np.float64)
            if "PCT_LAKE" in ds.data_vars
            else 0.0
        )
        lake = np.where(is_glaciated, 0.0, lake)
        if "PCT_LAKE" in ds.data_vars:
            ds["PCT_LAKE"].values = lake.astype(ds["PCT_LAKE"].dtype)
        natveg = np.where(is_paleo_land, np.clip(100.0 - glac - lake, 0.0, 100.0), 0.0)
        ds["PCT_NATVEG"].values = natveg.astype(ds["PCT_NATVEG"].dtype)

    if "PFTDATA_MASK" in ds.data_vars:
        ds["PFTDATA_MASK"].values = is_paleo_land.astype(ds["PFTDATA_MASK"].dtype)

    if "LANDFRAC_PFT" in ds.data_vars:
        ds["LANDFRAC_PFT"].values = paleo_land.astype(ds["LANDFRAC_PFT"].dtype)

    ds.attrs["paleo_adapted_history"] = (
        f"{datetime.now(timezone.utc).isoformat()}: adapted by "
        "scripts/paleo/build_paleo_clm_surfdata.py"
    )
    return ds


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-surfdata",
        type=Path,
        required=True,
        help="Input CLM surfdata NetCDF file",
    )
    parser.add_argument(
        "--paleo-topo",
        type=Path,
        required=True,
        help="Paleo topography NetCDF generated by build_paleo_topography.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output paleo CLM surfdata NetCDF path",
    )
    args = parser.parse_args(argv)

    ds_surf = xr.open_dataset(args.base_surfdata)
    ds_topo = xr.open_dataset(args.paleo_topo)
    ds_out = adapt_clm_surfdata_to_paleo(ds_surf, ds_topo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        var: {"zlib": True, "complevel": 4, "shuffle": True}
        for var in ds_out.data_vars
        if ds_out[var].dtype.kind in ("f", "i", "u") and ds_out[var].ndim >= 1
    }
    ds_out.to_netcdf(args.out, format="NETCDF4", encoding=encoding)
    print(f"Wrote compressed paleo CLM surfdata to {args.out}")


if __name__ == "__main__":
    main()
