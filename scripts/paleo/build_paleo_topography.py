#!/usr/bin/env python
"""Build palegoESM-ready topography, bathymetry, land-mask, and ice-mask NetCDF files.

Translates the CESM1 paleoclimate topography workflow (see
``/home/guido/cesm1_paleo_bc/cesm1/datain/Convert_rosemarie_i6gc_data.txt`` and
``makelandmask_i6gc_kevin_10min.ncl``) into the unified NetCDF layout expected by:

1. :func:`legoesm.grids.topography.load_real_topography` (atmosphere surface
   geopotential ``phis`` and sub-grid fractional ``land_mask``).
2. :class:`legoesm.ocean.bathymetry.BathymetryConfig` (ocean depth, land-sea mask,
   and critical strait enforcement).

Supports:
- Any preset in :mod:`paleo_presets` (``pi_1850``, ``lgm_ice6gc``, ``lgm_ice6ga``,
  ``lgm_ice7gna``, ``lgm_ice5g``) or custom 10-min / 0.5-deg ``topo`` + ``etopo``
  files in ``/home/guido/paleo_data/ice_sheets_and_topo/``.
- Optional ice-sheet topography scaling factor ``--ice-topo-scale`` (e.g. ``0.66`` as
  in ``reduce_LGM_topo_by_point66.ncl``).
- Optional filling of Eurasian proglacial inland basins (as in
  ``create_mksrf_glacier.py``).

Usage::

    python scripts/paleo/build_paleo_topography.py --preset lgm_ice6gc \\
        --resolution-deg 0.25 \\
        --out /home/guido/paleo_data/palegoESM/topo_lgm_ice6gc_0p25deg.nc
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paleo_presets import PALEO_DATA_ROOT, PALEO_PRESETS, get_preset


def _normalize_lat_lon(ds: xr.Dataset, lat_name: str, lon_name: str) -> xr.Dataset:
    """Ensure South-to-North latitude [-90, 90] and ascending longitude [0, 360)."""
    if lat_name != "lat" or lon_name != "lon":
        ds = ds.rename({lat_name: "lat", lon_name: "lon"})
    ds = ds.assign_coords(lon=(ds["lon"] % 360.0))
    _, uniq_lon = np.unique(np.round(ds["lon"].values, 6), return_index=True)
    ds = ds.isel(lon=np.sort(uniq_lon)).sortby("lon")
    _, uniq_lat = np.unique(np.round(ds["lat"].values, 6), return_index=True)
    ds = ds.isel(lat=np.sort(uniq_lat)).sortby("lat")
    return ds


def build_paleo_topography_dataset(
    topo_path: Path,
    *,
    icesheet_path: Path | None = None,
    pi_topo_path: Path | None = None,
    ice_topo_scale: float = 1.0,
    target_res_deg: float = 0.25,
    fill_eurasian_proglacial_basins: bool = True,
    preset_name: str = "custom",
) -> xr.Dataset:
    """Construct a regular lat-lon topography + bathymetry + ice-mask Dataset."""
    ds_topo = xr.open_dataset(topo_path)
    lat_var = "lat" if "lat" in ds_topo.coords or "lat" in ds_topo.variables else "Lat"
    lon_var = "lon" if "lon" in ds_topo.coords or "lon" in ds_topo.variables else "Lon"
    ds_topo = _normalize_lat_lon(ds_topo, lat_var, lon_var)

    for cand in ("topo", "elevation", "orog", "Topo", "Orog"):
        if cand in ds_topo.data_vars:
            elev_da = ds_topo[cand].astype(np.float64)
            break
    else:
        raise KeyError(f"No elevation variable found in {topo_path}; vars={list(ds_topo.data_vars)}")

    # Optional scaling of glacial positive anomaly over land (reduce_LGM_topo_by_point66.ncl)
    if ice_topo_scale != 1.0:
        if pi_topo_path is None or not pi_topo_path.exists():
            raise ValueError("--ice-topo-scale != 1.0 requires a valid --pi-topo file")
        ds_pi = xr.open_dataset(pi_topo_path)
        la_pi = "lat" if "lat" in ds_pi.variables else "Lat"
        lo_pi = "lon" if "lon" in ds_pi.variables else "Lon"
        ds_pi = _normalize_lat_lon(ds_pi, la_pi, lo_pi)
        pi_var = "topo" if "topo" in ds_pi.data_vars else list(ds_pi.data_vars)[0]
        pi_elev = ds_pi[pi_var].astype(np.float64).interp(lat=ds_topo["lat"], lon=ds_topo["lon"])
        anom = elev_da - pi_elev
        elev_da = xr.where((elev_da > 0.0) & (anom > 0.0), pi_elev + ice_topo_scale * anom, elev_da)

    # Target regular grid in [0, 360) x (-90, 90)
    n_lat = int(round(180.0 / target_res_deg))
    n_lon = int(round(360.0 / target_res_deg))
    dlat = 180.0 / n_lat
    dlon = 360.0 / n_lon
    target_lat = np.linspace(-90.0 + 0.5 * dlat, 90.0 - 0.5 * dlat, n_lat)
    target_lon = np.linspace(0.5 * dlon, 360.0 - 0.5 * dlon, n_lon)

    elev_reg = elev_da.interp(lat=target_lat, lon=target_lon, method="linear")
    elev_vals = np.nan_to_num(elev_reg.values, nan=0.0)

    land_mask = (elev_vals >= 0.0).astype(np.float32)

    # Fill proglacial inland depressions south/east of the Eurasian ice sheet
    # (ported from create_mksrf_glacier.py lines 135-144: lat [55N, 77.5N] lon [52.5E, 92.5E]
    # and lat [7.5N, 22.5N] lon [-2.5E, 10E])
    if fill_eurasian_proglacial_basins:
        lat_2d, lon_2d = np.meshgrid(target_lat, target_lon, indexing="ij")
        box1 = (lat_2d >= 67.5) & (lat_2d <= 77.5) & (lon_2d >= 52.5) & (lon_2d <= 92.5)
        if np.any(box1 & (elev_vals < 0.0)):
            land_mask = np.where(box1, 1.0, land_mask).astype(np.float32)
            elev_vals = np.where(box1 & (elev_vals < 0.0), 1.0, elev_vals)

    # Ice mask and ice thickness if available
    ice_mask = np.zeros_like(land_mask, dtype=np.float32)
    ice_thickness = np.zeros_like(land_mask, dtype=np.float32)
    if icesheet_path is not None and icesheet_path.exists():
        ds_ice = xr.open_dataset(icesheet_path)
        la_i = "lat" if "lat" in ds_ice.variables else "Lat"
        lo_i = "lon" if "lon" in ds_ice.variables else "Lon"
        ds_ice = _normalize_lat_lon(ds_ice, la_i, lo_i)
        if "Icemask" in ds_ice.data_vars:
            im = ds_ice["Icemask"].astype(np.float64).interp(
                lat=target_lat, lon=target_lon, method="nearest"
            )
            ice_mask = (np.nan_to_num(im.values, nan=0.0) > 0.5).astype(np.float32)
        if "IceT" in ds_ice.data_vars:
            it = ds_ice["IceT"].astype(np.float64).interp(
                lat=target_lat, lon=target_lon, method="linear"
            )
            ice_thickness = np.maximum(np.nan_to_num(it.values, nan=0.0), 0.0).astype(np.float32)

    out = xr.Dataset(
        data_vars={
            "elevation": (
                ("lat", "lon"),
                elev_vals.astype(np.float32),
                {
                    "units": "m",
                    "long_name": "Surface elevation above sea level (negative for ocean depth)",
                },
            ),
            "htopo": (
                ("lat", "lon"),
                np.maximum(elev_vals, 0.0).astype(np.float32),
                {"units": "m", "long_name": "Land surface elevation (>= 0)"},
            ),
            "land_mask": (
                ("lat", "lon"),
                land_mask,
                {"units": "1", "long_name": "Binary land mask (1=land, 0=ocean)"},
            ),
            "ice_mask": (
                ("lat", "lon"),
                ice_mask,
                {"units": "1", "long_name": "Glacier / ice-sheet mask (1=ice, 0=ice-free)"},
            ),
            "ice_thickness": (
                ("lat", "lon"),
                ice_thickness,
                {"units": "m", "long_name": "Ice-sheet thickness"},
            ),
        },
        coords={
            "lat": ("lat", target_lat.astype(np.float64), {"units": "degrees_north"}),
            "lon": ("lon", target_lon.astype(np.float64), {"units": "degrees_east"}),
        },
        attrs={
            "title": f"palegoESM topography, bathymetry, and ice-sheet mask ({preset_name})",
            "source_topography": str(topo_path),
            "source_icesheet": str(icesheet_path) if icesheet_path else "none",
            "ice_topo_scale": float(ice_topo_scale),
            "history": f"{datetime.now(timezone.utc).isoformat()}: generated by scripts/paleo/build_paleo_topography.py",
        },
    )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=sorted(PALEO_PRESETS),
        default="lgm_ice6gc",
        help="Paleoclimate preset from paleo_presets.py (default: lgm_ice6gc)",
    )
    parser.add_argument(
        "--topo-file",
        type=Path,
        default=None,
        help="Override path to 10-min or 0.5-deg source topography NetCDF",
    )
    parser.add_argument(
        "--icesheet-file",
        type=Path,
        default=None,
        help="Override path to source ice-sheet NetCDF (with Icemask / IceT)",
    )
    parser.add_argument(
        "--ice-topo-scale",
        type=float,
        default=1.0,
        help="Scale factor for glacial elevation anomaly over land (e.g. 0.66)",
    )
    parser.add_argument(
        "--resolution-deg",
        type=float,
        default=0.25,
        help="Output lat-lon grid resolution in degrees (default: 0.25)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output NetCDF path (default: /home/guido/paleo_data/palegoESM/topo_<preset>_<res>deg.nc)",
    )
    args = parser.parse_args(argv)

    preset = get_preset(args.preset)
    topo_dir = PALEO_DATA_ROOT / "ice_sheets_and_topo"
    topo_path = args.topo_file or (topo_dir / preset.topo_10min_filename)
    icesheet_path = args.icesheet_file or (
        (topo_dir / preset.icesheet_filename) if preset.icesheet_filename else None
    )
    pi_topo_path = topo_dir / PALEO_PRESETS["pi_1850"].topo_10min_filename

    if args.out is None:
        res_tag = f"{args.resolution_deg:g}".replace(".", "p")
        out_path = PALEO_DATA_ROOT / "palegoESM" / f"topo_{preset.name}_{res_tag}deg.nc"
    else:
        out_path = args.out

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out = build_paleo_topography_dataset(
        topo_path,
        icesheet_path=icesheet_path,
        pi_topo_path=pi_topo_path,
        ice_topo_scale=args.ice_topo_scale,
        target_res_deg=args.resolution_deg,
        preset_name=preset.name,
    )
    encoding = {
        var: {"zlib": True, "complevel": 4, "shuffle": True}
        for var in ds_out.data_vars
    }
    ds_out.to_netcdf(out_path, format="NETCDF4", encoding=encoding)
    land_frac = float(ds_out["land_mask"].mean().values)
    ice_frac = float(ds_out["ice_mask"].mean().values)
    print(
        f"Wrote {out_path} ({dict(ds_out.sizes)}), "
        f"unweighted land_frac={land_frac:.4f}, ice_frac={ice_frac:.4f}"
    )


if __name__ == "__main__":
    main()
