# Paleoclimate Boundary Conditions in `palegoESM` (`scripts/paleo/`)

This directory contains tools for building and configuring paleoclimate (e.g. LGM 21 ka, Mid-Holocene 6 ka, MIS3 38 ka, and Pre-Industrial 0 ka control) boundary conditions in `palegoESM`, adapted from Guido Vettoretti's CESM1 paleoclimate boundary-condition workflow.

- **Reference CESM1 archive**: [`https://github.com/guidov/cesm1-paleo-bc`](https://github.com/guidov/cesm1-paleo-bc) (local clone in [`/home/guido/cesm1_paleo_bc/README.md`](file:///home/guido/cesm1_paleo_bc/README.md))
- **External dataset catalog**: [`https://huggingface.co/datasets/guidov/palegoESM-data`](https://huggingface.co/datasets/guidov/palegoESM-data) (local catalog in [`/home/guido/paleo_data/README.md`](file:///home/guido/paleo_data/README.md))

---

## Why legoESM Changes Paleoclimate Modeling

Paleoclimate simulations (such as modeling the Last Glacial Maximum (LGM) or Deep Time eras) historically suffer from massive uncertainties in boundary conditions, sparse proxy data, and manual parameter tuning. legoESM addresses these problems directly:

- **Gradient-Based Calibration**: Instead of manually "tuning" parameters (like cloud microphysics or soil properties) over years of trial-and-error, you can take the exact mathematical gradient (`jax.grad`) through the entire coupled model. This allows automated, gradient-based optimization to align the model directly with paleoclimate proxy data.
- **Seamless Data Assimilation**: Paleoclimate data assimilation (combining tree rings, ice cores, or sediment data with physical models) becomes exceptionally clean. Variational data assimilation is native to legoESM because the code calculates its own gradients.
- **Swappable Physics Bricks**: If a specific legacy vegetation or ice sheet scheme fails under deep-time conditions (a common issue in traditional models), legoESM allows you to cleanly snap in alternative parameterizations or AI-driven emulators without rebuilding the whole system.
- **Massive Hardware Acceleration**: Running centennial or millennial paleoclimate scales is computationally punishing. legoESM is built from day one to scale seamlessly on GPUs and TPUs, significantly accelerating long-duration runs.

---

## 1. How CESM1 Paleo Boundary Conditions Map to `palegoESM`

| Component | CESM1 Workflow (`/home/guido/cesm1_paleo_bc`) | `palegoESM` Equivalent |
|---|---|---|
| **Topography & Land-Sea Mask** | `datain/Convert_rosemarie_i6gc_data.txt`, `makelandmask_*.ncl`, and `atm_s4/*/definesurf` ($\text{ICE}_{21\text{k}} - \text{ICE}_{0\text{k}} + \text{ETOPO}_{0\text{k}}$) | [`build_paleo_topography.py`](file:///home/guido/palegoESM/scripts/paleo/build_paleo_topography.py) builds a regular lat-lon NetCDF consumed directly by [`load_real_topography`](file:///home/guido/palegoESM/packages/core/legoesm/grids/topography.py) (which computes `phis` and sub-grid `land_mask`). |
| **Ocean Bathymetry & Straits** | `ocn_s1/mk_grid_gx1_step1.csh` + interactive `pykmteditnc4_v3.py` + `pyregionfixnc4_v3.py` | [`BathymetryConfig`](file:///home/guido/palegoESM/packages/ocean/legoesm/ocean/bathymetry.py) loads the same paleo topography/bathymetry NetCDF (`source="file"`), applies Laplacian smoothing, enforces `H_min`, and carves open straits (while naturally closing subaerial LGM sills like Bering Strait / Torres Strait). |
| **Ocean Salinity IC** | `ocn_s1/TS_INIC/*init_ts_const_salinity.ncl` ($34.7\text{ PSU}$ PI vs. $35.7\text{ PSU}$ LGM) | Set reference / initial ocean salinity to `preset.ocean_salinity_mean_psu` ($35.7\text{ PSU}$ at LGM, $+1.0\text{ PSU}$ glacial ocean-volume effect). |
| **Land Surface (`surfdata`)** | `lnd_s3/*/mkraw_s3c/create_mksrf_{glacier,pft,soilcol}.py` + `mksurfdata_s3d` | [`build_paleo_clm_surfdata.py`](file:///home/guido/palegoESM/scripts/paleo/build_paleo_clm_surfdata.py) imprints the paleo ice-sheet mask (`PCT_GLACIER=100%`, `PFT[0]=100%` bare ground under ice) and extrapolates coastal PFTs/soil texture onto exposed continental shelves via periodic 3-D unit-sphere KD-tree nearest-neighbor search. Loaded via [`CLMSurfaceParamProvider`](file:///home/guido/palegoESM/packages/land/legoesm/land/clm_surface_map.py). |
| **Orbital Forcing** | `env_run.xml`: `ORBITAL_YEAR = -19050` (LGM) or `1850` (PI) | [`OrbitalParameters`](file:///home/guido/palegoESM/packages/atmosphere/legoesm/atmosphere/physics/radiation/solar.py) uses the exact Berger (1978) / CESM `shr_orb_mod` convention (`eccentricity`, `obliquity_deg`, `long_perihelion_deg` = `mvelp`). Presets live in [`paleo_presets.py`](file:///home/guido/palegoESM/scripts/paleo/paleo_presets.py). |
| **Greenhouse Gases** | `cam.buildnml.csh`: `co2vmr = 190e-6`, `ch4vmr = 3.5e-7`, `n2ovmr = 2.0e-7` | [`RadiationConfig`](file:///home/guido/palegoESM/packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py) fields `co2_ppmv`, `ch4_ppbv`, `n2o_ppbv` (and `forcing.co2_ppmv` for land carbon). |
| **Coupler Remapping** | `cpl_s2/` offline SCRIP + 64-thread SciNet runoff weight smoothing | Computed online in JAX by `legoesm.coupler` from the model grid masks. |

---

## 2. Quick Start

### Step 0: Download Pre-Built Boundary Conditions from Hugging Face
All compressed NetCDF-4 `palegoESM` boundary condition datasets and 10-minute source reconstructions are hosted publicly on Hugging Face at **[`https://huggingface.co/datasets/guidov/palegoESM-data`](https://huggingface.co/datasets/guidov/palegoESM-data)** (with automatic fallback to [GitHub Release `v0.1.0-data`](https://github.com/guidov/palegoESM/releases/tag/v0.1.0-data)):

```bash
# Download ready-to-run 0.25-deg palegoESM topography & LGM CLM surfdata (~74 MB total)
python scripts/paleo/download_paleo_data.py --dest-dir data/paleo

# Also download the 10-arc-minute ICE-5G / ICE-6G_A / ICE-6G_C / ICE-7G_NA / ETOPO source files
python scripts/paleo/download_paleo_data.py --dest-dir data/paleo --include-source-reconstructions
```

### Step 1: Generate `palegoESM`-ready LGM / PI Topography & Bathymetry
Build compressed NetCDF-4 (`zlib=True, complevel=4`) 0.25° (or any resolution) files in `/home/guido/paleo_data/palegoESM/` from the 10-minute `ICE-6G_C`, `ICE-6G_A`, `ICE-7G_NA`, `ICE-5G`, or `ETOPO` datasets:

```bash
# LGM (21 ka) presets (ICE-6G_C, ICE-6G_A, ICE-7G_NA, ICE-5G)
python scripts/paleo/build_paleo_topography.py --preset lgm_ice6gc --resolution-deg 0.25
python scripts/paleo/build_paleo_topography.py --preset lgm_ice6ga --resolution-deg 0.25
python scripts/paleo/build_paleo_topography.py --preset lgm_ice7gna --resolution-deg 0.25
python scripts/paleo/build_paleo_topography.py --preset lgm_ice5g --resolution-deg 0.25

# LGM (21 ka) with 0.66x scaled ice-sheet elevation (as in reduce_LGM_topo_by_point66.ncl)
python scripts/paleo/build_paleo_topography.py --preset lgm_ice6ga --ice-topo-scale 0.66 \
    --out /home/guido/paleo_data/palegoESM/topo_lgm_ice6ga_scale0p66_0p25deg.nc

# Pre-Industrial (0 ka) ETOPO control
python scripts/paleo/build_paleo_topography.py --preset pi_1850 --resolution-deg 0.25
```

### Step 2: Adapt a CLM `surfdata` File for LGM Ice Sheets & Exposed Shelves
```bash
python scripts/paleo/build_paleo_clm_surfdata.py \
    --base-surfdata /home/guido/paleo_data/cesm1_bcs/lnd/surfdata_fv1_PI-CTRL_2013-05-08.nc \
    --paleo-topo /home/guido/paleo_data/palegoESM/topo_lgm_ice6gc_0p25deg.nc \
    --out /home/guido/paleo_data/palegoESM/surfdata_fv1_lgm_ice6gc.nc
```

### Step 3: Publish Updated Datasets to Hugging Face
If you generate new boundary conditions in `/home/guido/paleo_data/palegoESM/`, upload them to [`guidov/palegoESM-data`](https://huggingface.co/datasets/guidov/palegoESM-data):
```bash
python scripts/paleo/upload_paleo_data_hf.py
```

### Step 4: Use in Python / Model Config
```python
from scripts.paleo.paleo_presets import get_preset
from legoesm.atmosphere.physics.radiation.solar import OrbitalParameters

preset = get_preset("lgm_ice6gc")
orbit = OrbitalParameters(
    eccentricity=preset.eccentricity,
    obliquity_deg=preset.obliquity_deg,
    long_perihelion_deg=preset.long_perihelion_deg,
)
```

---

## 3. Repository & Upstream Sync Workflow

`/home/guido/palegoESM` is a Git worktree on branch `paleo` tracking two remotes:
- `palego`: `git@github.com:guidov/palegoESM.git` (pushes `paleo` to `main`)
- `origin`: `https://github.com/climate-federation/legoESM.git` (upstream `legoESM`)

To pull the latest upstream `legoESM` changes into `palegoESM` and push to GitHub:
```bash
cd /home/guido/palegoESM
git fetch origin main
git merge origin/main
git push palego paleo:main
```
