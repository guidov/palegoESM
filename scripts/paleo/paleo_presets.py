"""Paleoclimate epoch presets (orbital elements, GHGs, salinity, and dataset paths).

Follows PMIP3/PMIP4 and Guido Vettoretti's CESM1.0.5 glacial setup conventions
(see ``/home/guido/cesm1_paleo_bc/README.md`` and ``lgm_setup_2024/Readme``).

Orbital parameters use the Berger (1978) / CESM ``shr_orb_mod`` convention expected by
:class:`legoesm.atmosphere.physics.radiation.solar.OrbitalParameters`:
``long_perihelion_deg`` is ``mvelp`` (moving vernal equinox longitude of perihelion,
i.e. ``omega_bar + 180°``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

PALEO_DATA_ROOT = Path("/home/guido/paleo_data")


@dataclass(frozen=True)
class PaleoEpochPreset:
    """Complete paleoclimate boundary-condition specification for an epoch."""

    name: str
    description: str
    age_ka_bp: float
    cesm_orbital_year: int
    # Berger (1978) / CESM shr_orb_mod orbital elements
    eccentricity: float
    obliquity_deg: float
    long_perihelion_deg: float  # mvelp [deg], measured from moving vernal equinox
    # Well-mixed greenhouse gases
    co2_ppmv: float
    ch4_ppbv: float
    n2o_ppbv: float
    # Ocean initial / reference salinity offset relative to PI (34.7 PSU)
    ocean_salinity_mean_psu: float
    # Default 10-min combined (ICE_21k - ICE_0k + ETOPO) relief file in paleo_data
    topo_10min_filename: str
    # Optional full ice-sheet file containing IceT / Icemask
    icesheet_filename: str = ""


PALEO_PRESETS: Dict[str, PaleoEpochPreset] = {
    "pi_1850": PaleoEpochPreset(
        name="pi_1850",
        description="Pre-Industrial Control (1850 CE / 0 ka BP, CESM1 cesmpifv1/cesmpit31)",
        age_ka_bp=0.0,
        cesm_orbital_year=1850,
        eccentricity=0.016724,
        obliquity_deg=23.446,
        long_perihelion_deg=282.04,
        co2_ppmv=284.7,
        ch4_ppbv=791.0,
        n2o_ppbv=275.0,
        ocean_salinity_mean_psu=34.7,
        topo_10min_filename="etopo_10min_topo_v2.nc",
        icesheet_filename="orog_b_icetm_10min.0_StoN.nc",
    ),
    "mh_6ka": PaleoEpochPreset(
        name="mh_6ka",
        description="Mid-Holocene (6 ka BP, PMIP4)",
        age_ka_bp=6.0,
        cesm_orbital_year=-4050,
        eccentricity=0.018682,
        obliquity_deg=24.105,
        long_perihelion_deg=180.87,
        co2_ppmv=264.4,
        ch4_ppbv=597.0,
        n2o_ppbv=262.0,
        ocean_salinity_mean_psu=34.7,
        topo_10min_filename="etopo_10min_topo_v2.nc",
        icesheet_filename="orog_b_icetm_10min.0_StoN.nc",
    ),
    "lgm_ice6gc": PaleoEpochPreset(
        name="lgm_ice6gc",
        description="Last Glacial Maximum (21 ka BP, ICE-6G_C VM5a + ETOPO, PMIP4)",
        age_ka_bp=21.0,
        cesm_orbital_year=-19050,
        eccentricity=0.018994,
        obliquity_deg=22.949,
        long_perihelion_deg=294.42,
        co2_ppmv=190.0,
        ch4_ppbv=350.0,
        n2o_ppbv=200.0,
        ocean_salinity_mean_psu=35.7,
        topo_10min_filename="i6gcocnx1_21k-0k+etopo_10min_topo_v2_kevin.nc",
        icesheet_filename="orog_b_icetm_10min.21_StoN.nc",
    ),
    "lgm_ice6ga": PaleoEpochPreset(
        name="lgm_ice6ga",
        description="Last Glacial Maximum (21 ka BP, ICE-6G_A Interim VM5a, CESM1 cesmi6gafv1/t31)",
        age_ka_bp=21.0,
        cesm_orbital_year=-19050,
        eccentricity=0.018994,
        obliquity_deg=22.949,
        long_perihelion_deg=294.42,
        co2_ppmv=185.0,
        ch4_ppbv=350.0,
        n2o_ppbv=200.0,
        ocean_salinity_mean_psu=35.7,
        topo_10min_filename="i6gaocnx1_21k-0k+etopo_10min_topo_v2.nc",
        icesheet_filename="I6_Interim_VM5a_orog_b_icetm_0.5.21k_shift.nc",
    ),
    "lgm_ice7gna": PaleoEpochPreset(
        name="lgm_ice7gna",
        description="Last Glacial Maximum (21 ka BP, ICE-7G_NA Kevin + ETOPO)",
        age_ka_bp=21.0,
        cesm_orbital_year=-19050,
        eccentricity=0.018994,
        obliquity_deg=22.949,
        long_perihelion_deg=294.42,
        co2_ppmv=190.0,
        ch4_ppbv=350.0,
        n2o_ppbv=200.0,
        ocean_salinity_mean_psu=35.7,
        topo_10min_filename="i7gnaocnx1_21k-0k+etopo_10min_topo_v2_kevin.nc",
        icesheet_filename="orog_b_icetm_10min.21_StoN.nc",
    ),
    "lgm_ice5g": PaleoEpochPreset(
        name="lgm_ice5g",
        description="Last Glacial Maximum (21 ka BP, ICE-5G v1.2 + ETOPO, PMIP3)",
        age_ka_bp=21.0,
        cesm_orbital_year=-19050,
        eccentricity=0.018994,
        obliquity_deg=22.949,
        long_perihelion_deg=294.42,
        co2_ppmv=185.0,
        ch4_ppbv=350.0,
        n2o_ppbv=200.0,
        ocean_salinity_mean_psu=35.7,
        topo_10min_filename="i5gocnx1_21k-0k+etopo_10min_topo_v2.nc",
        icesheet_filename="ice5g_v1.2_21.0k_10min.nc",
    ),
}


def get_preset(name: str) -> PaleoEpochPreset:
    if name not in PALEO_PRESETS:
        raise KeyError(
            f"Unknown paleo preset {name!r}; available: {sorted(PALEO_PRESETS)}"
        )
    return PALEO_PRESETS[name]
