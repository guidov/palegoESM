#!/usr/bin/env python
"""Upload palegoESM paleoclimate boundary-condition datasets to Hugging Face Hub.

Requires a Hugging Face token (``HF_API_KEY`` or ``HF_TOKEN``) with ``repo.write``
permission on the ``guidov`` namespace (configure at https://huggingface.co/settings/tokens).

Usage::

    python scripts/paleo/upload_paleo_data_hf.py
"""

from __future__ import annotations

import os
from pathlib import Path
from huggingface_hub import HfApi

HF_REPO_ID = os.environ.get("PALEGOESM_HF_REPO", "guidov/palegoESM-data")
PALEO_DATA_ROOT = Path("/home/guido/paleo_data")

FILES_TO_UPLOAD = (
    "README.md",
    "palegoESM/topo_pi_1850_0p25deg.nc",
    "palegoESM/topo_lgm_ice6gc_0p25deg.nc",
    "palegoESM/topo_lgm_ice6ga_0p25deg.nc",
    "palegoESM/topo_lgm_ice7gna_0p25deg.nc",
    "palegoESM/topo_lgm_ice5g_0p25deg.nc",
    "palegoESM/surfdata_fv1_lgm_ice6gc.nc",
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


def main() -> None:
    token = os.environ.get("HF_API_KEY") or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", private=False, exist_ok=True)
    print(f"Verified dataset repo: https://huggingface.co/datasets/{HF_REPO_ID}")

    for rel_path in FILES_TO_UPLOAD:
        local_path = (PALEO_DATA_ROOT / rel_path).resolve()
        if not local_path.exists():
            print(f"[skip] Missing {local_path}")
            continue
        size_mb = local_path.stat().st_size / 1e6
        print(f"[upload] {rel_path} ({size_mb:.2f} MB)...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=rel_path,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
        )
    print(f"Done! Dataset published at https://huggingface.co/datasets/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
