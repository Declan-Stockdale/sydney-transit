"""
One-off download of the ABS Census data used for the service equity analysis.

These are static files, not an API, so there is nothing to schedule. Run this
once (and again only when a newer Census is released; 2026 Census results are
due from mid 2027).

Downloads into data/raw/census/ (skipping files already there):
    2021_GCP_SA2_for_NSW_short-header.zip
        2021 Census General Community Profile for every NSW Statistical Area
        Level 2 (SA2). Table G01 has usual resident population.
    SA2_2021_AUST_SHP_GDA2020.zip
        ASGS Edition 3 SA2 boundaries (shapefile). Gives each SA2's shape and
        area, and which ones are in Greater Sydney (GCC_CODE21 = "1GSYD").
    MB_2021_AUST_SHP_GDA2020.zip and Mesh_Block_Counts_2021.xlsx
        Mesh block boundaries and their usual resident counts.

The ABS does not need an API key.

Usage:
    python download_census.py
"""

from pathlib import Path

import requests

from tfnsw import DATA_DIR, setup_logging  # tfnsw also sets up Windows HTTPS certificates

CENSUS_DIR = DATA_DIR / "raw" / "census"
ASGS = (
    "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3"
    "/jul2021-jun2026/access-and-downloads/digital-boundary-files"
)
FILES = {
    "2021_GCP_SA2_for_NSW_short-header.zip":
        "https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_SA2_for_NSW_short-header.zip",
    "SA2_2021_AUST_SHP_GDA2020.zip": f"{ASGS}/SA2_2021_AUST_SHP_GDA2020.zip",
    # Mesh blocks are the smallest Census unit (roughly a street block). Their
    # resident counts let us measure how many people live far from a station.
    "MB_2021_AUST_SHP_GDA2020.zip": f"{ASGS}/MB_2021_AUST_SHP_GDA2020.zip",
    "Mesh_Block_Counts_2021.xlsx":
        "https://www.abs.gov.au/census/guide-census-data/mesh-block-counts/2021/Mesh%20Block%20Counts%2C%202021.xlsx",
}

log = setup_logging("download_census")


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.replace(dest)


def main() -> None:
    CENSUS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = CENSUS_DIR / name
        if dest.exists():
            log.info("%s already downloaded, skipping", name)
            continue
        log.info("Downloading %s", name)
        download(url, dest)
        log.info("Saved %s (%.1f MB)", dest, dest.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
