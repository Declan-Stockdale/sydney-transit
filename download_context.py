"""
One-off downloads of context data that sharpen the equity and demand analyses.

Static files; rerun only when a new release comes out (noted per file).
Saved to data/raw/context/:

    seifa_2021_sa2.xlsx
        ABS SEIFA 2021 by SA2: socio-economic disadvantage indexes (IRSD etc.).
        Next release with the 2026 Census, ~2028.
    erp_asgs2021.csv
        ABS Estimated Resident Population 2021 to the latest year (2025 at
        time of writing), every ASGS level including SA2, from the ABS data
        API. Updated each year around March.
    tzp24_population_by_tz.csv / tzp24_employment_by_tz.csv
        TfNSW Travel Zone Projections 2024: population and jobs per travel zone,
        2021 to 2066, in 5-year steps.
    tz21_shp.zip / tz21_concordances.zip
        Travel zone boundaries and their concordance to SA2s, to map TZP24.
    2021_WPP_SA2_for_NSW_short-header.zip
        2021 Census Working Population Profile: jobs counted at their place
        of work, by SA2. Used for access to jobs. (Origin-destination journey
        to work is only available through ABS TableBuilder, which needs a
        login, and TfNSW no longer publishes it.)
    station_entrances_2020.csv
        TfNSW train station entrance points, for walking distances to the
        actual way in rather than the station centre.

Households without a car come from table G34 of the Census DataPack that
download_census.py already fetched.

Sources: ABS (CC BY 4.0), Transport for NSW Open Data (CC BY 4.0).

Usage:
    python download_context.py
"""

import requests

from tfnsw import DATA_DIR, setup_logging  # tfnsw also sets up Windows HTTPS certificates

OUT_DIR = DATA_DIR / "raw" / "context"
TFNSW = "https://opendata.transport.nsw.gov.au/data/dataset"
FILES = {
    "seifa_2021_sa2.xlsx":
        "https://www.abs.gov.au/statistics/people/people-and-communities/socio-economic-indexes-areas-seifa-australia/2021/Statistical%20Area%20Level%202%2C%20Indexes%2C%20SEIFA%202021.xlsx",
    "erp_asgs2021.csv":
        "https://data.api.abs.gov.au/rest/data/ABS,ABS_ANNUAL_ERP_ASGS2021,1.2.0/all?startPeriod=2021",
    "tzp24_population_by_tz.csv":
        f"{TFNSW}/e35da583-c645-4e15-a7ca-ea6678ae2699/resource/62569f7f-53d4-4922-86be-e6f13e108777/download/tzp24-popd-by-age_sex-by-tz-2021-2066.csv",
    "tzp24_employment_by_tz.csv":
        f"{TFNSW}/65041f3f-c9bd-4731-b32b-05691272537e/resource/b1f00271-e244-458f-aad3-63a6fb4898f2/download/tzp24-employment-by-tz-by-industry-2021-2066.csv",
    "tz21_shp.zip":
        f"{TFNSW}/fd93096c-c7e2-4f33-9b98-264cbd75d9ce/resource/2f665f7e-e1d4-4caf-95a2-78f6795865a3/download/tz21_tfnsw_shp.zip",
    "tz21_concordances.zip":
        f"{TFNSW}/fd93096c-c7e2-4f33-9b98-264cbd75d9ce/resource/cff87607-f6fc-4e57-b58d-0ea6f59d8f75/download/tz21_concordances.zip",
    "2021_WPP_SA2_for_NSW_short-header.zip":
        "https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_WPP_SA2_for_NSW_short-header.zip",
    "station_entrances_2020.csv":
        f"{TFNSW}/3e875da4-5090-499f-9a1e-a566b6ef3559/resource/257179b0-aa8c-4172-8d76-9c77a1941a68/download/stationentrances2020_v4.csv",
}
# The ABS data API returns CSV only when asked for it in the Accept header
HEADERS = {"erp_asgs2021.csv": {"Accept": "application/vnd.sdmx.data+csv;labels=both"}}

log = setup_logging("download_context")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "sydney-transit portfolio project"
    for name, url in FILES.items():
        dest = OUT_DIR / name
        with session.get(url, headers=HEADERS.get(name, {}), stream=True, timeout=600) as resp:
            resp.raise_for_status()
            tmp = dest.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        tmp.replace(dest)
        log.info("Saved %s (%.1f MB)", name, dest.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
