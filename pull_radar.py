"""
Collect BOM rain radar images for Sydney: the highest-resolution rainfall data
available for free (about 0.5 km, a new frame every 5 minutes).

The BOM publishes these on its public FTP server but keeps only about 2 hours
of frames, so Task Scheduler runs this every 30 minutes. Each run downloads any
frames not already saved.

Products (Terrey Hills radar):
    IDR713  128 km range, 512 x 512 px (~0.5 km per pixel): all of Greater Sydney,
            Blue Mountains and Central Coast
    IDR714   64 km range (~0.25 km per pixel): the Sydney basin in more detail

Saved to data/raw/weather/radar/<product>/<YYYY-MM-DD>/<YYYYMMDDHHMM>.png (UTC).
The images use a fixed colour scale for rain rate; analysis converts pixel
colours to mm/h and samples them along each rail segment. Map layers (rail,
locations, background) are saved once to data/raw/weather/radar/<product>/layers/.

Data: Bureau of Meteorology radar. Attribute the Bureau and check the terms at
bom.gov.au/other/copyright.shtml before publishing images.

Usage:
    python pull_radar.py
"""

import ftplib
import io
from datetime import datetime, timezone

from tfnsw import DATA_DIR, setup_logging

FTP_HOST = "ftp.bom.gov.au"
RADAR_DIR = "/anon/gen/radar"
LAYERS_DIR = "/anon/gen/radar_transparencies"
PRODUCTS = ["IDR713", "IDR714"]
LAYERS = ["background", "locations", "rail", "range", "topography", "waterways"]
OUT_DIR = DATA_DIR / "raw" / "weather" / "radar"

log = setup_logging("pull_radar")


def fetch(ftp: ftplib.FTP, remote: str) -> bytes:
    buf = io.BytesIO()
    ftp.retrbinary(f"RETR {remote}", buf.write)
    return buf.getvalue()


def main() -> None:
    with ftplib.FTP(FTP_HOST, timeout=60) as ftp:
        ftp.login()  # anonymous
        names = ftp.nlst(RADAR_DIR)
        names = [n.rsplit("/", 1)[-1] for n in names]
        saved = 0
        for product in PRODUCTS:
            # Frames look like IDR713.T.202610260204.png (time in UTC)
            for name in sorted(n for n in names if n.startswith(f"{product}.T.") and n.endswith(".png")):
                stamp = name.split(".")[2]
                day = datetime.strptime(stamp, "%Y%m%d%H%M").strftime("%Y-%m-%d")
                path = OUT_DIR / product / day / f"{stamp}.png"
                if path.exists():
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(fetch(ftp, f"{RADAR_DIR}/{name}"))
                saved += 1

            layer_dir = OUT_DIR / product / "layers"
            for layer in LAYERS:
                path = layer_dir / f"{layer}.png"
                if not path.exists():
                    layer_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        path.write_bytes(fetch(ftp, f"{LAYERS_DIR}/{product}.{layer}.png"))
                    except ftplib.error_perm:
                        log.warning("No %s layer for %s", layer, product)
    log.info("Radar: saved %d new frames at %s", saved, datetime.now(timezone.utc).strftime("%H:%M UTC"))


if __name__ == "__main__":
    main()
