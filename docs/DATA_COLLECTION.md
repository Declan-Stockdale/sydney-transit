# Data collection guide

How raw data gets onto this machine, where it lands, and how to look after it.
Everything here runs locally on Windows for week 1. In week 2 the same scripts
move to a cloud VM, writing to Google Cloud Storage instead of `data/`.

## What is collected

| Data | Script | When | Where it lands |
| --- | --- | --- | --- |
| Realtime trip updates, alerts, vehicle positions (Sydney Trains, Metro, NSW TrainLink) | `collect_realtime.py` | Every 30 s, all the time | `data/raw/realtime/<kind>/<feed>/<date>/<HHMMSS>Z.pb.gz` |
| Static GTFS timetable (Sydney Trains, Metro, NSW TrainLink) | `pull_static.py` | Daily, 3:30am | `data/raw/static/<feed>/<date>.zip`, newest also extracted to `data/static/<feed>/` |
| Static GTFS bus timetable (all NSW buses) | `pull_static.py --feeds buses` | By hand, when refreshing access analysis | as above |
| Bundling finished days of realtime data | `compact_raw.py` | Daily, after the static pull | `data/raw/realtime/<kind>/<feed>/<date>.tar.xz` |
| BOM weather station observations (29 stations) | `pull_weather.py` | Every 6 h (BOM keeps only ~72 h) | `data/raw/weather/bom_obs/<wmo>/<date>/<HHMMSS>Z.json.gz` |
| Open-Meteo hourly weather at every rail station (~10 km grid) | `pull_weather.py` | Every 6 h; `--backfill` for history | `data/raw/weather/open_meteo/<recent or archive>/` |
| BOM rain radar, Sydney (~0.5 km, every 5 min) | `pull_radar.py` | Every 30 min (BOM keeps only ~2 h) | `data/raw/weather/radar/<IDR713 or IDR714>/<date>/<stamp>.png` |
| ABS Census 2021 (SA2 profile, SA2 and mesh block boundaries, mesh block counts) | `download_census.py` | Once, by hand | `data/raw/census/` |
| Park&Ride car park occupancy (41 car parks) | `pull_carpark.py` | Every 15 min | `data/raw/carpark/<date>/<HHMMSS>Z.json.gz` |
| Opal patronage: station entries/exits, trips by line and card type, peak loads | `download_patronage.py` | Monthly, by hand (TfNSW refreshes mid-month) | `data/raw/patronage/` |
| SEIFA 2021, ABS population estimates 2021 to 2025, TZP24 population and jobs forecasts, travel zones, Census jobs by workplace (WPP), station entrances | `download_context.py` | Once; rerun on new releases | `data/raw/context/` |
| OpenStreetMap walking network around all stations | `analysis/walk_access.py` (first run) | Once, cached | `data/raw/context/osm_walk.graphml` |

`<kind>` is `trip_updates`, `alerts` or `vehicle_positions`. `<feed>` is `sydneytrains`
or `metro`. Realtime dates and times are **UTC** (from each feed's header timestamp).
Static timetable dates are **Sydney dates**, because that is the service date they apply to.

All logs go to `data/logs/` (one file per script per day, kept 30 days).

## Endpoints (confirmed 2026-10-26)

| Feed | Sydney Trains | Metro |
| --- | --- | --- |
| Trip updates | `/v2/gtfs/realtime/sydneytrains` | `/v2/gtfs/realtime/metro` |
| Alerts | `/v2/gtfs/alerts/sydneytrains` | `/v2/gtfs/alerts/metro` |
| Vehicle positions | `/v2/gtfs/vehiclepos/sydneytrains` | `/v2/gtfs/vehiclepos/metro` |
| Static timetable | `/v1/gtfs/schedule/sydneytrains` | `/v2/gtfs/schedule/metro` |

| NSW TrainLink trip updates / vehicle positions / alerts | `/v1/gtfs/realtime/nswtrains`, `/v1/gtfs/vehiclepos/nswtrains`, `/v2/gtfs/alerts/nswtrains` | |
| NSW TrainLink timetable | `/v1/gtfs/schedule/nswtrains` | |
| Bus timetable | `/v1/gtfs/schedule/buses` | |

All on `https://api.transport.nsw.gov.au`. The v1 alerts paths return 401 for Sydney Trains and Metro; use v2.
NSW TrainLink has only v1 for trip updates, vehicle positions and the timetable, and only v2 for alerts.
For Metro use the **v2** timetable: the v1 one still downloads but stopped updating in
September 2024, and its trip_ids don't match the realtime feed. Sydney Trains has no v2
timetable.

## Compression: how the data is kept small

Raw snapshots are protobuf, and consecutive snapshots are nearly identical. Two layers:

1. **Per snapshot, as it arrives:** gzip. About 3 to 9x smaller depending on the feed.
   One file per snapshot means a crash can only ever lose the snapshot being written.
2. **Per day, overnight:** `compact_raw.py` packs each finished day into one `.tar.xz`.
   Because xz can see across snapshots, this is roughly 20 to 40x smaller than raw
   (measured on real data on 2026-10-26: 32x for Sydney Trains trip updates).
   The archive is checked byte for byte before the day folder is deleted.

Neither step changes the data: the `.pb` bytes inside are exactly what TfNSW sent,
so everything can be replayed later.

Snapshots are also **deduplicated**: if a feed's header timestamp hasn't changed since
the last save, the snapshot is skipped. Alerts change about every 90 s, so roughly
two thirds of alert polls are skipped.

Rough storage (30 s polling): about 250 MB/day of gzip files during the day, shrinking
to roughly 20 to 40 MB once compacted, plus about 15 MB/day of static timetable zips
when TfNSW publishes a new version. Plan for 15 to 20 GB a year.

Reading a compacted day in Python:

```python
import tarfile
from google.transit import gtfs_realtime_pb2

with tarfile.open("data/raw/realtime/trip_updates/sydneytrains/2026-10-26.tar.xz") as tf:
    for member in tf:
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(tf.extractfile(member).read())
```

Reading a not-yet-compacted snapshot: `gzip.decompress(path.read_bytes())`, then parse the same way.

## Scheduling (Windows Task Scheduler)

Register both tasks (once, from the project folder in PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File .\scheduling\register_tasks.ps1
Start-ScheduledTask -TaskName "SydneyTransit-CollectRealtime"
```

| Task | What it runs | Triggers |
| --- | --- | --- |
| `SydneyTransit-CollectRealtime` | `collect_realtime.py` | At logon, plus a check every 15 min that restarts it if it stopped |
| `SydneyTransit-Daily` | `pull_static.py`, then `compact_raw.py` | 3:30am daily, or as soon as possible if the PC was off |

Radar (every 30 min), car parks (every 15 min) and weather (every 6 h) are launched by
the collector itself as separate processes, not by Task Scheduler. Task Scheduler's
repeating triggers stalled on 26 Oct 2026 (radar and car parks stopped after 5:30pm
with the PC awake), and radar frames are only kept for 2 hours upstream.

Both run as you, only while you are logged on, with no window (they use `pythonw.exe`).

Useful commands:

```powershell
Get-ScheduledTask -TaskName "SydneyTransit-*" | Get-ScheduledTaskInfo   # last run, result
Get-Content data\logs\collect_realtime.log -Tail 20 -Wait               # watch the collector
Stop-ScheduledTask -TaskName "SydneyTransit-CollectRealtime"            # pause collecting
.\scheduling\unregister_tasks.ps1                                       # remove both tasks
```

A `LastTaskResult` of `0` means success. `267009` means "currently running", which is
normal for the collector.

**Known limitation:** nothing is collected while the PC is asleep, off or logged out.
Gaps show up as missing timestamps in the snapshot folders. This is acceptable for
week 1 exploration; the cloud VM in week 2 removes it. To reduce gaps meanwhile, set
Windows sleep to "Never" while plugged in (Settings > System > Power & sleep).

## Running scripts by hand

Activate the environment first (`.venv\Scripts\Activate.ps1`), then:

```powershell
python collect_realtime.py --minutes 5          # short test run
python collect_realtime.py --interval 15        # poll faster
python pull_static.py --feeds metro
python compact_raw.py --dry-run
python download_census.py
```

## Feed update intervals

Every hour the collector appends to `data/logs/feed_intervals.csv`: for each feed, how
many snapshots were new versus unchanged, and the median seconds between changes.
Use it to settle the polling interval and record the decision in `DECISIONS.md`.

## The API key

`TFNSW_API_KEY` lives in `.env` (ignored by git) and is loaded by `tfnsw.py`. No script
prints or logs it.
