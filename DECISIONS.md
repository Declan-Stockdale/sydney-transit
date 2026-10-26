# Decisions

A running log of design decisions and the reasons for them. Newest first.

## 2026-10-26: Context data, and what couldn't be had

**Added:** SEIFA 2021 (IRSD), Census households with no car (G34), ABS population
estimates to 2025 (ABS data API), TfNSW TZP24 population and jobs forecasts, 2021 Census
jobs by place of work (WPP), station entrances, OpenStreetMap walking network, and
live Park&Ride occupancy.

**Definition:** "transport-disadvantaged" SA2 = (IRSD decile 1 to 3 OR at least 15% of
dwellings without a car) AND (under half of residents within 1.5 km of a station OR under
4 walkable AM peak trains an hour), with at least 2,000 residents.

**Not available:** origin-destination journey to work. TfNSW withdrew JTW from 2016 on
because of ABS confidentiality changes, and 2021 OD needs an ABS TableBuilder login.
Jobs by place of work (WPP) and TZP24 jobs are used instead. The Rail Opal Assignment
Model (train-by-train loads, with actual arrival times) is published through a browser
site that returns 403 to scripts, so it needs a manual download.

## 2026-10-26: Observed arrival, service date and on-time threshold

**Decision:** The observed delay at a stop is the last prediction seen before the stop
drops out of the trip in the feed. Stops still listed in the final snapshot are left
out. "On time" means less than 5 minutes late, and early counts as 0 for
passenger-minutes.

**Why:** Checked on 2026-10-26, the Sydney Trains feed removes a stop from a trip once
it has been served, so the last prediction is the best available estimate.

**Service date:** Sydney Trains leaves `start_date` blank. The service date is taken as
the Sydney date three hours before the snapshot, because services run to about 3am.
Metro fills `start_date` itself. Each service date joins to the newest static zip
dated on or before it.

**Known issue:** over half of Sydney Trains predictions carry a delay of exactly 0,
probably meaning "no live estimate", so short delays may be understated. Check this
against vehicle positions.

## 2026-10-26: Three layers of weather data

**Decision:** Collect BOM station observations, Open-Meteo hourly weather at every rail
station, and BOM rain radar frames.

**Why:** Each fills a gap in the others. Stations (29 around Sydney) are measured truth
but sparse. Open-Meteo is gridded, hourly and can be backfilled, but its cells are about
10 to 15 km across (100 stations fall into 46 cells), too coarse for storm-scale rain.
Radar is about 0.5 km every 5 minutes, fine enough to colour individual track segments,
but the BOM keeps only 2 hours of it and there is no free archive, so it must be collected
from now on. The BOM also keeps only 72 hours of station observations.

**Notes:** Open-Meteo's `era5_land` model returned only nulls for precipitation around
Sydney, so the archive uses the default model mix. BOM radar pixels are colours on a
fixed rain-rate scale; converting them to mm/h happens in analysis.

## 2026-10-26: Metro timetable comes from the v2 endpoint

**Decision:** Download the Metro timetable from `/v2/gtfs/schedule/metro`, not v1.

**Why:** The v1 Metro feed still responds, but its feed_version is from September 2024,
its calendar ends 2024-12-26, and its route (`SMNW_M`) and trip_ids don't match the
realtime feed (`SMNW_M1`). The v2 feed covers Oct 2026 to Apr 2027 and matches realtime.
The stale v1 copy was deleted.

## 2026-10-26: Raw data storage and compression

**Decision:** Save every changed realtime snapshot untouched, gzip-compressed per file,
then bundle each finished UTC day into one `.tar.xz` per feed overnight.

**Why:** Raw snapshots must stay replayable for backfills. Measured on live feeds,
gzip per file gives 3 to 9x; daily `.tar.xz` bundles give roughly 20 to 40x, because
consecutive snapshots are nearly identical and xz can compress across them. Per-file
writes during the day mean a crash loses at most one snapshot.

**Alternatives considered:** Parquet or a database at ingestion (loses the raw bytes,
and parsing belongs in the transformation layer); zstd with a dictionary (similar
ratio but an extra dependency). Revisit when moving to Google Cloud Storage.

## 2026-10-26: Polling interval (provisional: 30 s)

**Decision:** Poll all six realtime endpoints every 30 seconds for now.

**Evidence so far:** A short sample on 2026-10-26 (about 2pm Sydney time) showed trip
updates and vehicle positions changing on every poll at 15 s and 30 s spacing, so they
refresh at least every 15 to 30 s. Alerts changed about every 90 s.

**Still to do:** Review `data/logs/feed_intervals.csv` after a full weekday including a
peak period, then confirm 30 s or move to 15 s. 15 s doubles storage and requests
(about 35,000 a day across six endpoints), which is still within TfNSW's daily limit.

## 2026-10-26: Keep every static timetable version

**Decision:** Download the static GTFS daily and keep a dated zip whenever its contents
change (checked by SHA-256).

**Why:** Sydney Trains trip_ids change daily, so realtime data must join to the timetable
for its own service date. Zips are dated by Sydney date because that is the service date.

## 2026-10-26: Collect vehicle positions as well

**Decision:** Collect vehicle positions alongside trip updates and alerts.

**Why:** Low cost (about 15 MB/day gzip before compaction) and useful for the stretch
reliability map and for checking predicted against actual train locations.

## 2026-10-26: HTTPS certificates from the Windows store

**Decision:** Use the `truststore` package so Python checks HTTPS certificates against
the Windows certificate store.

**Why:** Python's bundled certificate list failed with `CERTIFICATE_VERIFY_FAILED` on
this machine while Windows tools (curl) worked.
