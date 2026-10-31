# Decisions

A running log of design decisions and the reasons for them. Newest first.

## 2026-10-30: Operations analyses from the collected realtime data only

Seven dashboard pages built only from data already collected (no new sources): missed connections,
where delays start, slow stretches, bunching and gaps, dwell times, crowding and car park fill
(`python -m analysis.operations`). All are previews until a few weeks of data exist.

- **Bus times.** Bus arrivals are taken at stops within 250 m of a rail station only, the same way as
  trains (last prediction before the stop drops out, trusted only if the next snapshot came within 3
  minutes). They are the operator's own estimates, not GPS-checked like trains.
- **Connections.** A planned connection is the first departure on each other line and direction the
  timetable lets you catch (3 min train-to-train; 2 min + walk at 1.2 m/s x 1.3 detour for buses;
  planned waits over 20 min aren't connections). Counted per connection offered, not per passenger.
  Extra waits over an hour are dropped as gaps in the data.
- **Dwell.** The two position feeds differ: Metro reports a STOPPED_AT status every ~30 s; Sydney Trains
  positions snap to signalling locations, with platforms as their own locations ("Ourimbah 2 Loc"). Each
  gives a "platform time", compared within a mode and against each station's own midday value.
- **Occupancy.** Metro and buses report it on every fix; Sydney Trains on about half. `reliability.py`
  now stores it as missing when not reported (it previously read 0, "empty"). No feed reports levels
  above standing room only, or per-carriage loads.
- **Car parks.** Full = 95% of spaces, interpolated between 15-minute readings. Train service compared
  is the scheduled 6 to 10am departures on a typical weekday from the latest timetable, because the
  extracted Sydney Trains timetable only covers a few days.

## 2026-10-26: NSW TrainLink added; Newcastle and the Hunter as their own region

**Timetable.** The Sydney Trains GTFS already carries every NSW TrainLink rail trip (agency
`NSWTrains`: Blue Mountains, Central Coast & Newcastle, Hunter, South Coast, Southern Highlands,
and the regional XPT/Xplorer). Analysis now keeps all of them (previously the Hunter line and regional
trains were filtered out). The separate NSW TrainLink GTFS (`/v1/gtfs/schedule/nswtrains`) repeats a
subset of the same trains under different trip IDs and adds road coaches, so it is **not** added to
timetable counts (it would double count). It is still downloaded daily, because NSW TrainLink realtime
trip IDs only join to it.

**Realtime.** The Sydney Trains realtime feed has no Hunter line trips. NSW TrainLink's own feeds are
now collected too (`/v1/gtfs/realtime/nswtrains`, `/v1/gtfs/vehiclepos/nswtrains`,
`/v2/gtfs/alerts/nswtrains`). Coaches (route IDs not starting `4T.T.`) are dropped. Where a TrainLink
train also appears in the Sydney Trains feed (same station, same scheduled minute), the Sydney Trains
copy is kept. On the first day only 2 such duplicates appeared: the two realtime feeds mostly carry
different trains.

**Region.** Newcastle and the Hunter (ABS SA4s 111 and 106, 682,000 residents) are reported separately
from Greater Sydney so they don't change the Sydney equity statistics: PTAL rail only vs rail + bus,
share within an estimated 1.5 km walk of a station (straight line x 1.28; no footpath network built
for this region), and station-level service, demand and delays.

## 2026-10-26: Data audit fixes

- **Airport stations had no demand.** The patronage files call them "Domestic" and "International"
  (and "Bomaderry" rather than "Bomaderry (Nowra)"). An alias map in `demand_supply.clean_station`
  fixes the join.
- **Erskineville's long walk was an artefact.** It is missing from the 2020 station entrances file, so
  its centre point snapped to a single path node on the wrong side of the tracks (walk ratio 1.9). Stations
  without entrance records now use every path node within 150 m of the centre. Erskineville drops out of the
  long-detour list; the list is now Valley Heights, Bullaburra, Cheltenham, Birrong, Bella Vista, Turrella.
- **Suburb-centre artefact in jobs access.** One resident-weighted centre per suburb sometimes fell where no
  stop was within 800 m (e.g. Gosford - Springfield read as zero jobs). Each suburb now uses up to four origin
  points (one per quadrant, weighted by residents). Average jobs within 45 min: 122,000 rail only, 419,000 rail
  + bus (previously 141,000 and 440,000).
- **Stations outside Greater Sydney** (Newcastle, Southern Highlands, Bathurst) only carry their Sydney-bound
  services in this data (e.g. Newcastle's Hunter line is NSW TrainLink, outside the feeds used), which made
  Newcastle look thinly served and over-crowded. They are excluded from station-level dashboard views.
- **Night access can exceed peak access** in some suburbs (e.g. Baulkham Hills - East). Timetabled bus
  journeys are slower in peak traffic, so this is plausible, not an error. Time of day is now measured for
  every hour 5am to 11pm, weekday and Sunday.
- **PM peak** boardings per train are added (15:00 to 19:00), using the typical-day evening share.

## 2026-10-26: Standard access methods and their settings

| Method | Origin | Settings used |
| --- | --- | --- |
| Cumulative opportunities | Wachs & Kumagai 1973; UK DfT journey time statistics | Jobs, hospitals, tertiary campuses within 45 min; supermarkets within 20 min; weekday 8am |
| Gravity access | Hansen 1959 | Negative exponential decay, half-life 20 min, 60 min horizon; index with Sydney average = 100 |
| Temporal coverage | Time-of-day accessibility (Owen & Levinson, Access Across America) | Jobs within 45 min at weekday 7am, 9pm and Sunday 10am, each averaged over departures at :00, :10 and :20 |
| Transit gap index | Currie 2010 (Melbourne) | Need = mean z of inverse IRSD, no-car %, 65+ %, 15 to 24 %; supply = z of log PTAL; reported for high-need suburbs (need > +0.5 sd) |
| Excess wait time | Transport for London | Platforms with median scheduled gap of 10 min or less; gaps over 30 min dropped; Trains and Metro only (realtime) |
| Social exclusion lens | UK Social Exclusion Unit 2003; Lucas 2012 | Spearman correlation of access with 2021 Census unemployment; reported strictly as correlation |

Destinations come from OpenStreetMap (amenity=hospital, university, college; shop=supermarket).
Origins are given from memory and should be checked before publication. The transit gap needs
the high-need filter, because without it semi-rural fringe areas with average need dominate the
ranking through low supply alone.

## 2026-10-26: Scope change, buses added to access analysis (timetable only)

**Decision:** Add the TfNSW bus timetable (`/v1/gtfs/schedule/buses`) to the access,
equity and jobs analyses. Each is shown as rail only vs rail + bus. Reliability,
realtime collection and delay work stay Sydney Trains and Metro only. Light rail and
ferries remain out of scope.

**Why:** Rail-only access made bus-served areas (Northern Beaches, eastern suburbs)
look unserved, which overstated the equity gaps. With buses, residents with poor access
(PTAL 0 to 1b) fall from 92% to 32%. The transport-disadvantaged list shrinks from 57
suburbs (about 906,000 people) to 17 (about 257,000), concentrated in Campbelltown,
Mount Druitt, Bankstown and Wyong.

**Method:** Access uses Transport for London's PTAL method: walk limits 640 m (bus) and
960 m (rail), half-headway wait, reliability allowances of 2 min (bus) and 0.75 min
(rail), AM peak 7 to 9am. Walk distance is straight line x 1.28 (the measured median
detour). Jobs access is a connection-scan search over the full timetable (compiled with
numba): 45 minutes from 8am, 800 m access walk, 300 m walking transfers plus 1 minute.
The bus zip is about 108 MB, so it isn't part of the daily static pull; refresh it with
`python pull_static.py --feeds buses`.

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

**Update (same day): GPS-measured times replace predictions where possible.** About 66%
of Sydney Trains readings were exactly 0, and 41% of trips read 0 at every stop. Those
trains were live (98% appear in vehicle positions), so the zeros needed checking. The
observed time at a stop is now the vehicle-position GPS fix closest to the station
(within 300 m, within an hour of schedule): the last fix near the station for a
departure, and the first fix at a trip's final stop for its arrival. A trip's first stop
keeps the prediction, because trains sit at the origin platform under their new trip
number long before leaving, which made GPS departures from Central read minutes late.
GPS covers 82% of Sydney Trains and 47% of Metro train-stops, and the last prediction
fills the rest (`delay_source` column). Result: where the prediction was exactly 0, GPS
showed the train 2 or more minutes late under 1% of the time, so the zeros mostly were
"on time" rather than hidden delays. Sydney Trains' vehicle-position `stop_id` is a track location
(e.g. "Lidcombe.M16.1 Loc"), not a platform, so position is used rather than stop_id.

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
