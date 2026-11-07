"""
scripts/investigate_live_aqi_stations.py

Investigates live air-quality monitoring stations within 50 km of Amaravati (and across Andhra Pradesh).
1. Queries OpenAQ API v3 for locations near Amaravati, Vijayawada, and Guntur.
2. Retrieves latest measurements, timestamps, pollutants, and coordinates for each station.
3. Evaluates CPCB data sufficiency and calculates composite CPCB NAQI.
4. Ranks stations by freshness (primary) and distance (secondary).
5. Tests accessible CPCB/APPCB public endpoints with genuine HTTP requests.
6. Reports full telemetry findings without exposing secret credentials.
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime, timezone
import pytz
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction.current_air_quality_service import (
    CurrentAirQualityService,
    load_env_file,
    format_age_readable,
    IST,
)
from src.prediction.aqi_advisor import calculate_overall_cpcb_aqi
from src.prediction.data_loader import haversine_distance


def investigate_stations():
    load_env_file()
    srv = CurrentAirQualityService()
    headers = srv.get_headers()
    base_url = srv.base_url_v3

    print("=" * 85)
    print("      VAYUDRISHTI COMPREHENSIVE LIVE AQI STATION & PROVIDER INVESTIGATION")
    print("=" * 85)
    print(f"Provider: {srv.provider.upper()} | API Key: {srv.get_masked_api_key()}")
    print(f"Reference Point: Amaravati, Andhra Pradesh (Lat: 16.518, Lon: 80.518)")
    print(f"Local Time: {datetime.now(timezone.utc).astimezone(IST).strftime('%Y-%m-%d %I:%M:%S %p IST')}")
    print("-" * 85)

    # 1. Coordinate probes across Amaravati region (Amaravati, Vijayawada, Guntur)
    probe_points = [
        {"name": "Amaravati", "lat": 16.518, "lon": 80.518, "radius": 25000},
        {"name": "Vijayawada", "lat": 16.506, "lon": 80.648, "radius": 25000},
        {"name": "Guntur", "lat": 16.306, "lon": 80.436, "radius": 25000},
    ]

    discovered_locations = {}

    print("\n[STEP 1] PROBING OPENAQ V3 API FOR REGIONAL MONITORING LOCATIONS")
    for pt in probe_points:
        url = f"{base_url}/locations"
        params = {
            "coordinates": f"{pt['lat']:.4f},{pt['lon']:.4f}",
            "radius": pt["radius"],
            "limit": 20,
        }
        try:
            t0 = time.time()
            resp = requests.get(url, headers=headers, params=params, timeout=srv.timeout)
            dur = (time.time() - t0) * 1000.0
            print(f"  * Probe near {pt['name']} ({pt['lat']}, {pt['lon']}, r={pt['radius']}m):")
            print(f"      HTTP {resp.status_code} in {dur:.1f}ms")
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                print(f"      Found {len(results)} location(s)")
                for loc in results:
                    loc_id = loc.get("id")
                    if loc_id and loc_id not in discovered_locations:
                        discovered_locations[loc_id] = loc
            else:
                print(f"      Response: {resp.text[:120]}")
        except Exception as e:
            print(f"  * Probe near {pt['name']} failed: {type(e).__name__}: {str(e)[:100]}")

    # Also query OpenAQ locations in India matching keywords (Vijayawada, Amaravati, Guntur)
    search_keywords = ["Amaravati", "Vijayawada", "Guntur", "Tirupati", "Visakhapatnam"]
    print("\n[STEP 2] PROBING OPENAQ V3 API BY NAME / QUERY")
    for kw in search_keywords:
        url = f"{base_url}/locations"
        params = {"iso": "IN", "limit": 10, "name": kw}
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=srv.timeout)
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                print(f"  * Query '{kw}': {len(results)} match(es)")
                for loc in results:
                    loc_id = loc.get("id")
                    if loc_id and loc_id not in discovered_locations:
                        discovered_locations[loc_id] = loc
            else:
                # Try without name if name param is not supported, or log
                pass
        except Exception as e:
            pass

    print(f"\nTotal Unique OpenAQ Locations Discovered: {len(discovered_locations)}")

    # 2. Inspect Latest Observations & Timestamps for Each Location
    print("\n[STEP 3] RETRIEVING LATEST SENSOR MEASUREMENTS & FRESHNESS")
    station_evaluations = []

    for loc_id, loc in discovered_locations.items():
        stn_name = loc.get("name") or f"Location #{loc_id}"
        coords = loc.get("coordinates", {})
        stn_lat = float(coords.get("latitude", 0.0))
        stn_lon = float(coords.get("longitude", 0.0))
        dist_from_ref = round(haversine_distance(16.518, 80.518, stn_lat, stn_lon), 1)

        # Build sensor map
        sensor_map = {}
        for s in loc.get("sensors", []):
            s_id = s.get("id")
            p_info = s.get("parameter", {})
            if s_id and isinstance(p_info, dict):
                p_name = p_info.get("name") or p_info.get("displayName") or ""
                p_unit = p_info.get("units") or "µg/m³"
                sensor_map[s_id] = (p_name, p_unit)

        # Query /latest endpoint
        url_latest = f"{base_url}/locations/{loc_id}/latest"
        latest_items = []
        http_code = 0
        dur_ms = 0.0
        try:
            t0 = time.time()
            r_latest = requests.get(url_latest, headers=headers, timeout=srv.timeout)
            dur_ms = (time.time() - t0) * 1000.0
            http_code = r_latest.status_code
            if http_code == 200:
                latest_items = r_latest.json().get("results", [])
        except Exception as e:
            http_code = -1

        processed = srv.process_raw_measurements(
            latest_items,
            station_name=stn_name,
            city_name=loc.get("locality") or loc.get("city") or "Local Area",
            distance_km=dist_from_ref,
            provider_name="OpenAQ API v3",
            sensor_map=sensor_map,
        )

        pollutants = processed.get("pollutants", {})
        age_hours = processed.get("age_hours")
        obs_time_ist = processed.get("observation_time_ist", "N/A")
        badge = processed.get("status_badge", "UNKNOWN")
        aqi = processed.get("aqi")
        tier = processed.get("aqi_tier", "Unavailable")

        eval_entry = {
            "location_id": loc_id,
            "station_name": stn_name,
            "latitude": stn_lat,
            "longitude": stn_lon,
            "distance_km": dist_from_ref,
            "within_50km": dist_from_ref <= 50.0,
            "http_status": http_code,
            "latency_ms": dur_ms,
            "raw_records_count": len(latest_items),
            "pollutants": pollutants,
            "pollutants_count": len(pollutants),
            "has_cpcb_min": len(pollutants) >= 3 and ("PM2.5" in pollutants or "PM10" in pollutants),
            "age_hours": age_hours,
            "age_readable": format_age_readable(age_hours),
            "obs_time_ist": obs_time_ist,
            "aqi": aqi,
            "aqi_tier": tier,
            "status_badge": badge,
        }
        station_evaluations.append(eval_entry)

        print(f"\n  --- Station #{loc_id}: {stn_name} ---")
        print(f"      Distance from Amaravati: {dist_from_ref} km (Within 50km: {eval_entry['within_50km']})")
        print(f"      Coordinates: ({stn_lat:.4f}, {stn_lon:.4f})")
        print(f"      HTTP /latest Status: {http_code} ({dur_ms:.1f}ms) | Records: {len(latest_items)}")
        print(f"      Observed Timestamp (IST): {obs_time_ist}")
        print(f"      Age: {f'{age_hours:.1f}h' if age_hours is not None else 'N/A'} ({eval_entry['age_readable']})")
        print(f"      Badge: {badge} | AQI: {aqi} ({tier})")
        print(f"      Pollutants: {pollutants}")

    # 3. Investigate Official CPCB/APPCB Public Endpoints
    print("\n[STEP 4] INVESTIGATING OFFICIAL CPCB / APPCB DIRECT PUBLIC DATA ACCESS")
    cpcb_test_endpoints = [
        ("CPCB Official CCR Portal", "https://app.cpcbccr.com/caaqms/feed/feedData"),
        ("CPCB API Gateway", "https://airquality.cpcb.gov.in/aqi_dashboard/aqi_all_stations"),
        ("APPCB Portal Status", "https://pcb.ap.gov.in"),
        ("Data.gov.in Real-time AQI API", "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69?api-key=test&format=json"),
    ]

    cpcb_findings = []
    for label, ep in cpcb_test_endpoints:
        try:
            t0 = time.time()
            # Lightweight head/get request with 5s timeout
            r_cpcb = requests.get(ep, timeout=5.0, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            dur = (time.time() - t0) * 1000.0
            print(f"  * {label}:")
            print(f"      URL: {ep[:70]}...")
            print(f"      Status: HTTP {r_cpcb.status_code} in {dur:.1f}ms")
            cpcb_findings.append({
                "source": label,
                "url": ep,
                "status_code": r_cpcb.status_code,
                "accessible": r_cpcb.status_code == 200,
                "note": f"Responded with HTTP {r_cpcb.status_code}",
            })
        except Exception as e:
            print(f"  * {label}:")
            print(f"      URL: {ep[:70]}...")
            print(f"      Status: Connection Failed ({type(e).__name__}: {str(e)[:60]})")
            cpcb_findings.append({
                "source": label,
                "url": ep,
                "status_code": 0,
                "accessible": False,
                "note": f"{type(e).__name__}: {str(e)[:60]}",
            })

    # 4. Multi-Criteria Ranking of Candidates
    print("\n[STEP 5] RANKING CANDIDATES (FRESHNESS FIRST, DISTANCE SECOND)")
    def rank_key(s):
        age = s.get("age_hours")
        has_cpcb = s.get("has_cpcb_min", False)
        dist = s.get("distance_km", 999.0)

        if age is None or age >= 99990.0:
            tier = 4
            age_val = 99999.0
        elif age <= srv.freshness_live_hours:
            tier = 0 if has_cpcb else 1
            age_val = age
        elif age <= srv.freshness_recent_hours:
            tier = 2 if has_cpcb else 3
            age_val = age
        else:
            tier = 5
            age_val = age
        return (tier, age_val, dist)

    station_evaluations.sort(key=rank_key)

    print("\nRANKED CANDIDATE STATIONS:")
    print(f"{'Rank':<5} | {'Station Name':<35} | {'Dist (km)':<9} | {'Age':<12} | {'AQI':<6} | {'Tier':<14} | {'Observed IST'}")
    print("-" * 110)
    for idx, s in enumerate(station_evaluations, 1):
        age_str = f"{s['age_hours']:.1f}h" if s.get("age_hours") is not None else "N/A"
        aqi_str = str(s.get("aqi")) if s.get("aqi") is not None else "--"
        print(
            f"{idx:<5} | {s['station_name'][:35]:<35} | {s['distance_km']:>8.1f}km | "
            f"{age_str:<12} | {aqi_str:<6} | {s['aqi_tier'][:14]:<14} | {s['obs_time_ist']}"
        )

    # 5. Conclusion & Recommendations
    print("\n" + "=" * 85)
    print("                      INVESTIGATION SUMMARY & CONCLUSIONS")
    print("=" * 85)

    within_50 = [s for s in station_evaluations if s.get("within_50km")]
    live_within_50 = [s for s in within_50 if s.get("age_hours") is not None and s["age_hours"] <= 24.0]

    if live_within_50:
        best_live = live_within_50[0]
        print(f"  [FOUND] Newer valid monitoring station within 50 km:")
        print(f"          Station : {best_live['station_name']} (ID: {best_live['location_id']})")
        print(f"          Distance: {best_live['distance_km']} km")
        print(f"          Obs Time: {best_live['obs_time_ist']} (Age: {best_live['age_readable']})")
        print(f"          AQI     : {best_live['aqi']} ({best_live['aqi_tier']})")
    else:
        print("  [FINDING] No live (<24h) telemetry stations found within 50 km of Amaravati.")
        if within_50:
            print("  [DETAILS ON NEARBY STATIONS WITHIN 50 KM]:")
            for s in within_50:
                print(f"    - {s['station_name']} ({s['distance_km']} km): Last observed {s['obs_time_ist']} (Age: {s['age_readable']})")
        else:
            print("  [DETAILS]: OpenAQ returned no monitoring stations within 50 km.")

    print("\n  [OFFICIAL CPCB/APPCB PORTAL STATUS]:")
    for f in cpcb_findings:
        status_text = "Accessible (200 OK)" if f["accessible"] else f"Inaccessible / Blocked ({f['note']})"
        print(f"    - {f['source']}: {status_text}")

    print("=" * 85 + "\n")


if __name__ == "__main__":
    investigate_stations()
