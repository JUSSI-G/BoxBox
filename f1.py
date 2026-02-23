import struct
import csv
import os
import sys
import glob
from collections import defaultdict

h_format = "<HBBBBBQfIIBB"
h_size = struct.calcsize(h_format)

compounds = {
    16: "Soft", 17: "Medium", 18: "Hard",
    7:  "Inter", 8:  "Wet",
    0:  "Unknown"
}

def read_udp_dump(filepath):
    with open(filepath, "rb") as file:
        while True:
            len_bytes = file.read(2)
            if not len_bytes:
                break
            length = int.from_bytes(len_bytes, "little")
            packet = file.read(length)
            yield packet


# struct PacketHeader
# {
#     uint16    m_packetFormat;            // 2025
#     uint8     m_gameYear;                // Game year - last two digits e.g. 25
#     uint8     m_gameMajorVersion;        // Game major version - "X.00"
#     uint8     m_gameMinorVersion;        // Game minor version - "1.XX"
#     uint8     m_packetVersion;           // Version of this packet type, all start from 1
#     uint8     m_packetId;                // Identifier for the packet type, see below
#     uint64    m_sessionUID;              // Unique identifier for the session
#     float     m_sessionTime;             // Session timestamp
#     uint32    m_frameIdentifier;         // Identifier for the frame the data was retrieved on
#     uint32    m_overallFrameIdentifier;  // Overall identifier for the frame the data was retrieved
#                                          // on, doesn't go back after flashbacks
#     uint8     m_playerCarIndex;          // Index of player's car in the array
#     uint8     m_secondaryPlayerCarIndex; // Index of secondary player's car in the array (splitscreen)
#                                          // 255 if no second player
# };

def parse_header(packet):
    header = struct.unpack(h_format, packet[:h_size])

    return {
        "packetFormat": header[0],
        "gameYear": header[1],
        "packetId": header[5],
        "sessionUID": header[6],
        "sessionTime": header[7],
        "frameIdentifier": header[8],
        "overallFrameIdentifier": header[9],
        "playerCarIndex": header[10],
    }

def unpack_cars(fmt, packet, num_cars=22):
    size   = struct.calcsize(fmt)
    offset = h_size
    cars   = {}
    for i in range(num_cars):
        if offset + size > len(packet):
            break
        cars[i] = struct.unpack_from(fmt, packet, offset)
        offset += size
    return cars


# ── Packet 2 – Lap Data ───────────────────────────────────────────────────────
#
# struct LapData {
#   uint32 m_lastLapTimeInMS
#   uint32 m_currentLapTimeInMS
#   uint16 m_sector1TimeMSPart
#   uint8  m_sector1TimeMinutesPart
#   uint16 m_sector2TimeMSPart
#   uint8  m_sector2TimeMinutesPart
#   uint16 m_deltaToCarInFrontMSPart
#   uint8  m_deltaToCarInFrontMinutesPart
#   uint16 m_deltaToRaceLeaderMSPart
#   uint8  m_deltaToRaceLeaderMinutesPart
#   float  m_lapDistance
#   float  m_totalDistance
#   float  m_safetyCarDelta
#   uint8  m_carPosition
#   uint8  m_currentLapNum
#   uint8  m_pitStatus            // 0=none, 1=pitting, 2=in pit area
#   uint8  m_numPitStops
#   uint8  m_sector
#   uint8  m_currentLapInvalid
#   uint8  m_penalties
#   uint8  m_totalWarnings
#   uint8  m_cornerCuttingWarnings
#   uint8  m_numUnservedDriveThroughPens
#   uint8  m_numUnservedStopGoPens
#   uint8  m_gridPosition
#   uint8  m_driverStatus
#   uint8  m_resultStatus
#   uint8  m_pitLaneTimerActive
#   uint16 m_pitLaneTimeInLaneInMS
#   uint16 m_pitStopTimerInMS
#   uint8  m_pitStopShouldServePen
#   float  m_speedTrapFastestSpeed
#   uint8  m_speedTrapFastestLap
# };

lap_FMT = "<IIHBHBHBHBfffBBBBBBBBBBBBBBBHHBfB"

def parse_lap_data(packet):
    result = {}
    for i, d in unpack_cars(lap_FMT, packet).items():
        result[i] = {
            "last_lap_ms":        d[0],
            "current_lap_ms":     d[1],
            "sector1_ms":         d[2] + d[3],
            "sector2_ms":         d[4] + d[5],
            "lap_distance":       d[10],
            "position":           d[13],
            "current_lap":        d[14],
            "pit_status":         d[15],
            "num_pit_stops":      d[16],
            "lap_invalid":        d[18],
            "grid_position":      d[24],
            "result_status":      d[26],
            "pit_lane_time_ms":   d[28],
            "pit_stop_time_ms":   d[30],
        }
    return result

# ── Packet 6 – Car Telemetry ──────────────────────────────────────────────────
#
# struct CarTelemetryData {
#   uint16 m_speed
#   float  m_throttle
#   float  m_steer
#   float  m_brake
#   uint8  m_clutch
#   int8   m_gear
#   uint16 m_engineRPM
#   uint8  m_drs
#   uint8  m_revLightsPercent
#   uint16 m_revLightsBitValue           // F1 25 new field
#   uint16 m_brakesTemperature[4]        // RL RR FL FR
#   uint8  m_tyresSurfaceTemperature[4]  // RL RR FL FR
#   uint8  m_tyresInnerTemperature[4]    // RL RR FL FR
#   uint16 m_engineTemperature
#   float  m_tyresPressure[4]            // RL RR FL FR
#   uint8  m_surfaceType[4]
# };

tele_FMT = "<HfffBbHBBH4H4B4BH4f4B"

def parse_car_telemetry(packet):
    result = {}
    for i, d in unpack_cars(tele_FMT, packet).items():
        result[i] = {
            "speed_kmh":      d[0],
            "throttle":       round(d[1], 3),
            "steer":          round(d[2], 3),
            "brake":          round(d[3], 3),
            "gear":           d[5],
            "rpm":            d[6],
            "drs":            d[7],
            "brake_temp_rl":  d[10],
            "brake_temp_rr":  d[11],
            "brake_temp_fl":  d[12],
            "brake_temp_fr":  d[13],
            "tyre_surf_rl":   d[14],
            "tyre_surf_rr":   d[15],
            "tyre_surf_fl":   d[16],
            "tyre_surf_fr":   d[17],
            "tyre_inner_rl":  d[18],
            "tyre_inner_rr":  d[19],
            "tyre_inner_fl":  d[20],
            "tyre_inner_fr":  d[21],
            "engine_temp":    d[22],
            "tyre_press_rl":  round(d[23], 2),
            "tyre_press_rr":  round(d[24], 2),
            "tyre_press_fl":  round(d[25], 2),
            "tyre_press_fr":  round(d[26], 2),
        }
    return result

# ── Packet 7 – Car Status ─────────────────────────────────────────────────────
#
# struct CarStatusData {
#   uint8  m_tractionControl
#   uint8  m_antiLockBrakes
#   uint8  m_fuelMix
#   uint8  m_frontBrakeBias
#   uint8  m_pitLimiterStatus
#   float  m_fuelInTank
#   float  m_fuelCapacity
#   float  m_fuelRemainingLaps
#   uint16 m_maxRPM
#   uint16 m_idleRPM
#   uint8  m_maxGears
#   uint8  m_drsAllowed
#   uint16 m_drsActivationDistance
#   uint8  m_actualTyreCompound
#   uint8  m_visualTyreCompound
#   uint8  m_tyresAgeLaps
#   int8   m_vehicleFiaFlags
#   float  m_enginePowerICE
#   float  m_enginePowerMGUK
#   float  m_ersStoreEnergy
#   uint8  m_ersDeployMode       // 0=none, 1=medium, 2=hotlap, 3=overtake
#   float  m_ersHarvestedThisLapMGUK
#   float  m_ersHarvestedThisLapMGUH
#   float  m_ersDeployedThisLap
#   uint8  m_networkPaused
# };

status_FMT = "<BBBBBfffHHBBHBBBbfffBfffB"

def parse_car_status(packet):
    result = {}
    for i, d in unpack_cars(status_FMT, packet).items():
        # Sanity check for ERS (0 - 4,000,000 Joules)
        ers_store = round(d[19], 1)
        ers_deploy = round(d[23], 1)
        
        result[i] = {
            "fuel_in_tank":        round(d[5], 3),
            "fuel_remaining_laps": round(d[7], 2),
            "actual_compound":     compounds.get(d[13], f"id={d[13]}"),
            "visual_compound":     compounds.get(d[14], f"id={d[14]}"),
            "tyres_age_laps":      d[15],
            "ers_store_energy":    max(0.0, min(4000000.0, ers_store)),
            "ers_deploy_mode":     d[20],
            "ers_deployed":        max(0.0, min(4000000.0, ers_deploy)),
        }
    return result

# ── Packet 10 – Car Damage ────────────────────────────────────────────────────
#
# struct CarDamageData {
#   float  m_tyresWear[4]      // RL RR FL FR  (0-100 %)
#   uint8  m_tyresDamage[4]
#   uint8  m_brakesDamage[4]
#   uint8  m_tyreBlisters[4]   // F1 25 new field
#   uint8  m_frontLeftWingDamage
#   uint8  m_frontRightWingDamage
#   uint8  m_rearWingDamage
#   uint8  m_floorDamage
#   uint8  m_diffuserDamage
#   uint8  m_sidepodDamage
#   uint8  m_drsFault
#   uint8  m_ersFault
#   uint8  m_gearBoxDamage
#   uint8  m_engineDamage
#   uint8  m_engineMGUHWear
#   uint8  m_engineESWear
#   uint8  m_engineCEWear
#   uint8  m_engineICEWear
#   uint8  m_engineMGUKWear
#   uint8  m_engineTCWear
#   uint8  m_engineBlown
#   uint8  m_engineSeized
# };

damage_FMT = "<4f4B4B4BBBBBBBBBBBBBBBBBB"

def parse_car_damage(packet):
    result = {}
    for i, d in unpack_cars(damage_FMT, packet).items():
        # Helper to clip wear values between 0.0 and 100.0
        def clip_wear(val):
            try:
                clean_val = float(val)
                return round(max(0.0, min(100.0, clean_val)), 2)
            except:
                return 0.0

        result[i] = {
            "tyre_wear_rl":   clip_wear(d[0]),
            "tyre_wear_rr":   clip_wear(d[1]),
            "tyre_wear_fl":   clip_wear(d[2]),
            "tyre_wear_fr":   clip_wear(d[3]),
            "tyre_dmg_rl":    d[4],
            "tyre_dmg_rr":    d[5],
            "tyre_dmg_fl":    d[6],
            "tyre_dmg_fr":    d[7],
            "front_wing_l":   d[16],
            "front_wing_r":   d[17],
            "rear_wing":      d[18],
            "gearbox_damage": d[24],
        }
    return result

# ── Packet 4 – Participants ───────────────────────────────────────────────────
#
# struct ParticipantData {
#   uint8  m_aiControlled
#   uint8  m_driverId
#   uint8  m_networkId
#   uint8  m_teamId
#   uint8  m_myTeam
#   uint8  m_raceNumber
#   uint8  m_nationality
#   char   m_name[32]
#   uint8  m_yourTelemetry
#   uint8  m_showOnlineNames
#   uint16 m_techLevel
#   uint8  m_platform
#   uint8  m_numColours
#   uint8  m_liveryColours[4][3]   // 4 colours × 3 bytes (RGB)
# };

constructors = {
    0: "Mercedes", 1: "Ferrari", 2: "Red Bull", 3: "Williams",
    4: "Aston Martin", 5: "Alpine", 6: "RB", 7: "Haas",
    8: "McLaren", 9: "Sauber", 255: "Unknown"
}

PARTICIPANT_FMT = "<BBBBBBB32sBBHB"
PARTICIPANT_COLOUR_FMT = "<B12B"

def parse_participants(packet):
    base_size    = struct.calcsize(PARTICIPANT_FMT)
    colour_size  = struct.calcsize(PARTICIPANT_COLOUR_FMT)
    per_car_size = base_size + colour_size

    result = {}
    offset = h_size + 1  # +1 for m_numActiveCars byte after header

    for i in range(22):
        if offset + per_car_size > len(packet):
            break
        d = struct.unpack_from(PARTICIPANT_FMT, packet, offset)
        name = d[7].rstrip(b"\x00").decode("utf-8", errors="replace")
        result[i] = {
            "driver_name":   name,
            "team":          constructors.get(d[3], f"team={d[3]}"),
            "ai_controlled": bool(d[0]),
            "race_number":   d[5],
        }
        offset += per_car_size

    return result


def process_dump(filepath, min_lap_s=30, max_lap_s=600):
    # Rolling snapshots keyed by car_idx
    latest_telem  = {}   # car_idx -> dict
    latest_status = {}
    latest_damage = {}
    participants  = {}   # car_idx -> {driver_name, team, ...}

    # car_idx -> last known lap number (to detect crossings per car)
    last_lap_per_car = {}
    prev_lap_snapshot = {}  # car_idx -> last lap_data before crossing

    # (car_idx, lap) -> completed record
    lap_records = {}

    player_idx    = None
    packet_counts = defaultdict(int)
    total_packets = 0

    for packet in read_udp_dump(filepath):
        hdr = parse_header(packet)
        total_packets += 1
        pid = hdr["packetId"]
        packet_counts[pid] += 1

        if player_idx is None:
            player_idx = hdr["playerCarIndex"]

        # Participants — grab once, keep updating in case it changes
        if pid == 4:
            participants = parse_participants(packet)

        # Rolling snapshots for ALL cars
        elif pid == 6:
            latest_telem = parse_car_telemetry(packet)
        elif pid == 7:
            latest_status = parse_car_status(packet)
        elif pid == 10:
            latest_damage = parse_car_damage(packet)

        # Lap data — check every car for a lap crossing
        elif pid == 2:
            all_laps = parse_lap_data(packet)

            for car_idx, lap_data in all_laps.items():
                # 1. Skip inactive cars
                if lap_data["result_status"] <= 1 or lap_data["current_lap"] == 0:
                    continue

                current_lap = lap_data["current_lap"]
                last_lap    = last_lap_per_car.get(car_idx)

                # 2. Check for Lap Crossing (The "Aha!" moment)
                if last_lap is not None and current_lap > last_lap:
                    # We grab 'prev' which STILL holds the data from the very last
                    # frame of the previous lap (including those juicy sector times).
                    prev  = prev_lap_snapshot.get(car_idx, lap_data)
                    lap_s = lap_data["last_lap_ms"] / 1000
                    if min_lap_s <= lap_s <= max_lap_s:
                        info = participants.get(car_idx, {})
                        record = {
                            "car_idx":      car_idx,
                            "is_player":    car_idx == player_idx,
                            "ai_controlled":info.get("ai_controlled", True),
                            "driver_name":  info.get("driver_name", f"Car {car_idx}"),
                            "team":         info.get("team", "Unknown"),
                            "race_number":  info.get("race_number", 0),
                            "lap":          last_lap,
                            "lap_time_s":   round(lap_data["last_lap_ms"] / 1000, 3),
                            "lap_time_ms":  lap_data["last_lap_ms"],
                            "sector1_ms":   prev["sector1_ms"],  # From the snapshot!
                            "sector2_ms":   prev["sector2_ms"],  # From the snapshot!
                            "sector3_ms":   lap_data["last_lap_ms"] - (prev["sector1_ms"] + prev["sector2_ms"]),
                            "position":     lap_data["position"],
                            "num_pit_stops":    lap_data["num_pit_stops"],
                            "pit_stop_time_ms": lap_data["pit_stop_time_ms"],
                            "lap_invalid":      lap_data["lap_invalid"],
                            "grid_position":    lap_data["grid_position"],
                        }
                        record.update(latest_telem.get(car_idx, {}))
                        record.update(latest_status.get(car_idx, {}))
                        record.update(latest_damage.get(car_idx, {}))
                        record.pop("car_idx", None)  # already stored above
                        record["car_idx"] = car_idx  # put it back at the front

                        lap_records[(car_idx, last_lap)] = record

                # 3. CRITICAL: Update the snapshot AFTER the check
                # This ensures the NEXT time this loop runs, it has this frame's data
                prev_lap_snapshot[car_idx] = lap_data
                last_lap_per_car[car_idx] = current_lap

    print(f"Processed {total_packets} total packets")
    print(f"Packet breakdown: {dict(packet_counts)}")
    print(f"Player car index: {player_idx}")
    print(f"Total lap records: {len(lap_records)}")

    return sorted(lap_records.values(), key=lambda r: (r["lap"], r["position"]))


# ── CSV export ────────────────────────────────────────────────────────────────

def export_csv(records, filepath):
    if not records:
        print("No lap records to export.")
        return
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"Exported {len(records)} laps → {filepath}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dump_path = sys.argv[1]
    else:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        captures = sorted(glob.glob(os.path.join(BASE_DIR, "captures", "*.bin")))
        if not captures:
            print("No capture files found in captures/")
            sys.exit(1)
        dump_path = captures[-1]
        print(f"Using latest capture: {dump_path}")

    records = process_dump(dump_path)
    csv_path = os.path.splitext(dump_path)[0] + ".csv"
    export_csv(records, csv_path)

    print("\n── Race Leader History ──────────────────────────────────────────")
    # Filter only for the car in P1 at the end of each lap
    leaders = [r for r in records if r.get("position") == 1]
    
    for r in leaders:
        print(
            f"  Lap {r['lap']:>2} | Leader: {r['driver_name']:<12} | "
            f"Time: {r.get('lap_time_s', '?'):.3f}s | "
            f"Tyre: {r.get('actual_compound', '?')} ({r.get('tyres_age_laps', '?')}L)"
        )