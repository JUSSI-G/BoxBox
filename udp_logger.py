#UDP Logger to capture UDP data and save it to a file for further use.
#It automatically starts when it starts receiving packets and stops after it has been idle for 5 seconds.
import socket
import time
import struct
import threading
import pydirectinput
from datetime import datetime
import os

car_packet_id = 6
throttle_key = 'w'
min_speed = 30  

current_speed = 0
speed_lock = threading.Lock()

def parse_speed(data, player_index):
    h_format = "<HBBBBBQfIIBB"
    h_size = struct.calcsize(h_format)
    ct_format = "<HfffBbHBBH4H4B4BH4f4B"
    ct_size = struct.calcsize(ct_format)
    offset = h_size + player_index * ct_size
    if len(data) < offset + 2:
        return None
    return struct.unpack_from("<H", data, offset)[0] 

def auto_input(stop_event):
    throttle_held = False
    while not stop_event.is_set():
        with speed_lock:
            speed = current_speed

        if speed < min_speed and not throttle_held:
            pydirectinput.keyDown(throttle_key)
            throttle_held = True
            print(f"Throttle ON (speed: {speed} kmh)")
        elif speed >= min_speed and throttle_held:
            pydirectinput.keyUp(throttle_key)
            throttle_held = False
            print(f"Throttle OFF (speed: {speed} kmh)")

        stop_event.wait(0.1)

    if throttle_held:
        pydirectinput.keyUp(throttle_key)
    print("Auto input stopped")

def capture_udp_auto(port=20777, idle_stop_seconds=5):
    global current_speed

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(("127.0.0.1", port))
    udp_socket.settimeout(1.0)

    print("Waiting for UDP telemetry on port", port, "...")

    started = False
    last_packet_time = None
    packets = 0
    player_index = 0
    stop_event = threading.Event()
    auto_input_thread = None
    file_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(BASE_DIR, "captures"), exist_ok=True)
    savefile = os.path.join(BASE_DIR, "captures", f"udp_dump_{file_time}.bin")

    h_format = "<HBBBBBQfIIBB"
    h_size = struct.calcsize(h_format)

    with open(savefile, "wb") as f:
        while True:
            try:
                data, _ = udp_socket.recvfrom(4096)
            except socket.timeout:
                if started and last_packet_time:
                    if time.time() - last_packet_time > idle_stop_seconds:
                        break
                continue

            if len(data) >= h_size:
                header = struct.unpack_from(h_format, data)
                packet_id = header[5]
                player_index = header[10]

                if packet_id == car_packet_id:
                    speed = parse_speed(data, player_index)
                    if speed is not None:
                        with speed_lock:
                            current_speed = speed

            if not started:
                print("Telemetry detected. Recording started")
                print("Saving to", savefile)
                started = True
                auto_input_thread = threading.Thread(target=auto_input, args=(stop_event,), daemon=True)
                auto_input_thread.start()

            last_packet_time = time.time()
            f.write(len(data).to_bytes(2, "little"))
            f.write(data)
            packets += 1

    stop_event.set()
    if auto_input_thread:
        auto_input_thread.join()

    print("Recording stopped (idle timeout)")
    print("Captured packets:", packets)

capture_udp_auto()
