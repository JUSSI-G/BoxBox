#UDP Logger to capture UDP data and save it to a file for further use.
#It automatically starts when it starts receiving pacages and stops after it has been idle for 5 seconds.
import socket
import time
import threading
import pyautogui
from datetime import datetime
import os

def auto_input(stop_event, interval=40):
    print("Auto input active")
    while not stop_event.is_set():
        pyautogui.press('c')
        print("Key press sent")
        stop_event.wait(interval)
    print("Auto input stopped")

def capture_udp_auto(port=20777, idle_stop_seconds=5):
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(("127.0.0.1", port))
    udp_socket.settimeout(1.0)

    print("Waiting for UDP telemetry on port", port, "...")

    started = False
    last_packet_time = None
    packets = 0
    stop_event = threading.Event()
    auto_input_thread = None
    file_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(BASE_DIR, "captures"), exist_ok=True)
    savefile = os.path.join(BASE_DIR, "captures", f"udp_dump_{file_time}.bin")

    with open(savefile, "wb") as f:
        while True:
            try:
                data, _ = udp_socket.recvfrom(4096)
            except socket.timeout:
                if started and last_packet_time:
                    if time.time() - last_packet_time > idle_stop_seconds:
                        break
                continue

            if not started:
                print("Telemetry detected. Recording started")
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
