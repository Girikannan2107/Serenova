"""
Serenova Pico W Live Sensor Simulator
Simulates live HTTP POST stream of sensor telemetry from Raspberry Pi Pico W to FastAPI.
"""

import time
import math
import random
import requests
import argparse

BACKEND_URL = "http://127.0.0.1:8000/api/sensor/data"

def generate_sample(seq: int, t: float):
    # ECG waveform simulation (QRS-like pattern)
    heart_rate_hz = 1.25  # ~75 bpm
    ecg_phase = (t * heart_rate_hz) % 1.0
    if ecg_phase < 0.1:
        ecg_val = int(2048 + 1200 * math.sin(ecg_phase * 10 * math.pi))
    elif ecg_phase < 0.15:
        ecg_val = int(2048 - 400 * math.sin((ecg_phase - 0.1) * 20 * math.pi))
    elif ecg_phase < 0.25:
        ecg_val = int(2048 + 300 * math.sin((ecg_phase - 0.15) * 10 * math.pi))
    else:
        ecg_val = 2048 + random.randint(-20, 20)

    # PPG IR & Red (photoplethysmogram pulsatile wave)
    ppg_pulse = math.sin(t * heart_rate_hz * 2 * math.pi)
    ppg_ir = int(12500 + 1500 * ppg_pulse + random.randint(-50, 50))
    ppg_red = int(11800 + 1200 * ppg_pulse + random.randint(-40, 40))

    # Accelerometer (m/s^2 or g) & Gyroscope
    accel_x = round(0.05 + 0.15 * math.sin(t * 0.5) + random.uniform(-0.02, 0.02), 3)
    accel_y = round(0.12 + 0.10 * math.cos(t * 0.5) + random.uniform(-0.02, 0.02), 3)
    accel_z = round(9.80 + random.uniform(-0.05, 0.05), 3)

    gyro_x = round(0.01 * math.sin(t) + random.uniform(-0.005, 0.005), 4)
    gyro_y = round(0.02 * math.cos(t) + random.uniform(-0.005, 0.005), 4)
    gyro_z = round(0.01 + random.uniform(-0.005, 0.005), 4)

    # Skin Temperature (°C)
    temperature_c = round(36.7 + 0.2 * math.sin(t * 0.05) + random.uniform(-0.05, 0.05), 2)

    # Piezo acoustic / movement sensor
    piezo_spike = 1500 if random.random() < 0.08 else 0
    piezo = int(450 + 100 * math.sin(t * 3) + piezo_spike + random.randint(-15, 15))

    return {
        "device_id": "serenova-pico-01",
        "timestamp_ms": int(time.time() * 1000),
        "sequence": seq,
        "sensors": {
            "ppg_ir": ppg_ir,
            "ppg_red": ppg_red,
            "ecg": ecg_val,
            "accel_x": accel_x,
            "accel_y": accel_y,
            "accel_z": accel_z,
            "gyro_x": gyro_x,
            "gyro_y": gyro_y,
            "gyro_z": gyro_z,
            "temperature_c": temperature_c,
            "piezo": piezo
        }
    }

def run_simulation(target_url: str = BACKEND_URL, interval: float = 0.5, count: int = 0):
    print(f"[*] Starting Serenova Pico W Sensor Simulator -> {target_url}")
    print(f"[*] Publishing telemetry every {interval}s (Press Ctrl+C to stop)...")
    
    seq = 1
    t0 = time.time()
    sent = 0

    while True:
        try:
            t = time.time() - t0
            payload = generate_sample(seq, t)
            res = requests.post(target_url, json=payload, timeout=2.0)
            if res.status_code == 200:
                print(f"[+] Seq {seq:05d} | Temp: {payload['sensors']['temperature_c']}°C | ECG: {payload['sensors']['ecg']} | PPG_IR: {payload['sensors']['ppg_ir']} | Status: OK")
            else:
                print(f"[-] Seq {seq:05d} | HTTP {res.status_code}: {res.text}")
            
            seq += 1
            sent += 1
            if count > 0 and sent >= count:
                print(f"[*] Reached target count of {count} packets. Stopping.")
                break
            time.sleep(interval)
        except requests.exceptions.RequestException as e:
            print(f"[!] Network error sending telemetry: {e}")
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n[*] Simulator stopped by user.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serenova Pico W Live Sensor Simulator")
    parser.add_argument("--url", default=BACKEND_URL, help="FastAPI sensor ingest URL")
    parser.add_argument("--interval", type=float, default=0.5, help="Publish interval in seconds (default 0.5s)")
    parser.add_argument("--count", type=int, default=0, help="Number of packets to send (0 = infinite)")
    args = parser.parse_args()

    run_simulation(target_url=args.url, interval=args.interval, count=args.count)
