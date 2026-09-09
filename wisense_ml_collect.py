import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import serial


# --------------------------- SETTINGS ---------------------------
PORT = "COM7"
BAUD_RATE = 115200
TARGET_SUBCARRIERS = 64
CAPTURE_SECONDS = 20
COUNTDOWN_SECONDS = 5

OUTPUT_FILE = Path(__file__).resolve().with_name("wisense_posture_raw.csv")

LABELS = {
    "1": "EMPTY",
    "2": "STANDING",
    "3": "SITTING",
    "4": "MOVING",
}


def parse_csi(line):
    """Parse one WiSense CSI_DATA line into 64 normalized amplitudes."""
    if not line.startswith("CSI_DATA,"):
        return None

    try:
        metadata, raw_section = line.split(",[", 1)
        fields = metadata.split(",")

        if len(fields) < 5:
            return None

        sequence = int(fields[1])
        rssi = int(fields[2])
        raw_text = raw_section.rstrip("]\r\n")
        raw = np.fromstring(raw_text, sep=",", dtype=np.float64)

        if raw.size < TARGET_SUBCARRIERS * 2 or raw.size % 2 != 0:
            return None

        # ESP32 CSI byte order: imaginary, real, imaginary, real, ...
        imaginary = raw[0::2]
        real = raw[1::2]
        amplitude = np.hypot(real, imaginary)[:TARGET_SUBCARRIERS]

        valid = amplitude > 0
        if np.count_nonzero(valid) < 10:
            return None

        scale = np.median(amplitude[valid])
        if scale <= 0:
            return None

        amplitude = amplitude / scale
        amplitude[~valid] = 0.0
        return sequence, rssi, amplitude

    except (ValueError, IndexError):
        return None


def csv_header():
    base_columns = [
        "timestamp",
        "session_id",
        "participant",
        "distance_m",
        "label",
        "sequence",
        "rssi_dbm",
    ]
    subcarrier_columns = [f"sc_{index:02d}" for index in range(TARGET_SUBCARRIERS)]
    return base_columns + subcarrier_columns


def prepare_output_file():
    """Create the CSV header or reject an incompatible existing file."""
    expected_header = csv_header()

    if OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0:
        with OUTPUT_FILE.open("r", newline="", encoding="utf-8") as handle:
            existing_header = next(csv.reader(handle), [])

        if existing_header != expected_header:
            raise RuntimeError(
                f"Existing file has a different format: {OUTPUT_FILE}\n"
                "Rename that file and run the collector again."
            )
        return

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(expected_header)


def countdown(label):
    print(f"\nPrepare for {label}. Capture begins in:")
    for remaining in range(COUNTDOWN_SECONDS, 0, -1):
        print(f"  {remaining}...")
        time.sleep(1)


def capture_trial(receiver, label, participant, distance_m, trial_number):
    timestamp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{timestamp_id}_{participant}_{label}_T{trial_number:02d}"

    countdown(label)
    receiver.reset_input_buffer()

    print(
        f"Recording {label} for {CAPTURE_SECONDS} seconds. "
        "Do not change the ESP or chair positions."
    )

    start_time = time.monotonic()
    last_report_time = start_time
    saved_frames = 0

    with OUTPUT_FILE.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)

        while time.monotonic() - start_time < CAPTURE_SECONDS:
            try:
                line = receiver.readline().decode("utf-8", errors="ignore").strip()
            except serial.SerialException:
                raise

            parsed = parse_csi(line)
            if parsed is None:
                continue

            sequence, rssi, amplitude = parsed
            row = [
                datetime.now().isoformat(timespec="milliseconds"),
                session_id,
                participant,
                f"{distance_m:.2f}",
                label,
                sequence,
                rssi,
            ]
            row.extend(f"{value:.6f}" for value in amplitude)
            writer.writerow(row)
            saved_frames += 1

            now = time.monotonic()
            if now - last_report_time >= 1.0:
                elapsed = int(now - start_time)
                print(
                    f"  {elapsed:02d}/{CAPTURE_SECONDS}s | "
                    f"saved frames: {saved_frames} | RSSI: {rssi} dBm",
                    end="\r",
                )
                last_report_time = now

        handle.flush()

    print(
        f"\nSaved {saved_frames} frames for {label}. "
        f"Session: {session_id}"
    )

    if saved_frames < 80:
        print(
            "WARNING: Very few CSI frames were captured. Check that the sender "
            "is powered and the receiver is printing CSI_DATA lines."
        )


def read_distance():
    text = input("ESP-to-ESP distance in metres [2.0]: ").strip()
    if not text:
        return 2.0

    try:
        distance = float(text)
        if distance <= 0:
            raise ValueError
        return distance
    except ValueError:
        print("Invalid distance; using 2.0 metres.")
        return 2.0


def main():
    print("===== WiSense Labelled CSI Dataset Collector =====")
    print("Close Arduino Serial Monitor and the old Python dashboard first.")
    print("Keep both ESP positions, the chair and the room layout fixed.\n")

    participant = input("Participant code [P1]: ").strip().upper() or "P1"
    distance_m = read_distance()
    prepare_output_file()

    receiver = None
    trial_counts = {label: 0 for label in LABELS.values()}

    try:
        print(f"Opening receiver on {PORT}...")
        receiver = serial.Serial(PORT, BAUD_RATE, timeout=0.5)
        time.sleep(2)
        receiver.reset_input_buffer()
        print("Receiver connected. Keep the sender ESP powered.\n")

        while True:
            print("Choose the condition to record:")
            print("  1 - EMPTY room/zone")
            print("  2 - Person STANDING at the marked position")
            print("  3 - Person SITTING on the fixed chair")
            print("  4 - Person MOVING through the sensing zone")
            print("  Q - Finish collection")

            choice = input("Selection: ").strip().upper()
            if choice == "Q":
                break

            label = LABELS.get(choice)
            if label is None:
                print("Choose 1, 2, 3, 4 or Q.\n")
                continue

            input(f"Set up the {label} condition, then press Enter...")
            trial_counts[label] += 1
            capture_trial(
                receiver,
                label,
                participant,
                distance_m,
                trial_counts[label],
            )
            print()

    except serial.SerialException as error:
        print(f"\nSerial-port error: {error}")
        print(f"Confirm the receiver is on {PORT} and close Serial Monitor.")
    except RuntimeError as error:
        print(f"\nDataset error: {error}")
    except KeyboardInterrupt:
        print("\nCollection stopped by user.")
    finally:
        if receiver is not None and receiver.is_open:
            receiver.close()

    print(f"Dataset saved at: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
