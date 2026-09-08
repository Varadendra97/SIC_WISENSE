import time
from collections import deque

import numpy as np
import serial


# --------------------------- SETTINGS ---------------------------
PORT = "COM7"
BAUD_RATE = 115200

# The receiver prints about 10 CSI lines per second when it prints every fifth
# packet from a 50-packet/second sender.
WINDOW_SIZE = 20
WARMUP_SCORES = 50
CALIBRATION_SCORES = 60
REQUIRED_HIGH_READINGS = 3
RECENT_ACTIVITY_SECONDS = 30


def parse_csi(line):
    """Convert one CSI_DATA serial line into normalized CSI amplitudes."""
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

        if raw.size < 20 or raw.size % 2 != 0:
            return None

        # ESP32 CSI values arrive as imaginary, real, imaginary, real, ...
        imaginary = raw[0::2]
        real = raw[1::2]
        amplitude = np.hypot(real, imaginary)

        valid = amplitude > 0
        if np.count_nonzero(valid) < 10:
            return None

        # Normalization reduces changes caused only by overall signal strength.
        scale = np.median(amplitude[valid])
        if scale <= 0:
            return None

        amplitude = amplitude / scale
        amplitude[~valid] = 0.0

        return sequence, rssi, amplitude

    except (ValueError, IndexError):
        return None


def calculate_activity_score(csi_window):
    """Measure short-term variation across usable CSI subcarriers."""
    matrix = np.asarray(csi_window, dtype=np.float64)
    usable_subcarriers = np.all(matrix > 0, axis=0)

    if np.count_nonzero(usable_subcarriers) < 10:
        return None

    variation = np.std(matrix[:, usable_subcarriers], axis=0)
    return float(np.mean(variation) * 100.0)


def calculate_fixed_limits(calibration_scores):
    """Calculate a robust baseline and one fixed threshold."""
    scores = np.asarray(calibration_scores, dtype=np.float64)
    baseline = float(np.median(scores))

    # Ignore the highest 10% of idle readings, which can contain short spikes.
    quiet_upper = float(np.percentile(scores, 90))
    margin = max(2.0, baseline * 0.20)

    # The cap prevents one noisy calibration from producing values such as 66
    # when the real idle score is only around 11-16.
    candidate = quiet_upper + margin
    safety_cap = baseline + max(8.0, baseline * 0.60)
    fixed_threshold = min(candidate, safety_cap)
    fixed_threshold = max(fixed_threshold, baseline + 2.0)

    return baseline, fixed_threshold


def main():
    csi_window = deque(maxlen=WINDOW_SIZE)
    calibration_scores = []

    baseline = None
    threshold = None
    warmup_count = 0
    high_count = 0
    last_movement_time = 0.0
    last_print_time = 0.0
    receiver = None

    print(f"Opening ESP32 receiver on {PORT}...")

    try:
        receiver = serial.Serial(PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        receiver.reset_input_buffer()

        print("ESP32 receiver connected!")
        print("AUTOMATIC CALIBRATION STARTED")
        print("Keep the sensing area empty and do not touch the ESP32s.\n")

        while True:
            line = receiver.readline().decode("utf-8", errors="ignore").strip()
            result = parse_csi(line)

            if result is None:
                continue

            _sequence, rssi, amplitude = result

            # CSI length can occasionally change. Start a fresh window if it does.
            if csi_window and amplitude.size != csi_window[0].size:
                csi_window.clear()

            csi_window.append(amplitude)

            if len(csi_window) < WINDOW_SIZE:
                print(
                    f"\rCollecting CSI window: {len(csi_window)}/{WINDOW_SIZE}",
                    end="",
                    flush=True,
                )
                continue

            score = calculate_activity_score(csi_window)
            if score is None:
                continue

            # Discard startup readings while the ESP32 link and CSI window settle.
            if threshold is None and warmup_count < WARMUP_SCORES:
                warmup_count += 1
                print(
                    f"\rStabilizing signal: {warmup_count}/{WARMUP_SCORES} "
                    f"| Score: {score:6.2f}",
                    end="",
                    flush=True,
                )
                continue

            # Calibrate once, then keep both values fixed until the next restart.
            if threshold is None:
                calibration_scores.append(score)
                print(
                    f"\rCalibrating: {len(calibration_scores)}/{CALIBRATION_SCORES} "
                    f"| Current score: {score:6.2f}",
                    end="",
                    flush=True,
                )

                if len(calibration_scores) >= CALIBRATION_SCORES:
                    baseline, threshold = calculate_fixed_limits(calibration_scores)
                    idle_low = float(np.percentile(calibration_scores, 10))
                    idle_high = float(np.percentile(calibration_scores, 90))
                    print("\n\nCALIBRATION COMPLETED!")
                    print(f"Stable idle range: {idle_low:.2f} to {idle_high:.2f}")
                    print(f"Fixed baseline : {baseline:.2f}")
                    print(f"Fixed threshold: {threshold:.2f}")
                    print("These values will remain fixed for this run.")
                    print("Walk between the two ESP32 boards now.\n")

                continue

            current_time = time.time()

            if score > threshold:
                high_count += 1
            else:
                high_count = 0

            if high_count >= REQUIRED_HIGH_READINGS:
                last_movement_time = current_time
                status = "GREEN | MOVEMENT"
            elif (
                last_movement_time > 0
                and current_time - last_movement_time < RECENT_ACTIVITY_SECONDS
            ):
                status = "YELLOW | RECENT"
            else:
                status = "RED | NO MOVEMENT"

            # Short, separate lines remain readable in the VS Code terminal.
            if current_time - last_print_time >= 0.50:
                print(
                    f"Score:{score:6.2f} | Base:{baseline:6.2f} | "
                    f"Limit:{threshold:6.2f} | RSSI:{rssi:4d} | {status}"
                )
                last_print_time = current_time

    except KeyboardInterrupt:
        print("\nWiSense program stopped.")
    except serial.SerialException as error:
        print(f"\nSerial-port error: {error}")
        print("Close Arduino Serial Monitor and check that PORT is COM7.")
    finally:
        if receiver is not None and receiver.is_open:
            receiver.close()
        print("Receiver COM port closed.")


if __name__ == "__main__":
    main()
