"""Live terminal test. Keep collector, trainer and your model beside this file."""
import argparse
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import serial
import sklearn
from wisense_ml_collect import parse_csi
from wisense_ml_train import features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', default='COM7')
    p.add_argument('--model', default=str(Path(__file__).with_name('wisense_posture_model.joblib')))
    args = p.parse_args()
    bundle = joblib.load(args.model)
    if bundle.get('feature_version') != 'mean_std_iqr_absdiff_v1':
        raise ValueError('Unsupported model features; use the supplied WiSense trainer.')
    if bundle.get('sklearn_version') != sklearn.__version__:
        raise ValueError('Use the same scikit-learn version as training, or retrain in this environment.')
    model = bundle['model']
    bins = np.asarray(bundle['bin_indices'], dtype=int)
    size = int(bundle['window_size'])
    if size != 12 or bins.size < 10 or np.any((bins < 0) | (bins >= 64)):
        raise ValueError('Unexpected model window or CSI bin configuration.')
    print('WiSense live posture test | Ctrl+C to stop')
    print('Predictions are experimental. Model support is NOT measured accuracy.')
    print('Hold each condition for 20 seconds; allow two complete windows after changing posture.')
    with serial.Serial(args.port, 115200, timeout=0.1) as receiver:
        time.sleep(2)
        receiver.reset_input_buffer()
        print(f'Connected to {args.port}. Waiting for CSI...')
        buffer = b''
        frames = []
        last_valid = time.monotonic()
        last_notice = last_valid
        last_sequence = None
        while True:
            chunk = receiver.read(min(max(receiver.in_waiting, 1), 8192))
            buffer += chunk
            now = time.monotonic()
            if now - last_valid > 1:
                frames.clear()
            if now - last_valid > 3 and now - last_notice > 3:
                print('NO DATA | Check sender power and CSI serial output.')
                last_notice = now
            if len(buffer) > 65536:
                buffer = b''
                frames.clear()
                print('Serial framing error: waiting for a complete line.')
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                parsed = parse_csi(line.decode('utf-8', errors='ignore').strip())
                if parsed is None:
                    continue
                seq, rssi, amplitude = parsed
                if seq == last_sequence:
                    continue
                if last_sequence is not None and seq < last_sequence:
                    frames.clear()
                last_sequence = seq
                x = amplitude[bins]
                if not np.isfinite(x).all() or np.any(x <= 0):
                    frames.clear()
                    print('INVALID CSI | Waiting for usable bins; no prediction.')
                    continue
                last_valid = time.monotonic()
                frames.append(x)
                if len(frames) < size:
                    continue
                f = features(np.asarray(frames)).reshape(1, -1)
                frames.clear()
                probabilities = model.predict_proba(f)[0]
                top = int(np.argmax(probabilities))
                label = str(model.classes_[top])
                support = float(probabilities[top])
                # No tuned confidence cutoff: show raw predictions for honest testing.
                print(f'{datetime.now():%H:%M:%S} | {label:8s} | '
                      f'Model support: {support:.0%} | RSSI: {rssi} dBm', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped. Receiver COM port closed.')
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f'Live test stopped: {error}')
