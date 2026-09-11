"""WiSense synchronized dashboard. Receiver controls physical LEDs and main state.
Run: py wisense_dashboard_synced.py --port COM7
Use WiSense_Receiver_Synced.ino; room LED GPIO32 turns off after 15 idle seconds.
ML remains EMPTY/STANDING/MOVING using your three-class model.
"""
import argparse
import csv
import math
import queue
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

# User-adjustable settings. Changing the main window requires recalibration,
# which happens automatically. Do not change the ML window without retraining.
MAIN_WINDOW = 8
WARMUP_SCORES = 50
CALIBRATION_SCORES = 60
HIGH_READINGS = 2
LOW_READINGS = 2
RECENT_ACTIVITY_SECONDS = 1.0
NO_DATA_SECONDS = 2.0
MAX_FRAME_GAP = 1.0
UI_INTERVAL_MS = 80
GRAPH_INTERVAL_SECONDS = 0.25
LOG_INTERVAL_SECONDS = 0.5
LIGHT_ACK_TIMEOUT = 3.0
BAUD = 115200
BG, CARD, BORDER = '#080f1c', '#111e30', '#24354b'
TEXT, MUTED, BLUE = '#edf4fc', '#93a7bd', '#4acbff'
GREEN, RED, YELLOW, PURPLE = '#43df9a', '#ff7385', '#f7c766', '#b49aff'
COLORS = {'MOVEMENT': GREEN, 'NO_MOVEMENT': RED, 'RECENT_MOVEMENT': YELLOW}
ROOT = Path(__file__).resolve().parent
CSV_HEADER = ['timestamp', 'activity_score', 'baseline', 'threshold',
              'rssi_dbm', 'status', 'movement_events']


def parse_csi(line):
    """Match the collector's first-64 positive-median amplitude normalization.
    Preserve full lines and validate their reported length before parsing.
    """
    if not line.startswith('CSI_DATA,') or not line.endswith(']'):
        return None
    try:
        meta, values = line.split(',[', 1)
        fields = meta.split(',')
        if len(fields) != 5:
            return None
        seq, rssi, channel, length = map(int, fields[1:])
        raw = np.array([int(v) for v in values[:-1].split(',')], dtype=float)
        if len(raw) != length or length < 128 or length % 2:
            return None
        if np.any((raw < -128) | (raw > 127)):
            return None
        amplitude = np.hypot(raw[0::2], raw[1::2])[:64]
        valid = amplitude > 0
        if valid.sum() < 10:
            return None
        amplitude /= np.median(amplitude[valid])
        return seq, rssi, amplitude
    except (ValueError, IndexError):
        return None


def activity_score(frames):
    matrix = np.asarray(frames)
    usable = np.all(matrix > 0, axis=0)
    if usable.sum() < 10:
        return None
    return float(matrix[:, usable].std(axis=0).mean() * 100)


def fixed_limits(scores):
    baseline = float(np.median(scores))
    candidate = float(np.percentile(scores, 90)) + max(2, baseline * .2)
    cap = baseline + max(8, baseline * .6)
    return baseline, max(min(candidate, cap), baseline + 2)


def ml_features(x):
    """Exactly mean_std_iqr_absdiff_v1 from wisense_ml_train.py."""
    return np.concatenate([x.mean(axis=0), x.std(axis=0),
                           np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0),
                           np.abs(np.diff(x, axis=0)).mean(axis=0)])


def put_latest(q, item):
    """A slow ML worker must not build a backlog of old windows."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


class Detector:
    """All methods are called under Engine.lock; no UI or I/O operations here."""
    def __init__(self):
        self.events = 0
        self.last_event = '--'
        self.reset()

    def reset(self):
        self.frames = deque(maxlen=MAIN_WINDOW)
        self.history = deque(maxlen=600)
        self.calibration = []
        self.warmup = 0
        self.baseline = self.threshold = self.score = None
        self.high = self.low = 0
        self.moving = False
        self.last_movement = None

    def gap(self):
        self.frames.clear()
        self.high = self.low = 0
        self.moving = False
        self.score = None
        self.last_movement = None
        # Never mix disconnected periods in startup calibration.
        if self.threshold is None:
            self.warmup = 0
            self.calibration.clear()

    def process(self, amplitude, now):
        self.frames.append(amplitude)
        if len(self.frames) < MAIN_WINDOW:
            return
        self.score = activity_score(self.frames)
        if self.score is None:
            return
        self.history.append((now, self.score))
        if self.threshold is None:
            if self.warmup < WARMUP_SCORES:
                self.warmup += 1
            else:
                self.calibration.append(self.score)
                if len(self.calibration) >= CALIBRATION_SCORES:
                    self.baseline, self.threshold = fixed_limits(self.calibration)
            return
        if self.score > self.threshold:
            self.high += 1
            self.low = 0
            if self.high >= HIGH_READINGS:
                if not self.moving:
                    self.events += 1
                    self.last_event = datetime.now().strftime('%H:%M:%S')
                self.moving = True
                self.last_movement = now
        else:
            self.high = 0
            self.low += 1
            if self.low >= LOW_READINGS:
                self.moving = False

    def status(self, now):
        if self.threshold is None:
            if len(self.frames) < MAIN_WINDOW:
                return 'CALIBRATING', f'Collecting CSI  {len(self.frames)}/{MAIN_WINDOW}'
            if self.warmup < WARMUP_SCORES:
                return 'CALIBRATING', f'Stabilizing  {self.warmup}/{WARMUP_SCORES}'
            return 'CALIBRATING', f'Calibrating  {len(self.calibration)}/{CALIBRATION_SCORES}'
        if self.score is None:
            return 'WAITING', 'Collecting fresh CSI'
        if self.moving:
            return 'MOVEMENT', 'Movement detected'
        if self.last_movement is not None and now - self.last_movement < RECENT_ACTIVITY_SECONDS:
            return 'RECENT_MOVEMENT', 'Recent movement'
        return 'NO_MOVEMENT', 'No movement detected'


class HomeLight:
    """Receiver-armed room-light automation. Calls are protected by Engine.lock."""
    def __init__(self):
        self.enabled = False
        self.want_on = False
        self.ack_on = None
        self.last_ack = None
        self.armed_ack = False

    def cancel(self):
        self.want_on = False

    def toggle(self):
        self.enabled = not self.enabled
        self.cancel()

    def acknowledge(self, on, now):
        self.ack_on = on
        self.last_ack = now

    def acknowledge_arm(self, armed, now):
        self.armed_ack = armed
        self.last_ack = now

    def firmware_ready(self, now):
        return self.last_ack is not None and now - self.last_ack < LIGHT_ACK_TIMEOUT

    def tick(self, now, healthy, calibrated, moving):
        if not self.enabled or not healthy or not calibrated or not self.firmware_ready(now):
            self.cancel()
            return
        if moving:
            self.want_on = True

    def display(self, now):
        if not self.firmware_ready(now):
            return 'Receiver lamp control not confirmed - upload the updated receiver sketch.'
        if not self.enabled:
            return 'LIGHT OFF - enable Home Automation for the room-light demo'
        if self.want_on:
            return 'LIGHT ON - disable Home Automation to switch it off'
        return 'Armed - waiting for movement after main calibration'


class Engine:
    def __init__(self, port, model_path, data_dir=ROOT, demo=False):
        self.port, self.model_path, self.data_dir, self.demo = port, Path(model_path), Path(data_dir), demo
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.reset_request = threading.Event()
        self.ml_active = threading.Event()
        self.detector = Detector()
        self.receiver_state_time = None
        self.receiver_state_code = 'WAITING'
        self.room_remaining_ms = 0
        self.ml_queue = queue.Queue(maxsize=1)
        self.log_queue = queue.Queue(maxsize=128)
        self.threads = []
        self.link = 'Connecting'
        self.last_frame = None
        self.rssi = None
        self.times = deque(maxlen=100)
        self.sequence = None
        self.rejected = 0
        self.generation = 0
        self.ml_message = 'Open this tab to load the model'
        self.ml_result = None
        self.ml_history = deque(maxlen=16)
        self.log_error = ''
        self.log_dropped = 0
        self.ml_error = False
        self.ml_ready = False
        self.light = HomeLight()
        self.last_light_command = 0.0
        self.last_light_sent = None

    def toggle_light(self):
        with self.lock:
            self.light.toggle()

    def _service_light(self, receiver, now):
        with self.lock:
            healthy = (self.last_frame is not None and now-self.last_frame < NO_DATA_SECONDS
                       and self.link == 'Connected')
            calibrated = True  # Receiver owns calibration.

            if self.demo:
                self.light.acknowledge_arm(self.light.enabled, now)
                self.light.acknowledge(self.light.want_on, now)

            # Arm/disarm the receiver-side automation. The receiver then
            # turns GPIO32 on from the exact same movement state that drives
            # the green LED, eliminating the old ~12 s Python-side lag.
            if not self.light.enabled or not healthy or not calibrated:
                command = b'AUTO:0\n'
            else:
                command = b'AUTO:1\n'

            # Send the ARM command as a heartbeat every second. DISARM is
            # sent immediately when Home Automation is turned off.
            if self.last_light_sent == command and now-self.last_light_command < 1.0:
                return

            if receiver is not None:
                if receiver.write(command) != len(command):
                    raise OSError('Incomplete lamp-control command; check USB connection.')
            self.last_light_command = now
            self.last_light_sent = command


    def start(self):
        for target in (self._logger, self._ml_worker, self._reader):
            thread = threading.Thread(target=target, daemon=True)
            self.threads.append(thread)
            thread.start()

    def set_ml_active(self, active):
        # Invalidate in-flight predictions when the user changes tabs.
        with self.lock:
            self.generation += 1
            self.ml_result = None
            self.ml_message = 'Collecting 12 fresh CSI frames' if self.ml_ready else self.ml_message
            if active:
                self.ml_active.set()
            else:
                self.ml_active.clear()

    def enqueue_log(self, kind, reading):
        if self.demo:
            return
        try:
            self.log_queue.put_nowait((kind, reading))
        except queue.Full:
            self.log_dropped += 1

    def _accept_receiver_state(self, line, now):
        try:
            fields = line.split(',')
            if len(fields) != 11 or fields[1] not in {'WAITING', 'CALIBRATING', 'MOVEMENT', 'NO_MOVEMENT'}:
                return False
            score, baseline, threshold = map(float, fields[2:5])
            cal, warm, room, armed, remaining, events = map(int, fields[5:])
            if not all(math.isfinite(v) and v >= 0 for v in (score, baseline, threshold)):
                return False
            if room not in (0, 1) or armed not in (0, 1) or not 0 <= remaining <= 15000 or events < 0:
                return False
        except ValueError:
            return False
        self.receiver_state_code, self.receiver_state_time = fields[1], now
        d = self.detector
        ready = fields[1] in {'MOVEMENT', 'NO_MOVEMENT'}
        d.score = score if ready else None
        d.baseline, d.threshold = (baseline, threshold) if ready else (None, None)
        d.moving = fields[1] == 'MOVEMENT'
        d.warmup = warm
        if events > d.events:
            d.last_event = datetime.now().strftime('%H:%M:%S')
        d.events = events
        if ready:
            d.history.append((now, score))
        self.room_remaining_ms = remaining
        self.light.acknowledge(bool(room), now)
        self.light.acknowledge_arm(bool(armed), now)
        self.receiver_calibration = cal
        return True

    def _main_status(self, now):
        if self.demo:
            return self.detector.status(now)
        if self.receiver_state_time is None or now-self.receiver_state_time > 1.0:
            return 'NO_DATA', 'Waiting for receiver state — upload the matching receiver sketch'
        code = self.receiver_state_code
        if code == 'CALIBRATING':
            return code, f'Calibrating receiver  {self.receiver_calibration}/60'
        return code, {'MOVEMENT': 'Movement detected', 'NO_MOVEMENT': 'No movement detected',
                      'WAITING': 'Waiting for fresh CSI'}[code]

    def _room_text(self, now):
        if self.receiver_state_time is None or now-self.receiver_state_time > 1.0:
            return 'Room state unavailable — check receiver firmware and USB'
        if not self.light.enabled:
            return 'Disabling automation...' if self.light.armed_ack else 'Home Automation OFF'
        if not self.light.armed_ack:
            return 'Waiting for receiver to confirm automation'
        if self.light.ack_on:
            if self.room_remaining_ms:
                return f'LIGHT ON — off in {math.ceil(self.room_remaining_ms/1000)} s without movement'
            return 'LIGHT ON — movement detected'
        return 'Armed — turns on with green; off after 15 seconds without movement'

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            d = self.detector
            age = None if self.last_frame is None else now - self.last_frame
            healthy = age is not None and age < NO_DATA_SECONDS and self.link == 'Connected'
            code, title = self._main_status(now)
            if not healthy:
                code, title = 'NO_DATA', 'No fresh CSI' if self.last_frame else 'Waiting for receiver'
            rate = 0.0
            if healthy and len(self.times) > 1:
                span = self.times[-1] - self.times[0]
                rate = (len(self.times) - 1) / span if span > 0 else 0
            ml = self.ml_result
            if not healthy or ml is not None and now - ml['monotonic'] > 3:
                ml = None
            return dict(code=code, title=title, score=d.score if healthy else None,
                        baseline=d.baseline, threshold=d.threshold,
                        events=d.events, last_event=d.last_event,
                        rssi=self.rssi if healthy else None, age=age, rate=rate,
                        healthy=healthy, link=self.link, history=list(d.history),
                        ml=ml, ml_message=self.ml_message, ml_error=self.ml_error,
                        ml_history=list(self.ml_history), log_error=self.log_error +
                        (f' | Dropped log entries: {self.log_dropped}' if self.log_dropped else ''),
                        rejected=self.rejected, light_enabled=self.light.enabled,
                        light_text=self._room_text(now),
                        light_on=(self.light.ack_on is True and self.receiver_state_time is not None and now-self.receiver_state_time <= 1.0))

    def _reader(self):
        receiver = None
        buffer = b''
        ml_frames = []
        frame_generation = -1
        last_log = 0.0
        rng = np.random.default_rng(42)
        demo_sequence = 0
        try:
            if not self.demo:
                import serial
                receiver = serial.Serial(self.port, BAUD, timeout=.05, write_timeout=.2)
                self.stop.wait(2)
                receiver.reset_input_buffer()
                receiver.write(b'CALIBRATE\n')
            with self.lock:
                self.link = 'Connected'
            while not self.stop.is_set():
                if self.reset_request.is_set():
                    with self.lock:
                        self.detector.reset()
                        self.generation += 1
                        self.ml_result = None
                        self.light.cancel()
                    self.reset_request.clear()
                    ml_frames.clear()
                    buffer = b''
                    if receiver:
                        receiver.reset_input_buffer()
                        receiver.write(b'CALIBRATE\n')
                        with self.lock:
                            self.receiver_state_time = None
                self._service_light(receiver, time.monotonic())
                if self.demo:
                    self.stop.wait(.15)
                    demo_sequence += 1
                    sigma = .03 if demo_sequence < 140 or demo_sequence % 90 < 60 else .35
                    amp = np.maximum(.05, 1 + rng.normal(0, sigma, 64))
                    amp[28:37] = 0
                    packets = [(demo_sequence, -64, amp)]
                else:
                    # A long OS backlog is stale data: discard explicitly and mark the gap.
                    if receiver.in_waiting > BAUD / 10:
                        receiver.reset_input_buffer()
                        buffer = b''
                        with self.lock:
                            self.detector.gap()
                            self.generation += 1
                            self.ml_result = None
                            self.last_frame = None
                            self.times.clear()
                        ml_frames.clear()
                        continue
                    buffer += receiver.read(min(max(receiver.in_waiting, 1), 8192))
                    if len(buffer) > 65536:
                        buffer = b''
                        with self.lock:
                            self.rejected += 1
                    packets = []
                    while b'\n' in buffer:
                        raw, buffer = buffer.split(b'\n', 1)
                        line = raw.decode('utf-8', errors='ignore').strip()
                        if line.startswith('RSTATE,'):
                            with self.lock:
                                self._accept_receiver_state(line, time.monotonic())
                            continue
                        if line in ('LIGHT_ACK:0', 'LIGHT_ACK:1'):
                            with self.lock:
                                self.light.acknowledge(line.endswith('1'), time.monotonic())
                            continue
                        if line in ('LIGHT_ACK:ARM', 'LIGHT_ACK:DISARM'):
                            with self.lock:
                                self.light.acknowledge_arm(line == 'LIGHT_ACK:ARM', time.monotonic())
                            continue
                        result = parse_csi(line)
                        if result:
                            packets.append(result)
                        elif line.startswith('CSI_DATA,'):
                            with self.lock:
                                self.rejected += 1
                for seq, rssi, amplitude in packets:
                    now = time.monotonic()
                    with self.lock:
                        if seq == self.sequence:
                            continue
                        if (self.last_frame is not None and now - self.last_frame > MAX_FRAME_GAP
                                or self.sequence is not None and seq < self.sequence):
                            self.detector.gap()
                            self.generation += 1
                            self.ml_result = None
                            self.times.clear()
                        self.sequence = seq
                        self.last_frame, self.rssi = now, rssi
                        self.times.append(now)
                        if self.demo:
                            self.detector.process(amplitude, now)
                        generation = self.generation
                        if self.ml_active.is_set() and self.ml_ready:
                            if generation != frame_generation:
                                ml_frames.clear()
                                frame_generation = generation
                            ml_frames.append(amplitude)
                            if len(ml_frames) == 12:
                                put_latest(self.ml_queue, (generation, now, np.array(ml_frames), rssi))
                                ml_frames.clear()
                        else:
                            ml_frames.clear()
                        d = self.detector
                        code, _ = self._main_status(now)
                        if code in ('MOVEMENT', 'NO_MOVEMENT') and d.threshold is not None and d.score is not None and now - last_log >= LOG_INTERVAL_SECONDS:
                            row = (datetime.now().isoformat(timespec='milliseconds'),
                                   d.score, d.baseline, d.threshold, rssi, code, d.events, 'UNKNOWN')
                            self.enqueue_log('main', row)
                            last_log = now
        except Exception as error:
            with self.lock:
                self.link = f'Receiver error: {error}'
                self.ml_result = None
                self.light.cancel()
            print(self.link)
        finally:
            if receiver is not None:
                try:
                    receiver.write(b'AUTO:0\n')
                except Exception:
                    pass  # Receiver firmware also turns the lamp off after its 5 s watchdog.
                receiver.close()

    def _ml_worker(self):
        model = bins = None
        while not self.stop.is_set():
            if not self.ml_active.wait(.2):
                continue
            if model is None:
                try:
                    if self.demo:
                        raise ValueError('Preview mode: ML is disabled for synthetic CSI.')
                    import joblib
                    import sklearn
                    if not self.model_path.is_file():
                        raise ValueError('Place wisense_posture_model.joblib beside this script, then restart.')
                    # Load only your own trusted model files: joblib uses pickle.
                    b = joblib.load(self.model_path)
                    model = b['model']
                    if set(model.classes_) not in ({'STANDING', 'MOVING'}, {'EMPTY', 'STANDING', 'MOVING'}):
                        raise ValueError('Use a STANDING/MOVING or EMPTY/STANDING/MOVING model.')
                    if b.get('feature_version') != 'mean_std_iqr_absdiff_v1' or b.get('window_size') != 12:
                        raise ValueError('Model features do not match the WiSense trainer.')
                    if b.get('normalization') != 'collector_per_packet_positive_median_first64':
                        raise ValueError('Model normalization does not match this CSI parser.')
                    if b.get('sklearn_version') != sklearn.__version__:
                        raise ValueError('scikit-learn version differs from training. Retrain using this Python installation.')
                    bins = np.asarray(b['bin_indices'], dtype=int)
                    if bins.ndim != 1 or len(bins) < 10 or np.any((bins < 0) | (bins >= 64)):
                        raise ValueError('Invalid model subcarrier indices.')
                    if model.n_features_in_ != 4 * len(bins):
                        raise ValueError('Unexpected number of model features.')
                    model.n_jobs = 1  # Avoid creating a CPU thread pool on every small prediction.
                    with self.lock:
                        self.ml_ready = True
                        self.ml_message = 'Collecting 12 fresh CSI frames'
                except Exception as error:
                    with self.lock:
                        self.ml_error = True
                        self.ml_message = str(error)
                    return  # Main detection and storage remain active.
            try:
                generation, timestamp, matrix, rssi = self.ml_queue.get(timeout=.2)
            except queue.Empty:
                continue
            with self.lock:
                if generation != self.generation or not self.ml_active.is_set():
                    continue
            if time.monotonic() - timestamp > NO_DATA_SECONDS:
                continue
            x = matrix[:, bins]
            if not np.isfinite(x).all() or np.any(x <= 0):
                with self.lock:
                    self.ml_result = None
                    self.ml_message = 'Invalid CSI window: waiting for fresh data'
                continue
            try:
                probability = model.predict_proba(ml_features(x).reshape(1, -1))[0]
                top = int(np.argmax(probability))
                result = dict(label=str(model.classes_[top]), support=float(probability[top]),
                              probabilities=dict(zip(map(str, model.classes_), map(float, probability))),
                              timestamp=datetime.now().strftime('%H:%M:%S'), monotonic=timestamp,
                              rssi=rssi)
                with self.lock:
                    if generation != self.generation or not self.ml_active.is_set():
                        continue
                    self.ml_result = result
                    self.ml_history.appendleft(result)
                    self.ml_message = 'Live model output'
                    self.enqueue_log('ml', (datetime.now().isoformat(timespec='milliseconds'),
                                           result['label'], result['support'], rssi))
            except Exception as error:
                with self.lock:
                    self.ml_error = True
                    self.ml_result = None
                    self.ml_message = f'ML prediction failed: {error}'
                return

    def _logger(self):
        if self.demo:
            return
        db = handle = None
        csv_error = db_error = ''
        try:
            db = sqlite3.connect(self.data_dir / 'wisense.db', timeout=.5)
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('''CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                activity_score REAL NOT NULL, baseline REAL NOT NULL, threshold REAL NOT NULL,
                rssi_dbm INTEGER, status TEXT NOT NULL, movement_events INTEGER NOT NULL, posture TEXT)''')
            db.execute('CREATE INDEX IF NOT EXISTS idx_sensor_readings_timestamp ON sensor_readings(timestamp)')
            db.execute('''CREATE TABLE IF NOT EXISTS ml_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                prediction TEXT NOT NULL, model_support REAL NOT NULL, rssi_dbm INTEGER)''')
            db.commit()
        except sqlite3.Error as error:
            db_error = f'SQLite: {error}'
            if db is not None:
                db.close()
            db = None
        try:
            path = self.data_dir / 'wisense_log.csv'
            exists = path.exists() and path.stat().st_size > 0
            if exists:
                with path.open(newline='', encoding='utf-8') as f:
                    if next(csv.reader(f), []) != CSV_HEADER:
                        raise ValueError('Existing CSV header differs; CSV logging disabled to protect it.')
            handle = path.open('a', newline='', encoding='utf-8')
            writer = csv.writer(handle)
            if not exists:
                writer.writerow(CSV_HEADER)
                handle.flush()
        except (OSError, ValueError) as error:
            csv_error = f'CSV: {error}'
        try:
            # Drain entries until BOTH producers have stopped, including their final rows.
            while (not self.stop.is_set() or not self.log_queue.empty() or
                   any(t.is_alive() and t is not threading.current_thread() for t in self.threads)):
                self.log_error = ' | '.join(e for e in (db_error, csv_error) if e)
                try:
                    kind, row = self.log_queue.get(timeout=.1)
                except queue.Empty:
                    continue
                if db is not None:
                    try:
                        if kind == 'main':
                            db.execute('''INSERT INTO sensor_readings
                                (timestamp,activity_score,baseline,threshold,rssi_dbm,status,movement_events,posture)
                                VALUES (?,?,?,?,?,?,?,?)''', row)
                        else:
                            db.execute('''INSERT INTO ml_predictions
                                (timestamp,prediction,model_support,rssi_dbm) VALUES (?,?,?,?)''', row)
                        db.commit()
                        db_error = ''
                    except sqlite3.Error as error:
                        db.rollback()
                        db_error = f'SQLite write failed: {error}'
                if handle is not None and kind == 'main':
                    try:
                        writer.writerow(row[:7])
                        handle.flush()
                        csv_error = ''
                    except OSError as error:
                        csv_error = f'CSV write failed: {error}'
        finally:
            if handle is not None:
                handle.close()
            if db is not None:
                db.close()


class Dashboard:
    def __init__(self, root, engine):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.tk, self.root, self.engine = tk, root, engine
        self.tab = 'main'
        self.closed = False
        self.last_graph = 0
        self.last_ml_stamp = None
        root.title('WiSense 2.0 | Movement Monitor')
        root.geometry('1220x800')
        root.minsize(1000, 720)
        root.configure(bg=BG)
        root.option_add('*Font', ('Segoe UI', 10))
        top = tk.Frame(root, bg=BG)
        top.pack(fill='x', padx=30, pady=(22, 12))
        tk.Label(top, text='WiSense', fg=TEXT, bg=BG,
                 font=('Segoe UI', 26, 'bold')).pack(side='left')
        tk.Label(top, text=' /  INDOOR SENSING', fg=MUTED, bg=BG,
                 font=('Segoe UI', 11)).pack(side='left', padx=16)
        self.badge = tk.Label(top, text='CONNECTING', bg=CARD, fg=BLUE, padx=14, pady=8)
        self.badge.pack(side='right')
        nav = tk.Frame(root, bg=BG)
        nav.pack(fill='x', padx=30, pady=(0, 18))
        self.buttons = {}
        for key, title in [('main', 'Main Dashboard'), ('ml', 'ML Experiment')]:
            b = tk.Button(nav, text=title, command=lambda k=key: self.select_tab(k),
                          relief='flat', bd=0, padx=22, pady=10, cursor='hand2',
                          activebackground=BORDER, activeforeground=TEXT)
            b.pack(side='left', padx=(0, 8))
            self.buttons[key] = b
        tk.Button(nav, text='Recalibrate main  [R]', command=engine.reset_request.set,
                  bg=CARD, fg=MUTED, relief='flat', padx=14, pady=10,
                  activebackground=BORDER, activeforeground=TEXT, cursor='hand2').pack(side='right')
        self.footer = tk.Label(root, text='', bg=BG, fg=MUTED, anchor='w', padx=30, pady=12)
        self.footer.pack(side='bottom', fill='x')
        self.container = tk.Frame(root, bg=BG)
        self.container.pack(fill='both', expand=True, padx=30)
        self.pages = {k: tk.Frame(self.container, bg=BG) for k in ('main', 'ml')}
        main = self.pages['main']
        self.banner = tk.Frame(main, bg='#142a42', padx=24, pady=15)
        self.banner.pack(fill='x', pady=(0, 14))
        self.status = tk.Label(self.banner, text='Waiting for CSI', bg='#142a42', fg=BLUE,
                               font=('Segoe UI', 24, 'bold'), anchor='w')
        self.status.pack(fill='x')
        self.subtitle = tk.Label(self.banner, text='Keep the sensing area empty during calibration.',
                                 bg='#142a42', fg=MUTED, anchor='w')
        self.subtitle.pack(fill='x', pady=(5, 0))
        automation = tk.Frame(main, bg=CARD, padx=14, pady=8,
                              highlightbackground=BORDER, highlightthickness=1)
        automation.pack(fill='x', pady=(0, 12))
        self.light_button = tk.Button(automation, text='Home Automation: OFF',
                                      command=engine.toggle_light, bg=BORDER, fg=TEXT,
                                      relief='flat', padx=14, pady=8, cursor='hand2',
                                      activebackground=BLUE, activeforeground=BG)
        self.light_button.pack(side='left')
        self.lamp_icon = tk.Canvas(automation, bg=CARD, width=74, height=26, highlightthickness=0)
        self.lamp_icon.pack(side='left', padx=14)
        self.lamp_shape = self.lamp_icon.create_rectangle(7, 6, 67, 20, fill=BORDER, outline=MUTED, width=2)
        self.lamp_icon.create_line(11, 3, 11, 23, fill=MUTED, width=2)
        self.lamp_icon.create_line(63, 3, 63, 23, fill=MUTED, width=2)
        self.light_note = tk.Label(automation, text='', bg=CARD, fg=MUTED, anchor='w',
                                  wraplength=650, justify='left')
        self.light_note.pack(side='left', fill='x', expand=True)
        metrics = tk.Frame(main, bg=BG)
        metrics.pack(fill='x', pady=(0, 16))
        self.values = {}
        for i, (key, label) in enumerate([('score', 'LIVE SCORE'), ('baseline', 'BASELINE'),
                 ('threshold', 'THRESHOLD'), ('rssi', 'WI-FI SIGNAL'), ('events', 'EVENTS'), ('last', 'LAST EVENT')]):
            metrics.columnconfigure(i, weight=1, uniform='cards')
            card = tk.Frame(metrics, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
            card.grid(row=0, column=i, sticky='nsew', padx=(0 if i == 0 else 5, 0 if i == 5 else 5))
            tk.Label(card, text=label, fg=MUTED, bg=CARD,
                     font=('Segoe UI', 9)).pack(anchor='w', padx=12, pady=(13, 4))
            value = tk.Label(card, text='--', fg=TEXT, bg=CARD, font=('Segoe UI', 20, 'bold'))
            value.pack(anchor='w', padx=12, pady=(0, 12))
            self.values[key] = value
            if key == 'rssi':
                self.signal = tk.Canvas(card, bg=CARD, width=46, height=12, highlightthickness=0)
                self.signal.place(relx=1, x=-54, y=13)
                self.bars = [self.signal.create_rectangle(2+i*10, 10-i*3, 8+i*10, 12,
                                                          fill=BORDER, outline='') for i in range(4)]
        graph_card = tk.Frame(main, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        graph_card.pack(fill='both', expand=True)
        tk.Label(graph_card, text='ACTIVITY OVER TIME', bg=CARD, fg=TEXT,
                 font=('Segoe UI', 11, 'bold')).pack(anchor='w', padx=18, pady=(14, 0))
        self.figure = Figure(figsize=(10, 3), dpi=100, facecolor=CARD)
        self.axis = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=.07, right=.98, bottom=.20, top=.92)
        self.axis.set_facecolor(CARD)
        for spine in self.axis.spines.values():
            spine.set_color(BORDER)
        self.axis.tick_params(colors=MUTED, labelsize=9)
        self.axis.grid(color=BORDER, alpha=.65, linewidth=.6)
        self.axis.set_xlabel('Seconds ago', color=MUTED, fontsize=9)
        self.axis.set_ylabel('CSI variation', color=MUTED, fontsize=9)
        self.axis.set_xlim(-60, 0)
        self.axis.set_ylim(0, 30)
        self.score_line, = self.axis.plot([], [], color=BLUE, lw=1.8, label='Live score')
        self.base_line, = self.axis.plot([], [], color=MUTED, lw=1, ls='--', label='Baseline')
        self.threshold_line, = self.axis.plot([], [], color=YELLOW, lw=1.2, ls='--', label='Threshold')
        legend = self.axis.legend(loc='upper right', facecolor=CARD, edgecolor=BORDER,
                                   ncol=3, fontsize=8)
        for t in legend.get_texts():
            t.set_color(TEXT)
        self.canvas = FigureCanvasTkAgg(self.figure, graph_card)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        ml = self.pages['ml']
        tk.Label(ml, text='POSTURE EXPERIMENT', bg=BG, fg=PURPLE,
                 font=('Segoe UI', 11, 'bold')).pack(anchor='w')
        tk.Label(ml, text='Fixed-setup experiment. EMPTY needs a model trained with EMPTY data.\n'
                 'Sitting is outside this model. Model support is not measured accuracy.',
                 bg=BG, fg=MUTED, justify='left').pack(anchor='w', pady=(6, 14))
        mlcard = tk.Frame(ml, bg=CARD, padx=24, pady=20,
                          highlightbackground=BORDER, highlightthickness=1)
        mlcard.pack(fill='x')
        self.ml_title = tk.Label(mlcard, text='Waiting for model', bg=CARD, fg=PURPLE,
                                font=('Segoe UI', 28, 'bold'), anchor='w')
        self.ml_title.pack(fill='x')
        self.ml_info = tk.Label(mlcard, text='', bg=CARD, fg=MUTED, anchor='w',
                               justify='left', wraplength=950)
        self.ml_info.pack(fill='x', pady=(6, 14))
        self.support_labels, self.support_bars = {}, {}
        style = ttk.Style(root)
        style.theme_use('clam')
        for label, color in [('EMPTY', MUTED), ('STANDING', PURPLE), ('MOVING', GREEN)]:
            row = tk.Frame(mlcard, bg=CARD)
            row.pack(fill='x', pady=4)
            val = tk.Label(row, text=label, width=22, anchor='w', bg=CARD, fg=color)
            val.pack(side='left')
            style.configure(label+'.Horizontal.TProgressbar', background=color,
                            troughcolor=BORDER, bordercolor=CARD, lightcolor=color, darkcolor=color)
            bar = ttk.Progressbar(row, style=label+'.Horizontal.TProgressbar', maximum=100)
            bar.pack(side='left', fill='x', expand=True)
            self.support_labels[label], self.support_bars[label] = val, bar
        tk.Label(ml, text='RECENT PREDICTIONS', bg=BG, fg=MUTED,
                 font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(18, 8))
        style.configure('ML.Treeview', background=CARD, fieldbackground=CARD, foreground=TEXT,
                        rowheight=28, borderwidth=0, font=('Segoe UI', 10))
        style.configure('ML.Treeview.Heading', background=BORDER, foreground=TEXT,
                        font=('Segoe UI', 10, 'bold'))
        self.table = ttk.Treeview(ml, columns=('time', 'prediction', 'support', 'rssi'),
                                 show='headings', height=6, style='ML.Treeview')
        for col, title in [('time', 'TIME'), ('prediction', 'PREDICTION'),
                           ('support', 'MODEL SUPPORT'), ('rssi', 'RSSI')]:
            self.table.heading(col, text=title)
            self.table.column(col, width=180, anchor='center')
        self.table.pack(fill='both', expand=True)
        self.select_tab('main')
        root.bind('<KeyPress-r>', lambda e: engine.reset_request.set())
        root.bind('<KeyPress-R>', lambda e: engine.reset_request.set())
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.update()

    def select_tab(self, key):
        self.tab = key
        for name, page in self.pages.items():
            page.pack_forget()
            self.buttons[name].configure(bg=BLUE if name == key else CARD,
                                          fg=BG if name == key else MUTED)
        self.pages[key].pack(fill='both', expand=True)
        self.engine.set_ml_active(key == 'ml')
        self.last_graph = 0
        self.last_ml_stamp = None

    def update(self):
        if self.closed:
            return
        s = self.engine.snapshot()
        now = time.monotonic()
        badge = 'PREVIEW / SYNTHETIC' if self.engine.demo else ('CSI LIVE' if s['healthy'] else 'NO DATA')
        self.badge.configure(text=badge, fg=GREEN if s['healthy'] else YELLOW)
        age = '--' if s['age'] is None else f"{s['age']:.1f}s"
        storage = 'Preview: logging OFF' if self.engine.demo else 'SQLite + CSV'
        self.footer.configure(text=f"{storage}   |   {s['rate']:.1f} frames/s   |   Last CSI: {age}   |   "
                              + (s['log_error'] or s['link']), fg=YELLOW if s['log_error'] else MUTED)
        if self.tab == 'main':
            self.light_button.configure(text='Home Automation: ON' if s['light_enabled'] else 'Home Automation: OFF',
                                        bg=BLUE if s['light_enabled'] else BORDER,
                                        fg=BG if s['light_enabled'] else TEXT)
            self.light_note.configure(text=s['light_text'])
            self.lamp_icon.itemconfigure(self.lamp_shape, fill='#fff3b0' if s['light_on'] else BORDER)
            color = COLORS.get(s['code'], BLUE)
            self.status.configure(text=s['title'], fg=color)
            note = 'Keep the area empty; baseline and threshold lock after calibration.'
            if s['code'] in COLORS:
                note = 'Receiver: two high readings turn green on; two low readings turn red on.'
            elif s['code'] == 'NO_DATA':
                note = 'Check sender power, receiver USB and COM port. Old readings are not treated as live.'
            self.subtitle.configure(text=note)
            for key in ('score', 'baseline', 'threshold'):
                self.values[key].configure(text='--' if s[key] is None else f'{s[key]:.2f}')
            self.values['score'].configure(fg=color)
            self.values['events'].configure(text=str(s['events']))
            self.values['last'].configure(text=s['last_event'])
            rssi = s['rssi']
            self.values['rssi'].configure(text='--' if rssi is None else f'{rssi} dBm')
            count = 0 if rssi is None else 4 if rssi >= -55 else 3 if rssi >= -65 else 2 if rssi >= -75 else 1
            for i, bar in enumerate(self.bars):
                self.signal.itemconfigure(bar, fill=(GREEN if count >= 3 else YELLOW) if i < count else BORDER)
            if now - self.last_graph >= GRAPH_INTERVAL_SECONDS:
                h = [(t-now, v) for t, v in s['history'] if t >= now-60]
                self.score_line.set_data([x for x, _ in h], [v for _, v in h])
                for line, value in [(self.base_line, s['baseline']), (self.threshold_line, s['threshold'])]:
                    line.set_data([-60, 0] if value is not None else [], [value, value] if value is not None else [])
                target = max([10] + [v*1.2 for _, v in h] + [v*1.2 for v in (s['threshold'],) if v is not None])
                # Stable rounded bounds rather than axis rescaling at every sample.
                current = self.axis.get_ylim()[1]
                if target > current or target < current*.55:
                    self.axis.set_ylim(0, math.ceil(target/10)*10)
                self.canvas.draw_idle()
                self.last_graph = now
        else:
            ml = s['ml']
            title = ml['label'] if ml else 'Model unavailable' if s['ml_error'] else 'Waiting for fresh CSI'
            color = GREEN if title == 'MOVING' else MUTED if title == 'EMPTY' else PURPLE
            self.ml_title.configure(text=title, fg=color)
            info = s['ml_message']
            if ml:
                info = f"Model support: {ml['support']:.0%}   |   Updated: {ml['timestamp']}   |   12-frame window"
            elif not s['healthy'] and not s['ml_error']:
                info = 'No fresh CSI. Check the receiver and sender.'
            self.ml_info.configure(text=info)
            for label in self.support_bars:
                supported = ml is not None and label in ml['probabilities']
                value = ml['probabilities'][label]*100 if supported else 0
                self.support_bars[label]['value'] = value
                self.support_labels[label].configure(text=f'{label}   {value:.0f}%' if supported else label+'   --')
            stamp = s['ml_history'][0]['monotonic'] if s['ml_history'] else None
            if stamp != self.last_ml_stamp:
                self.table.delete(*self.table.get_children())
                for r in s['ml_history']:
                    self.table.insert('', 'end', values=(r['timestamp'], r['label'], f"{r['support']:.0%}", r['rssi']))
                self.last_ml_stamp = stamp
        self.root.after(UI_INTERVAL_MS, self.update)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.engine.stop.set()
        self.badge.configure(text='CLOSING')
        self._wait_close()

    def _wait_close(self):
        if any(t.is_alive() for t in self.engine.threads):
            self.root.after(100, self._wait_close)
        else:
            self.root.destroy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--port', default='COM7')
    p.add_argument('--model', default=str(ROOT / 'wisense_posture_model.joblib'))
    p.add_argument('--demo', action='store_true', help='Synthetic UI preview; no serial, ML or logging')
    args = p.parse_args()
    import tkinter as tk
    root = tk.Tk()
    engine = Engine(args.port, args.model, demo=args.demo)
    dashboard = Dashboard(root, engine)
    engine.start()
    try:
        root.mainloop()
    except KeyboardInterrupt:
        dashboard.close()
        root.mainloop()
    finally:
        engine.stop.set()


if __name__ == '__main__':
    main()
