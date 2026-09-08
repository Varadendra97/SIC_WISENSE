import csv
import sqlite3
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle


# --------------------------- SETTINGS ---------------------------
APP_VERSION = "1.6 SQLITE"

PORT = "COM7"
BAUD_RATE = 115200
TARGET_SUBCARRIERS = 64

WINDOW_SIZE = 12
WARMUP_SCORES = 50
CALIBRATION_SCORES = 60
REQUIRED_HIGH_READINGS = 2
RECENT_ACTIVITY_SECONDS = 30

PLOT_HISTORY = 150
HEATMAP_HISTORY = 80
SERIAL_LINES_PER_UPDATE = 40
UPDATE_INTERVAL_MS = 300
LOG_INTERVAL_SECONDS = 0.50

# CSV backup/export
LOG_FILE = Path(__file__).resolve().with_name("wisense_log.csv")

# SQLite database
DB_FILE = Path(__file__).resolve().with_name("wisense.db")


# --------------------------- CSI PARSING ---------------------------

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

        raw = np.fromstring(
            raw_text,
            sep=",",
            dtype=np.float64
        )

        if raw.size < 20 or raw.size % 2 != 0:
            return None

        # ESP32 CSI order:
        # imaginary, real, imaginary, real, ...
        imaginary = raw[0::2]
        real = raw[1::2]

        amplitude = np.hypot(real, imaginary)

        # Use the first 64 subcarriers consistently.
        if amplitude.size < TARGET_SUBCARRIERS:
            return None

        amplitude = amplitude[:TARGET_SUBCARRIERS]

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


# --------------------------- ACTIVITY SCORE ---------------------------

def calculate_activity_score(csi_window):
    """Measure short-term variation across usable CSI subcarriers."""

    matrix = np.asarray(
        csi_window,
        dtype=np.float64
    )

    usable_subcarriers = np.all(
        matrix > 0,
        axis=0
    )

    if np.count_nonzero(usable_subcarriers) < 10:
        return None

    variation = np.std(
        matrix[:, usable_subcarriers],
        axis=0
    )

    return float(
        np.mean(variation) * 100.0
    )


# --------------------------- CALIBRATION ---------------------------

def calculate_fixed_limits(calibration_scores):
    """Calculate a stable baseline and one fixed movement threshold."""

    scores = np.asarray(
        calibration_scores,
        dtype=np.float64
    )

    baseline = float(
        np.median(scores)
    )

    quiet_upper = float(
        np.percentile(scores, 90)
    )

    margin = max(
        2.0,
        baseline * 0.20
    )

    candidate = quiet_upper + margin

    safety_cap = baseline + max(
        8.0,
        baseline * 0.60
    )

    threshold = min(
        candidate,
        safety_cap
    )

    threshold = max(
        threshold,
        baseline + 2.0
    )

    return baseline, threshold


# --------------------------- DASHBOARD ---------------------------

class WiSenseDashboard:

    def __init__(self, receiver):

        self.receiver = receiver

        # CSI rolling data
        self.csi_window = deque(
            maxlen=WINDOW_SIZE
        )

        self.csi_history = deque(
            maxlen=HEATMAP_HISTORY
        )

        self.score_history = deque(
            maxlen=PLOT_HISTORY
        )

        self.calibration_scores = []

        # Processing state
        self.warmup_count = 0

        self.baseline = None
        self.threshold = None
        self.live_score = None
        self.rssi = None

        self.high_count = 0

        self.last_movement_time = 0.0

        self.movement_events = 0
        self.was_moving = False

        self.last_event_text = "--"

        # Serial errors
        self.serial_error = None

        # Logging state
        self.last_log_time = 0.0
        self.log_error = None

        # SQLite
        self.db_conn = None
        self.db_error = None

        self._init_database()

        # Status
        self.phase_text = "WAITING FOR CSI"
        self.phase_color = "#2563eb"

        # --------------------------- FIGURE ---------------------------

        self.figure = plt.figure(
            figsize=(14, 8),
            facecolor="#08111f"
        )

        try:
            self.figure.canvas.manager.set_window_title(
                f"WiSense {APP_VERSION} - Device-Free Movement Dashboard"
            )
        except AttributeError:
            pass

        grid = self.figure.add_gridspec(
            3,
            2,
            height_ratios=(0.9, 0.9, 4.0),
            hspace=0.32,
            wspace=0.25,
        )

        self.status_axis = self.figure.add_subplot(
            grid[0, :]
        )

        self.metrics_axis = self.figure.add_subplot(
            grid[1, :]
        )

        self.score_axis = self.figure.add_subplot(
            grid[2, 0]
        )

        self.heatmap_axis = self.figure.add_subplot(
            grid[2, 1]
        )

        self._build_status_panel()
        self._build_metrics_panel()
        self._build_score_plot()
        self._build_heatmap()

        self.figure.suptitle(
            f"WiSense | Device-Free Indoor Movement Detection | {APP_VERSION}",
            color="#f8fafc",
            fontsize=17,
            fontweight="bold",
            y=0.98,
        )

        self.figure.text(
            0.5,
            0.018,
            "SQLite + CSV  |  Press R to recalibrate  |  "
            "Close the window or press Ctrl+C to stop",
            ha="center",
            color="#94a3b8",
            fontsize=9,
        )

        self.figure.subplots_adjust(
            top=0.91,
            bottom=0.09,
            left=0.07,
            right=0.96
        )

        self.figure.canvas.mpl_connect(
            "key_press_event",
            self._on_key_press
        )

        self.figure.canvas.mpl_connect(
            "close_event",
            self._on_close
        )

        self.animation = None

    # --------------------------- SQLITE ---------------------------

    def _init_database(self):
        """Create the local SQLite database and table."""

        try:

            self.db_conn = sqlite3.connect(
                DB_FILE
            )

            self.db_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sensor_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    activity_score REAL NOT NULL,
                    baseline REAL NOT NULL,
                    threshold REAL NOT NULL,
                    rssi_dbm INTEGER,
                    status TEXT NOT NULL,
                    movement_events INTEGER NOT NULL,
                    posture TEXT
                )
                """
            )

            self.db_conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sensor_readings_timestamp
                ON sensor_readings(timestamp)
                """
            )

            self.db_conn.commit()

            print(
                f"SQLite database ready: {DB_FILE}"
            )

            self.db_error = None

        except sqlite3.Error as error:

            self.db_error = str(error)

            print(
                f"SQLite database error: {error}"
            )

            self.db_conn = None

    def _write_database_row(self, reading):
        """Insert one processed sensor reading into SQLite."""

        if self.db_conn is None:
            return

        try:

            self.db_conn.execute(
                """
                INSERT INTO sensor_readings (
                    timestamp,
                    activity_score,
                    baseline,
                    threshold,
                    rssi_dbm,
                    status,
                    movement_events,
                    posture
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reading["timestamp"],
                    reading["score"],
                    reading["baseline"],
                    reading["threshold"],
                    reading["rssi"],
                    reading["status"],
                    reading["events"],
                    reading["posture"],
                ),
            )

            self.db_conn.commit()

            self.db_error = None

        except sqlite3.Error as error:

            if self.db_error is None:
                print(
                    f"SQLite logging error: {error}"
                )

            self.db_error = str(error)

    # --------------------------- STATUS PANEL ---------------------------

    def _build_status_panel(self):

        self.status_axis.set_axis_off()

        self.status_box = Rectangle(
            (0, 0),
            1,
            1,
            transform=self.status_axis.transAxes,
            facecolor=self.phase_color,
            edgecolor="none",
        )

        self.status_axis.add_patch(
            self.status_box
        )

        self.status_text_artist = self.status_axis.text(
            0.5,
            0.5,
            self.phase_text,
            transform=self.status_axis.transAxes,
            ha="center",
            va="center",
            color="#ffffff",
            fontsize=21,
            fontweight="bold",
        )

    # --------------------------- METRICS PANEL ---------------------------

    def _build_metrics_panel(self):

        self.metrics_axis.set_axis_off()

        labels = (
            "LIVE SCORE",
            "BASELINE",
            "THRESHOLD",
            "RSSI",
            "EVENTS",
            "LAST EVENT",
        )

        positions = np.linspace(
            0.07,
            0.93,
            len(labels)
        )

        self.metric_values = []

        for position, label in zip(
            positions,
            labels
        ):

            self.metrics_axis.text(
                position,
                0.76,
                label,
                transform=self.metrics_axis.transAxes,
                ha="center",
                va="center",
                color="#94a3b8",
                fontsize=9,
            )

            value_artist = self.metrics_axis.text(
                position,
                0.30,
                "--",
                transform=self.metrics_axis.transAxes,
                ha="center",
                va="center",
                color="#f8fafc",
                fontsize=15,
                fontweight="bold",
            )

            self.metric_values.append(
                value_artist
            )

        # RSSI signal bars
        rssi_position = positions[3]

        self.metric_values[3].set_position(
            (
                rssi_position + 0.025,
                0.30
            )
        )

        self.rssi_bars = []

        bar_start_x = (
            rssi_position - 0.065
        )

        for index in range(4):

            height = (
                0.10 +
                index * 0.075
            )

            signal_bar = Rectangle(
                (
                    bar_start_x + index * 0.012,
                    0.20
                ),
                0.008,
                height,
                transform=self.metrics_axis.transAxes,
                facecolor="#334155",
                edgecolor="none",
            )

            self.metrics_axis.add_patch(
                signal_bar
            )

            self.rssi_bars.append(
                signal_bar
            )

    # --------------------------- AXIS STYLING ---------------------------

    def _style_plot_axis(self, axis):

        axis.set_facecolor(
            "#0f1b2d"
        )

        axis.tick_params(
            colors="#cbd5e1",
            labelsize=9
        )

        for spine in axis.spines.values():

            spine.set_color(
                "#334155"
            )

        axis.grid(
            color="#334155",
            alpha=0.35,
            linewidth=0.8
        )

    # --------------------------- SCORE GRAPH ---------------------------

    def _build_score_plot(self):

        self._style_plot_axis(
            self.score_axis
        )

        self.score_axis.set_title(
            "Activity Score",
            color="#f8fafc",
            fontsize=12,
            fontweight="bold"
        )

        self.score_axis.set_xlabel(
            "Recent processed windows",
            color="#cbd5e1"
        )

        self.score_axis.set_ylabel(
            "CSI variation score",
            color="#cbd5e1"
        )

        self.score_axis.set_xlim(
            0,
            PLOT_HISTORY - 1
        )

        self.score_axis.set_ylim(
            0,
            30
        )

        (
            self.score_line,
        ) = self.score_axis.plot(
            [],
            [],
            color="#38bdf8",
            linewidth=2.0,
            label="Live score"
        )

        (
            self.baseline_line,
        ) = self.score_axis.plot(
            [],
            [],
            color="#e2e8f0",
            linewidth=1.3,
            linestyle="--",
            label="Baseline"
        )

        (
            self.threshold_line,
        ) = self.score_axis.plot(
            [],
            [],
            color="#fb923c",
            linewidth=1.8,
            linestyle="--",
            label="Threshold"
        )

        legend = self.score_axis.legend(
            loc="upper left",
            facecolor="#0f1b2d",
            edgecolor="#334155",
            fontsize=8,
        )

        for text_artist in legend.get_texts():

            text_artist.set_color(
                "#e2e8f0"
            )

    # --------------------------- HEATMAP ---------------------------

    def _build_heatmap(self):

        self._style_plot_axis(
            self.heatmap_axis
        )

        self.heatmap_axis.grid(
            False
        )

        self.heatmap_axis.set_title(
            "CSI Subcarrier Heatmap",
            color="#f8fafc",
            fontsize=12,
            fontweight="bold"
        )

        self.heatmap_axis.set_xlabel(
            "Recent CSI packets",
            color="#cbd5e1"
        )

        self.heatmap_axis.set_ylabel(
            "Subcarrier index",
            color="#cbd5e1"
        )

        initial = np.ones(
            (64, 2),
            dtype=np.float64
        )

        self.heatmap_image = self.heatmap_axis.imshow(
            initial,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
            vmin=0.5,
            vmax=1.5,
            extent=(
                0,
                HEATMAP_HISTORY,
                0,
                initial.shape[0]
            ),
        )

        colorbar = self.figure.colorbar(
            self.heatmap_image,
            ax=self.heatmap_axis,
            pad=0.025,
            fraction=0.05
        )

        colorbar.set_label(
            "Normalized amplitude",
            color="#cbd5e1",
            fontsize=9
        )

        colorbar.ax.tick_params(
            colors="#cbd5e1",
            labelsize=8
        )

        colorbar.outline.set_edgecolor(
            "#334155"
        )

    # --------------------------- CALIBRATION RESET ---------------------------

    def reset_calibration(self):

        self.csi_window.clear()
        self.score_history.clear()
        self.calibration_scores.clear()

        self.warmup_count = 0

        self.baseline = None
        self.threshold = None
        self.live_score = None

        self.high_count = 0

        self.last_movement_time = 0.0

        self.was_moving = False

        self.last_log_time = 0.0

        self.phase_text = (
            "RECALIBRATING - KEEP AREA EMPTY"
        )

        self.phase_color = "#2563eb"

    # --------------------------- KEY PRESS ---------------------------

    def _on_key_press(self, event):

        if event.key and event.key.lower() == "r":

            self.reset_calibration()

    # --------------------------- WINDOW CLOSE ---------------------------

    def _on_close(self, _event):

        if (
            self.receiver is not None
            and self.receiver.is_open
        ):

            self.receiver.close()

        if self.db_conn is not None:

            self.db_conn.close()

            self.db_conn = None

    # --------------------------- CSI PROCESSING ---------------------------

    def process_csi(
        self,
        rssi,
        amplitude
    ):
        """Process one already-parsed CSI amplitude frame."""

        # Protect rolling window from changing CSI lengths.
        if (
            self.csi_window
            and amplitude.size
            != self.csi_window[0].size
        ):

            self.csi_window.clear()
            self.csi_history.clear()

        self.rssi = rssi

        self.csi_window.append(
            amplitude
        )

        self.csi_history.append(
            amplitude
        )

        # Wait for enough samples.
        if len(self.csi_window) < WINDOW_SIZE:

            self.phase_text = (
                f"COLLECTING CSI "
                f"{len(self.csi_window)}/"
                f"{WINDOW_SIZE}"
            )

            self.phase_color = "#2563eb"

            return

        # Calculate activity score.
        score = calculate_activity_score(
            self.csi_window
        )

        if score is None:
            return

        self.live_score = score

        self.score_history.append(
            score
        )

        # Warm-up period.
        if (
            self.threshold is None
            and self.warmup_count < WARMUP_SCORES
        ):

            self.warmup_count += 1

            self.phase_text = (
                f"STABILIZING "
                f"{self.warmup_count}/"
                f"{WARMUP_SCORES} "
                "- KEEP AREA EMPTY"
            )

            self.phase_color = "#2563eb"

            return

        # Calibration.
        if self.threshold is None:

            self.calibration_scores.append(
                score
            )

            self.phase_text = (
                f"CALIBRATING "
                f"{len(self.calibration_scores)}/"
                f"{CALIBRATION_SCORES} "
                "- KEEP AREA EMPTY"
            )

            self.phase_color = "#d97706"

            if (
                len(self.calibration_scores)
                >= CALIBRATION_SCORES
            ):

                (
                    self.baseline,
                    self.threshold
                ) = calculate_fixed_limits(
                    self.calibration_scores
                )

                self.phase_text = (
                    "RED | NO MOVEMENT"
                )

                self.phase_color = "#b91c1c"

            return

        # --------------------------- MOVEMENT DETECTION ---------------------------

        current_time = time.time()

        if score > self.threshold:

            self.high_count += 1

        else:

            self.high_count = 0

        moving = (
            self.high_count
            >= REQUIRED_HIGH_READINGS
        )

        if moving:

            self.last_movement_time = (
                current_time
            )

            self.phase_text = (
                "GREEN | MOVEMENT DETECTED"
            )

            self.phase_color = "#15803d"

            if not self.was_moving:

                self.movement_events += 1

                self.last_event_text = (
                    datetime.now().strftime(
                        "%H:%M:%S"
                    )
                )

        elif (
            self.last_movement_time > 0
            and
            current_time
            - self.last_movement_time
            < RECENT_ACTIVITY_SECONDS
        ):

            self.phase_text = (
                "YELLOW | RECENT MOVEMENT"
            )

            self.phase_color = "#ca8a04"

        else:

            self.phase_text = (
                "RED | NO MOVEMENT"
            )

            self.phase_color = "#b91c1c"

        self.was_moving = moving

    # --------------------------- SERIAL READER ---------------------------

    def _consume_serial(self):

        processed = 0

        try:

            while (
                self.receiver.in_waiting > 0
                and
                processed
                < SERIAL_LINES_PER_UPDATE
            ):

                line = (
                    self.receiver.readline()
                    .decode(
                        "utf-8",
                        errors="ignore"
                    )
                    .strip()
                )

                result = parse_csi(
                    line
                )

                processed += 1

                if result is None:
                    continue

                (
                    _sequence,
                    rssi,
                    amplitude
                ) = result

                self.process_csi(
                    rssi,
                    amplitude
                )

        except serial.SerialException as error:

            self.serial_error = str(error)

            self.phase_text = (
                "SERIAL CONNECTION ERROR"
            )

            self.phase_color = "#7f1d1d"

    # --------------------------- FORMATTING ---------------------------

    @staticmethod
    def _format_number(
        value,
        digits=2
    ):

        if value is None:
            return "--"

        return (
            f"{value:.{digits}f}"
        )

    # --------------------------- RSSI VISUAL ---------------------------

    @staticmethod
    def _rssi_visual(rssi):
        """Return mobile-style bar count and colour for RSSI."""

        if rssi is None:

            return 0, "#334155"

        if rssi >= -55:

            return 4, "#22c55e"

        if rssi >= -65:

            return 3, "#22c55e"

        if rssi >= -75:

            return 2, "#eab308"

        if rssi >= -85:

            return 1, "#f97316"

        return 1, "#ef4444"

    # --------------------------- METRICS UPDATE ---------------------------

    def _update_metrics(self):

        values = (
            self._format_number(
                self.live_score
            ),

            self._format_number(
                self.baseline
            ),

            self._format_number(
                self.threshold
            ),

            (
                "--"
                if self.rssi is None
                else f"{self.rssi} dBm"
            ),

            str(
                self.movement_events
            ),

            self.last_event_text,
        )

        for (
            artist,
            value
        ) in zip(
            self.metric_values,
            values
        ):

            artist.set_text(
                value
            )

        (
            active_bars,
            active_colour
        ) = self._rssi_visual(
            self.rssi
        )

        for (
            index,
            signal_bar
        ) in enumerate(
            self.rssi_bars
        ):

            signal_bar.set_facecolor(
                active_colour
                if index < active_bars
                else "#334155"
            )

        self.metric_values[3].set_color(
            active_colour
            if self.rssi is not None
            else "#f8fafc"
        )

    # --------------------------- DATABASE + CSV LOGGING ---------------------------

    def _write_log_row(self):
        """Save one reading to SQLite and CSV."""

        # Don't log until calibration is complete.
        if (
            self.threshold is None
            or self.live_score is None
        ):

            return

        current_time = time.time()

        # Logging interval.
        if (
            current_time
            - self.last_log_time
            < LOG_INTERVAL_SECONDS
        ):

            return

        # Convert dashboard status into database status.
        if self.phase_text.startswith(
            "GREEN"
        ):

            status = "MOVEMENT"

        elif self.phase_text.startswith(
            "YELLOW"
        ):

            status = "RECENT_MOVEMENT"

        else:

            status = "NO_MOVEMENT"

        timestamp = (
            datetime.now()
            .isoformat(
                timespec="milliseconds"
            )
        )

        # One common reading object.
        reading = {

            "timestamp": timestamp,

            "score": round(
                float(self.live_score),
                3
            ),

            "baseline": round(
                float(self.baseline),
                3
            ),

            "threshold": round(
                float(self.threshold),
                3
            ),

            "rssi": (
                None
                if self.rssi is None
                else int(self.rssi)
            ),

            "status": status,

            "events": int(
                self.movement_events
            ),

            "posture": "UNKNOWN",
        }

        # --------------------------- SQLITE ---------------------------

        self._write_database_row(
            reading
        )

        # --------------------------- CSV BACKUP ---------------------------

        try:

            needs_header = (
                not LOG_FILE.exists()
                or LOG_FILE.stat().st_size == 0
            )

            with LOG_FILE.open(
                "a",
                newline="",
                encoding="utf-8"
            ) as log_handle:

                writer = csv.writer(
                    log_handle
                )

                if needs_header:

                    writer.writerow(
                        (
                            "timestamp",
                            "activity_score",
                            "baseline",
                            "threshold",
                            "rssi_dbm",
                            "status",
                            "movement_events",
                        )
                    )

                writer.writerow(
                    (
                        timestamp,

                        f"{self.live_score:.3f}",

                        f"{self.baseline:.3f}",

                        f"{self.threshold:.3f}",

                        self.rssi,

                        status,

                        self.movement_events,
                    )
                )

            self.log_error = None

        except OSError as error:

            if self.log_error is None:

                print(
                    f"CSV logging error: {error}"
                )

            self.log_error = str(error)

        # Update logging time after writing.
        self.last_log_time = current_time

    # --------------------------- SCORE GRAPH UPDATE ---------------------------

    def _update_score_plot(self):

        scores = np.asarray(
            self.score_history,
            dtype=np.float64
        )

        count = scores.size

        if count == 0:

            self.score_line.set_data(
                [],
                []
            )

            self.baseline_line.set_data(
                [],
                []
            )

            self.threshold_line.set_data(
                [],
                []
            )

            return

        x_values = np.arange(
            count
        )

        self.score_line.set_data(
            x_values,
            scores
        )

        # Baseline line.
        if self.baseline is not None:

            self.baseline_line.set_data(
                x_values,
                np.full(
                    count,
                    self.baseline,
                    dtype=np.float64
                )
            )

        else:

            self.baseline_line.set_data(
                [],
                []
            )

        # Threshold line.
        if self.threshold is not None:

            self.threshold_line.set_data(
                x_values,
                np.full(
                    count,
                    self.threshold,
                    dtype=np.float64
                )
            )

        else:

            self.threshold_line.set_data(
                [],
                []
            )

        # Automatically scale Y axis.
        visible_values = list(
            scores
        )

        if self.baseline is not None:

            visible_values.append(
                self.baseline
            )

        if self.threshold is not None:

            visible_values.append(
                self.threshold
            )

        upper_limit = max(
            5.0,
            max(visible_values) * 1.25
        )

        self.score_axis.set_ylim(
            0,
            upper_limit
        )

        self.score_axis.set_xlim(
            0,
            max(
                PLOT_HISTORY - 1,
                count - 1
            )
        )

    # --------------------------- HEATMAP UPDATE ---------------------------

    def _update_heatmap(self):

        if not self.csi_history:
            return

        matrix = np.asarray(
            self.csi_history,
            dtype=np.float64
        ).T

        self.heatmap_image.set_data(
            matrix
        )

        self.heatmap_image.set_extent(
            (
                0,
                HEATMAP_HISTORY,
                0,
                matrix.shape[0]
            )
        )

        self.heatmap_axis.set_xlim(
            0,
            HEATMAP_HISTORY
        )

        self.heatmap_axis.set_ylim(
            0,
            matrix.shape[0]
        )

        positive = matrix[
            matrix > 0
        ]

        if positive.size:

            low, high = np.percentile(
                positive,
                (5, 95)
            )

            if high - low < 0.10:

                midpoint = (
                    high + low
                ) / 2.0

                low = (
                    midpoint - 0.05
                )

                high = (
                    midpoint + 0.05
                )

            self.heatmap_image.set_clim(
                float(low),
                float(high)
            )

    # --------------------------- MAIN UPDATE ---------------------------

    def update(self, _frame_number):

        # Read CSI from ESP32.
        self._consume_serial()

        # Update status panel.
        self.status_box.set_facecolor(
            self.phase_color
        )

        self.status_text_artist.set_text(
            self.phase_text
        )

        # Update metrics.
        self._update_metrics()

        # Store data.
        self._write_log_row()

        # Update graph.
        self._update_score_plot()

        # Update heatmap.
        self._update_heatmap()

        return ()

    # --------------------------- RUN DASHBOARD ---------------------------

    def run(self):

        self.animation = FuncAnimation(
            self.figure,
            self.update,
            interval=UPDATE_INTERVAL_MS,
            cache_frame_data=False,
        )

        plt.show()


# --------------------------- MAIN ---------------------------

def main():

    receiver = None
    dashboard = None

    try:

        print(
            f"WiSense dashboard {APP_VERSION}"
        )

        print(
            f"Opening ESP32 receiver on {PORT}..."
        )

        receiver = serial.Serial(
            PORT,
            BAUD_RATE,
            timeout=0
        )

        time.sleep(2)

        receiver.reset_input_buffer()

        print(
            "Connected. Opening WiSense dashboard..."
        )

        dashboard = WiSenseDashboard(
            receiver
        )

        dashboard.run()

    except serial.SerialException as error:

        print(
            f"Serial-port error: {error}"
        )

        print(
            "Close Arduino Serial Monitor "
            "and confirm that the receiver is COM7."
        )

    except KeyboardInterrupt:

        print(
            "\nWiSense dashboard stopped."
        )

    finally:

        if (
            receiver is not None
            and receiver.is_open
        ):

            receiver.close()

        if (
            dashboard is not None
            and dashboard.db_conn is not None
        ):

            dashboard.db_conn.close()

            dashboard.db_conn = None

        print(
            "Receiver COM port closed."
        )


# --------------------------- START ---------------------------

if __name__ == "__main__":

    main()