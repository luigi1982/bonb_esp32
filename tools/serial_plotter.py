#!/usr/bin/env python3
"""
Serial Plotter + Logger
Live scrolling plot (Arduino-style) + CSV logging in one tool.
Checkboxes to show/hide any signal. Logs everything to CSV regardless.
"""

import argparse
import csv
import re
import sys
import time
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("[ERROR] pyserial not installed. Run: pip install pyserial")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.widgets import CheckButtons
except ImportError:
    print("[ERROR] matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)

SCRIPT_DIR      = Path(__file__).parent
FLUSH_EVERY     = 50
RECONNECT_DELAY = 2.0
READ_TIMEOUT    = 1.0
WINDOW_SEC      = 10      # seconds of data visible in plot
MAX_BUFFER      = 10000   # max samples kept in memory per signal
HR_WINDOW_SEC   = 30      # rolling window for the heart-rate readout
HR_HISTORY_SEC  = HR_WINDOW_SEC + 5   # how long beat timestamps are kept
# Minimum comma-separated fields for a line to count as real telemetry (see
# parse_line) -- our BCG CSV always has 9; short interleaved log lines
# (prio:/stack: task dumps, wifi_sta:, UPLOAD_TIME:, etc.) have 1-2.
MIN_TELEMETRY_FIELDS = 5

_TS_PREFIX = re.compile(r'^\[(\d{2}:\d{2}:\d{2}\.\d+)\]\s*')

# Signals hidden by default (noisy / not useful to plot continuously)
DEFAULT_HIDDEN = {'gate', 'beat', 'breath'}


# ---------------------------------------------------------------------------
# Parsing (same as serial_logger)
# ---------------------------------------------------------------------------

_KEY_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def parse_line(raw_line):
    m = _TS_PREFIX.match(raw_line)
    device_ts = m.group(1) if m else ''
    payload = raw_line[m.end():] if m else raw_line

    tokens = payload.split(',')
    # Real telemetry rows are always multiple comma-separated key:value
    # pairs (e.g. bcg_classic.c's 9-field CSV). Short lines are almost always
    # an unrelated interleaved log line (battery/WiFi/GPS/modem/OTA status,
    # task-list "prio:/stack:" dumps, etc.) that happens to also use a
    # comma+colon style. Accepting one mints a bogus one-off "column" that
    # never gets cleaned up (its key is often unique per line, e.g. an
    # embedded timestamp), snowballing into phantom columns/checkboxes.
    # MIN_TELEMETRY_FIELDS is a heuristic, not a guarantee -- raise it if a
    # future stream needs fewer real fields than this filters out.
    if len(tokens) < MIN_TELEMETRY_FIELDS:
        return (None, None)

    fields = {}
    for token in tokens:
        token = token.strip()
        if ':' in token:
            k, _, v = token.partition(':')
            k, v = k.strip(), v.strip()
            # Reject keys that aren't clean identifiers (e.g. "I (12345) IP5316"
            # from a stray log line) -- real CSV keys are always plain names.
            if k and _KEY_RE.match(k):
                fields[k] = v
    return (None, None) if not fields else (device_ts, fields)


# ---------------------------------------------------------------------------
# CSV sink (same as serial_logger)
# ---------------------------------------------------------------------------

class CsvSink:
    def __init__(self):
        self._fh = self._writer = self._path = None
        self._columns = []

    def _open(self, columns):
        self._close()
        self._columns = ['timestamp_iso', 'device_ts'] + columns
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = SCRIPT_DIR / f"serial_log_{ts}.csv"
        self._fh = open(self._path, 'w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._fh, fieldnames=self._columns,
                                      extrasaction='ignore', restval='')
        self._writer.writeheader()
        self._fh.flush()
        print(f"[LOG] {self._path.name}")

    def write(self, host_ts, device_ts, fields):
        new_keys = [k for k in fields if k not in self._columns]
        if not self._fh:
            self._open(list(fields.keys()))
        elif new_keys:
            print(f"[NEW COLUMNS] {new_keys} — rotating CSV")
            existing = [c for c in self._columns if c not in ('timestamp_iso', 'device_ts')]
            self._close()
            self._open(existing + new_keys)
        row = {'timestamp_iso': host_ts, 'device_ts': device_ts}
        row.update(fields)
        self._writer.writerow(row)

    def flush(self):
        if self._fh:
            self._fh.flush()

    def _close(self):
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = self._writer = None

    def close(self):
        self._close()


# ---------------------------------------------------------------------------
# Port helpers (same as serial_logger)
# ---------------------------------------------------------------------------

def _find_usb_port():
    for p in serial.tools.list_ports.comports():
        dev = p.device.lower()
        if any(k in dev for k in ('usb', 'acm', 'usbserial', 'tty.usbmodem', 'tty.usbserial')):
            return p.device
    return None


def connect(baud, after_disconnect=False):
    if after_disconnect:
        time.sleep(1.5)
        print('[WAIT] Device disconnected — waiting…', flush=True)
    elif _find_usb_port() is None:
        print('[WAIT] No USB device — plug in your device…', flush=True)
    while True:
        port = _find_usb_port()
        if port:
            try:
                ser = serial.Serial(port, baud, timeout=READ_TIMEOUT)
                print(f'[OK] Connected: {port} at {baud} baud')
                return ser, port
            except Exception:
                pass
        time.sleep(RECONNECT_DELAY)


def _safe_close(ser):
    try:
        ser.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Shared state between serial thread and plot thread
# ---------------------------------------------------------------------------

class SharedData:
    def __init__(self):
        self.lock       = threading.Lock()
        self.times      = deque(maxlen=MAX_BUFFER)   # host time as float seconds
        self.signals    = {}   # key → deque of float values
        self.columns    = []   # ordered list of signal names (set on first data)
        self.ready      = False
        self.line_count = 0
        self.t0         = None  # reference time (first sample)
        self.beat_times = deque()   # host-seconds timestamp of each 'beat' event (BCG heartbeat)


shared = SharedData()


def compute_hr(beat_times, t_now, window_sec=HR_WINDOW_SEC):
    """BPM from the mean inter-beat interval over the last `window_sec` of
    beat events. Returns (bpm_or_None, n_beats_in_window). None before
    t_now reaches window_sec (not enough elapsed time) or with <2 beats."""
    if t_now < window_sec:
        return None, 0
    recent = [bt for bt in beat_times if bt >= t_now - window_sec]
    if len(recent) < 2:
        return None, len(recent)
    diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
    mean_ibi = sum(diffs) / len(diffs)
    if mean_ibi <= 0:
        return None, len(recent)
    return 60.0 / mean_ibi, len(recent)


# ---------------------------------------------------------------------------
# Serial reader thread
# ---------------------------------------------------------------------------

def serial_thread(baud, fixed_port):
    sink = CsvSink()
    ser, port = connect(baud) if fixed_port is None else (None, fixed_port)
    if ser is None:
        ser, port = connect(baud)

    while True:
        try:
            raw = ser.readline()
            if not raw:
                continue
            if not raw.endswith(b'\n'):
                # readline() hit READ_TIMEOUT before a full line arrived --
                # this is a partial fragment, not a complete line. Treating
                # it as one splits real telemetry mid-field and glues its
                # leftover bytes onto the *next* line's front, which is
                # exactly the "res_fia_filt______ax..." garbage this device's
                # busier console (interleaved WiFi/GPS/modem/battery logs
                # competing for the 115200 baud UART) can produce. Drop it.
                continue

            host_ts  = datetime.now().isoformat(timespec='milliseconds')
            now_sec  = time.monotonic()

            try:
                line = raw.decode('utf-8', errors='replace').rstrip('\r\n')
            except Exception:
                line = repr(raw)

            device_ts, fields = parse_line(line)
            if not fields:
                continue

            # Once a real schema is established (first valid line), ignore
            # any keys outside it. This is the actual fix for cross-talk
            # from OTHER legitimate multi-field log lines (OTA/upload
            # progress, task dumps, etc.) that happen to also match the
            # "key:value,key:value,..." shape MIN_TELEMETRY_FIELDS alone
            # can't tell apart from real telemetry -- they'd otherwise mint
            # a permanent new column/checkbox the first time they appear.
            if shared.columns:
                fields = {k: v for k, v in fields.items() if k in shared.columns}
                if not fields:
                    continue

            sink.write(host_ts, device_ts, fields)
            shared.line_count += 1
            if shared.line_count % FLUSH_EVERY == 0:
                sink.flush()

            with shared.lock:
                if shared.t0 is None:
                    shared.t0 = now_sec

                t = now_sec - shared.t0

                # Register any new columns
                for k in fields:
                    if k not in shared.signals:
                        shared.signals[k] = deque(maxlen=MAX_BUFFER)
                        shared.columns.append(k)

                shared.times.append(t)

                for k, buf in shared.signals.items():
                    try:
                        buf.append(float(fields.get(k, float('nan'))))
                    except ValueError:
                        buf.append(float('nan'))

                # BCG classical detector: a nonzero 'beat' field is one
                # confirmed heartbeat event (bcg_task.c / bcg_classic.c only
                # emit it for one sample per beat). Track timestamps for the
                # rolling HR_WINDOW_SEC readout below.
                if 'beat' in fields:
                    try:
                        beat_val = float(fields['beat'])
                    except ValueError:
                        beat_val = 0.0
                    if beat_val != 0.0:
                        shared.beat_times.append(t)
                        cutoff = t - HR_HISTORY_SEC
                        while shared.beat_times and shared.beat_times[0] < cutoff:
                            shared.beat_times.popleft()

                shared.ready = True

        except (serial.SerialException, OSError) as exc:
            print(f'\n[DISCONNECT] {exc}')
            sink.flush()
            _safe_close(ser)
            ser, port = connect(baud, after_disconnect=True)


# ---------------------------------------------------------------------------
# Live plot
# ---------------------------------------------------------------------------

COLORS = [
    '#00ff41', '#ff6b6b', '#4ecdc4', '#ffe66d', '#a29bfe',
    '#fd79a8', '#81ecec', '#fab1a0', '#74b9ff', '#55efc4',
]

# Fixed colors for specific signals (override the palette-by-index default
# below) — beat/breath are the BCG detector's event markers, so give them
# fixed, semantically-obvious colors instead of whatever falls out of column
# order.
FIXED_COLORS = {
    'beat':   '#ff3b30',   # red
    'breath': '#30d158',   # green
}


def run_plot(window_sec):
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    ax.tick_params(colors='#cccccc')
    ax.spines[:].set_color('#444444')
    ax.set_xlabel('Time (s)', color='#cccccc')
    ax.set_ylabel('Value', color='#cccccc')
    ax.set_title('Serial Plotter', color='#ffffff', fontsize=12)
    ax.grid(True, color='#333333', linewidth=0.5)

    plt.subplots_adjust(left=0.08, right=0.75, top=0.93, bottom=0.1)

    # Heart-rate readout: BPM from the 'beat' column over a rolling
    # HR_WINDOW_SEC window, sits above the checkbox panel (which occupies
    # the vertically-centered band of the right-hand margin).
    hr_readout = fig.text(0.97, 0.97, f'HR: measuring… ({HR_WINDOW_SEC}s)',
                           ha='right', va='top', fontsize=13,
                           color='#ff3b30', fontweight='bold')
    hr_last_console_bpm = [None]   # mutable box so the closure below can write it

    lines      = {}   # signal_name → Line2D
    check_ax   = None
    check_btn  = None
    visible    = {}   # signal_name → bool

    def build_checkboxes(columns):
        nonlocal check_ax, check_btn, visible

        if check_ax:
            check_ax.remove()

        n = len(columns)
        h = min(0.04 * n + 0.05, 0.85)
        check_ax = plt.axes([0.77, 0.5 - h / 2, 0.21, h])
        check_ax.set_facecolor('#2d2d2d')

        states = [visible.get(c, c not in DEFAULT_HIDDEN) for c in columns]
        for c, s in zip(columns, states):
            visible[c] = s

        # matplotlib >=3.7 dropped CheckButtons.labels/.rectangles in favor of
        # styling via constructor kwargs (label_props/frame_props apply to
        # every checkbox, so each value list is broadcast to length n).
        check_btn = CheckButtons(
            check_ax, columns, states,
            label_props={'color': ['#cccccc'] * n, 'fontsize': [8] * n},
            frame_props={'facecolor': ['#444444'] * n, 'edgecolor': ['#888888'] * n},
        )

        def on_check(label):
            visible[label] = not visible.get(label, True)
            if label in lines:
                lines[label].set_visible(visible[label])
            fig.canvas.draw_idle()

        check_btn.on_clicked(on_check)

    known_columns = []

    def animate(_frame):
        nonlocal known_columns

        with shared.lock:
            if not shared.ready:
                return
            times      = list(shared.times)
            cols       = list(shared.columns)
            buffers    = {k: list(v) for k, v in shared.signals.items()}
            beat_times = list(shared.beat_times)

        # New columns appeared — add lines + rebuild checkboxes
        new_cols = [c for c in cols if c not in known_columns]
        if new_cols:
            for i, c in enumerate(cols):
                if c not in lines:
                    color = FIXED_COLORS.get(c, COLORS[i % len(COLORS)])
                    ln, = ax.plot([], [], color=color, linewidth=1.0,
                                  label=c, alpha=0.9)
                    ln.set_visible(visible.get(c, c not in DEFAULT_HIDDEN))
                    lines[c] = ln
            known_columns = cols
            build_checkboxes(cols)
            ax.legend(loc='upper left', fontsize=7,
                      facecolor='#2d2d2d', edgecolor='#555555',
                      labelcolor='#cccccc')

        if not times:
            return

        t_now  = times[-1]
        t_min  = t_now - window_sec

        # Heart-rate readout: BPM from 'beat' events over the last
        # HR_WINDOW_SEC. Shows a countdown until that much time has
        # elapsed, then a live-updating number.
        bpm, n_beats = compute_hr(beat_times, t_now)
        if t_now < HR_WINDOW_SEC:
            hr_readout.set_text(f'HR: measuring… {HR_WINDOW_SEC - int(t_now)}s')
            hr_readout.set_color('#888888')
        elif bpm is None:
            hr_readout.set_text('HR: -- (no beats)')
            hr_readout.set_color('#888888')
        else:
            hr_readout.set_text(f'HR: {bpm:.0f} BPM')
            hr_readout.set_color('#ff3b30')
            rounded = round(bpm)
            if hr_last_console_bpm[0] != rounded:
                hr_last_console_bpm[0] = rounded
                print(f'[HR] {bpm:.0f} BPM  ({n_beats} beats in last {HR_WINDOW_SEC}s, t={t_now:.0f}s)')

        # Slice to visible window
        idx = [i for i, t in enumerate(times) if t >= t_min]
        if not idx:
            return
        t_win = [times[i] for i in idx]

        for c, ln in lines.items():
            if c not in buffers:
                continue
            buf = buffers[c]
            y_win = [buf[i] for i in idx if i < len(buf)]
            if len(y_win) != len(t_win):
                continue
            ln.set_data(t_win, y_win)

        ax.set_xlim(t_min, t_now)

        # Auto y-scale from visible signals
        vis_vals = []
        for c, ln in lines.items():
            if visible.get(c, True) and c in buffers:
                buf = buffers[c]
                vis_vals += [buf[i] for i in idx if i < len(buf)]

        finite = [v for v in vis_vals if v == v]  # drop nan
        if finite:
            ymin, ymax = min(finite), max(finite)
            pad = (ymax - ymin) * 0.1 or 0.1
            ax.set_ylim(ymin - pad, ymax + pad)

    ani = animation.FuncAnimation(fig, animate, interval=50, cache_frame_data=False)
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Serial Plotter + Logger — live plot and CSV logging.'
    )
    parser.add_argument('--port', '-p', default=None)
    parser.add_argument('--baud', '-b', type=int, default=115200)
    parser.add_argument('--window', '-w', type=int, default=10,
                        help='Seconds of data shown in plot (default: 10)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List serial ports and exit')
    args = parser.parse_args()

    if args.list:
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            print(f'{p.device:<25} {p.description}')
        sys.exit(0)

    t = threading.Thread(target=serial_thread, args=(args.baud, args.port),
                         daemon=True)
    t.start()

    run_plot(args.window)
    print(f'\n[STOP] {shared.line_count} rows captured')


if __name__ == '__main__':
    main()
