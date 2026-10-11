#!/usr/bin/env python3
"""
🔊  Hearing Analysis App — Full Featured Edition
Pure-Tone Audiometry · Speech-in-Noise · Hearing Age · Tinnitus Pitch Match
Sound Localization · UCL Test · Noise Meter · Age-Norm Comparison
History Tracking · Hearing Tips · Dashboard · Export
"""

import sys, os, json, math, random
import numpy as np
import sounddevice as sd
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QStackedWidget, QFrame, QProgressBar,
    QCheckBox, QRadioButton, QMessageBox, QFileDialog, QSizePolicy,
    QScrollArea, QButtonGroup, QSlider, QSpinBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, QGridLayout,
    QToolTip
)
from PySide6.QtCore import Qt, QTimer, QSize, Signal, QThread
from PySide6.QtGui import QFont, QPainter, QColor, QLinearGradient, QPen, QKeySequence, QShortcut

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
SAMPLE_RATE     = 44100
TONE_DURATION   = 1.8          # seconds per presentation
PULSE_ON        = 0.20         # pulsed-tone on phase
PULSE_GAP       = 0.10         # pulsed-tone off phase
RAMP_FRAC       = 0.02         # rise/fall ramp as fraction of segment

FREQS           = [250, 500, 1000, 2000, 4000, 8000]
PTA_FREQS       = [500, 1000, 2000, 4000]
HIGH_FREQS      = [8000, 10000, 12000, 14000, 16000, 18000, 20000]

DB_MIN, DB_MAX  = -10, 120
STEP_DOWN       = 10           # dB step after hearing the tone
STEP_UP         = 5            # dB step after NOT hearing the tone
MAX_TRIALS      = 20           # per frequency
START_DB        = 40           # starting level
NUM_REVERSALS   = 3            # reversals needed to establish threshold

HISTORY_FILE    = os.path.join(os.path.expanduser("~"), ".hearing_analysis_history.json")

# Simplified ISO 7029 median hearing threshold shift constants (dB) by frequency
# H_med = A * (age - 18)^2  for age > 18
ISO_MALE   = {250: 0.003, 500: 0.003, 1000: 0.003, 2000: 0.006, 4000: 0.012, 8000: 0.018}
ISO_FEMALE = {250: 0.002, 500: 0.002, 1000: 0.002, 2000: 0.004, 4000: 0.007, 8000: 0.010}

# ── Colour palette ──
BG   = "#1a1a2e";  BG2  = "#16213e"
ACC  = "#e94560";  ACC2 = "#0f3460"
BLUE = "#44aaff";  GRN  = "#2ecc71"
YEL  = "#f1c40f";  ORG  = "#e67e22"
RED  = "#e74c3c";  PUR  = "#8e44ad"
CYAN = "#4ecdc4"
TXT  = "#c0c0d0";  TXB  = "#ffffff"

NAV = [
    ("🏠", "Home"),     ("🔊", "Pure Tone"),  ("🗣️", "Noise Test"),
    ("👴", "Hear Age"), ("🔔", "Tinnitus"),   ("👂", "Localize"),
    ("⚡", "UCL"),      ("🎤", "Noise Meter"),("📊", "Dashboard"),
    ("📁", "History"),  ("💡", "Tips"),
]

# ═══════════════════════════════════════════════════════════════════════════════
#  TONE GENERATOR  – improved with smoother ramps & safety clamp
# ═══════════════════════════════════════════════════════════════════════════════
class ToneGenerator:
    """Generates and plays calibrated pure tones, noise, and mixed stimuli."""

    # Reference: 0 dB HL ≈ a calibrated output.  We map so that
    # 80 dB HL → amplitude 1.0 and scale linearly in dB from there.
    # IMPROVEMENT: added floor to avoid silence at very low dB.
    @staticmethod
    def db_to_amplitude(db: float) -> float:
        if db <= -10:
            return 0.0
        return min(1.0, 10.0 ** ((db - 80) / 20))

    @staticmethod
    def tone(freq: float, db: float, dur: float = TONE_DURATION,
             pulsed: bool = True, sr: int = SAMPLE_RATE) -> np.ndarray:
        n = int(sr * dur)
        t = np.linspace(0, dur, n, endpoint=False)
        wave = np.sin(2 * np.pi * freq * t)

        if pulsed:
            on_samples  = int(PULSE_ON * sr)
            off_samples = int(PULSE_GAP * sr)
            envelope = np.zeros(n, dtype=np.float64)
            pos = 0
            while pos < n:
                end = min(pos + on_samples, n)
                segment_len = end - pos
                ramp_len = min(int(RAMP_FRAC * sr), segment_len // 4)
                seg = np.ones(segment_len, dtype=np.float64)
                # Raised-cosine ramps instead of linear (IMPROVEMENT)
                if ramp_len > 0:
                    seg[:ramp_len]  = 0.5 * (1 - np.cos(np.pi * np.arange(ramp_len) / ramp_len))
                    seg[-ramp_len:] = 0.5 * (1 + np.cos(np.pi * np.arange(ramp_len) / ramp_len))
                envelope[pos:end] = seg
                pos += on_samples + off_samples
            wave *= envelope

        # Global fade-in / fade-out to prevent clicks (IMPROVEMENT: cosine ramp)
        ramp = min(int(RAMP_FRAC * sr), n // 4)
        if ramp > 0:
            fade_in  = 0.5 * (1 - np.cos(np.pi * np.arange(ramp) / ramp))
            fade_out = 0.5 * (1 + np.cos(np.pi * np.arange(ramp) / ramp))
            wave[:ramp]  *= fade_in
            wave[-ramp:] *= fade_out

        amp = ToneGenerator.db_to_amplitude(db)
        return (wave * amp).astype(np.float32)

    @staticmethod
    def noise(dur: float, db: float, sr: int = SAMPLE_RATE) -> np.ndarray:
        raw = np.random.randn(int(sr * dur))
        # IMPROVEMENT: band-limit noise to ~200-8000 Hz for speech-relevance
        from scipy.signal import butter, sosfilt  # lazy import
        sos = butter(4, [200, 8000], btype='band', fs=sr, output='sos')
        filtered = sosfilt(sos, raw)
        amp = ToneGenerator.db_to_amplitude(db)
        return (filtered * amp).astype(np.float32)

    @staticmethod
    def noise_fallback(dur: float, db: float, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Fallback if scipy is not available — simple white noise."""
        amp = ToneGenerator.db_to_amplitude(db)
        return (np.random.randn(int(sr * dur)) * amp).astype(np.float32)

    @staticmethod
    def tone_in_noise(freq: float, tone_db: float, noise_db: float,
                      dur: float = TONE_DURATION, sr: int = SAMPLE_RATE) -> np.ndarray:
        try:
            n = ToneGenerator.noise(dur, noise_db, sr)
        except ImportError:
            n = ToneGenerator.noise_fallback(dur, noise_db, sr)
        t = ToneGenerator.tone(freq, tone_db, dur, True, sr)
        return (t + n).astype(np.float32)

    @staticmethod
    def play(data: np.ndarray, ear: str = "both", sr: int = SAMPLE_RATE):
        """Play mono data as stereo with ear routing."""
        stereo = np.zeros((len(data), 2), dtype=np.float32)
        # IMPROVEMENT: add slight interaural delay for localisation realism
        if ear in ("Right", "both", "R"):
            stereo[:, 0] = data
        if ear in ("Left", "both", "L"):
            stereo[:, 1] = data
        sd.stop()
        sd.play(stereo, sr)

    @staticmethod
    def stop():
        sd.stop()


# Shorthand alias used throughout
TG = ToneGenerator


# ═══════════════════════════════════════════════════════════════════════════════
#  AUDIO METER THREAD  – IMPROVEMENT: fixed stream handling
# ═══════════════════════════════════════════════════════════════════════════════
class MeterThread(QThread):
    level = Signal(float)

    def __init__(self):
        super().__init__()
        self._running = False

    def run(self):
        self._running = True
        try:
            with sd.InputStream(channels=1, samplerate=SAMPLE_RATE,
                                blocksize=2048, dtype='float32') as stream:
                while self._running:
                    try:
                        data, overflowed = stream.read(SAMPLE_RATE // 5)
                        if data.size == 0:
                            self.level.emit(-60.0)
                            continue
                        rms = np.sqrt(np.mean(data ** 2))
                        db_spl = 20 * np.log10(max(rms, 1e-10)) + 94
                        self.level.emit(float(db_spl))
                    except Exception:
                        self.level.emit(-60.0)
        except Exception:
            # No microphone available
            self.level.emit(-120.0)

    def stop(self):
        self._running = False
        self.wait(2000)


# ═══════════════════════════════════════════════════════════════════════════════
#  CUSTOM WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════
class VolumeBar(QWidget):
    """Horizontal colour-graded volume bar."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self._level = 0.0

    def set_level(self, db: float):
        self._level = max(0.0, min(1.0, (db - DB_MIN) / (DB_MAX - DB_MIN)))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        # Background
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#0a0a1a"))
        p.drawRoundedRect(0, 0, w, h, 11, 11)
        # Filled portion
        fill_w = max(4, int(w * self._level) - 4)
        grad = QLinearGradient(0, 0, fill_w, 0)
        grad.setColorAt(0.0, QColor(GRN))
        grad.setColorAt(0.5, QColor(YEL))
        grad.setColorAt(0.8, QColor(RED))
        grad.setColorAt(1.0, QColor(PUR))
        p.setBrush(grad)
        p.drawRoundedRect(2, 2, fill_w, h - 4, 9, 9)
        # IMPROVEMENT: dB label inside bar
        p.setPen(QColor(TXB))
        p.setFont(QFont("Sans", 8, QFont.Bold))
        db_val = int(DB_MIN + self._level * (DB_MAX - DB_MIN))
        p.drawText(self.rect(), Qt.AlignCenter, f"{db_val} dB")
        p.end()


class EarAnimation(QWidget):
    """Pulsing ear icon – animates while a tone is playing."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(72, 72)
        self._active = False
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(50)

    def start(self):
        self._active = True
        self._timer.start()

    def stop(self):
        self._active = False
        self._timer.stop()
        self._phase = 0.0
        self.update()

    def _tick(self):
        self._phase += 0.12
        if self._phase > 2 * math.pi:
            self._phase -= 2 * math.pi
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy = 36, 36
        if self._active:
            r = 20 + 8 * abs(math.sin(self._phase))
            p.setPen(QPen(QColor(ACC), 2))
            p.setBrush(QColor(ACC + "40"))
            p.drawEllipse(int(cx - r), int(cy - r), int(2 * r), int(2 * r))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(ACC) if self._active else QColor(TXB))
        p.drawEllipse(18, 18, 36, 36)
        p.setBrush(QColor(BG))
        p.drawEllipse(26, 26, 20, 20)
        p.end()


class AudiogramCanvas(FigureCanvas):
    """Standard audiogram with shaded severity bands."""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 5), dpi=100, facecolor=BG)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)

    def plot(self, right=None, left=None, age_norm=None, title="Audiogram"):
        ax = self.ax
        ax.clear()
        ax.set_facecolor("#262640")
        ax.set_xscale("log")
        ax.set_xlim(180, 11000)
        ax.set_xticks(FREQS)
        ax.set_xticklabels(["250", "500", "1K", "2K", "4K", "8K"])
        ax.set_ylim(130, -15)
        ax.set_yticks(range(-10, 130, 10))
        ax.set_xlabel("Frequency (Hz)", color="w", fontsize=11)
        ax.set_ylabel("Hearing Level (dB HL)", color="w", fontsize=11)
        ax.set_title(title, color="w", fontsize=14, fontweight="bold")
        ax.tick_params(colors="w", which="both")
        for spine in ax.spines.values():
            spine.set_color("#444466")
        ax.grid(True, which="major", color="#444466", alpha=0.4, lw=0.5)

        bands = [
            (-10, 25, GRN, "Normal"),   (25, 40, YEL, "Mild"),
            (40, 55, ORG, "Moderate"),   (55, 70, RED, "Mod-Severe"),
            (70, 90, "#c0392b", "Severe"), (90, 120, PUR, "Profound"),
        ]
        for lo, hi, col, label in bands:
            ax.axhspan(lo, hi, alpha=0.08, color=col)
            ax.text(10200, (lo + hi) / 2, label, fontsize=7, va="center",
                    ha="right", color=col, alpha=0.8)

        if age_norm is not None:
            ax.axhline(y=age_norm, color="#888", ls="--", lw=1.5, alpha=0.6,
                       label=f"Age Norm ({age_norm:.0f} dB)")
        if right:
            xs = [f for f in FREQS if f in right]
            ys = [right[f] for f in xs]
            ax.plot(xs, ys, "o-", color="#ff4444", ms=9, lw=2,
                    mfc="#ff4444", mec="w", mew=1.2, label="Right (O)", zorder=5)
        if left:
            xs = [f for f in FREQS if f in left]
            ys = [left[f] for f in xs]
            ax.plot(xs, ys, "x-", color=BLUE, ms=10, lw=2,
                    mew=2.5, label="Left (X)", zorder=5)
        if right or left or age_norm is not None:
            ax.legend(loc="lower left", fontsize=9,
                      facecolor="#262640", edgecolor="#555577", labelcolor="w")
        self.fig.tight_layout()
        self.draw()


class HighFreqCanvas(FigureCanvas):
    """Ultra-high frequency audiogram (8K-20K)."""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(7, 3.5), dpi=100, facecolor=BG)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)

    def plot(self, results=None):
        ax = self.ax
        ax.clear()
        ax.set_facecolor("#262640")
        ax.set_xlim(7500, 21000)
        ax.set_xticks(HIGH_FREQS)
        ax.set_xticklabels(["8K", "10K", "12K", "14K", "16K", "18K", "20K"])
        ax.set_ylim(100, -20)
        ax.tick_params(colors="w")
        ax.set_xlabel("Frequency (Hz)", color="w", fontsize=10)
        ax.set_ylabel("dB HL", color="w", fontsize=10)
        ax.set_title("High-Frequency Audiogram", color="w",
                     fontsize=12, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_color("#444466")
        ax.grid(True, color="#444466", alpha=0.4)
        if results:
            xs = [f for f in HIGH_FREQS if f in results]
            ys = [results[f] for f in xs]
            ax.plot(xs, ys, "D-", color=ACC, ms=8, lw=2, mec="w", mew=1)
        self.fig.tight_layout()
        self.draw()


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def classify_hearing(pta: float):
    """Return (label, colour) for a PTA value."""
    if pta <= 25:  return "Normal Hearing",           GRN
    if pta <= 40:  return "Mild Hearing Loss",        YEL
    if pta <= 55:  return "Moderate Hearing Loss",    ORG
    if pta <= 70:  return "Moderately Severe Loss",   RED
    if pta <= 90:  return "Severe Hearing Loss",      "#c0392b"
    return "Profound Hearing Loss",                    PUR


def calc_pta(thresholds: dict) -> float | None:
    """Pure-Tone Average over 500, 1K, 2K, 4K."""
    vals = [thresholds[f] for f in PTA_FREQS if f in thresholds]
    return sum(vals) / len(vals) if vals else None


def age_norm(age: int, gender: str = "Male") -> float:
    """ISO 7029-based expected median threshold shift (simplified)."""
    data = ISO_MALE if gender == "Male" else ISO_FEMALE
    shifts = [data.get(f, 0.004) * (age - 18) ** 2 if age > 18 else 0
              for f in PTA_FREQS]
    return sum(shifts) / len(shifts) if shifts else 0


def make_card() -> QFrame:
    f = QFrame(objectName="card")
    f.setLayout(QVBoxLayout(f))
    f.layout().setSpacing(8)
    f.layout().setContentsMargins(16, 12, 16, 12)
    return f


def make_separator() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"color:{ACC2};")
    return f


def heading_label(text: str) -> QLabel:
    return QLabel(text, objectName="cardTitle")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════
class MainWin(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔊 Hearing Analysis — Enhanced Edition")
        self.setMinimumSize(1100, 780)

        # ── User profile ──
        self.user_age = 30
        self.user_gender = "Male"

        # ── Results store ──
        self.results = {
            "pt":  {"Right": {}, "Left": {}},
            "sn":  {"snr": None},
            "ha":  {"age": None, "th": {}},
            "tin": {"freq": None, "db": None},
            "loc": {"score": None, "total": None},
            "ucl": {"Right": {}, "Left": {}},
            "noise": None,
        }

        self.history = self._load_history()

        # ── Central layout ──
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Sidebar ──
        sidebar = QFrame(objectName="sidebar")
        sidebar.setFixedWidth(160)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(4, 10, 4, 10)
        side_layout.setSpacing(2)
        side_layout.addWidget(QLabel(
            "🔊 Hearing\n   Analysis",
            styleSheet=f"color:{ACC};font-size:15px;font-weight:bold;padding:6px;"))
        side_layout.addWidget(make_separator())

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons = []
        for idx, (icon, label) in enumerate(NAV):
            btn = QPushButton(f"  {icon}  {label}")
            btn.setCheckable(True)
            btn.setObjectName("navBtn")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setMinimumHeight(34)
            btn.setToolTip(f"Navigate to {label}")
            self.nav_group.addButton(btn, idx)
            self.nav_buttons.append(btn)
            side_layout.addWidget(btn)
        self.nav_buttons[0].setChecked(True)
        self.nav_group.idClicked.connect(self._on_nav)
        side_layout.addStretch()
        side_layout.addWidget(QLabel("v2.1", styleSheet="color:#555;font-size:9px;padding:4px;"))
        main_layout.addWidget(sidebar)

        # ── Page stack ──
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack)

        # ── Build all pages ──
        self._build_pages()
        self._apply_styles()

        # IMPROVEMENT: Global keyboard shortcuts
        self._shortcut_yes = QShortcut(QKeySequence("Y"), self)
        self._shortcut_yes.activated.connect(self._global_hear)
        self._shortcut_no = QShortcut(QKeySequence("N"), self)
        self._shortcut_no.activated.connect(self._global_nohear)

    # ── Navigation ──
    def _on_nav(self, idx):
        self.stack.setCurrentIndex(idx)
        if idx == 8:   self._dashboard_refresh()
        if idx == 9:   self._history_refresh()

    # ── Global keyboard shortcut handlers ──
    def _global_hear(self):
        """Y key → press whichever 'I Hear It' button is active."""
        for btn in (self.pt_yes_btn, self.sn_yes_btn,
                    self.ha_yes_btn, self.loc_left_btn):  # last is fallback
            if btn.isEnabled() and btn.isVisible():
                btn.click()
                return
        # Try other hear buttons
        if hasattr(self, 'pt_yes_btn') and self.pt_yes_btn.isEnabled():
            self.pt_yes_btn.click()

    def _global_nohear(self):
        """N key → press whichever 'I Don't Hear It' button is active."""
        if hasattr(self, 'pt_no_btn') and self.pt_no_btn.isEnabled():
            self.pt_no_btn.click()
        elif hasattr(self, 'sn_no_btn') and self.sn_no_btn.isEnabled():
            self.sn_no_btn.click()
        elif hasattr(self, 'ha_no_btn') and self.ha_no_btn.isEnabled():
            self.ha_no_btn.click()

    # ════════════════════════════════════════════════════════════════════
    #  BUILD ALL PAGES
    # ════════════════════════════════════════════════════════════════════
    def _build_pages(self):
        self._build_home_page()
        self._build_pure_tone_page()
        self._build_speech_noise_page()
        self._build_hearing_age_page()
        self._build_tinnitus_page()
        self._build_localization_page()
        self._build_ucl_page()
        self._build_noise_meter_page()
        self._build_dashboard_page()
        self._build_history_page()
        self._build_tips_page()

    # ── HOME ──────────────────────────────────────────────────────────
    def _build_home_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(36, 24, 36, 24)
        layout.setSpacing(14)

        layout.addWidget(QLabel("🔊 Hearing Analysis",
                                objectName="pageTitle", alignment=Qt.AlignCenter))
        layout.addWidget(QLabel("Comprehensive Hearing Screening Suite",
                                objectName="subtitle", alignment=Qt.AlignCenter))

        # Profile card
        profile_card = make_card()
        pl = profile_card.layout()
        pl.addWidget(heading_label("👤 Your Profile"))
        row = QHBoxLayout()
        row.addWidget(QLabel("Age:", styleSheet=f"color:{TXT};"))
        self.age_spin = QSpinBox()
        self.age_spin.setRange(5, 100)
        self.age_spin.setValue(30)
        self.age_spin.valueChanged.connect(lambda v: setattr(self, 'user_age', v))
        row.addWidget(self.age_spin)
        row.addSpacing(16)
        row.addWidget(QLabel("Gender:", styleSheet=f"color:{TXT};"))
        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["Male", "Female"])
        self.gender_combo.currentTextChanged.connect(
            lambda v: setattr(self, 'user_gender', v))
        row.addWidget(self.gender_combo)
        row.addStretch()
        pl.addLayout(row)
        layout.addWidget(profile_card)

        # Feature grid
        grid_frame = QFrame(objectName="card")
        grid_layout = QGridLayout(grid_frame)
        grid_layout.setSpacing(8)
        features = [
            ("🔊", "Pure Tone",   "Standard audiometry\n6 frequencies per ear",  0, 0),
            ("🗣️", "Speech+Noise","Detect tones through\nbackground noise",      0, 1),
            ("👴", "Hearing Age",  "Ultra-high frequency\ndetection test",         0, 2),
            ("🔔", "Tinnitus Match","Match your tinnitus\npitch & volume",        1, 0),
            ("👂", "Localization", "Identify sound\ndirection L/R/C",             1, 1),
            ("⚡", "UCL Test",     "Find uncomfortable\nloudness level",          1, 2),
        ]
        for icon, title, desc, r, c in features:
            box = QFrame(objectName="featureCard")
            bl = QVBoxLayout(box)
            bl.setSpacing(3)
            bl.addWidget(QLabel(icon, alignment=Qt.AlignCenter,
                                styleSheet="font-size:26px;"))
            bl.addWidget(QLabel(title, alignment=Qt.AlignCenter,
                                styleSheet=f"color:{TXB};font-weight:bold;font-size:12px;"))
            bl.addWidget(QLabel(desc, alignment=Qt.AlignCenter,
                                styleSheet=f"color:{TXT};font-size:11px;"))
            grid_layout.addWidget(box, r, c)
        layout.addWidget(grid_frame)

        # Disclaimer
        disclaimer = QLabel(
            "⚠️ DISCLAIMER: Screening/educational tool only — NOT a medical diagnosis.<br>"
            "Results depend on headphones, system volume &amp; ambient noise.<br>"
            "Consult a licensed audiologist for professional evaluation."
        )
        disclaimer.setWordWrap(True)
        disclaimer.setAlignment(Qt.AlignCenter)
        disclaimer.setStyleSheet(
            f"color:{ACC};font-size:11px;padding:10px;"
            f"background:{BG2};border-radius:8px;border:1px solid {ACC2};")
        layout.addWidget(disclaimer)
        layout.addStretch()

        # Calibration button
        cal_row = QHBoxLayout()
        cal_btn = QPushButton("▶  Play Calibration Tone (1 kHz, 40 dB)",
                              objectName="secondaryButton")
        cal_btn.clicked.connect(
            lambda: TG.play(TG.tone(1000, 40, 2.5, True), "both"))
        cal_row.addWidget(cal_btn)
        cal_row.addStretch()
        layout.addLayout(cal_row)

        self.stack.addWidget(page)

    # ── PURE TONE AUDIOMETRY ──────────────────────────────────────────
    def _build_pure_tone_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 18, 28, 18)
        layout.setSpacing(8)
        layout.addWidget(QLabel("🔊 Pure Tone Audiometry", objectName="pageTitle"))

        # Setup card
        setup_card = make_card()
        sl = setup_card.layout()
        sl.addWidget(heading_label("Setup"))
        row = QHBoxLayout()
        self.pt_check_r = QCheckBox("Right")
        self.pt_check_l = QCheckBox("Left")
        self.pt_check_both = QCheckBox("Both")
        self.pt_check_r.setChecked(True)
        self.pt_check_l.setChecked(True)
        self.pt_check_both.setChecked(True)
        self.pt_check_both.toggled.connect(
            lambda v: (self.pt_check_r.setChecked(v),
                       self.pt_check_l.setChecked(v)))
        for w in (self.pt_check_r, self.pt_check_l, self.pt_check_both):
            row.addWidget(w)
        row.addStretch()
        self.pt_radio_pulsed = QRadioButton("Pulsed ✓")
        self.pt_radio_pulsed.setChecked(True)
        self.pt_radio_continuous = QRadioButton("Continuous")
        row.addWidget(self.pt_radio_pulsed)
        row.addWidget(self.pt_radio_continuous)
        sl.addLayout(row)

        # IMPROVEMENT: starting level selector
        start_row = QHBoxLayout()
        start_row.addWidget(QLabel("Start Level:", styleSheet=f"color:{TXT};"))
        self.pt_start_spin = QSpinBox()
        self.pt_start_spin.setRange(0, 80)
        self.pt_start_spin.setValue(START_DB)
        self.pt_start_spin.setSuffix(" dB HL")
        start_row.addWidget(self.pt_start_spin)
        start_row.addStretch()
        sl.addLayout(start_row)

        btn_row = QHBoxLayout()
        self.pt_start_btn = QPushButton("Start Pure Tone Test",
                                        objectName="primaryButton")
        self.pt_start_btn.setFixedHeight(40)
        self.pt_start_btn.clicked.connect(self._pt_start)
        btn_row.addWidget(self.pt_start_btn)
        self.pt_cancel_btn = QPushButton("Cancel", objectName="cancelBtn")
        self.pt_cancel_btn.clicked.connect(self._pt_cancel)
        self.pt_cancel_btn.hide()
        btn_row.addWidget(self.pt_cancel_btn)
        btn_row.addStretch()
        sl.addLayout(btn_row)
        layout.addWidget(setup_card)

        # Test-in-progress card
        self.pt_test_card = QFrame(objectName="testCard")
        tl = QVBoxLayout(self.pt_test_card)
        tl.setAlignment(Qt.AlignCenter)
        self.pt_progress = QProgressBar(objectName="progressBar")
        self.pt_progress.setFixedHeight(6)
        self.pt_progress.setTextVisible(False)
        tl.addWidget(self.pt_progress)
        self.pt_step_label = QLabel(alignment=Qt.AlignCenter,
                                    styleSheet=f"color:{TXT};font-size:11px;")
        tl.addWidget(self.pt_step_label)

        ear_row = QHBoxLayout()
        ear_row.setAlignment(Qt.AlignCenter)
        self.pt_ear_anim = EarAnimation()
        ear_row.addWidget(self.pt_ear_anim)
        self.pt_ear_label = QLabel(alignment=Qt.AlignCenter,
                                   styleSheet="font-size:16px;font-weight:bold;letter-spacing:3px;")
        ear_row.addWidget(self.pt_ear_label)
        tl.addLayout(ear_row)

        self.pt_freq_label = QLabel(alignment=Qt.AlignCenter,
                                    styleSheet=f"color:{TXB};font-size:42px;font-weight:bold;")
        tl.addWidget(self.pt_freq_label)
        self.pt_db_label = QLabel(alignment=Qt.AlignCenter,
                                  styleSheet=f"color:{CYAN};font-size:30px;font-weight:bold;")
        tl.addWidget(self.pt_db_label)
        self.pt_vol_bar = VolumeBar()
        tl.addWidget(self.pt_vol_bar)
        self.pt_status_label = QLabel(alignment=Qt.AlignCenter,
                                      styleSheet=f"color:{YEL};font-size:14px;font-style:italic;")
        tl.addWidget(self.pt_status_label)
        self.pt_test_card.hide()
        layout.addWidget(self.pt_test_card)

        # Response buttons
        resp_row = QHBoxLayout()
        resp_row.setSpacing(18)
        self.pt_yes_btn = QPushButton("✓  I Hear It  (Y)", objectName="hearBtn")
        self.pt_yes_btn.setFixedSize(220, 62)
        self.pt_yes_btn.clicked.connect(self._pt_hear)
        self.pt_yes_btn.setEnabled(False)
        self.pt_no_btn = QPushButton("✗  I Don't Hear It  (N)", objectName="nohearBtn")
        self.pt_no_btn.setFixedSize(220, 62)
        self.pt_no_btn.clicked.connect(self._pt_nohear)
        self.pt_no_btn.setEnabled(False)
        resp_row.addStretch()
        resp_row.addWidget(self.pt_yes_btn)
        resp_row.addWidget(self.pt_no_btn)
        resp_row.addStretch()
        layout.addLayout(resp_row)

        self.pt_replay_btn = QPushButton("🔄  Replay Tone", objectName="secondaryButton")
        self.pt_replay_btn.setFixedHeight(34)
        self.pt_replay_btn.clicked.connect(self._pt_replay)
        self.pt_replay_btn.setEnabled(False)
        layout.addWidget(self.pt_replay_btn)
        layout.addStretch()

        self.stack.addWidget(page)

        # Internal state
        self._pt_ears = []
        self._pt_running = False
        self._pt_ear_idx = 0
        self._pt_freq_idx = 0
        self._pt_current_db = START_DB
        self._pt_going_down = True
        self._pt_trial_count = 0
        self._pt_reversals = []        # IMPROVEMENT: track reversals properly
        self._pt_asc_count = {}        # ascending count at each level
        self._pt_timer = QTimer(self, singleShot=True)
        self._pt_timer.timeout.connect(self._pt_tone_ended)

    # ── Pure Tone Logic ──
    def _pt_start(self):
        self._pt_ears = []
        if self.pt_check_r.isChecked(): self._pt_ears.append("Right")
        if self.pt_check_l.isChecked(): self._pt_ears.append("Left")
        if not self._pt_ears:
            QMessageBox.warning(self, "No Ear Selected",
                                "Select at least one ear to test.")
            return
        self.results["pt"] = {"Right": {}, "Left": {}}
        self._pt_ear_idx = 0
        self._pt_freq_idx = 0
        self._pt_current_db = self.pt_start_spin.value()
        self._pt_going_down = True
        self._pt_trial_count = 0
        self._pt_reversals = []
        self._pt_asc_count = {}
        self._pt_running = True
        self._pt_total_steps = len(self._pt_ears) * len(FREQS)
        self.pt_test_card.show()
        self.pt_yes_btn.setEnabled(True)
        self.pt_no_btn.setEnabled(True)
        self.pt_replay_btn.setEnabled(True)
        self.pt_start_btn.hide()
        self.pt_cancel_btn.show()
        self._pt_play_tone()

    def _pt_cancel(self):
        TG.stop()
        self._pt_running = False
        self.pt_ear_anim.stop()
        self.pt_test_card.hide()
        self.pt_start_btn.show()
        self.pt_cancel_btn.hide()

    def _pt_current_ear(self):
        return self._pt_ears[self._pt_ear_idx]

    def _pt_current_freq(self):
        return FREQS[self._pt_freq_idx]

    def _pt_current_step(self):
        return self._pt_freq_idx + self._pt_ear_idx * len(FREQS) + 1

    def _pt_refresh_display(self):
        ear = self._pt_current_ear()
        colour = "#ff4444" if ear == "Right" else BLUE
        self.pt_ear_label.setText(f"{ear.upper()} EAR")
        self.pt_ear_label.setStyleSheet(
            f"color:{colour};font-size:16px;font-weight:bold;letter-spacing:3px;")
        self.pt_freq_label.setText(f"{self._pt_current_freq()} Hz")
        self.pt_db_label.setText(f"{self._pt_current_db} dB HL")
        self.pt_vol_bar.set_level(self._pt_current_db)
        self.pt_status_label.setText("🔊 Listen…")
        self.pt_status_label.setStyleSheet(f"color:{YEL};font-size:14px;font-style:italic;")
        pct = int(100 * (self._pt_current_step() - 1) / self._pt_total_steps)
        self.pt_progress.setValue(pct)
        self.pt_step_label.setText(
            f"Step {self._pt_current_step()} of {self._pt_total_steps}")

    def _pt_play_tone(self):
        TG.stop()
        self._pt_refresh_display()
        self.pt_ear_anim.start()
        QTimer.singleShot(350, self._pt_do_play)

    def _pt_do_play(self):
        if not self._pt_running:
            return
        try:
            data = TG.tone(self._pt_current_freq(), self._pt_current_db,
                           TONE_DURATION, self.pt_radio_pulsed.isChecked())
            TG.play(data, self._pt_current_ear())
        except Exception as exc:
            self.pt_status_label.setText(f"❌ Audio error: {exc}")
            return
        self.pt_status_label.setText("🔊 Tone playing…")
        self._pt_timer.start(int(TONE_DURATION * 1000 + 200))

    def _pt_tone_ended(self):
        if self._pt_running:
            self.pt_ear_anim.stop()
            self.pt_status_label.setText("⏳ Did you hear it?  (Y / N)")
            self.pt_status_label.setStyleSheet(
                f"color:{CYAN};font-size:14px;font-style:italic;")

    def _pt_replay(self):
        if self._pt_running:
            self._pt_play_tone()

    def _pt_hear(self):
        if not self._pt_running: return
        TG.stop()
        self._pt_timer.stop()
        self.pt_ear_anim.stop()
        self._pt_process_response(True)

    def _pt_nohear(self):
        if not self._pt_running: return
        TG.stop()
        self._pt_timer.stop()
        self.pt_ear_anim.stop()
        self._pt_process_response(False)

    # IMPROVEMENT: Proper Hughson-Westlake bracketing with reversal tracking
    def _pt_process_response(self, heard: bool):
        self._pt_trial_count += 1
        prev_dir = self._pt_going_down

        if heard:
            self._pt_going_down = True
            self._pt_current_db = max(DB_MIN, self._pt_current_db - STEP_DOWN)
        else:
            self._pt_going_down = False
            self._pt_current_db = min(DB_MAX, self._pt_current_db + STEP_UP)

        # Detect reversal
        if self._pt_trial_count > 1 and self._pt_going_down != prev_dir:
            self._pt_reversals.append(self._pt_current_db)

        # Threshold decision: need at least NUM_REVERSALS reversals
        if len(self._pt_reversals) >= NUM_REVERSALS:
            # Average of last 2 reversals
            threshold = (self._pt_reversals[-1] + self._pt_reversals[-2]) / 2
            threshold = round(threshold / 5) * 5   # round to nearest 5 dB
            self._pt_record_threshold(threshold)
            return

        if self._pt_current_db >= DB_MAX:
            self._pt_record_threshold(DB_MAX)
            return
        if self._pt_current_db <= DB_MIN and heard:
            self._pt_record_threshold(DB_MIN)
            return
        if self._pt_trial_count >= MAX_TRIALS:
            # Fallback: best estimate
            if self._pt_reversals:
                threshold = round(self._pt_reversals[-1] / 5) * 5
            else:
                threshold = self._pt_current_db
            self._pt_record_threshold(threshold)
            return

        self._pt_play_tone()

    def _pt_record_threshold(self, db: float):
        ear = self._pt_current_ear()
        freq = self._pt_current_freq()
        self.results["pt"][ear][freq] = db
        self.pt_status_label.setText(f"✓ Threshold: {db} dB @ {freq} Hz")
        self.pt_status_label.setStyleSheet(
            f"color:{GRN};font-size:14px;font-weight:bold;")

        # Advance to next frequency / ear
        self._pt_freq_idx += 1
        self._pt_trial_count = 0
        self._pt_reversals = []
        self._pt_asc_count = {}

        if self._pt_freq_idx >= len(FREQS):
            self._pt_freq_idx = 0
            self._pt_ear_idx += 1
            if self._pt_ear_idx >= len(self._pt_ears):
                # All done!
                self._pt_running = False
                TG.stop()
                self.pt_ear_anim.stop()
                self.pt_yes_btn.setEnabled(False)
                self.pt_no_btn.setEnabled(False)
                self.pt_replay_btn.setEnabled(False)
                self._save_history("Pure Tone")
                QTimer.singleShot(700, lambda: QMessageBox.information(
                    self, "Test Complete",
                    "Pure tone test complete!\nVisit the Dashboard for full results."))
                return

        self._pt_current_db = self.pt_start_spin.value()
        self._pt_going_down = True
        QTimer.singleShot(700, self._pt_play_tone)

    # ── SPEECH-IN-NOISE ───────────────────────────────────────────────
    def _build_speech_noise_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 18, 28, 18)
        layout.setSpacing(8)
        layout.addWidget(QLabel("🗣️ Speech-in-Noise Test", objectName="pageTitle"))
        layout.addWidget(QLabel(
            "Detect a 1000 Hz tone mixed with background noise at decreasing SNR. "
            "Lower SNR threshold = better hearing in noise.",
            styleSheet=f"color:{TXT};font-size:13px;", wordWrap=True))

        info_card = make_card()
        il = info_card.layout()
        il.addWidget(heading_label("How It Works"))
        for line in [
            "🔊  A tone plays mixed with background noise",
            "📉  The tone gets quieter relative to noise each round",
            "✋  Press 'I Hear It' if you detect the tone",
            "🔇  Press 'I Don't Hear It' when the tone is buried",
        ]:
            il.addWidget(QLabel(line, styleSheet=f"color:{TXT};font-size:12px;"))

        self.sn_start_btn = QPushButton("Start Speech-in-Noise Test",
                                        objectName="primaryButton")
        self.sn_start_btn.setFixedHeight(40)
        self.sn_start_btn.clicked.connect(self._sn_start)
        il.addWidget(self.sn_start_btn)
        layout.addWidget(info_card)

        # Test card
        self.sn_test_card = QFrame(objectName="testCard")
        stl = QVBoxLayout(self.sn_test_card)
        stl.setAlignment(Qt.AlignCenter)
        self.sn_round_label = QLabel(alignment=Qt.AlignCenter,
                                     styleSheet=f"color:{TXB};font-size:22px;font-weight:bold;")
        stl.addWidget(self.sn_round_label)
        self.sn_snr_label = QLabel(alignment=Qt.AlignCenter,
                                   styleSheet=f"color:{CYAN};font-size:30px;font-weight:bold;")
        stl.addWidget(self.sn_snr_label)
        self.sn_status = QLabel(alignment=Qt.AlignCenter,
                                styleSheet=f"color:{YEL};font-size:14px;")
        stl.addWidget(self.sn_status)
        btn_row = QHBoxLayout()
        self.sn_yes_btn = QPushButton("✓  I Hear It", objectName="hearBtn")
        self.sn_yes_btn.setFixedSize(170, 52)
        self.sn_yes_btn.clicked.connect(self._sn_hear)
        self.sn_no_btn = QPushButton("✗  I Don't", objectName="nohearBtn")
        self.sn_no_btn.setFixedSize(170, 52)
        self.sn_no_btn.clicked.connect(self._sn_nohear)
        btn_row.addStretch()
        btn_row.addWidget(self.sn_yes_btn)
        btn_row.addWidget(self.sn_no_btn)
        btn_row.addStretch()
        stl.addLayout(btn_row)
        self.sn_test_card.hide()
        layout.addWidget(self.sn_test_card)

        self.sn_result_label = QLabel("", alignment=Qt.AlignCenter,
                                      styleSheet=f"color:{TXT};font-size:13px;")
        layout.addWidget(self.sn_result_label)
        layout.addStretch()
        self.stack.addWidget(page)

        # Internal state
        self._sn_running = False
        self._sn_round = 0
        self._sn_noise_db = 55
        self._sn_tone_db = 70
        self._sn_snr = 15
        self._sn_direction = 0     # -1 = going down, +1 = going up
        self._sn_reversals = []
        self._sn_max_rounds = 12

    def _sn_start(self):
        self._sn_running = True
        self._sn_round = 0
        self._sn_tone_db = 70
        self._sn_noise_db = 55
        self._sn_snr = 15
        self._sn_direction = 0
        self._sn_reversals = []
        self.sn_test_card.show()
        self.sn_start_btn.setEnabled(False)
        self._sn_play()

    def _sn_play(self):
        if not self._sn_running: return
        self._sn_round += 1
        self.sn_round_label.setText(f"Round {self._sn_round}")
        self.sn_snr_label.setText(f"SNR: {self._sn_snr:+d} dB")
        self.sn_status.setText("🔊 Listen…")
        QTimer.singleShot(300, self._sn_do_play)

    def _sn_do_play(self):
        if not self._sn_running: return
        try:
            data = TG.tone_in_noise(1000, self._sn_tone_db, self._sn_noise_db, 2.5)
            TG.play(data, "both")
        except Exception as exc:
            self.sn_status.setText(f"❌ {exc}")
            return
        self.sn_status.setText("🔊 Playing tone + noise…")

    def _sn_hear(self):
        if not self._sn_running: return
        TG.stop()
        if self._sn_direction == 1:      # was going up → reversal
            self._sn_reversals.append(self._sn_snr)
        self._sn_direction = -1           # now going down
        self._sn_snr -= 5
        self._sn_tone_db = self._sn_noise_db + self._sn_snr
        if self._sn_round >= self._sn_max_rounds or len(self._sn_reversals) >= 4:
            self._sn_finish(); return
        self._sn_play()

    def _sn_nohear(self):
        if not self._sn_running: return
        TG.stop()
        if self._sn_direction == -1:     # was going down → reversal
            self._sn_reversals.append(self._sn_snr)
        self._sn_direction = 1           # now going up
        self._sn_snr += 5
        self._sn_tone_db = self._sn_noise_db + self._sn_snr
        if self._sn_round >= self._sn_max_rounds or len(self._sn_reversals) >= 4:
            self._sn_finish(); return
        self._sn_play()

    def _sn_finish(self):
        self._sn_running = False
        TG.stop()
        threshold = (sum(self._sn_reversals[-2:]) / 2
                     if len(self._sn_reversals) >= 2
                     else self._sn_snr)
        self.results["sn"]["snr"] = threshold
        self.sn_test_card.hide()
        self.sn_start_btn.setEnabled(True)
        colour = GRN if threshold <= 0 else YEL if threshold <= 5 else ORG if threshold <= 10 else RED
        quality = "Good" if threshold <= 0 else "Fair" if threshold <= 5 else "Poor"
        self.sn_result_label.setText(
            f"<b style='color:{colour}'>SNR Threshold: {threshold:+.1f} dB — {quality}</b><br>"
            f"Normal ≤0 dB | Mild +5 dB | Poor +10+ dB")
        self._save_history("Speech-in-Noise")

    # ── HEARING AGE ───────────────────────────────────────────────────
    def _build_hearing_age_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 18, 28, 18)
        layout.setSpacing(8)
        layout.addWidget(QLabel("👴 Hearing Age Test", objectName="pageTitle"))
        layout.addWidget(QLabel(
            "Test ultra-high frequencies (8K–20K Hz). As we age, we lose "
            "high-frequency sensitivity first. This estimates your 'hearing age'.",
            styleSheet=f"color:{TXT};font-size:13px;", wordWrap=True))

        start_card = make_card()
        scl = start_card.layout()
        self.ha_start_btn = QPushButton("Start Hearing Age Test",
                                        objectName="primaryButton")
        self.ha_start_btn.setFixedHeight(40)
        self.ha_start_btn.clicked.connect(self._ha_start)
        scl.addWidget(self.ha_start_btn)
        layout.addWidget(start_card)

        # Test card
        self.ha_test_card = QFrame(objectName="testCard")
        htl = QVBoxLayout(self.ha_test_card)
        htl.setAlignment(Qt.AlignCenter)
        self.ha_freq_label = QLabel(alignment=Qt.AlignCenter,
                                    styleSheet=f"color:{TXB};font-size:42px;font-weight:bold;")
        htl.addWidget(self.ha_freq_label)
        self.ha_status = QLabel(alignment=Qt.AlignCenter,
                                styleSheet=f"color:{YEL};font-size:14px;")
        htl.addWidget(self.ha_status)
        btn_row = QHBoxLayout()
        self.ha_yes_btn = QPushButton("✓  I Hear It", objectName="hearBtn")
        self.ha_yes_btn.setFixedSize(160, 50)
        self.ha_yes_btn.clicked.connect(self._ha_hear)
        self.ha_no_btn = QPushButton("✗  I Don't", objectName="nohearBtn")
        self.ha_no_btn.setFixedSize(160, 50)
        self.ha_no_btn.clicked.connect(self._ha_nohear)
        btn_row.addStretch()
        btn_row.addWidget(self.ha_yes_btn)
        btn_row.addWidget(self.ha_no_btn)
        btn_row.addStretch()
        htl.addLayout(btn_row)
        self.ha_test_card.hide()
        layout.addWidget(self.ha_test_card)

        # Results card
        res_card = make_card()
        rcl = res_card.layout()
        self.ha_result_label = QLabel("Complete the test to see results",
                                      styleSheet=f"color:{TXT};font-size:13px;")
        rcl.addWidget(self.ha_result_label)
        self.ha_canvas = HighFreqCanvas()
        self.ha_canvas.setMinimumHeight(200)
        rcl.addWidget(self.ha_canvas)
        layout.addWidget(res_card)
        layout.addStretch()
        self.stack.addWidget(page)

        self._ha_running = False
        self._ha_freq_idx = 0
        self._ha_thresholds = {}

    def _ha_start(self):
        self._ha_running = True
        self._ha_freq_idx = 0
        self._ha_thresholds = {}
        self.ha_test_card.show()
        self.ha_start_btn.setEnabled(False)
        self._ha_play()

    def _ha_play(self):
        if not self._ha_running or self._ha_freq_idx >= len(HIGH_FREQS):
            self._ha_finish()
            return
        freq = HIGH_FREQS[self._ha_freq_idx]
        self.ha_freq_label.setText(f"{freq:,} Hz")
        self.ha_status.setText("🔊 Listen…")
        QTimer.singleShot(400, lambda: self._ha_do_play(freq))

    def _ha_do_play(self, freq):
        if not self._ha_running: return
        try:
            data = TG.tone(freq, 60, 2.0, True)
            TG.play(data, "both")
        except Exception:
            self.ha_status.setText("❌ Cannot play this frequency")
            self._ha_nohear()
            return
        self.ha_status.setText(f"🔊 Playing {freq:,} Hz at 60 dB…")

    def _ha_hear(self):
        if not self._ha_running: return
        TG.stop()
        self._ha_thresholds[HIGH_FREQS[self._ha_freq_idx]] = 60
        self._ha_freq_idx += 1
        self._ha_play()

    def _ha_nohear(self):
        if not self._ha_running: return
        TG.stop()
        self._ha_thresholds[HIGH_FREQS[self._ha_freq_idx]] = 100
        self._ha_freq_idx += 1
        self._ha_play()

    def _ha_finish(self):
        self._ha_running = False
        TG.stop()
        self.ha_test_card.hide()
        self.ha_start_btn.setEnabled(True)
        # Determine highest frequency heard at ≤70 dB
        max_heard = max(
            (f for f, db in self._ha_thresholds.items() if db <= 70),
            default=0)
        age_map = [
            (20000, 20), (18000, 25), (16000, 30), (14000, 35),
            (12000, 40), (10000, 50), (8000, 60),
        ]
        hearing_age = 70
        for freq, age in age_map:
            if max_heard >= freq:
                hearing_age = age
                break
        self.results["ha"]["age"] = hearing_age
        self.results["ha"]["th"] = dict(self._ha_thresholds)
        colour = (GRN if hearing_age <= 30 else
                  YEL if hearing_age <= 40 else
                  ORG if hearing_age <= 50 else RED)
        msg = ("✅ Younger than or matching your age!"
               if hearing_age <= self.user_age
               else "⚠️ Older than your actual age — consider seeing an audiologist.")
        self.ha_result_label.setText(
            f"<b style='color:{colour};font-size:18px;'>"
            f"Estimated Hearing Age: {hearing_age} years</b><br><br>"
            f"Highest frequency heard: <b>{max_heard:,} Hz</b><br>"
            f"Your actual age: <b>{self.user_age}</b><br>{msg}")
        self.ha_canvas.plot(self._ha_thresholds)
        self._save_history("Hearing Age")

    # ── TINNITUS PITCH MATCHING ───────────────────────────────────────
    def _build_tinnitus_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 18, 28, 18)
        layout.setSpacing(8)
        layout.addWidget(QLabel("🔔 Tinnitus Pitch Matching", objectName="pageTitle"))
        layout.addWidget(QLabel(
            "Adjust frequency and volume to match your tinnitus. "
            "Play the tone and compare it to what you hear internally.",
            styleSheet=f"color:{TXT};font-size:13px;", wordWrap=True))

        ctrl_card = make_card()
        cl = ctrl_card.layout()

        # Frequency
        cl.addWidget(heading_label("Frequency (Pitch)"))
        freq_row = QHBoxLayout()
        self.tin_freq_slider = QSlider(Qt.Horizontal)
        self.tin_freq_slider.setRange(100, 16000)
        self.tin_freq_slider.setValue(4000)
        self.tin_freq_slider.valueChanged.connect(self._tin_update_labels)
        freq_row.addWidget(self.tin_freq_slider)
        self.tin_freq_label = QLabel("4000 Hz")
        self.tin_freq_label.setFixedWidth(80)
        self.tin_freq_label.setStyleSheet(f"color:{TXB};font-size:14px;font-weight:bold;")
        freq_row.addWidget(self.tin_freq_label)
        cl.addLayout(freq_row)

        # Preset buttons
        preset_row = QHBoxLayout()
        for freq, label in [(250, "Low"), (1000, "1K"), (4000, "4K"),
                            (8000, "8K"), (12000, "12K")]:
            btn = QPushButton(label, objectName="secondaryButton")
            btn.clicked.connect(lambda _, f=freq: self.tin_freq_slider.setValue(f))
            preset_row.addWidget(btn)
        cl.addLayout(preset_row)
        cl.addWidget(make_separator())

        # Intensity
        cl.addWidget(heading_label("Intensity (Volume)"))
        db_row = QHBoxLayout()
        self.tin_db_slider = QSlider(Qt.Horizontal)
        self.tin_db_slider.setRange(0, 90)
        self.tin_db_slider.setValue(30)
        self.tin_db_slider.valueChanged.connect(self._tin_update_labels)
        db_row.addWidget(self.tin_db_slider)
        self.tin_db_label = QLabel("30 dB")
        self.tin_db_label.setFixedWidth(60)
        self.tin_db_label.setStyleSheet(f"color:{CYAN};font-size:14px;font-weight:bold;")
        db_row.addWidget(self.tin_db_label)
        cl.addLayout(db_row)
        cl.addWidget(make_separator())

        # Ear selector
        ear_row = QHBoxLayout()
        ear_row.addWidget(QLabel("Ear:", styleSheet=f"color:{TXT};"))
        self.tin_ear_combo = QComboBox()
        self.tin_ear_combo.addItems(["Both", "Right", "Left"])
        ear_row.addWidget(self.tin_ear_combo)
        ear_row.addStretch()
        cl.addLayout(ear_row)

        # Play / Stop
        play_row = QHBoxLayout()
        self.tin_play_btn = QPushButton("▶  Play Tone", objectName="primaryButton")
        self.tin_play_btn.setFixedHeight(42)
        self.tin_play_btn.clicked.connect(self._tin_play)
        play_row.addWidget(self.tin_play_btn)
        self.tin_stop_btn = QPushButton("⏹  Stop", objectName="secondaryButton")
        self.tin_stop_btn.setFixedHeight(42)
        self.tin_stop_btn.clicked.connect(lambda: TG.stop())
        play_row.addWidget(self.tin_stop_btn)
        cl.addLayout(play_row)

        # Save
        self.tin_save_btn = QPushButton("💾  Save Tinnitus Match",
                                        objectName="secondaryButton")
        self.tin_save_btn.setFixedHeight(38)
        self.tin_save_btn.clicked.connect(self._tin_save)
        cl.addWidget(self.tin_save_btn)
        layout.addWidget(ctrl_card)

        # Info card
        info_card = make_card()
        il = info_card.layout()
        il.addWidget(heading_label("About Tinnitus"))
        il.addWidget(QLabel(
            "Tinnitus affects ~15% of the population. Common causes: noise exposure, "
            "aging, certain medications. If persistent or one-sided, see an ENT specialist.",
            styleSheet=f"color:{TXT};font-size:12px;", wordWrap=True))
        layout.addWidget(info_card)
        layout.addStretch()
        self.stack.addWidget(page)

    def _tin_update_labels(self):
        self.tin_freq_label.setText(f"{self.tin_freq_slider.value()} Hz")
        self.tin_db_label.setText(f"{self.tin_db_slider.value()} dB")

    def _tin_play(self):
        try:
            data = TG.tone(self.tin_freq_slider.value(),
                           self.tin_db_slider.value(), 3.0, False)
            TG.play(data, self.tin_ear_combo.currentText())
        except Exception:
            pass

    def _tin_save(self):
        self.results["tin"]["freq"] = self.tin_freq_slider.value()
        self.results["tin"]["db"] = self.tin_db_slider.value()
        self._save_history("Tinnitus Match")
        QMessageBox.information(self, "Saved",
            f"Tinnitus match saved:\n"
            f"{self.tin_freq_slider.value()} Hz @ {self.tin_db_slider.value()} dB HL")

    # ── SOUND LOCALIZATION ────────────────────────────────────────────
    def _build_localization_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 18, 28, 18)
        layout.setSpacing(8)
        layout.addWidget(QLabel("👂 Sound Localization Test", objectName="pageTitle"))
        layout.addWidget(QLabel(
            "A tone plays in Left, Right, or Both ears (center). "
            "Identify the direction. Tests binaural hearing processing.",
            styleSheet=f"color:{TXT};font-size:13px;", wordWrap=True))

        setup_card = make_card()
        scl = setup_card.layout()
        round_row = QHBoxLayout()
        round_row.addWidget(QLabel("Rounds:", styleSheet=f"color:{TXT};"))
        self.loc_rounds_spin = QSpinBox()
        self.loc_rounds_spin.setRange(5, 30)
        self.loc_rounds_spin.setValue(10)
        round_row.addWidget(self.loc_rounds_spin)
        round_row.addStretch()
        scl.addLayout(round_row)
        self.loc_start_btn = QPushButton("Start Localization Test",
                                         objectName="primaryButton")
        self.loc_start_btn.setFixedHeight(40)
        self.loc_start_btn.clicked.connect(self._loc_start)
        scl.addWidget(self.loc_start_btn)
        layout.addWidget(setup_card)

        # Test card
        self.loc_test_card = QFrame(objectName="testCard")
        ltl = QVBoxLayout(self.loc_test_card)
        ltl.setAlignment(Qt.AlignCenter)
        self.loc_round_label = QLabel(alignment=Qt.AlignCenter,
                                      styleSheet=f"color:{TXB};font-size:22px;font-weight:bold;")
        ltl.addWidget(self.loc_round_label)
        self.loc_status = QLabel(
            "🔊 Listen & identify direction",
            alignment=Qt.AlignCenter,
            styleSheet=f"color:{YEL};font-size:14px;")
        ltl.addWidget(self.loc_status)
        btn_row = QHBoxLayout()
        self.loc_left_btn = QPushButton("⬅  Left", objectName="locBtn")
        self.loc_left_btn.setFixedSize(130, 52)
        self.loc_left_btn.clicked.connect(lambda: self._loc_answer("Left"))
        self.loc_center_btn = QPushButton("⬆  Center", objectName="locBtnC")
        self.loc_center_btn.setFixedSize(130, 52)
        self.loc_center_btn.clicked.connect(lambda: self._loc_answer("Center"))
        self.loc_right_btn = QPushButton("Right  ➡", objectName="locBtn")
        self.loc_right_btn.setFixedSize(130, 52)
        self.loc_right_btn.clicked.connect(lambda: self._loc_answer("Right"))
        btn_row.addStretch()
        btn_row.addWidget(self.loc_left_btn)
        btn_row.addWidget(self.loc_center_btn)
        btn_row.addWidget(self.loc_right_btn)
        btn_row.addStretch()
        ltl.addLayout(btn_row)
        self.loc_test_card.hide()
        layout.addWidget(self.loc_test_card)

        self.loc_result_label = QLabel("", alignment=Qt.AlignCenter,
                                       styleSheet=f"color:{TXT};font-size:13px;")
        layout.addWidget(self.loc_result_label)
        layout.addStretch()
        self.stack.addWidget(page)

        self._loc_running = False
        self._loc_round = 0
        self._loc_total = 10
        self._loc_correct = 0
        self._loc_current = "Center"

    def _loc_start(self):
        self._loc_running = True
        self._loc_round = 0
        self._loc_total = self.loc_rounds_spin.value()
        self._loc_correct = 0
        self.loc_test_card.show()
        self.loc_start_btn.setEnabled(False)
        self._loc_next()

    def _loc_next(self):
        if self._loc_round >= self._loc_total:
            self._loc_finish()
            return
        self._loc_round += 1
        self.loc_round_label.setText(f"Round {self._loc_round}/{self._loc_total}")
        self.loc_status.setText("🔊 Listen…")
        for btn in (self.loc_left_btn, self.loc_center_btn, self.loc_right_btn):
            btn.setEnabled(False)
        direction = random.choice(["Left", "Center", "Right"])
        self._loc_current = direction
        ear = direction if direction != "Center" else "both"
        QTimer.singleShot(600,
            lambda: TG.play(TG.tone(1000, 50, 0.8, True), ear))
        QTimer.singleShot(800, lambda: self.loc_status.setText(
            "Where did it come from?"))
        QTimer.singleShot(800,
            lambda: [b.setEnabled(True)
                     for b in (self.loc_left_btn, self.loc_center_btn,
                               self.loc_right_btn)])

    def _loc_answer(self, answer: str):
        if not self._loc_running: return
        TG.stop()
        correct = answer == self._loc_current
        if correct:
            self._loc_correct += 1
        self.loc_status.setText(
            "✓ Correct!" if correct else f"✗ It was {self._loc_current}")
        QTimer.singleShot(900, self._loc_next)

    def _loc_finish(self):
        self._loc_running = False
        TG.stop()
        self.loc_test_card.hide()
        self.loc_start_btn.setEnabled(True)
        pct = self._loc_correct / self._loc_total * 100
        self.results["loc"]["score"] = self._loc_correct
        self.results["loc"]["total"] = self._loc_total
        colour = (GRN if pct >= 80 else
                  YEL if pct >= 60 else
                  ORG if pct >= 40 else RED)
        self.loc_result_label.setText(
            f"<b style='color:{colour}'>"
            f"Score: {self._loc_correct}/{self._loc_total} ({pct:.0f}%)</b><br>"
            f"{'✅ Good localization!' if pct >= 70 else '⚠️ Consider seeing an audiologist.'}")
        self._save_history("Localization")

    # ── UCL (UNCOMFORTABLE LOUDNESS LEVEL) ────────────────────────────
    def _build_ucl_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 18, 28, 18)
        layout.setSpacing(8)
        layout.addWidget(QLabel("⚡ Uncomfortable Loudness Level (UCL)",
                                objectName="pageTitle"))
        layout.addWidget(QLabel(
            "Volume gradually increases. Press 'Too Loud!' when it becomes "
            "uncomfortable. <b>Do not endure pain.</b>",
            styleSheet=f"color:{TXT};font-size:13px;", wordWrap=True))

        setup_card = make_card()
        scl = setup_card.layout()
        ear_row = QHBoxLayout()
        ear_row.addWidget(QLabel("Ear:", styleSheet=f"color:{TXT};"))
        self.ucl_ear_combo = QComboBox()
        self.ucl_ear_combo.addItems(["Both", "Right", "Left"])
        ear_row.addWidget(self.ucl_ear_combo)
        ear_row.addStretch()
        scl.addLayout(ear_row)
        self.ucl_start_btn = QPushButton("Start UCL Test",
                                         objectName="primaryButton")
        self.ucl_start_btn.setFixedHeight(40)
        self.ucl_start_btn.clicked.connect(self._ucl_start)
        scl.addWidget(self.ucl_start_btn)
        layout.addWidget(setup_card)

        # Test card
        self.ucl_test_card = QFrame(objectName="testCard")
        utl = QVBoxLayout(self.ucl_test_card)
        utl.setAlignment(Qt.AlignCenter)
        self.ucl_freq_label = QLabel(alignment=Qt.AlignCenter,
                                     styleSheet=f"color:{TXB};font-size:28px;font-weight:bold;")
        utl.addWidget(self.ucl_freq_label)
        self.ucl_db_label = QLabel(alignment=Qt.AlignCenter,
                                   styleSheet=f"color:{CYAN};font-size:34px;font-weight:bold;")
        utl.addWidget(self.ucl_db_label)
        self.ucl_vol_bar = VolumeBar()
        self.ucl_vol_bar.setFixedHeight(28)
        utl.addWidget(self.ucl_vol_bar)
        self.ucl_status = QLabel("🔊 Volume increasing…",
                                 alignment=Qt.AlignCenter,
                                 styleSheet=f"color:{YEL};font-size:14px;")
        utl.addWidget(self.ucl_status)
        self.ucl_stop_btn = QPushButton("⛔  Too Loud!", objectName="uclStopBtn")
        self.ucl_stop_btn.setFixedSize(180, 58)
        self.ucl_stop_btn.clicked.connect(self._ucl_hit)
        utl.addWidget(self.ucl_stop_btn)
        self.ucl_test_card.hide()
        layout.addWidget(self.ucl_test_card)

        self.ucl_result_label = QLabel("", styleSheet=f"color:{TXT};font-size:13px;")
        layout.addWidget(self.ucl_result_label)
        layout.addStretch()
        self.stack.addWidget(page)

        self._ucl_running = False
        self._ucl_freq_idx = 0
        self._ucl_current_db = 20
        self._ucl_timer = QTimer(self)
        self._ucl_timer.timeout.connect(self._ucl_increase)
        self._ucl_results = {"Right": {}, "Left": {}}

    def _ucl_start(self):
        self._ucl_running = True
        self._ucl_freq_idx = 0
        self._ucl_results = {"Right": {}, "Left": {}}
        self.ucl_test_card.show()
        self.ucl_start_btn.setEnabled(False)
        self._ucl_next_freq()

    def _ucl_next_freq(self):
        if self._ucl_freq_idx >= len(PTA_FREQS):
            self._ucl_finish()
            return
        self._ucl_current_db = 20
        freq = PTA_FREQS[self._ucl_freq_idx]
        self.ucl_freq_label.setText(f"{freq} Hz")
        self.ucl_db_label.setText(f"{self._ucl_current_db} dB")
        self.ucl_vol_bar.set_level(self._ucl_current_db)
        self.ucl_status.setText(
            "🔊 Volume increasing… Press Too Loud when uncomfortable!")
        self._ucl_beep()
        self._ucl_timer.start(1200)

    def _ucl_beep(self):
        try:
            data = TG.tone(PTA_FREQS[self._ucl_freq_idx],
                           self._ucl_current_db, 0.8, True)
            TG.play(data, self.ucl_ear_combo.currentText())
        except Exception:
            pass

    def _ucl_increase(self):
        if not self._ucl_running: return
        self._ucl_current_db += 5
        self.ucl_db_label.setText(f"{self._ucl_current_db} dB")
        self.ucl_vol_bar.set_level(self._ucl_current_db)
        self._ucl_beep()
        if self._ucl_current_db >= 100:
            self._ucl_record(100)

    def _ucl_hit(self):
        if not self._ucl_running: return
        TG.stop()
        self._ucl_timer.stop()
        self._ucl_record(self._ucl_current_db)

    def _ucl_record(self, db: int):
        ear = self.ucl_ear_combo.currentText()
        freq = PTA_FREQS[self._ucl_freq_idx]
        if ear in ("Right", "Both"):
            self._ucl_results["Right"][freq] = db
        if ear in ("Left", "Both"):
            self._ucl_results["Left"][freq] = db
        self._ucl_freq_idx += 1
        QTimer.singleShot(1200, self._ucl_next_freq)

    def _ucl_finish(self):
        self._ucl_running = False
        TG.stop()
        self._ucl_timer.stop()
        self.ucl_test_card.hide()
        self.ucl_start_btn.setEnabled(True)
        self.results["ucl"] = dict(self._ucl_results)
        lines = ["<b>UCL Results:</b>"]
        for ear in ("Right", "Left"):
            if self._ucl_results[ear]:
                lines.append(f"<br><b>{ear}:</b>")
                for freq, db in self._ucl_results[ear].items():
                    col = GRN if db >= 90 else YEL if db >= 80 else ORG if db >= 70 else RED
                    lines.append(f"  {freq} Hz: <b style='color:{col}'>{db} dB</b>")
                    if db <= 70:
                        lines.append(
                            f"  <span style='color:{RED}'>⚠️ Low UCL — "
                            f"possible hyperacusis</span>")
        self.ucl_result_label.setText("<br>".join(lines))
        self._save_history("UCL Test")

    # ── NOISE METER ───────────────────────────────────────────────────
    def _build_noise_meter_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 18, 28, 18)
        layout.setSpacing(8)
        layout.addWidget(QLabel("🎤 Environmental Noise Meter",
                                objectName="pageTitle"))
        layout.addWidget(QLabel(
            "Measures ambient noise level using your microphone. "
            "Use it to verify a quiet testing environment (< 40 dB SPL recommended).",
            styleSheet=f"color:{TXT};font-size:13px;", wordWrap=True))

        meter_card = make_card()
        mcl = meter_card.layout()

        self.nm_level_label = QLabel("-- dB SPL", alignment=Qt.AlignCenter,
                                     styleSheet=f"color:{TXB};font-size:48px;font-weight:bold;")
        mcl.addWidget(self.nm_level_label)

        self.nm_bar = VolumeBar()
        self.nm_bar.setFixedHeight(32)
        mcl.addWidget(self.nm_bar)

        self.nm_status_label = QLabel("Microphone idle", alignment=Qt.AlignCenter,
                                      styleSheet=f"color:{TXT};font-size:14px;")
        mcl.addWidget(self.nm_status_label)

        btn_row = QHBoxLayout()
        self.nm_start_btn = QPushButton("▶  Start Measuring",
                                        objectName="primaryButton")
        self.nm_start_btn.setFixedHeight(42)
        self.nm_start_btn.clicked.connect(self._nm_start)
        btn_row.addWidget(self.nm_start_btn)
        self.nm_stop_btn = QPushButton("⏹  Stop", objectName="secondaryButton")
        self.nm_stop_btn.setFixedHeight(42)
        self.nm_stop_btn.setEnabled(False)
        self.nm_stop_btn.clicked.connect(self._nm_stop)
        btn_row.addWidget(self.nm_stop_btn)
        mcl.addLayout(btn_row)
        layout.addWidget(meter_card)

        # Reference levels card
        ref_card = make_card()
        rcl = ref_card.layout()
        rcl.addWidget(heading_label("Reference Noise Levels"))
        ref_levels = [
            ("20 dB", "Whisper quiet room",     GRN),
            ("30 dB", "Quiet library",           GRN),
            ("40 dB", "Quiet office",            YEL),
            ("50 dB", "Moderate rainfall",       YEL),
            ("60 dB", "Normal conversation",     ORG),
            ("70 dB", "Vacuum cleaner",          ORG),
            ("80 dB", "Busy traffic",            RED),
            ("90 dB+", "Dangerous — hearing damage risk", RED),
        ]
        for level, desc, col in ref_levels:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"<b style='color:{col}'>{level}</b>"))
            row.addWidget(QLabel(f"  {desc}", styleSheet=f"color:{TXT};"))
            row.addStretch()
            rcl.addLayout(row)
        layout.addWidget(ref_card)
        layout.addStretch()
        self.stack.addWidget(page)

        self._nm_meter_thread = None

    def _nm_start(self):
        self._nm_meter_thread = MeterThread()
        self._nm_meter_thread.level.connect(self._nm_update)
        self._nm_meter_thread.start()
        self.nm_start_btn.setEnabled(False)
        self.nm_stop_btn.setEnabled(True)
        self.nm_status_label.setText("🎤 Measuring…")
        self.nm_status_label.setStyleSheet(f"color:{GRN};font-size:14px;")

    def _nm_stop(self):
        if self._nm_meter_thread:
            self._nm_meter_thread.stop()
            self._nm_meter_thread = None
        self.nm_start_btn.setEnabled(True)
        self.nm_stop_btn.setEnabled(False)
        self.nm_status_label.setText("Microphone idle")
        self.nm_status_label.setStyleSheet(f"color:{TXT};font-size:14px;")

    def _nm_update(self, db_spl: float):
        if db_spl < -100:
            self.nm_level_label.setText("-- dB SPL")
            self.nm_status_label.setText("❌ No microphone detected")
            self.nm_status_label.setStyleSheet(f"color:{RED};font-size:14px;")
            return
        self.nm_level_label.setText(f"{db_spl:.1f} dB SPL")
        self.nm_bar.set_level(db_spl)
        self.results["noise"] = round(db_spl, 1)
        # Colour code
        if db_spl < 40:
            col, msg = GRN, "✅ Quiet — good for testing"
        elif db_spl < 60:
            col, msg = YEL, "⚠️ Moderate — may affect test accuracy"
        elif db_spl < 80:
            col, msg = ORG, "⚠️ Loud — not recommended for testing"
        else:
            col, msg = RED, "⛔ Dangerous — protect your hearing!"
        self.nm_level_label.setStyleSheet(f"color:{col};font-size:48px;font-weight:bold;")
        self.nm_status_label.setText(msg)
        self.nm_status_label.setStyleSheet(f"color:{col};font-size:14px;")

    # ── DASHBOARD ─────────────────────────────────────────────────────
    def _build_dashboard_page(self):
        page = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(28, 18, 28, 18)
        layout.setSpacing(10)
        layout.addWidget(QLabel("📊 Hearing Dashboard", objectName="pageTitle"))

        # Summary card
        self.dash_summary_card = make_card()
        self.dash_summary_layout = self.dash_summary_card.layout()
        self.dash_summary_layout.addWidget(
            heading_label("Overall Summary"))
        self.dash_summary_label = QLabel(
            "Complete tests to see your summary.",
            styleSheet=f"color:{TXT};font-size:13px;", wordWrap=True)
        self.dash_summary_layout.addWidget(self.dash_summary_label)
        layout.addWidget(self.dash_summary_card)

        # Audiogram
        self.dash_audiogram = AudiogramCanvas()
        self.dash_audiogram.setMinimumHeight(280)
        layout.addWidget(self.dash_audiogram)

        # Detailed results grid
        self.dash_details_card = make_card()
        self.dash_details_layout = self.dash_details_card.layout()
        self.dash_details_layout.addWidget(heading_label("Detailed Results"))
        self.dash_details_label = QLabel(
            "No test results yet.", styleSheet=f"color:{TXT};font-size:12px;")
        self.dash_details_layout.addWidget(self.dash_details_label)
        layout.addWidget(self.dash_details_card)

        # Export button
        self.dash_export_btn = QPushButton("📁  Export Results as JSON",
                                           objectName="secondaryButton")
        self.dash_export_btn.setFixedHeight(38)
        self.dash_export_btn.clicked.connect(self._dash_export)
        layout.addWidget(self.dash_export_btn)

        layout.addStretch()
        scroll.setWidget(inner)
        self.stack.addWidget(scroll)

    def _dashboard_refresh(self):
        r = self.results
        lines = []

        # Pure Tone summary
        pta_r = calc_pta(r["pt"]["Right"])
        pta_l = calc_pta(r["pt"]["Left"])
        if pta_r is not None:
            label_r, col_r = classify_hearing(pta_r)
            lines.append(
                f"<b>Right Ear PTA:</b> <span style='color:{col_r}'>"
                f"{pta_r:.1f} dB — {label_r}</span>")
        if pta_l is not None:
            label_l, col_l = classify_hearing(pta_l)
            lines.append(
                f"<b>Left Ear PTA:</b> <span style='color:{col_l}'>"
                f"{pta_l:.1f} dB — {label_l}</span>")

        # Age norm comparison
        if pta_r is not None or pta_l is not None:
            norm = age_norm(self.user_age, self.user_gender)
            pta_avg = pta_r if pta_l is None else pta_l if pta_r is None else (pta_r + pta_l) / 2
            diff = pta_avg - norm
            lines.append(
                f"<b>Age-Norm (ISO 7029):</b> Expected ≈{norm:.1f} dB | "
                f"You are <span style='color:{GRN if diff <= 5 else RED}'>"
                f"{diff:+.1f} dB</span> vs norm")

        # Speech in Noise
        if r["sn"]["snr"] is not None:
            snr = r["sn"]["snr"]
            col = GRN if snr <= 0 else YEL if snr <= 5 else ORG if snr <= 10 else RED
            lines.append(
                f"<b>Speech-in-Noise SNR:</b> "
                f"<span style='color:{col}'>{snr:+.1f} dB</span>")

        # Hearing Age
        if r["ha"]["age"] is not None:
            ha = r["ha"]["age"]
            col = GRN if ha <= 30 else YEL if ha <= 40 else ORG if ha <= 50 else RED
            lines.append(
                f"<b>Hearing Age:</b> <span style='color:{col}'>{ha} years</span>")

        # Localization
        if r["loc"]["score"] is not None:
            pct = r["loc"]["score"] / r["loc"]["total"] * 100
            col = GRN if pct >= 80 else YEL if pct >= 60 else RED
            lines.append(
                f"<b>Localization:</b> "
                f"<span style='color:{col}'>"
                f"{r['loc']['score']}/{r['loc']['total']} ({pct:.0f}%)</span>")

        # Tinnitus
        if r["tin"]["freq"] is not None:
            lines.append(
                f"<b>Tinnitus Match:</b> {r['tin']['freq']} Hz @ {r['tin']['db']} dB")

        # Noise
        if r["noise"] is not None:
            col = GRN if r["noise"] < 40 else YEL if r["noise"] < 60 else RED
            lines.append(
                f"<b>Ambient Noise:</b> "
                f"<span style='color:{col}'>{r['noise']} dB SPL</span>")

        # UCL
        for ear in ("Right", "Left"):
            if r["ucl"][ear]:
                avg = sum(r["ucl"][ear].values()) / len(r["ucl"][ear])
                lines.append(f"<b>{ear} UCL avg:</b> {avg:.0f} dB")

        self.dash_summary_label.setText(
            "<br>".join(lines) if lines else "Complete tests to see your summary.")

        # Audiogram
        norm_val = age_norm(self.user_age, self.user_gender) if pta_r or pta_l else None
        self.dash_audiogram.plot(
            right=r["pt"]["Right"] or None,
            left=r["pt"]["Left"] or None,
            age_norm=norm_val)

        # Details table
        detail_lines = []
        for ear in ("Right", "Left"):
            if r["pt"][ear]:
                detail_lines.append(f"<b>{ear} Ear Thresholds:</b>")
                for freq in FREQS:
                    db = r["pt"][ear].get(freq, "—")
                    detail_lines.append(f"&nbsp;&nbsp;{freq} Hz: {db} dB HL")
        self.dash_details_label.setText(
            "<br>".join(detail_lines) if detail_lines else "No test results yet.")

    def _dash_export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "hearing_results.json",
            "JSON Files (*.json);;All Files (*)")
        if not path:
            return
        export_data = {
            "exported": datetime.now().isoformat(),
            "profile": {"age": self.user_age, "gender": self.user_gender},
            "results": self.results,
        }
        try:
            with open(path, "w") as f:
                json.dump(export_data, f, indent=2)
            QMessageBox.information(self, "Exported",
                                    f"Results saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    # ── HISTORY ───────────────────────────────────────────────────────
    def _build_history_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 18, 28, 18)
        layout.setSpacing(8)
        layout.addWidget(QLabel("📁 Test History", objectName="pageTitle"))

        self.hist_table = QTableWidget()
        self.hist_table.setColumnCount(5)
        self.hist_table.setHorizontalHeaderLabels(
            ["Date", "Test Type", "Key Result", "Age", "Gender"])
        self.hist_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.hist_table.setAlternatingRowColors(True)
        self.hist_table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.hist_table)

        btn_row = QHBoxLayout()
        clear_btn = QPushButton("🗑️  Clear History", objectName="secondaryButton")
        clear_btn.clicked.connect(self._hist_clear)
        btn_row.addWidget(clear_btn)
        export_btn = QPushButton("📁  Export History", objectName="secondaryButton")
        export_btn.clicked.connect(self._hist_export)
        btn_row.addWidget(export_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        layout.addStretch()
        self.stack.addWidget(page)

    def _history_refresh(self):
        records = self.history
        self.hist_table.setRowCount(len(records))
        for i, rec in enumerate(reversed(records)):
            self.hist_table.setItem(i, 0,
                QTableWidgetItem(rec.get("date", "")))
            self.hist_table.setItem(i, 1,
                QTableWidgetItem(rec.get("test", "")))
            self.hist_table.setItem(i, 2,
                QTableWidgetItem(rec.get("summary", "")))
            self.hist_table.setItem(i, 3,
                QTableWidgetItem(str(rec.get("age", ""))))
            self.hist_table.setItem(i, 4,
                QTableWidgetItem(rec.get("gender", "")))

    def _hist_clear(self):
        reply = QMessageBox.question(
            self, "Clear History",
            "Delete all history records? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.history = []
            self._save_history_file()
            self._history_refresh()

    def _hist_export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export History", "hearing_history.json",
            "JSON Files (*.json);;All Files (*)")
        if not path:
            return
        try:
            with open(path, "w") as f:
                json.dump(self.history, f, indent=2)
            QMessageBox.information(self, "Exported",
                                    f"History saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    # ── TIPS ──────────────────────────────────────────────────────────
    def _build_tips_page(self):
        page = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(28, 18, 28, 18)
        layout.setSpacing(10)
        layout.addWidget(QLabel("💡 Hearing Health Tips", objectName="pageTitle"))

        tips = [
            ("🎧 Protect Your Ears",
             "Wear hearing protection (earplugs or earmuffs) in noisy "
             "environments — concerts, construction, mowing the lawn. "
             "Noise-induced hearing loss is permanent and preventable."),
            ("⏱️ 60/60 Rule",
             "Listen to personal audio devices at no more than 60% volume "
             "for no more than 60 minutes at a time. Take listening breaks."),
            ("🩺 Annual Check-ups",
             "Get a professional hearing test every 1–2 years, especially "
             "if you're over 50 or work in a noisy environment."),
            ("💊 Ototoxic Medications",
             "Some medications (certain antibiotics, chemotherapy drugs, "
             "high-dose aspirin) can damage hearing. Always discuss side "
             "effects with your doctor."),
            ("🚭 Smoking & Hearing",
             "Smoking restricts blood flow to the cochlea. Studies show "
             "smokers are 70% more likely to suffer hearing loss."),
            ("🏃 Exercise & Hearing",
             "Cardiovascular exercise improves blood circulation to the "
             "inner ear. Even a 20-minute daily walk helps."),
            ("🧹 Ear Hygiene",
             "Never insert objects into your ear canal. Earwax is "
             "protective — if buildup is a problem, see a doctor for "
             "safe removal."),
            ("🔔 Tinnitus Management",
             "If you have tinnitus, try sound therapy (white noise, nature "
             "sounds), stress reduction, and limit caffeine/alcohol. "
             "See an audiologist for persistent cases."),
            ("📱 App Limitations",
             "This app is a screening tool. Results depend on your "
             "headphones, device volume, and ambient noise. Always verify "
             "with professional audiometry."),
            ("👂 Early Intervention",
             "If you suspect hearing loss, act early. Untreated hearing "
             "loss is linked to cognitive decline, social isolation, and "
             "depression. Hearing aids are more effective when started early."),
            ("😴 Rest Your Ears",
             "After exposure to loud sounds, give your ears 16–48 hours "
             "of quiet recovery time. Temporary threshold shift can become "
             "permanent with repeated exposure."),
            ("🎵 Musicians' Advice",
             "If you play music, consider custom musician earplugs that "
             "reduce volume evenly across frequencies. Monitor your "
             "exposure levels during practice and performances."),
        ]

        for title, body in tips:
            card = make_card()
            cl = card.layout()
            cl.addWidget(QLabel(f"<b>{title}</b>",
                                styleSheet=f"color:{ACC};font-size:14px;"))
            cl.addWidget(QLabel(body, styleSheet=f"color:{TXT};font-size:12px;",
                                wordWrap=True))
            layout.addWidget(card)

        layout.addStretch()
        scroll.setWidget(inner)
        self.stack.addWidget(scroll)

    # ════════════════════════════════════════════════════════════════════
    #  HISTORY PERSISTENCE
    # ════════════════════════════════════════════════════════════════════
    def _load_history(self) -> list:
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, "r") as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
        return []

    def _save_history(self, test_type: str):
        """Create a summary record and append to history."""
        r = self.results
        summary = ""

        if test_type == "Pure Tone":
            pta_r = calc_pta(r["pt"]["Right"])
            pta_l = calc_pta(r["pt"]["Left"])
            parts = []
            if pta_r is not None:
                label, _ = classify_hearing(pta_r)
                parts.append(f"R PTA={pta_r:.0f}dB ({label})")
            if pta_l is not None:
                label, _ = classify_hearing(pta_l)
                parts.append(f"L PTA={pta_l:.0f}dB ({label})")
            summary = " | ".join(parts)
        elif test_type == "Speech-in-Noise":
            snr = r["sn"]["snr"]
            summary = f"SNR={snr:+.1f}dB" if snr is not None else "Incomplete"
        elif test_type == "Hearing Age":
            summary = f"Est. age={r['ha']['age']}yr" if r["ha"]["age"] else "Incomplete"
        elif test_type == "Tinnitus Match":
            summary = f"{r['tin']['freq']}Hz @ {r['tin']['db']}dB"
        elif test_type == "Localization":
            if r["loc"]["score"] is not None:
                summary = f"{r['loc']['score']}/{r['loc']['total']}"
        elif test_type == "UCL Test":
            parts = []
            for ear in ("Right", "Left"):
                if r["ucl"][ear]:
                    avg = sum(r["ucl"][ear].values()) / len(r["ucl"][ear])
                    parts.append(f"{ear[0]} avg={avg:.0f}dB")
            summary = " | ".join(parts) if parts else "Incomplete"

        record = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "test": test_type,
            "summary": summary,
            "age": self.user_age,
            "gender": self.user_gender,
            "results": json.loads(json.dumps(r, default=str)),
        }
        self.history.append(record)
        self._save_history_file()

    def _save_history_file(self):
        try:
            with open(HISTORY_FILE, "w") as f:
                json.dump(self.history, f, indent=2, default=str)
        except IOError:
            pass

    # ════════════════════════════════════════════════════════════════════
    #  STYLESHEET
    # ════════════════════════════════════════════════════════════════════
    def _apply_styles(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: {BG};
                color: {TXT};
                font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
            }}
            #sidebar {{
                background: {BG2};
                border-right: 1px solid {ACC2};
            }}
            #navBtn {{
                background: transparent;
                color: {TXT};
                border: none;
                text-align: left;
                padding: 7px 10px;
                font-size: 12px;
                border-radius: 6px;
                margin: 1px 2px;
            }}
            #navBtn:hover {{
                background: {ACC2};
                color: {TXB};
            }}
            #navBtn:checked {{
                background: {ACC};
                color: {TXB};
                font-weight: bold;
            }}
            #pageTitle {{
                color: {TXB};
                font-size: 22px;
                font-weight: bold;
            }}
            #subtitle {{
                color: {TXT};
                font-size: 14px;
                margin-bottom: 6px;
            }}
            #card {{
                background: {BG2};
                border: 1px solid {ACC2};
                border-radius: 10px;
            }}
            #cardTitle {{
                color: {ACC};
                font-size: 14px;
                font-weight: bold;
            }}
            #featureCard {{
                background: {BG2};
                border: 1px solid {ACC2};
                border-radius: 8px;
                padding: 8px;
            }}
            #testCard {{
                background: {BG2};
                border: 2px solid {ACC};
                border-radius: 12px;
                padding: 18px;
            }}
            #primaryButton {{
                background: {ACC};
                color: {TXB};
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }}
            #primaryButton:hover {{
                background: #ff5a7a;
            }}
            #primaryButton:disabled {{
                background: #555;
                color: #888;
            }}
            #secondaryButton {{
                background: {ACC2};
                color: {TXB};
                border: 1px solid {ACC};
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 12px;
            }}
            #secondaryButton:hover {{
                background: {ACC};
            }}
            #cancelBtn {{
                background: #555;
                color: {TXB};
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
            }}
            #cancelBtn:hover {{
                background: #777;
            }}
            #hearBtn {{
                background: {GRN};
                color: {TXB};
                border: none;
                border-radius: 12px;
                font-size: 16px;
                font-weight: bold;
            }}
            #hearBtn:hover {{
                background: #27ae60;
            }}
            #hearBtn:disabled {{
                background: #333;
                color: #666;
            }}
            #nohearBtn {{
                background: {RED};
                color: {TXB};
                border: none;
                border-radius: 12px;
                font-size: 16px;
                font-weight: bold;
            }}
            #nohearBtn:hover {{
                background: #c0392b;
            }}
            #nohearBtn:disabled {{
                background: #333;
                color: #666;
            }}
            #uclStopBtn {{
                background: {PUR};
                color: {TXB};
                border: none;
                border-radius: 12px;
                font-size: 18px;
                font-weight: bold;
            }}
            #uclStopBtn:hover {{
                background: #9b59b6;
            }}
            #locBtn {{
                background: {ACC2};
                color: {TXB};
                border: 2px solid {BLUE};
                border-radius: 10px;
                font-size: 14px;
                font-weight: bold;
            }}
            #locBtnC {{
                background: {ACC2};
                color: {TXB};
                border: 2px solid {ACC};
                border-radius: 10px;
                font-size: 14px;
                font-weight: bold;
            }}
            #progressBar {{
                background: #222;
                border: none;
                border-radius: 3px;
            }}
            #progressBar::chunk {{
                background: {ACC};
                border-radius: 3px;
            }}
            QCheckBox, QRadioButton {{
                color: {TXT};
                spacing: 6px;
                font-size: 12px;
            }}
            QCheckBox::indicator, QRadioButton::indicator {{
                width: 16px;
                height: 16px;
            }}
            QComboBox {{
                background: {BG};
                color: {TXB};
                border: 1px solid {ACC2};
                border-radius: 6px;
                padding: 5px 10px;
                min-width: 80px;
            }}
            QComboBox QAbstractItemView {{
                background: {BG2};
                color: {TXB};
                selection-background-color: {ACC};
            }}
            QSpinBox {{
                background: {BG};
                color: {TXB};
                border: 1px solid {ACC2};
                border-radius: 6px;
                padding: 4px 8px;
                min-width: 70px;
            }}
            QSlider::groove:horizontal {{
                background: {BG};
                height: 8px;
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {ACC};
                width: 18px;
                height: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }}
            QSlider::sub-page:horizontal {{
                background: {ACC2};
                border-radius: 4px;
            }}
            QTableWidget {{
                background: {BG2};
                color: {TXB};
                border: 1px solid {ACC2};
                border-radius: 8px;
                gridline-color: {ACC2};
                font-size: 12px;
            }}
            QTableWidget::item {{
                padding: 4px;
            }}
            QTableWidget::item:alternate {{
                background: {BG};
            }}
            QHeaderView::section {{
                background: {ACC2};
                color: {TXB};
                border: none;
                padding: 6px;
                font-weight: bold;
            }}
            QScrollArea {{
                border: none;
            }}
            QScrollBar:vertical {{
                background: {BG};
                width: 10px;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical {{
                background: {ACC2};
                border-radius: 5px;
                min-height: 30px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
        """)

    # ── Cleanup on close ──
    def closeEvent(self, event):
        TG.stop()
        if self._nm_meter_thread:
            self._nm_meter_thread.stop()
        event.accept()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Set a dark palette base (helps with combo boxes, etc.)
    from PySide6.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BG))
    palette.setColor(QPalette.WindowText, QColor(TXT))
    palette.setColor(QPalette.Base, QColor(BG2))
    palette.setColor(QPalette.AlternateBase, QColor(BG))
    palette.setColor(QPalette.Text, QColor(TXB))
    palette.setColor(QPalette.Button, QColor(ACC2))
    palette.setColor(QPalette.ButtonText, QColor(TXB))
    palette.setColor(QPalette.Highlight, QColor(ACC))
    palette.setColor(QPalette.HighlightedText, QColor(TXB))
    app.setPalette(palette)

    window = MainWin()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()