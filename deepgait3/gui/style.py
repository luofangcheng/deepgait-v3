"""Deepgait GUI style: color palette, fonts, and QSS helpers.

Theme: modern flat, deep green primary (echoing FTIR 525 nm green light).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

PRIMARY = "#2E7D32"       # deep green — main brand
ACCENT = "#4CAF50"        # bright green — buttons, highlights
PRIMARY_DARK = "#1B5E20"  # darker green — pressed states
BG = "#F5F5F5"            # light gray background
SURFACE = "#FFFFFF"       # card/panel background
TEXT = "#212121"          # primary text
TEXT_SECONDARY = "#757575"  # labels, hints
BORDER = "#E0E0E0"        # dividers, borders
ERROR = "#D32F2F"         # errors
WARNING = "#F9A825"       # warnings

# ---------------------------------------------------------------------------
# QSS helpers
# ---------------------------------------------------------------------------

def _table_style() -> str:
    return f"""
    QTableWidget {{
        background-color: {SURFACE};
        alternate-background-color: {BG};
        gridline-color: {BORDER};
        border: 1px solid {BORDER};
        font-size: 13px;
    }}
    QTableWidget::item {{
        padding: 4px;
    }}
    QHeaderView::section {{
        background-color: {PRIMARY};
        color: white;
        padding: 6px;
        font-weight: bold;
        border: none;
    }}
    """


def _button_style() -> str:
    return f"""
    QPushButton {{
        background-color: {PRIMARY};
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        font-size: 13px;
    }}
    QPushButton:hover {{
        background-color: {ACCENT};
    }}
    QPushButton:pressed {{
        background-color: {PRIMARY_DARK};
    }}
    QPushButton:disabled {{
        background-color: {BORDER};
        color: {TEXT_SECONDARY};
    }}
    """


def _groupbox_style() -> str:
    return f"""
    QGroupBox {{
        font-weight: bold;
        border: 1px solid {BORDER};
        border-radius: 4px;
        margin-top: 8px;
        padding-top: 8px;
        font-size: 13px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 4px;
        color: {PRIMARY};
    }}
    """


def _lineedit_style() -> str:
    return f"""
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        padding: 6px;
        border: 1px solid {BORDER};
        border-radius: 3px;
        background-color: {SURFACE};
        font-size: 13px;
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border: 1px solid {PRIMARY};
    }}
    """


def _progressbar_style() -> str:
    return f"""
    QProgressBar {{
        border: 1px solid {BORDER};
        border-radius: 3px;
        text-align: center;
        font-size: 12px;
    }}
    QProgressBar::chunk {{
        background-color: {ACCENT};
    }}
    """


def _tabwidget_style() -> str:
    return f"""
    QTabWidget::pane {{
        border: 1px solid {BORDER};
        background-color: {SURFACE};
    }}
    QTabBar::tab {{
        background-color: {BG};
        padding: 10px 20px;
        margin-right: 2px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        font-size: 13px;
    }}
    QTabBar::tab:selected {{
        background-color: {PRIMARY};
        color: white;
        font-weight: bold;
    }}
    QTabBar::tab:hover:!selected {{
        background-color: {BORDER};
    }}
    """


def _slider_style() -> str:
    return f"""
    QSlider::groove:horizontal {{
        border: 1px solid {BORDER};
        height: 6px;
        background: {BG};
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {PRIMARY};
        border: none;
        width: 14px;
        margin: -4px 0;
        border-radius: 7px;
    }}
    QSlider::sub-page:horizontal {{
        background: {ACCENT};
        border-radius: 3px;
    }}
    """


def apply_style(app) -> None:
    """Apply deepgait stylesheet to a QApplication instance."""
    qss = (
        _button_style()
        + _table_style()
        + _groupbox_style()
        + _lineedit_style()
        + _progressbar_style()
        + _tabwidget_style()
        + _slider_style()
        + f"""
        QWidget {{
            font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif;
            color: {TEXT};
        }}
        QMainWindow {{
            background-color: {BG};
        }}
        QLabel {{
            font-size: 13px;
        }}
        QTextEdit {{
            background-color: {SURFACE};
            border: 1px solid {BORDER};
            font-size: 12px;
        }}
        QListWidget {{
            background-color: {SURFACE};
            border: 1px solid {BORDER};
            font-size: 13px;
        }}
        QCheckBox {{
            font-size: 13px;
        }}
        """
    )
    app.setStyleSheet(qss)
