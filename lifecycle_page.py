# -*- coding: utf-8 -*-
"""
lifecycle_page.py: coastal bridge-bent lifecycle failure-mode GUI.

- 100,000 independent Monte Carlo samples by default (35 fixed dimensions);
  one random quantile per parameter and sample, fixed over the full lifetime.
- Structural and corrosion inputs, stochastic scour, material deterioration,
  machine-learning feature mapping and transition criteria are retained.
- Plots: scour depth, corrosion level, annual mode probabilities, transition
  types, 1x2 transition-time fitted distributions, and 1x2 all-sample CDFs.
- Best-fit transition legend reports fitted distribution parameters and
  empirical mean/COV of observed first-transition time samples (not fit moments).
"""

import io
import os
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MaxNLocator
from scipy import stats, special

warnings.filterwarnings("ignore")


# ============================================================
# 1. Global settings -- synchronized with lifetime-transition code
# ============================================================
RANDOM_SEED = 2026
N_SAMPLES = 100000
YEARS_FULL = np.arange(0, 101, 1, dtype=int)

MIN_PERSIST_YEARS = 3
USE_TRANSITION_PROB_MARGIN = True
TRANSITION_PROB_MARGIN = 0.05
MIN_TRANSITION_YEAR = 1

FORCE_SCOUR_MONOTONIC = True
FORCE_CORROSION_MONOTONIC = True
SD_DISCRETIZE_STEP = None  # continuous scour depth; no 0.5 m discretization
SD_COV_DEFAULT = 0.27
TRANSITION_HIST_BIN_WIDTH = 5

# Reference: Transition_lifetime_A4_editable_FiveDistributions_CDF_GOF.py
# Fits are computed only for confirmed, observed first transitions within 100 y.
FIT_MIN_SAMPLES = 20
FIT_GOF_TEST_FRACTION = 0.20
FIT_GOF_SEED = 2026
FIT_GOF_MIN_EXPECTED = 5.0
FIT_GOF_ALPHA = 0.05
DEFAULT_CDF_THRESHOLD_FFF = 20.0  # interface unit: percent
DEFAULT_CDF_THRESHOLD_CSF = 10.0  # interface unit: percent

# The original GUI uses the following 35-dimension random-input order.
# Preserve the mapping even though GUI table rows are reordered. For MC this
# also matches the independent random-variable order in the reference script.
ORIGINAL_RANDOM_INPUT_ORDER = (
    "N", "Dp", "rho_pl", "alpha", "S_Dp", "Dr", "Hp_Dc",
    "Dc_Dp", "rho_cl", "rho_ps", "fyl", "fc", "rho_cs", "t",
    "d_l", "fyt", "d_t", "Acs", "ecs", "Ccr", "D0", "kc",
    "kt", "ke", "n_val", "X1", "lam_corr", "R", "SD_val",
    "lambda_sd", "B_val", "p_val", "q_val", "r_val", "s_val",
)

DISPLAY_NAME = {
    "FFF": "FFF",
    "PFF": "CFF",
    "CFF": "CFF",
    "PSF": "CSF",
    "CSF": "CSF",
}

# Current transition-code defaults for Case2
CURRENT_CASE2_DEFAULTS = {
    "N": ("Uniform", 3.0, None, 2.0, 4.0),
    "Dp": ("Normal", 1.2, 1.2 * 0.10, 0.6, 1.8),
    "rho_pl": ("Normal", 0.010, 0.010 * 0.27, 0.005, 0.015),
    "alpha": ("Normal", 0.15, 0.15 * 0.12, 0.05, 0.25),
    "S_Dp": ("Normal", 3.0, 3.0 * 0.15, 2.5, 3.5),
    "Dr": ("Uniform", 0.55, None, 0.35, 0.75),
    "Hp_Dc": ("Normal", 3.0, 3.0 * 0.26, 1.0, 5.0),
    "Dc_Dp": ("Normal", 2.0, 2.0 * 0.10, 1.5, 3.0),
    "rho_cl": ("Normal", 0.010, 0.010 * 0.27, 0.005, 0.015),
    "rho_ps": ("Normal", 0.008, 0.008 * 0.42, 0.003, 0.013),
    "fyl": ("Lognormal", 400.0, 400.0 * 0.106, 300.0, 500.0),
    "fc": ("Lognormal", 40.0, 40.0 * 0.20, 20.0, 60.0),
    "rho_cs": ("Normal", 0.008, 0.008 * 0.42, 0.003, 0.013),
    "t": ("Normal", 0.05, 0.05 * 0.20, 0.04, 0.08),
    "d_l": ("Normal", 0.025, 0.025 * 0.10, 0.018, 0.032),
    "fyt": ("Lognormal", 350.0, 350.0 * 0.106, 250.0, 450.0),
    "d_t": ("Normal", 0.016, 0.016 * 0.10, 0.010, 0.020),
    "Acs": ("Normal", 7.758, 1.360, 1e-6, None),
    "ecs": ("Normal", 0.0, 1.105, None, None),
    "Ccr": ("Normal", 0.900, 0.150, 1e-6, None),
    "D0": ("Normal", 473.0, 43.20, 1e-6, None),
    "kc": ("Normal", 0.800, 0.100, 1e-6, None),
    "kt": ("Normal", 0.850, 0.024, 1e-6, None),
    "ke": ("Normal", 1.000, 0.300, 1e-6, None),
    "n_val": ("Beta", 0.250, 0.050, None, None),
    "X1": ("Lognormal", 1.000, 0.050, None, None),
    "lam_corr": ("Deterministic", 2.000, 0.000, None, None),
    "R": ("Gumbel", 5.560, 1.160, 1e-6, None),
    "lambda_sd": ("Deterministic", 2.000, 0.000, None, None),
    "B_val": ("Deterministic", 2.260, 0.0, None, None),
    "p_val": ("Deterministic", 1.093, 0.0, None, None),
    "q_val": ("Deterministic", 0.021, 0.0, None, None),
    "r_val": ("Deterministic", 0.269, 0.0, None, None),
    "s_val": ("Deterministic", 2.135, 0.0, None, None),
}


# ============================================================
# 2. Model loading and label helpers
# ============================================================
@st.cache_resource
def load_numpy_assets(model_path: str = "model_assets_numpy.pkl"):
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


def canonical_label(label):
    s = str(label)
    if s == "CFF":
        return "PFF"
    if s == "CSF":
        return "PSF"
    return s


def display_label(label):
    return DISPLAY_NAME.get(str(label), DISPLAY_NAME.get(canonical_label(label), str(label)))


def get_label_names(assets):
    if "label_names" in assets:
        return np.array(assets["label_names"]).astype(str)
    if "le" in assets:
        return np.array(assets["le"].classes_).astype(str)
    return np.array(["FFF", "PFF", "PSF"], dtype=str)


def softmax_stable(z):
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def sigmoid_stable(z):
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def forward_softmax_numpy(X_raw, assets):
    scaler = assets["scaler"]
    weights = assets["weights"]

    a = scaler.transform(X_raw)
    for w, b in weights[:-1]:
        a = np.maximum(0.0, np.dot(a, w) + b)

    w_out, b_out = weights[-1]
    logits = np.dot(a, w_out) + b_out
    probs = softmax_stable(logits)
    pred_idx = np.argmax(probs, axis=1)
    return pred_idx, probs


def forward_sigmoid_numpy(X_scaled, weights):
    a = X_scaled
    for w, b in weights[:-1]:
        a = np.maximum(0.0, np.dot(a, w) + b)
    w_out, b_out = weights[-1]
    z = np.dot(a, w_out) + b_out
    return sigmoid_stable(z.reshape(-1))


def forward_hierarchical_numpy(X_raw, assets):
    scaler = assets["scaler"]
    X_scaled = scaler.transform(X_raw)
    label_names = get_label_names(assets)

    idx_fff = int(assets["idx_fff"])
    idx_pff = int(assets["idx_pff"])
    idx_psf = int(assets["idx_psf"])
    pff_threshold = float(assets["pff_threshold"])
    psf_threshold = float(assets["psf_threshold"])

    p_pff = forward_sigmoid_numpy(X_scaled, assets["stage1_weights"])
    p_psf_cond = forward_sigmoid_numpy(X_scaled, assets["stage2_weights"])

    probs = np.zeros((len(X_raw), len(label_names)), dtype=float)
    probs[:, idx_pff] = p_pff
    probs[:, idx_fff] = (1.0 - p_pff) * (1.0 - p_psf_cond)
    probs[:, idx_psf] = (1.0 - p_pff) * p_psf_cond

    pred_idx = np.full(len(X_raw), idx_fff, dtype=int)
    is_pff = p_pff >= pff_threshold
    pred_idx[is_pff] = idx_pff
    non_pff = ~is_pff
    pred_idx[non_pff & (p_psf_cond >= psf_threshold)] = idx_psf
    pred_idx[non_pff & (p_psf_cond < psf_threshold)] = idx_fff
    return pred_idx, probs


def predict_model(X_raw, assets):
    if "stage1_weights" in assets and "stage2_weights" in assets:
        return forward_hierarchical_numpy(X_raw, assets)
    return forward_softmax_numpy(X_raw, assets)


# ============================================================
# 3. Matplotlib / CSS
# ============================================================
def setup_matplotlib_font():
    candidate_files = [
        Path(__file__).parent / "fonts" / "times.ttf",
        Path(__file__).parent / "fonts" / "Times New Roman.ttf",
        Path(__file__).parent / "times.ttf",
        Path(__file__).parent / "Times New Roman.ttf",
    ]

    for font_file in candidate_files:
        if font_file.exists():
            font_manager.fontManager.addfont(str(font_file))
            prop = font_manager.FontProperties(fname=str(font_file))
            font_name = prop.get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["font.serif"] = [font_name]
            return font_name, prop

    available = {f.name for f in font_manager.fontManager.ttflist}
    if "Times New Roman" in available:
        prop = font_manager.FontProperties(family="Times New Roman")
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        return "Times New Roman", prop

    fallback = "STIXGeneral"
    prop = font_manager.FontProperties(family=fallback)
    plt.rcParams["font.family"] = fallback
    plt.rcParams["font.serif"] = [fallback, "DejaVu Serif"]
    return fallback, prop


GLOBAL_FONT_NAME, GLOBAL_FONT_PROP = setup_matplotlib_font()
plt.rcParams["font.family"] = "Times New Roman" if GLOBAL_FONT_NAME == "Times New Roman" else GLOBAL_FONT_NAME
plt.rcParams["font.serif"] = ["Times New Roman", "STIXGeneral", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"


# ============================================================
# GUI presentation controls (edit these values as needed)
# ============================================================
FORMULA_BAR_HEIGHT_PX = 142       # Formula reference panel minimum height (px)
FORMULA_SPACING_EM = 1.5          # Horizontal spacing BETWEEN the four aligned formulas (em)
FORMULA_TITLE_PADDING_EM = 0.25   # Extra horizontal room around each heading (em)
FORMULA_HEADING_ROW_GAP_EM = 1.1  # Distance between captions and formulas (em)
GLOSSARY_ROW_PADDING_TOP_PX = 8     # Space ABOVE each parameter/description (px)
GLOSSARY_ROW_PADDING_BOTTOM_PX = 11 # Space BELOW each parameter, BEFORE the divider (px)
GLOSSARY_ROW_GAP_PX = 1.40             # Space after the divider (px)
PARAMETER_ROW_GAP_REM = 0.22     # Vertical gap between parameter rows (rem)
PARAMETER_NAME_SHIFT_PX = -8      # Positive = move parameter symbols downward (px)

# Figure/header appearance -- editing these will NOT affect any MC results.
MAIN_TITLE_FONT_MAX_PX = 44       # Main heading upper size (px)
DASHBOARD_CAPTION_HEIGHT_PX = 50  # Each plot caption's line height (px)
DASHBOARD_CAPTION_IMAGE_GAP_REM = 0.08  # Caption to chart image (rem)
TRANSITION_TYPES_BOTTOM_FRAC = 0.265   # Smaller = less blank below type labels


def inject_css():
    st.markdown(
        """
        <style>
        html, body, *,
        [data-testid="stAppViewContainer"],
        [data-testid="stMarkdownContainer"],
        [data-testid="stWidgetLabel"],
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        .stText, .stMarkdown,
        p, span, label, button, input, textarea,
        div, h1, h2, h3, h4, h5, h6,
        table, th, td, li, ul, ol {
            font-family: 'Times New Roman', serif !important;
        }
        .block-container {
            padding-top: 1.0rem;
            padding-bottom: 2.0rem;
            max-width: 98% !important;
        }
        hr {
            margin-top: 5px;
            margin-bottom: 10px;
            border-top: 1px solid #ddd;
        }
        div[data-baseweb="input"] input {
            text-align: center !important;
            font-family: 'Times New Roman', serif !important;
            font-size: 16px !important;
        }
        div[data-baseweb="select"] div {
            font-family: 'Times New Roman', serif !important;
            font-size: 16px !important;
        }
        ul[data-baseweb="menu"] li, [role="listbox"] li {
            font-family: 'Times New Roman', serif !important;
            font-size: 16px !important;
        }
        .section-header {
            color: #D70051;
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 5px;
            font-family: 'Times New Roman', serif;
        }
        .col-header {
            text-align: center;
            color: #333;
            font-size: 16px;
            font-weight: bold;
            font-family: 'Times New Roman', serif;
        }
        .plot-title {
            text-align: center;
            font-family: 'Times New Roman', serif;
            font-weight: bold;
            font-size: 17px;
            margin-bottom: 2px;
        }

        /* ================== DASHBOARD LAYOUT / TYPOGRAPHY ================== */
        /* The Streamlit top toolbar overlays the first ~50px on some browsers.
           Reserve an explicit top offset so the main heading is never cropped. */
        .block-container {
            max-width: 100% !important;
            padding: 3.0rem 1.15rem 2rem !important;
        }
        .st-key-dashboard_header,
        .st-key-dashboard_header [data-testid="stMarkdownContainer"] {
            overflow: visible !important;
            width: 100% !important;
        }
        .dashboard-main-title {
            display: block !important;
            width: 100% !important;
            min-height: 54px;
            box-sizing: border-box;
            padding: 0 8px 12px;
            margin: 0 auto;
            font-family: 'Times New Roman', serif !important;
            font-size: clamp(31px, 2.38vw, __MAIN_TITLE_MAX_PX__px) !important;
            line-height: 1.18 !important;
            text-align: center !important;
            color: #42474e !important;
            font-weight: 700 !important;
            white-space: normal;
            overflow: visible;
        }
        /* Independent horizontal rule immediately beneath the full-width title. */
        hr.dashboard-title-divider {
            width: 100% !important;
            height: 0;
            margin: 4px 0 12px !important;
            border: 0 !important;
            border-top: 1.0px solid #aeb6c0 !important;
            opacity: 1;
        }
        .section-header {
            color: #D70051;
            font-size: clamp(15px, 1.12vw, 19px);
            font-weight: bold;
            line-height: 1.16;
            margin: 0 0 6px;
        }
        .col-header {
            color: #333;
            font-size: clamp(15px, .86vw, 15px);
            font-weight: 600;
            text-align: center;
            white-space: nowrap;
            line-height: 1.2;
        }
        /* Read-only Range labels follow the same 40px row rhythm as inputs. */
        .parameter-range {
            display: flex;
            height: 40px;
            min-height: 40px;
            align-items: center;
            justify-content: center;
            text-align: center;
            font-size: clamp(11px, .9vw, 15px);
            line-height: 1.2;
            margin: 0;
            color: #444;
        }
        /* All 35 Parameter labels: bold KaTeX glyphs, exactly centered
           vertically in the same 40px row as select/number-input controls. */
        .st-key-structure_card [data-testid="stMarkdown"]:has(.katex),
        .st-key-corrosion_card [data-testid="stMarkdown"]:has(.katex),
        .st-key-scour_card [data-testid="stMarkdown"]:has(.katex) {
            height: 40px !important;
            min-height: 40px !important;
            margin: 0 !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            transform: translateY(__PARAM_NAME_SHIFT_PX__px);
        }
        .st-key-structure_card [data-testid="stMarkdownContainer"] p:has(.katex),
        .st-key-corrosion_card [data-testid="stMarkdownContainer"] p:has(.katex),
        .st-key-scour_card [data-testid="stMarkdownContainer"] p:has(.katex) {
            height: 40px !important;
            min-height: 40px !important;
            width: 100% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1.2 !important;
            text-align: center !important;
        }
        .st-key-structure_card [data-testid="stMarkdownContainer"] p .katex,
        .st-key-corrosion_card [data-testid="stMarkdownContainer"] p .katex,
        .st-key-scour_card [data-testid="stMarkdownContainer"] p .katex {
            font-size: clamp(12px, .95vw, 16px) !important;
            font-weight: 700 !important;
        }
        hr.parameter-divider {
            margin: 1px 0 4px;
            border: none;
            border-top: 1px solid #d9dce0;
        }
        /* One LaTeX expression containing four formulas, separated by space.
           Edit FORMULA_BAR_HEIGHT_PX near inject_css() to resize this area.
           Narrow browsers can scroll the single line horizontally instead of
           clipping tall superscripts or overlapping the Simulate button. */
        .st-key-equation_toolbar {
            box-sizing: border-box;
            min-height: __FORMULA_BAR_HEIGHT_PX__px;
            padding: 8px 0 !important;
            overflow: visible !important;
        }
        .st-key-equation_toolbar [data-testid="stVerticalBlock"] {
            gap: 0 !important;
        }
        .st-key-equation_toolbar [data-testid="stElementContainer"] {
            overflow: visible !important;
        }
        .st-key-equation_toolbar [data-testid="stLatex"] {
            box-sizing: border-box;
            width: 100% !important;
            min-height: 88px;
            display: block !important;
            overflow-x: auto !important;
            overflow-y: hidden !important;
            margin: 0 !important;
            padding: 20px 4px !important;
        }
        .st-key-equation_toolbar [data-testid="stLatex"] .katex-display {
            box-sizing: border-box;
            width: max-content;
            min-width: 100%;
            margin: 0 auto !important;
            padding: 9px 4px !important;
            overflow: visible !important;
        }
        .st-key-equation_toolbar [data-testid="stLatex"] .katex {
            font-size: clamp(10px, .70vw, 14px) !important;
            white-space: nowrap;
        }
        /* Compact card rows, while retaining enough actual control height. */
        .st-key-structure_card [data-testid="stVerticalBlock"],
        .st-key-corrosion_card [data-testid="stVerticalBlock"],
        .st-key-scour_card [data-testid="stVerticalBlock"],
        .st-key-other_card [data-testid="stVerticalBlock"] {
            gap: __PARAM_ROW_GAP_REM__rem !important;
        }
        .st-key-structure_card [data-testid="stHorizontalBlock"],
        .st-key-corrosion_card [data-testid="stHorizontalBlock"],
        .st-key-scour_card [data-testid="stHorizontalBlock"] {
            gap: .68rem !important;
            align-items: center !important;
        }
        /* Narrow the grey fields within their grid cells, leaving clear
           gaps between Parameter, Distribution, Mean and dispersion columns.
           Placeholder dashes are REAL disabled text inputs, so they get the
           exact same height, baseline and corner radius as editable fields. */
        .st-key-structure_card [data-testid="stSelectbox"],
        .st-key-corrosion_card [data-testid="stSelectbox"],
        .st-key-scour_card [data-testid="stSelectbox"] {
            width: 93% !important;
            margin: 0 auto !important;
        }
        .st-key-structure_card [data-testid="stNumberInput"],
        .st-key-corrosion_card [data-testid="stNumberInput"],
        .st-key-scour_card [data-testid="stNumberInput"],
        .st-key-structure_card [data-testid="stTextInput"],
        .st-key-corrosion_card [data-testid="stTextInput"],
        .st-key-scour_card [data-testid="stTextInput"] {
            width: 95% !important;
            margin: 0 auto !important;
        }
        /* Identical 40px heights for Distribution selects, editable numerical
           controls and disabled dash fields; no Markdown-based fake boxes. */
        .st-key-structure_card [data-baseweb="select"] > div,
        .st-key-corrosion_card [data-baseweb="select"] > div,
        .st-key-scour_card [data-baseweb="select"] > div,
        .st-key-structure_card [data-baseweb="input"],
        .st-key-corrosion_card [data-baseweb="input"],
        .st-key-scour_card [data-baseweb="input"] {
            height: 40px !important;
            min-height: 40px !important;
            max-height: 40px !important;
            box-sizing: border-box !important;
            border-radius: 7px !important;
        }
        .st-key-structure_card [data-baseweb="select"] [role="combobox"],
        .st-key-corrosion_card [data-baseweb="select"] [role="combobox"],
        .st-key-scour_card [data-baseweb="select"] [role="combobox"] {
            display: flex !important;
            align-items: center !important;
            justify-content: space-between !important;
            height: 40px !important;
            min-height: 40px !important;
            max-height: 40px !important;
            padding: 0 8px !important;
            line-height: normal !important;
            box-sizing: border-box !important;
            white-space: nowrap !important;
            text-align: center !important;
        }
        /* Center the displayed choice in its grey box, not at its top-left.
           Keep the separate dropdown chevron at the right edge. */
        .st-key-structure_card [data-baseweb="select"] [role="combobox"] > div:first-child,
        .st-key-corrosion_card [data-baseweb="select"] [role="combobox"] > div:first-child,
        .st-key-scour_card [data-baseweb="select"] [role="combobox"] > div:first-child {
            flex: 1 1 auto !important;
            min-width: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
            padding: 0 !important;
        }
        .st-key-structure_card [data-baseweb="select"] [role="combobox"] > div:first-child *,
        .st-key-corrosion_card [data-baseweb="select"] [role="combobox"] > div:first-child *,
        .st-key-scour_card [data-baseweb="select"] [role="combobox"] > div:first-child * {
            text-align: center !important;
            line-height: 1.2 !important;
        }
        .st-key-structure_card [data-baseweb="input"] input,
        .st-key-corrosion_card [data-baseweb="input"] input,
        .st-key-scour_card [data-baseweb="input"] input {
            height: 40px !important;
            min-height: 40px !important;
            max-height: 40px !important;
            padding: 0 5px !important;
            line-height: normal !important;
            text-align: center !important;
            box-sizing: border-box !important;
        }
        /* Placeholder "–" must align exactly with the adjacent input boxes.
           Keep disabled styling legible, rather than greyed-out and shifted. */
        .st-key-structure_card [data-testid="stTextInput"] input:disabled,
        .st-key-corrosion_card [data-testid="stTextInput"] input:disabled,
        .st-key-scour_card [data-testid="stTextInput"] input:disabled {
            text-align: center !important;
            color: #777 !important;
            -webkit-text-fill-color: #777 !important;
            opacity: 1 !important;
            cursor: default !important;
        }
        .st-key-structure_card [data-baseweb="select"] *,
        .st-key-corrosion_card [data-baseweb="select"] *,
        .st-key-scour_card [data-baseweb="select"] *,
        .st-key-structure_card [data-baseweb="input"] *,
        .st-key-corrosion_card [data-baseweb="input"] *,
        .st-key-scour_card [data-baseweb="input"] * {
            font-size: clamp(12px, .91vw, 15px) !important;
        }
        /* Preserve Streamlit's native +/- stepper controls for editable
           Mean, COV, and St. dev. numeric fields; do not hide the buttons. */
        .st-key-structure_card [data-testid="stNumberInputStepUp"],
        .st-key-structure_card [data-testid="stNumberInputStepDown"],
        .st-key-corrosion_card [data-testid="stNumberInputStepUp"],
        .st-key-corrosion_card [data-testid="stNumberInputStepDown"],
        .st-key-scour_card [data-testid="stNumberInputStepUp"],
        .st-key-scour_card [data-testid="stNumberInputStepDown"] {
            display: flex !important;
            visibility: visible !important;
            align-items: center !important;
            justify-content: center !important;
            flex-shrink: 0 !important;
        }
        /* Bold only the three labels in the Other parameters card. */
        .st-key-other_card [data-testid="stWidgetLabel"] p,
        .st-key-other_card [data-testid="stWidgetLabel"] label {
            font-family: 'Times New Roman', serif !important;
            font-weight: 700 !important;
            font-size: clamp(12px, .90vw, 16px) !important;
        }
        /* Keep every caption directly above its chart and vertically aligned.
           The plot axes are offset inside each 4.25-inch figure; 14% left
           padding centers a caption over the visible axes, not its PNG. */
        .dashboard-chart-title {
            box-sizing: border-box;
            width: 100%;
            height: __DASHBOARD_CAPTION_HEIGHT_PX__px;
            min-height: __DASHBOARD_CAPTION_HEIGHT_PX__px;
            display: flex;
            justify-content: center;
            align-items: center;
            text-align: center;
            font-weight: bold;
            font-size: clamp(12px, 1.04vw, 17px);
            line-height: 1.15;
            color: #333;
            margin: 0 !important;
            padding: 0 0 0 14%;
        }
        /* Streamlit inserts extra vertical space between the caption Markdown
           and st.image in a column; remove only that local gap. */
        .st-key-dashboard_plots [data-testid="stColumn"] > [data-testid="stVerticalBlock"],
        .st-key-dashboard_plots [data-testid="stColumn"] [data-testid="stVerticalBlock"] {
            gap: __DASHBOARD_CAPTION_IMAGE_GAP_REM__rem !important;
            row-gap: __DASHBOARD_CAPTION_IMAGE_GAP_REM__rem !important;
        }
        .st-key-dashboard_plots [data-testid="stMarkdown"] {
            width: 100% !important;
            margin: 0 !important;
        }
        .st-key-dashboard_plots [data-testid="stImage"] img {
            display: block;
            width: 100%;
            height: auto;
        }
        .dashboard-notes {
            font-size: 14px;
            color: #555;
            line-height: 1.5;
            padding: 4px 2px;
        }
        /* Bottom-of-dashboard mathematical formulas and reference material. */
        /* Match the exact typography used by Simulation results. */
        .section-header,
        .reference-section-title,
        .reference-card-title {
            font-family: 'Times New Roman', serif !important;
            font-size: clamp(15px, 1.12vw, 19px) !important;
            font-weight: 700 !important;
            line-height: 1.16 !important;
            color: #D70051;
        }
        .reference-section-title {
            margin: 14px 0 8px;
            padding: 0 0 6px;
            border-bottom: 1px solid #d9dce0;
        }
        .reference-card-title {
            padding: 1px 0 9px;
            margin: 0 0 10px;
            border-bottom: 1px solid #dedfe2;
        }
        /* One valid KaTeX array: standard {cccc} columns, no unsupported @{}
           column modifiers. Each caption shares the center of its equation. */
        .st-key-equation_toolbar {
            box-sizing: border-box;
            min-height: __FORMULA_BAR_HEIGHT_PX__px !important;
            padding: 4px 0 5px !important;
            overflow: visible !important;
        }
        .st-key-equation_toolbar [data-testid="stLatex"] {
            box-sizing: border-box;
            width: 100% !important;
            min-height: 118px !important;
            padding: 25px 9px 25px !important;
            margin: 0 !important;
            overflow-x: auto !important;
            overflow-y: auto !important;
        }
        .st-key-equation_toolbar [data-testid="stLatex"] .katex-display {
            box-sizing: border-box;
            width: max-content;
            min-width: 100%;
            padding: 6px 4px !important;
            margin: 0 auto !important;
            overflow: visible !important;
        }
        .st-key-equation_toolbar [data-testid="stLatex"] .katex {
            font-size: clamp(11px, .79vw, 16px) !important;
            white-space: nowrap;
        }
        /* Parameter glossary: both main columns must START at the same height.
           Previous align-items:center vertically centered the entire 17-row
           structural column alongside the 18-row right column. */
        .st-key-parameter_glossary [data-testid="stHorizontalBlock"] {
            align-items: flex-start !important;
            gap: .55rem !important;
        }
        .glossary-group-title {
            box-sizing: border-box;
            display: flex;
            align-items: center;
            width: 100%;
            min-height: 36px;
            font: 700 clamp(14px, .94vw, 17px) 'Times New Roman', serif !important;
            line-height: 1.18;
            color: #464b52;
            background: #f4f5f7;
            padding: 7px 11px;
            border-radius: 5px;
            margin: 0 0 7px !important;
        }
        /* Each glossary parameter is its own native Streamlit container.
           This keeps KaTeX symbols and descriptions in the SAME row and gives
           the requested thin separator and editable vertical breathing space. */
        .st-key-parameter_glossary [class*="st-key-glossary_row_"] {
            box-sizing: border-box !important;
            border-bottom: 1px solid #dce1e7 !important;
            border-radius: 0 !important;
            margin: 0 0 __GLOSSARY_ROW_GAP_PX__px !important;
            padding: __GLOSSARY_ROW_PADDING_TOP_PX__px 4px __GLOSSARY_ROW_PADDING_BOTTOM_PX__px !important;
            width: 100% !important;
        }
        .st-key-parameter_glossary [class*="st-key-glossary_row_"] [data-testid="stVerticalBlock"] {
            gap: 0 !important;
            row-gap: 0 !important;
        }
        .st-key-parameter_glossary [class*="st-key-glossary_row_"] [data-testid="stHorizontalBlock"] {
            align-items: center !important;
            gap: .55rem !important;
        }
        .st-key-parameter_glossary [class*="st-key-glossary_row_"] [data-testid="stMarkdownContainer"] p:has(.katex) {
            min-height: 31px;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        .st-key-parameter_glossary [class*="st-key-glossary_row_"] [data-testid="stMarkdownContainer"] p .katex {
            font-size: clamp(13px, .95vw, 17px) !important;
            font-weight: 700 !important;
        }
        .glossary-meaning {
            box-sizing: border-box;
            min-height: 31px;
            display: flex;
            align-items: center;
            padding: 0 4px !important;
            margin: 0 !important;
            line-height: 1.35;
            font: 400 clamp(13px, .86vw, 15px) 'Times New Roman', serif !important;
            color: #4b5563;
            overflow-wrap: anywhere;
        }
        .st-key-parameter_glossary [data-testid="stVerticalBlock"] {
            gap: .08rem !important;
        }
        /* Outer pair stretches to the taller side. The description card then
           fills the available column height, while its rows keep the above
           adjustable padding and the separator below every parameter. */
        .st-key-reference_materials [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]) {
            align-items: stretch !important;
        }
        .st-key-reference_materials [data-testid="stColumn"]:has(.st-key-parameter_glossary) {
            display: flex !important;
            flex-direction: column !important;
            align-self: stretch !important;
        }
        .st-key-reference_materials [data-testid="stColumn"]:has(.st-key-parameter_glossary) > [data-testid="stVerticalBlock"] {
            height: 100% !important;
            flex: 1 1 auto !important;
            display: flex !important;
            flex-direction: column !important;
        }
        .st-key-reference_materials [data-testid="stColumn"]:has(.st-key-parameter_glossary)
        [data-testid="stElementContainer"]:has(.st-key-parameter_glossary) {
            flex: 1 1 auto !important;
            display: flex !important;
            flex-direction: column !important;
        }
        .st-key-reference_materials .st-key-parameter_glossary {
            flex: 1 1 auto !important;
            min-height: 100% !important;
        }
        @media(max-width: 900px) {
            .glossary-meaning { font-size: 13px !important; }
        }
        /* Two main actions: larger and bold labels, without affecting other inputs. */
        .st-key-simulate_action [data-testid="stButton"] button,
        .st-key-download_action [data-testid="stDownloadButton"] button {
            min-height: 48px !important;
            font-family: 'Times New Roman', serif !important;
            font-size: clamp(18px, 1.2vw, 22px) !important;
            font-weight: 700 !important;
            line-height: 1.2 !important;
        }
        .st-key-simulate_action button *,
        .st-key-download_action button * {
            font-family: 'Times New Roman', serif !important;
            font-size: inherit !important;
            font-weight: 700 !important;
        }
        .st-key-sfm_reference [data-testid="stImage"] img,
        .st-key-schematic_reference [data-testid="stImage"] img {
            max-width: 100%;
            height: auto;
            object-fit: contain;
        }
        .dashboard-author-footer {
            border-top: 1px solid #d9dce0;
            margin: 18px auto 3px;
            padding: 15px 8px 5px;
            font: 700 clamp(13px, .9vw, 16px) 'Times New Roman', serif !important;
            text-align: center;
            color: #525862;
            line-height: 1.6;
        }
        .author-contact-gap { display: inline-block; width: 28px; }
        @media(max-width: 900px) {
            .glossary-columns { grid-template-columns: minmax(0, 1fr); }
            .author-contact-gap { width: 12px; }
        }
        @media(max-width: 1250px) {
            .col-header { font-size: 11px; }
            .parameter-range { font-size: 11px; }
            .dashboard-chart-title { font-size: 12px; }
        }
        </style>
        """
        .replace("__FORMULA_BAR_HEIGHT_PX__", str(FORMULA_BAR_HEIGHT_PX))
        .replace("__GLOSSARY_ROW_PADDING_TOP_PX__", str(GLOSSARY_ROW_PADDING_TOP_PX))
        .replace("__GLOSSARY_ROW_PADDING_BOTTOM_PX__", str(GLOSSARY_ROW_PADDING_BOTTOM_PX))
        .replace("__GLOSSARY_ROW_GAP_PX__", str(GLOSSARY_ROW_GAP_PX))
        .replace("__PARAM_ROW_GAP_REM__", str(PARAMETER_ROW_GAP_REM))
        .replace("__PARAM_NAME_SHIFT_PX__", str(PARAMETER_NAME_SHIFT_PX))
        .replace("__MAIN_TITLE_MAX_PX__", str(MAIN_TITLE_FONT_MAX_PX))
        .replace("__DASHBOARD_CAPTION_HEIGHT_PX__", str(DASHBOARD_CAPTION_HEIGHT_PX))
        .replace("__DASHBOARD_CAPTION_IMAGE_GAP_REM__", str(DASHBOARD_CAPTION_IMAGE_GAP_REM)),
        unsafe_allow_html=True,
    )



def find_all_crossovers(
    years,
    prob_a,
    prob_b,
    label_a,
    label_b,
):
    """Find all intersections of two annual probability curves."""
    crossovers = []
    diff = np.asarray(prob_a) - np.asarray(prob_b)

    for i in range(len(years) - 1):
        if diff[i] == 0:
            if i > 0 and diff[i - 1] != 0:
                slope_a = prob_a[i + 1] - prob_a[i]
                slope_b = prob_b[i + 1] - prob_b[i]

                if diff[i - 1] > 0 and slope_a < slope_b:
                    crossovers.append(
                        (
                            round(float(years[i]), 2),
                            f"{label_a} to {label_b}",
                        )
                    )
                elif diff[i - 1] < 0 and slope_a > slope_b:
                    crossovers.append(
                        (
                            round(float(years[i]), 2),
                            f"{label_b} to {label_a}",
                        )
                    )

        elif diff[i] * diff[i + 1] < 0:
            t_cross = (
                years[i]
                - diff[i]
                * (years[i + 1] - years[i])
                / (diff[i + 1] - diff[i])
            )

            slope_a = prob_a[i + 1] - prob_a[i]
            slope_b = prob_b[i + 1] - prob_b[i]

            if slope_a < slope_b:
                desc = f"{label_a} to {label_b}"
            else:
                desc = f"{label_b} to {label_a}"

            crossovers.append(
                (round(float(t_cross), 2), desc)
            )

    return crossovers

# Uniform, one-step-larger figure typography for axes and legends.
FIGURE_TEXT_FONT_SIZE = 12
FIGURE_TITLE_FONT_SIZE = 13


def apply_academic_style(ax):
    ax.xaxis.label.set_fontproperties(GLOBAL_FONT_PROP)
    ax.yaxis.label.set_fontproperties(GLOBAL_FONT_PROP)
    ax.title.set_fontproperties(GLOBAL_FONT_PROP)
    ax.xaxis.label.set_fontsize(FIGURE_TEXT_FONT_SIZE)
    ax.yaxis.label.set_fontsize(FIGURE_TEXT_FONT_SIZE)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(GLOBAL_FONT_PROP)
        label.set_fontsize(FIGURE_TEXT_FONT_SIZE)

    leg = ax.get_legend()
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontproperties(GLOBAL_FONT_PROP)
            text.set_fontsize(FIGURE_TEXT_FONT_SIZE)

    ax.tick_params(
        axis="both",
        direction="in",
        top=True,
        right=True,
        labelsize=FIGURE_TEXT_FONT_SIZE,
        width=0.8,
        length=3.5,
    )


def set_axis_labels(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontproperties=GLOBAL_FONT_PROP,
                  fontsize=FIGURE_TEXT_FONT_SIZE)
    ax.set_ylabel(ylabel, fontproperties=GLOBAL_FONT_PROP,
                  fontsize=FIGURE_TEXT_FONT_SIZE)


def style_thin_black_axes(ax, *, frame_linewidth=0.5,
                          tick_linewidth=0.5, tick_length=3.5):
    """Apply independently configurable frames/ticks to the two SFM 1x2 panels."""
    ax.tick_params(axis="both", which="both",
                   width=tick_linewidth, length=tick_length,
                   colors="black", direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(frame_linewidth)


# ============================================================
# 4. Sampling -- same rules as current lifetime-transition code
# ============================================================
def get_samples(u_array, dist_type, mean, std, p_min, p_max):
    u_array = np.clip(
        np.asarray(u_array, dtype=float),
        1e-10,
        1.0 - 1e-10
    )

    if dist_type == "Deterministic":
        return np.full_like(u_array, float(mean), dtype=float)

    if dist_type == "DiscreteUniform":
        lo = int(round(p_min))
        hi = int(round(p_max))
        values = np.arange(lo, hi + 1, dtype=float)
        idx = np.floor(u_array * len(values)).astype(int)
        idx = np.clip(idx, 0, len(values) - 1)
        return values[idx]

    if dist_type == "Uniform":
        # IMPORTANT: identical to current transition code:
        # Uniform uses lower/upper bounds directly; mean/std are ignored.
        lower = float(p_min)
        upper = float(p_max)
        return lower + u_array * (upper - lower)

    lower = -np.inf if p_min is None else float(p_min)
    upper = np.inf if p_max is None else float(p_max)

    if dist_type == "Normal":
        if mean is None or std is None or std <= 0:
            raise ValueError(f"Invalid Normal parameters: mean={mean}, std={std}")

        if np.isneginf(lower) and np.isposinf(upper):
            s = stats.norm.ppf(u_array, loc=mean, scale=std)
        else:
            a = (lower - mean) / std
            b = (upper - mean) / std
            s = stats.truncnorm.ppf(
                u_array, a, b, loc=mean, scale=std
            )
            s = np.clip(s, lower, upper)

    elif dist_type == "Lognormal":
        if mean is None or mean <= 0 or std is None or std < 0:
            raise ValueError(
                f"Invalid Lognormal parameters: mean={mean}, std={std}"
            )

        if std == 0:
            s = np.full_like(u_array, float(mean), dtype=float)
        else:
            sigma2 = np.log(1.0 + (std / mean) ** 2)
            sigma_ln = np.sqrt(sigma2)
            mu_ln = np.log(mean) - 0.5 * sigma2
            base_dist = stats.lognorm(
                s=sigma_ln,
                scale=np.exp(mu_ln)
            )

            if p_min is None and p_max is None:
                s = base_dist.ppf(u_array)
            else:
                lower_cdf = (
                    0.0
                    if p_min is None or lower <= 0.0
                    else float(base_dist.cdf(lower))
                )
                upper_cdf = (
                    1.0
                    if p_max is None or np.isposinf(upper)
                    else float(base_dist.cdf(upper))
                )
                u_trunc = lower_cdf + u_array * (upper_cdf - lower_cdf)
                u_trunc = np.clip(u_trunc, 1e-12, 1.0 - 1e-12)
                s = base_dist.ppf(u_trunc)
                s = np.clip(s, lower, upper)

    elif dist_type == "Beta":
        var = std ** 2
        temp = mean * (1.0 - mean) / var - 1.0
        a = mean * temp
        b = (1.0 - mean) * temp
        if a <= 0 or b <= 0:
            raise ValueError(
                f"Invalid Beta parameters: mean={mean}, std={std}"
            )
        s = stats.beta.ppf(u_array, a, b)

    elif dist_type == "Gumbel":
        # R follows F(r)=exp{-exp[-alpha0*(r-mu0)]}
        mu0 = float(mean)
        alpha0 = float(std)
        s = stats.gumbel_r.ppf(
            u_array,
            loc=mu0,
            scale=1.0 / alpha0
        )
        if p_min is not None or p_max is not None:
            s = np.clip(s, lower, upper)

    else:
        raise ValueError(f"Unsupported distribution: {dist_type}")

    if not np.all(np.isfinite(s)):
        raise ValueError(f"{dist_type} sampling produced NaN/Inf.")

    return np.asarray(s, dtype=float)


def generate_samples_dict(all_inputs, n_samples=N_SAMPLES, seed=RANDOM_SEED):
    # Independent (ordinary) Monte Carlo sampling, NOT Latin hypercube.
    # One random quantile per parameter and sample; keep this U for all years.
    rng = np.random.default_rng(seed)
    U = rng.random((n_samples, len(all_inputs)))

    samples_dict = {}
    for idx, p in enumerate(all_inputs):
        if p["id"] == "SD_val":
            # SD_val is only a fixed MC quantile driver for the scour path.
            samples_dict[p["id"]] = U[:, idx].copy()
            continue

        # N is displayed as Uniform, but represents the integer-valued
        # uniform distribution over {2, 3, 4}; all other Uniform
        # parameters retain their original continuous sampling.
        sampling_dist = (
            "DiscreteUniform"
            if p["id"] == "N" and p["dist"] == "Uniform"
            else p["dist"]
        )
        samples_dict[p["id"]] = get_samples(
            U[:, idx],
            sampling_dist,
            p["mean"],
            p["std"],
            p["min"],
            p["max"],
        )

    return samples_dict, U


# ============================================================
# 5. Corrosion and scour path generation
# ============================================================
def compute_T_init(
    cover_mm, D0, k_e, k_t, k_c, n,
    C_cr, C0_arr, X1
):
    valid_mask = (
        (C0_arr > C_cr)
        & (C0_arr > 0)
        & (C_cr > 0)
    )

    safe_C0 = np.where(valid_mask, C0_arr, 2.0)
    safe_Ccr = np.where(valid_mask, C_cr, 1.0)
    ratio = np.clip(
        safe_Ccr / safe_C0,
        1e-12,
        1.0 - 1e-12
    )
    inv_erf = special.erfinv(1.0 - ratio)

    denom = (
        4.0
        * k_e
        * k_t
        * k_c
        * D0
        * (1.0 ** n)
        * (inv_erf ** 2)
    )

    valid_denom = denom > 0
    safe_denom = np.where(valid_denom, denom, 1.0)
    base_val = np.maximum(
        cover_mm ** 2 / safe_denom,
        1e-12
    )

    T = X1 * (
        base_val
        ** (
            1.0
            / np.maximum(1.0 - n, 1e-6)
        )
    )
    return np.where(
        valid_mask & valid_denom,
        T,
        np.inf
    )


def pitting_corrosion_matrix(
    d_rein_mm,
    T_init_arr,
    cover_mm,
    years_arr,
    R_arr,
    lambda_arr,
    w_b=0.5,
):
    i_corr0 = (
        37.8
        * lambda_arr
        * (1.0 - w_b) ** (-1.64)
        / cover_mm
    )

    A0 = np.pi * d_rein_mm ** 2 / 4.0
    corr_rate = np.zeros(
        (len(d_rein_mm), len(years_arr)),
        dtype=float
    )

    for y_idx, yr in enumerate(years_arr):
        t_p = np.where(
            np.isinf(T_init_arr),
            0.0,
            np.maximum(0.0, yr - T_init_arr)
        )

        diam_loss_uniform = (
            2.0
            * (
                0.0116
                * 0.85
                * i_corr0
                * (t_p ** 0.71)
                / 0.71
            )
        )

        d_rem_uniform = np.maximum(
            0.0,
            d_rein_mm - diam_loss_uniform
        )
        Au = np.pi * d_rem_uniform ** 2 / 4.0

        p_val = R_arr * (diam_loss_uniform / 2.0)
        A_rem = np.copy(A0)
        mask_p = p_val > 0

        if np.any(mask_p):
            p_v = np.minimum(
                p_val[mask_p],
                d_rein_mm[mask_p]
            )
            dr = d_rein_mm[mask_p]
            inner_val = np.maximum(
                0.0,
                1.0 - (p_v / dr) ** 2
            )
            a = 2.0 * p_v * np.sqrt(inner_val)

            theta1 = 2.0 * np.arcsin(
                np.clip(a / dr, -1.0, 1.0)
            )
            A1 = 0.5 * (
                theta1 * (0.5 * dr) ** 2
                - a * (0.5 * dr - p_v ** 2 / dr)
            )

            theta2 = np.where(
                p_v > 1e-12,
                2.0 * np.arcsin(
                    np.clip(a / (2.0 * p_v), -1.0, 1.0)
                ),
                0.0,
            )
            A2 = 0.5 * (
                theta2 * p_v ** 2
                - a * p_v ** 2 / dr
            )

            ADP = np.where(
                p_v <= dr / np.sqrt(2.0),
                A0[mask_p] - A1 - A2,
                A1 - A2,
            )

            Ap = (
                1.0 - a / (2.0 * dr)
            ) * (
                Au[mask_p] - A0[mask_p]
            ) + ADP

            A_rem[mask_p] = np.clip(
                Ap,
                0.0,
                A0[mask_p]
            )

        corr_rate[:, y_idx] = np.clip(
            1.0 - A_rem / A0,
            0.0,
            1.0
        )

    if FORCE_CORROSION_MONOTONIC:
        corr_rate = np.maximum.accumulate(
            corr_rate,
            axis=1
        )

    return corr_rate


def generate_corrosion_paths(samples_dict, years_arr):
    t_samples = np.asarray(samples_dict["t"], dtype=float)
    dl_samples = np.asarray(samples_dict["d_l"], dtype=float)
    dt_samples = np.asarray(samples_dict["d_t"], dtype=float)

    C0 = (
        np.asarray(samples_dict["Acs"], dtype=float) * 0.5
        + np.asarray(samples_dict["ecs"], dtype=float)
    )

    tc_mm = t_samples * 1000.0
    cover_stir_mm = np.maximum(
        tc_mm - dt_samples * 1000.0,
        15.0
    )

    T_init_long = compute_T_init(
        tc_mm,
        samples_dict["D0"],
        samples_dict["ke"],
        samples_dict["kt"],
        samples_dict["kc"],
        samples_dict["n_val"],
        samples_dict["Ccr"],
        C0,
        samples_dict["X1"],
    )

    T_init_stir = compute_T_init(
        cover_stir_mm,
        samples_dict["D0"],
        samples_dict["ke"],
        samples_dict["kt"],
        samples_dict["kc"],
        samples_dict["n_val"],
        samples_dict["Ccr"],
        C0,
        samples_dict["X1"],
    )

    corr_long = pitting_corrosion_matrix(
        dl_samples * 1000.0,
        T_init_long,
        tc_mm,
        years_arr,
        samples_dict["R"],
        samples_dict["lam_corr"],
    )

    corr_stir = pitting_corrosion_matrix(
        dt_samples * 1000.0,
        T_init_stir,
        cover_stir_mm,
        years_arr,
        samples_dict["R"],
        samples_dict["lam_corr"],
    )

    return (
        corr_long,
        corr_stir,
        T_init_long,
        T_init_stir,
    )


def generate_scour_paths(
    samples_dict,
    U,
    all_inputs,
    years_arr,
    sd_cov,
):
    """
    Continuous stochastic scour depth with a fixed lifecycle MC quantile.

    For each sample i, U_SD,i is generated once and retained for all years.
    The yearly distribution is:

        SD_mean(t) = lambda_sd * B *
                     {p[1-exp(-q t)] + r[1-exp(-s t)]}

        SD(t) ~ Truncated Normal(mean=SD_mean(t),
                                 std=COV_SD*SD_mean(t),
                                 range=[0, 8] m)

    No 0.5 m discretization is used.
    """
    input_ids = [p["id"] for p in all_inputs]
    sd_idx = input_ids.index("SD_val")
    U_SD = np.clip(U[:, sd_idx], 1e-10, 1.0 - 1e-10)

    p_arr = np.asarray(samples_dict["p_val"], dtype=float)
    q_arr = np.asarray(samples_dict["q_val"], dtype=float)
    r_arr = np.asarray(samples_dict["r_val"], dtype=float)
    s_arr = np.asarray(samples_dict["s_val"], dtype=float)
    B_arr = np.asarray(samples_dict["B_val"], dtype=float)
    lambda_sd_arr = np.asarray(samples_dict["lambda_sd"], dtype=float)

    raw = np.zeros((len(U_SD), len(years_arr)), dtype=float)

    for y_idx, yr in enumerate(years_arr):
        sd_mean = (
            lambda_sd_arr * B_arr *
            (p_arr * (1.0 - np.exp(-q_arr * yr))
             + r_arr * (1.0 - np.exp(-s_arr * yr)))
        )

        zero_mask = np.isclose(sd_mean, 0.0, atol=1e-14)
        dynamic_std = np.maximum(sd_mean * float(sd_cov), 1e-10)
        a = (0.0 - sd_mean) / dynamic_std
        b = (8.0 - sd_mean) / dynamic_std

        sd_samples = stats.truncnorm.ppf(
            U_SD,
            a,
            b,
            loc=sd_mean,
            scale=dynamic_std,
        )
        sd_samples = np.where(zero_mask, 0.0, sd_samples)
        raw[:, y_idx] = np.clip(sd_samples, 0.0, 8.0)

    if FORCE_SCOUR_MONOTONIC:
        raw = np.maximum.accumulate(raw, axis=1)

    return raw


# ============================================================
# 6. 20D assembly and lifecycle prediction
# ============================================================
def build_base_feature_matrix(samples_dict, n_samples):
    X_fixed = np.zeros((n_samples, 20), dtype=float)

    mapping = [
        ("N", 0),
        ("Dp", 1),
        ("rho_pl", 2),
        ("alpha", 3),
        ("S_Dp", 4),
        ("Dr", 5),
        ("Hp_Dc", 7),
        ("Dc_Dp", 8),
        ("rho_cl", 9),
        ("rho_ps", 10),
        ("fyl", 11),
        ("fc", 12),
        ("rho_cs", 13),
        ("t", 16),
        ("d_l", 17),
        ("fyt", 18),
        ("d_t", 19),
    ]

    for key, col_idx in mapping:
        X_fixed[:, col_idx] = samples_dict[key]

    return X_fixed


def run_lifecycle_prediction(
    assets,
    all_inputs,
    sd_cov,
    n_samples=N_SAMPLES,
    seed=RANDOM_SEED,
):
    label_names = get_label_names(assets)

    # Preserve the original 35-dimensional parameter-to-random-column mapping.
    by_id = {p["id"]: p for p in all_inputs}
    if set(by_id) != set(ORIGINAL_RANDOM_INPUT_ORDER) or len(by_id) != len(all_inputs):
        raise ValueError("Lifecycle GUI inputs must contain the original 35 unique parameter IDs.")
    all_inputs = [by_id[pid] for pid in ORIGINAL_RANDOM_INPUT_ORDER]

    samples_dict, U = generate_samples_dict(
        all_inputs,
        n_samples=n_samples,
        seed=seed,
    )

    (
        corr_long,
        corr_stir,
        T_init_long,
        T_init_stir,
    ) = generate_corrosion_paths(
        samples_dict,
        YEARS_FULL
    )

    scour_depths = generate_scour_paths(
        samples_dict,
        U,
        all_inputs,
        YEARS_FULL,
        sd_cov=sd_cov,
    )

    X_base = build_base_feature_matrix(
        samples_dict,
        n_samples
    )

    n_classes = len(label_names)
    n_years = len(YEARS_FULL)

    mode_paths = np.zeros(
        (n_samples, n_years),
        dtype=int
    )
    prob_paths = np.zeros(
        (n_samples, n_years, n_classes),
        dtype=float
    )
    annual_probs = np.zeros(
        (n_years, n_classes),
        dtype=float
    )

    for y_idx, yr in enumerate(YEARS_FULL):
        X_year = X_base.copy()
        X_year[:, 6] = scour_depths[:, y_idx]
        X_year[:, 14] = corr_stir[:, y_idx]
        X_year[:, 15] = corr_long[:, y_idx]

        pred_idx, probs = predict_model(
            X_year,
            assets
        )

        mode_paths[:, y_idx] = pred_idx
        prob_paths[:, y_idx, :] = probs
        annual_probs[y_idx, :] = (
            np.bincount(
                pred_idx,
                minlength=n_classes
            ).astype(float)
            / n_samples
        )

    transition_df = get_transition_table(
        mode_paths,
        prob_paths,
        label_names,
        YEARS_FULL,
    )

    hist_df = get_grouped_transition_histogram(
        transition_df,
        bin_width=TRANSITION_HIST_BIN_WIDTH,
    )

    return {
        "label_names": label_names,
        "samples_dict": samples_dict,
        "U": U,
        "T_init_long": T_init_long,
        "T_init_stir": T_init_stir,
        "corr_long": corr_long,
        "corr_stir": corr_stir,
        "scour_depths": scour_depths,
        "mode_paths": mode_paths,
        "prob_paths": prob_paths,
        "annual_probs": annual_probs,
        "transition_df": transition_df,
        "hist_df": hist_df,
    }


# ============================================================
# 7. Individual transition-time extraction
# ============================================================
def get_transition_table(
    mode_paths,
    prob_paths,
    label_names,
    years_arr,
):
    n_samples, n_years = mode_paths.shape
    label_names = np.asarray(label_names).astype(str)

    initial_modes = mode_paths[:, 0].copy()
    transition_years = np.full(
        n_samples,
        years_arr[-1],
        dtype=int
    )
    transition_flag = np.zeros(
        n_samples,
        dtype=bool
    )
    transition_modes = initial_modes.copy()
    transition_margin = np.full(
        n_samples,
        np.nan,
        dtype=float
    )

    search_end = (
        n_years
        if MIN_PERSIST_YEARS <= 1
        else n_years - MIN_PERSIST_YEARS + 1
    )

    # Vectorized implementation of the original first-transition scan.
    # Each sample retains its year-0 mode, 3-year persistence criterion,
    # probability-margin test, and earliest valid event year exactly as before.
    unresolved = np.ones(n_samples, dtype=bool)
    for j in range(1, search_end):
        yr = int(years_arr[j])
        if yr < MIN_TRANSITION_YEAR:
            continue
        new_modes = mode_paths[:, j]
        valid = unresolved & (new_modes != initial_modes)
        if not np.any(valid):
            continue
        for offset in range(max(1, MIN_PERSIST_YEARS)):
            valid &= mode_paths[:, j + offset] == new_modes
            if USE_TRANSITION_PROB_MARGIN:
                idx = np.flatnonzero(valid)
                if idx.size == 0:
                    break
                margins = (prob_paths[idx, j + offset, new_modes[idx]]
                           - prob_paths[idx, j + offset, initial_modes[idx]])
                valid[idx[margins < TRANSITION_PROB_MARGIN]] = False
        idx = np.flatnonzero(valid)
        if idx.size == 0:
            continue
        transition_years[idx] = yr
        transition_flag[idx] = True
        transition_modes[idx] = new_modes[idx]
        transition_margin[idx] = (prob_paths[idx, j, new_modes[idx]]
                                  - prob_paths[idx, j, initial_modes[idx]])
        unresolved[idx] = False
        if not unresolved.any():
            break

    initial_labels_raw = label_names[initial_modes]
    transition_labels_raw = label_names[transition_modes]

    initial_labels = np.array(
        [canonical_label(x) for x in initial_labels_raw],
        dtype=object
    )
    transition_labels = np.array(
        [canonical_label(x) for x in transition_labels_raw],
        dtype=object
    )

    transition_type = []
    for i in range(n_samples):
        if transition_flag[i]:
            transition_type.append(
                f"{initial_labels[i]}->{transition_labels[i]}"
            )
        else:
            transition_type.append(
                "No transition within 100 years"
            )

    return pd.DataFrame({
        "Sample_ID": np.arange(1, n_samples + 1),
        "Initial_Mode": initial_labels,
        "Initial_Mode_Display": [
            display_label(x)
            for x in initial_labels
        ],
        "Transition_Mode": transition_labels,
        "Transition_Mode_Display": [
            display_label(x)
            for x in transition_labels
        ],
        "Transition_Type": transition_type,
        "Transition_Year": transition_years,
        "Transition_Flag": transition_flag,
        "Transition_Probability_Margin": transition_margin,
    })


def get_grouped_transition_histogram(
    transition_df,
    bin_width=5,
):
    df_tr = transition_df[
        transition_df["Transition_Flag"]
    ].copy()

    groups = {
        "To_FFF": [
            "PFF->FFF",
            "PSF->FFF",
        ],
        "To_CSF": [
            "PFF->PSF",
            "FFF->PSF",
        ],
    }

    display = {
        "To_FFF":
            "To FFF (CFF→FFF + CSF→FFF)",
        "To_CSF":
            "To CSF (CFF→CSF + FFF→CSF)",
    }

    edges = np.arange(
        0,
        100 + bin_width,
        bin_width,
        dtype=float
    )

    rows = []
    for group_name, types in groups.items():
        years = df_tr.loc[
            df_tr["Transition_Type"].isin(types),
            "Transition_Year",
        ].to_numpy(dtype=float)

        counts, _ = np.histogram(
            years,
            bins=edges
        )
        n_group = len(years)

        for k, count in enumerate(counts):
            rows.append({
                "Target_Group": group_name,
                "Target_Group_Display": display[group_name],
                "Bin_Start": edges[k],
                "Bin_End": edges[k + 1],
                "Bin_Center": 0.5 * (
                    edges[k] + edges[k + 1]
                ),
                "Count": int(count),
                "Probability_within_group":
                    count / n_group
                    if n_group > 0
                    else 0.0,
                "Probability_in_all_samples":
                    count / len(transition_df),
                "N_group": int(n_group),
            })

    return pd.DataFrame(rows)


# ============================================================
# 8. UI builders
# ============================================================
# True mathematical subscripts, superscripts and roman (upright) units.
# Streamlit's markdown parser is unreliable for nested HTML <sub> in tiny rows.
PARAMETER_MATH_LABELS = {
    "Dc_Dp": r"D_c/D_p",
    "Hp_Dc": r"H_p/D_c",
    "alpha": r"\alpha",
    "d_l": r"d_l\,\mathrm{(m)}",
    "d_t": r"d_s\,\mathrm{(m)}",
    "t": r"t_c\,\mathrm{(m)}",
    "rho_cl": r"\rho_{c,l}",
    "rho_cs": r"\rho_{c,s}",
    "N": r"N",
    "Dp": r"D_p\,\mathrm{(m)}",
    "rho_pl": r"\rho_{p,l}",
    "rho_ps": r"\rho_{p,s}",
    "S_Dp": r"S_p/D_p",
    "fyl": r"f_{yl}\,\mathrm{(MPa)}",
    "fyt": r"f_{ys}\,\mathrm{(MPa)}",
    "fc": r"f_c\,\mathrm{(MPa)}",
    "Dr": r"D_r",
    "Acs": r"A_{cs}",
    "ecs": r"\varepsilon_{cs}",
    "Ccr": r"C_{cr}",
    "D0": r"D_0",
    "kc": r"k_c",
    "kt": r"k_t",
    "ke": r"k_e",
    "n_val": r"n",
    "X1": r"X_I",
    "R": r"R",
    "lam_corr": r"\lambda_{corr}",
    "SD_val": r"SD\,\mathrm{(m)}",
    "lambda_sd": r"\lambda_{sd}",
    "B_val": r"B\,\mathrm{(m)}",
    "p_val": r"p",
    "q_val": r"q",
    "r_val": r"r",
    "s_val": r"s",
}


def format_parameter_math(parameter_id):
    """Bold KaTeX math symbols while preserving existing subscripts and units."""
    expression = PARAMETER_MATH_LABELS[parameter_id]
    # Keep units in upright text; only mathematical symbols use bold math glyphs.
    symbol, sep, unit = expression.partition(r"\,\mathrm{")
    if sep:
        return rf"\boldsymbol{{{symbol}}}\,\mathrm{{{unit}"
    return rf"\boldsymbol{{{expression}}}"


def render_param_section(title, params_config, use_std=False):
    """Compact 5-column parameter editor; unchanged simulation input mapping.

    The original description text remains in the configuration tuple as
    metadata; the distribution selector is correctly headed Distribution.
    """
    if title:
        st.markdown(
            f"<div class='section-header'>{title}</div>",
            unsafe_allow_html=True,
        )

    # A narrower Distribution column plus wider inter-column gaps.
    column_widths = [1.15, 1.46, 1.12, 1.12, 1.22]
    headers = ["Parameter", "Distribution", "Mean",
               "Std" if use_std else "COV", "Range"]
    header_cols = st.columns(column_widths, gap="small")
    for col, label in zip(header_cols, headers):
        col.markdown(
            f"<div class='col-header'>{label}</div>",
            unsafe_allow_html=True,
        )
    st.markdown("<hr class='parameter-divider'>", unsafe_allow_html=True)

    user_vals = []
    for cfg in params_config:
        (p_id, html_name, _description, rng_str, p_min, p_max,
         p_mean, p_dist, p_disp, p_step, p_fmt, dist_opts) = cfg

        c1, c2, c3, c4, c5 = st.columns(column_widths, gap="small")
        # Plain Markdown MathText: renders subscripts and upright unit text
        # properly, independent of HTML sanitization and browser scaling.
        c1.markdown("$" + format_parameter_math(p_id) + "$")
        with c2:
            dist_val = st.selectbox(
                label=f"{p_id}_dist",
                options=dist_opts,
                index=dist_opts.index(p_dist),
                label_visibility="collapsed",
            )

        mean_disabled = p_mean is None or dist_val in ("Uniform", "DiscreteUniform")
        if p_mean is None:
            # An actual disabled BaseWeb input (not a Markdown <div>): same
            # dimensions/vertical alignment as editable Mean and COV fields.
            mean_val = 0.0
            with c3:
                st.text_input(
                    label=f"{p_id}_mean_dash", value="–",
                    disabled=True, label_visibility="collapsed",
                )
        else:
            with c3:
                kwargs = {}
                if p_min is not None:
                    kwargs["min_value"] = float(p_min)
                if p_max is not None:
                    kwargs["max_value"] = float(p_max)
                mean_val = st.number_input(
                    label=f"{p_id}_mean",
                    value=float(p_mean),
                    step=float(p_step),
                    format=p_fmt,
                    disabled=mean_disabled,
                    label_visibility="collapsed",
                    **kwargs,
                )

        disp_disabled = dist_val in ("Deterministic", "Uniform", "DiscreteUniform")
        if disp_disabled:
            disp_val = 0.0
            with c4:
                st.text_input(
                    label=f"{p_id}_disp_dash", value="–",
                    disabled=True, label_visibility="collapsed",
                )
        else:
            with c4:
                disp_val = st.number_input(
                    label=f"{p_id}_disp",
                    min_value=0.0,
                    value=float(p_disp),
                    step=0.05,
                    format="%.3f",
                    label_visibility="collapsed",
                )

        c5.markdown(
            f"<div class='parameter-range'>{rng_str}</div>",
            unsafe_allow_html=True,
        )

        if use_std:
            std_val = float(disp_val)
        elif disp_disabled:
            std_val = 0.0
        else:
            std_val = float(mean_val) * float(disp_val)

        user_vals.append({
            "id": p_id,
            "mean": float(mean_val),
            "std": std_val,
            "raw_disp": float(p_disp if p_id == "SD_val" else disp_val),
            "dist": dist_val,
            "min": p_min,
            "max": p_max,
        })
    return user_vals


# ============================================================
# 9. Plot helpers
# ============================================================
# ============================================================
# 9. GUI figures -- matching the uploaded plotting references
# ============================================================
def set_dynamic_y_axis(ax, maximum, *, probability=False):
    """Set zero-based, data-dependent limits and readable major tick spacing.

    Include plotted fills/curves and, on CDF figures, the threshold line in
    maximum. Limits do not clip valid plotted data. Probability cannot exceed 1.
    """
    v = float(maximum)
    if not np.isfinite(v) or v <= 0:
        v = 0.02 if probability else 0.05
    desired = v * 1.08  # small headroom above the tallest observed element
    if probability:
        desired = min(desired, 1.0)
    approx_step = desired / 5.0
    power = 10.0 ** np.floor(np.log10(approx_step))
    nice = np.asarray([1., 2., 2.5, 5., 10.]) * power
    step = float(nice[np.searchsorted(nice, approx_step, side="left")])
    upper = float(np.ceil((desired - 1e-12 * step) / step) * step)
    if probability:
        upper = min(1.0, upper)
    upper = max(upper, step)
    n_steps = int(np.floor(upper / step + 1e-9))
    ticks = np.arange(n_steps + 1, dtype=float) * step
    if len(ticks) == 0 or ticks[-1] < upper - step * 0.1:
        ticks = np.append(ticks, upper)
    ax.set_ylim(0.0, upper)
    ax.set_yticks(np.round(ticks, 10))


def make_corrosion_plot(years, corr_long, corr_stir):
    """Reference: population mean and ±1 standard deviation (not median/16-84%)."""
    fig, ax = plt.subplots(figsize=(6, 3.3), dpi=220)
    ymax = 0.0
    for matrix, line_color, shade_color, alpha, text_label in [
        (corr_stir, "#3B82F6", "#BFDBFE", 0.62, "Transverse reinforcement"),
        (corr_long, "#E45756", "#FBC4C4", 0.55, "Longitudinal reinforcement"),
    ]:
        mu = np.mean(matrix, axis=0)
        sd = np.std(matrix, axis=0, ddof=0)
        ax.plot(years, mu, color=line_color, linewidth=2.2,
                label=f"{text_label} (mean)")
        shaded_upper = np.clip(mu + sd, 0, 1)
        ymax = max(ymax, float(np.max(shaded_upper)), float(np.max(mu)))
        ax.fill_between(years, np.clip(mu - sd, 0, 1), shaded_upper,
                        color=shade_color, alpha=alpha,
                        label=f"{text_label} (mean ± 1 Std)")
    set_axis_labels(ax, r"Service time, $t$ (years)", r"Corrosion level, $Q_{corr}$")
    ax.set_xlim(0, 100)
    set_dynamic_y_axis(ax, ymax, probability=True)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(False)
    apply_academic_style(ax)
    fig.tight_layout(pad=0.35)
    return fig


def make_scour_plot(years, samples_dict, sd_cov):
    """Reference: theoretical mean + 1 SD band, no markers on scour curve."""
    fig, ax = plt.subplots(figsize=(6, 3.3), dpi=220)
    x = np.asarray(years, dtype=float)
    def first(name):
        return float(np.asarray(samples_dict[name], dtype=float)[0])
    # Original GUI: B_val is the unadjusted B; lambda_sd is a separate factor.
    mean = (first("lambda_sd") * first("B_val") *
            (first("p_val") * (1.0 - np.exp(-first("q_val") * x))
             + first("r_val") * (1.0 - np.exp(-first("s_val") * x))))
    dev = float(sd_cov) * mean
    low, high = np.clip(mean - dev, 0, 8), np.clip(mean + dev, 0, 8)
    ax.fill_between(x, low, high, color="#B8E0E6", alpha=0.65,
                    label="Mean ± 1 Std", zorder=1)
    ax.plot(x, mean, color="#1F7A8C", linewidth=2.2, label="Mean", zorder=4)
    # Unlike the reference's marked-year illustration, no point markers,
    # year-specific guide lines, or annotations are added here.
    set_axis_labels(ax, r"Service time, $t$ (years)", r"Scour depth, $SD$ (m)")
    ax.set_xlim(0, 100)
    set_dynamic_y_axis(ax, max(float(np.max(high)), float(np.max(mean))))
    ax.set_xticks(np.arange(0, 101, 20))
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(x) for x in ("Mean", "Mean ± 1 Std") if x in labels]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              frameon=False, loc="upper left")
    ax.grid(False)
    apply_academic_style(ax)
    fig.tight_layout(pad=0.35)
    return fig


def make_failure_probability_plot(
    years,
    annual_probs,
    label_names,
):
    fig, ax = plt.subplots(
        figsize=(6, 3.3),
        dpi=220
    )

    # User-specified RGB colors /255:
    # FFF [70,192,115], CFF [60,117,189], CSF [224,47,98]
    colors = {
        "FFF": "#46C073",
        "PFF": "#3C75BD",
        "CFF": "#3C75BD",
        "PSF": "#E02F62",
        "CSF": "#E02F62",
    }

    for idx, name in enumerate(label_names):
        canonical = canonical_label(name)
        ax.plot(
            years,
            annual_probs[:, idx],
            color=colors.get(
                str(name),
                colors.get(canonical, "#555555"),
            ),
            lw=2.4,
            label=display_label(name),
        )

    set_axis_labels(
        ax,
        r"Service time, $t$ (years)",
        r"Probability, $P_{\mathrm{SFM}}$"
    )
    ax.set_xlim(0, 100)
    set_dynamic_y_axis(ax, float(np.max(annual_probs)), probability=True)

    # Legend order requested: CFF, FFF, CSF (curve drawing order unchanged).
    handles, labels = ax.get_legend_handles_labels()
    legend_map = dict(zip(labels, handles))
    wanted = ["CFF", "FFF", "CSF"]
    ordered = [name for name in wanted if name in legend_map]
    ax.legend([legend_map[name] for name in ordered], ordered,
              frameon=False, loc="upper right", ncol=3)
    ax.grid(False)
    apply_academic_style(ax)
    fig.tight_layout(pad=0.35)
    return fig


# ============================================================
# 9a. SFM transition type probabilities and best-fit transition times
# ============================================================
def _gof_group_cutpoints(max_year, bin_width):
    """与 np.histogram(整数年, bins=0,5,...,100) 完全相同的离散年分组。

    [0,5)含0,1,2,3,4年，底层连续潜在时间区间应是[0,4.5]；
    [5,10)含5,...,9年，连续区间应是[4.5,9.5]；
    最后[95,100]包括100年，潜在时间区间为[94.5,100]。
    """
    edges = np.arange(0.0, max_year + 0.5*bin_width, bin_width)
    if abs(edges[-1]-max_year)>1e-8:
        raise ValueError('max_year必须可被bin_width整除')
    cuts = np.r_[0.0, edges[1:-1]-0.5, max_year]
    return edges, cuts

def _heldout_chi_square_gof(observed_years, fitted_bin_prob, min_expected=5.0):
    """独立验证样本的Pearson χ²检验；相邻分组自动合并使期望频数≥5。

    参数来自独立训练子集，因此自由度为合并后组数-1（不是组数-1-k）。
    为防止空分组干扰和过度乐观，禁止仅因实际次数为0就删除分组。
    """
    observed_counts = np.asarray(observed_years, dtype=float)
    probs = np.asarray(fitted_bin_prob, dtype=float)
    n = int(round(observed_counts.sum()))
    if n <= 0 or not np.isfinite(probs).all() or np.any(probs < 0):
        return np.nan, np.nan, 0, 'Invalid counts or probabilities'
    probs = probs / probs.sum()
    expected = n * probs
    # 从左到右合并连续分组；右侧尾部不足时合并入最后一组。
    merged_o=[]; merged_e=[]; o_acc=0.0; e_acc=0.0
    for o,e in zip(observed_counts,expected):
        o_acc+=o; e_acc+=e
        if e_acc >= min_expected:
            merged_o.append(o_acc); merged_e.append(e_acc)
            o_acc=0.0; e_acc=0.0
    if e_acc > 0:
        if not merged_o:
            return np.nan,np.nan,0,'Too few expected counts'
        merged_o[-1]+=o_acc; merged_e[-1]+=e_acc
    n_groups=len(merged_o)
    df=n_groups-1  # 训练数据独立估计全部参数，不扣减拟合参数个数
    if df < 1:
        return np.nan,np.nan,df,'Too few merged bins'
    o_arr=np.asarray(merged_o)
    e_arr=np.asarray(merged_e)
    chi2=float(np.sum((o_arr-e_arr)**2/e_arr))
    p=float(stats.chi2.sf(chi2, df))
    return chi2,p,df,'OK'

def _fit_transition_times_best_distribution(years_group, bin_width=5.0,
                                             max_year=100.0, *, _training_only=False):
    """离散整数年极大似然估计 + AIC筛选 + 独立验证集Pearson χ²拟合优度P值。

    1) 候选分布5种；全部参数采用整数年份区间概率进行MLE；
    2) 从观测转移事件随机分出80%训练/20%验证，只用于模型拟合统计；
    3) 用训练集AIC选择优选分布，保持验证集对选择和参数估计独立；
    4) 用验证集按5年分组的Pearson卡方计算P值，不把拟合优度P值解释为模型概率；
    5) 最终绘图参数在全部观测数据上重新拟合，以利用全部信息。
    分布拟合仍条件化在0-100年已发生转移的样本，不包括右删失样本。
    """
    from scipy.optimize import minimize

    observed = np.asarray(years_group, dtype=float)
    observed = observed[np.isfinite(observed) & (observed >= 0) & (observed <= max_year)]
    if len(observed) < FIT_MIN_SAMPLES:
        return None, pd.DataFrame()

    # 训练拟合与独立验证。在递归训练阶段不再次分割，也不计算P值。
    train_table=None
    heldout_years=None
    if not _training_only and len(observed) >= 2*FIT_MIN_SAMPLES:
        rng=np.random.default_rng(FIT_GOF_SEED)
        perm=rng.permutation(len(observed))
        n_val=max(FIT_MIN_SAMPLES, int(round(len(observed)*FIT_GOF_TEST_FRACTION)))
        n_val=min(n_val, len(observed)-FIT_MIN_SAMPLES)
        train_years=observed[perm[n_val:]]
        heldout_years=observed[perm[:n_val]]
        _,train_table=_fit_transition_times_best_distribution(
            train_years,bin_width,max_year,_training_only=True)

    years_int = np.rint(observed).astype(int)
    uniq, cnt = np.unique(years_int, return_counts=True)
    lower = np.clip(uniq.astype(float) - 0.5, 0.0, max_year)
    upper = np.clip(uniq.astype(float) + 0.5, 0.0, max_year)
    count_total = len(observed)
    mean_v = max(float(np.mean(observed)),0.1)
    sd_v = max(float(np.std(observed)),0.4)

    # 仅保留原有5种候选分布：Normal、Lognormal、Gamma、Weibull、Beta(0-100)。
    candidate_cfg = {
        'Normal': {
            'start': [mean_v,np.log(sd_v)],
            'bounds': [(-150.0,250.0),(np.log(0.05),np.log(1000.0))],
        },
        'Lognormal': {
            'start': [np.log(max(float(np.std(np.log(np.maximum(observed,0.5)))),0.2)),
                      float(np.mean(np.log(np.maximum(observed,0.5))))],
            'bounds': [(np.log(0.05),np.log(10.0)),(np.log(0.05),np.log(2000.0))],
        },
        'Gamma': {
            'start': [np.log(np.clip((mean_v/sd_v)**2,0.1,100.0)),
                      np.log(np.clip(sd_v**2/max(mean_v,0.2),0.1,1000.0))],
            'bounds': [(np.log(0.02),np.log(120.0)),(np.log(0.02),np.log(10000.0))],
        },
        'Weibull': {
            'start': [np.log(1.5),np.log(np.clip(mean_v/special.gamma(1+1/1.5),0.1,2000.0))],
            'bounds': [(np.log(0.05),np.log(50.0)),(np.log(0.1),np.log(5000.0))],
        },
        'Beta_0_100': {
            'start': [np.log(1.5),np.log(1.5)],
            'bounds': [(np.log(0.02),np.log(200.0)),(np.log(0.02),np.log(200.0))],
        },
    }

    def dist_from_theta(name,theta):
        if name=='Normal':
            mu,sigma=float(theta[0]),float(np.exp(theta[1]))
            return stats.norm(loc=mu,scale=sigma), {'mu':mu,'sigma':sigma}
        if name=='Lognormal':
            shape,scale=map(float,np.exp(theta))
            return stats.lognorm(s=shape,loc=0,scale=scale), {'sigma_ln':shape,'scale':scale}
        if name=='Gamma':
            shape,scale=map(float,np.exp(theta))
            return stats.gamma(a=shape,loc=0,scale=scale), {'shape_k':shape,'scale_theta':scale}
        if name=='Weibull':
            shape,scale=map(float,np.exp(theta))
            return stats.weibull_min(c=shape,loc=0,scale=scale),{'shape_k':shape,'scale_lambda':scale}
        if name=='Beta_0_100':
            a,b=map(float,np.exp(theta))
            return stats.beta(a=a,b=b,loc=0,scale=max_year),{'a':a,'b':b}
        raise ValueError(name)

    def conditional_interval_mass(dist,lo,hi):
        lo=np.asarray(lo,dtype=float)
        hi=np.asarray(hi,dtype=float)
        total=max(float(dist.cdf(max_year)-dist.cdf(0.0)),0.0)
        if total < 1e-160 or not np.isfinite(total):
            return None
        cdf_diff=dist.cdf(hi)-dist.cdf(lo)
        sf_diff=dist.sf(lo)-dist.sf(hi)
        prob=np.maximum(cdf_diff,sf_diff)/total
        if not np.all(np.isfinite(prob)) or np.any(prob < -1e-9):
            return None
        return np.maximum(prob,0.0)

    # 必须使用与整数年直方图一致的4.5/9.5/...分界，而非简单[0,5],[5,10]
    edges,cuts=_gof_group_cutpoints(max_year,bin_width)
    hist_counts,_=np.histogram(observed,bins=edges)
    hist_obs=hist_counts/float(count_total)
    fitted={}; entries=[]
    for name,cfg in candidate_cfg.items():
        def negative_log_likelihood(theta):
            try:
                dist,_=dist_from_theta(name,theta)
                mass=conditional_interval_mass(dist,lower,upper)
                if mass is None: return 1e100
                loss=-float(np.dot(cnt,np.log(np.maximum(mass,1e-300))))
                return loss if np.isfinite(loss) else 1e100
            except (FloatingPointError,ValueError,OverflowError):
                return 1e100
        bounds=cfg['bounds']
        blo=np.array([a for a,b in bounds]); bhi=np.array([b for a,b in bounds])
        starts=[np.clip(np.asarray(cfg['start'],dtype=float),blo,bhi)]
        if name=='Beta_0_100':
            starts.extend([np.log([0.3,1.5]),np.log([2.0,0.5])])
        elif name in ('Gamma','Weibull'):
            starts.append(np.clip(np.array([np.log(0.6),np.log(max(mean_v,1.0))]),blo,bhi))
        elif name=='Lognormal':
            starts.append(np.array([np.log(1.3),np.log(max(mean_v,1.0))]))
        best_opt=None
        for guess in starts:
            opt=minimize(negative_log_likelihood,np.clip(guess,blo,bhi),
                         method='L-BFGS-B',bounds=bounds,
                         options={'maxiter':230,'ftol':1e-10})
            if np.isfinite(opt.fun) and (best_opt is None or opt.fun<best_opt.fun):
                best_opt=opt
        if best_opt is None or best_opt.fun>=1e90:
            entries.append({'Distribution':name,'N_Observed':count_total,'Status':'FAILED',
                            'AIC':np.nan,'BIC':np.nan,'GOF_Pvalue':np.nan})
            continue
        dist,params=dist_from_theta(name,best_opt.x)
        fit_bin=conditional_interval_mass(dist,cuts[:-1],cuts[1:])
        if fit_bin is None or fit_bin.sum()<=0:
            entries.append({'Distribution':name,'N_Observed':count_total,'Status':'FAILED',
                            'AIC':np.nan,'BIC':np.nan,'GOF_Pvalue':np.nan})
            continue
        fit_bin=np.asarray(fit_bin,dtype=float)/np.sum(fit_bin)
        loglik=-float(best_opt.fun)
        k=len(params)
        aic=float(2*k-2*loglik)
        bic=float(np.log(count_total)*k-2*loglik)
        rmse=float(np.sqrt(np.mean((hist_obs-fit_bin)**2)))
        ks5=float(np.max(np.abs(np.cumsum(hist_obs)-np.cumsum(fit_bin))))
        rowspec={'Distribution':name,'N_Observed':count_total,'Status':'OK',
                 'LogLikelihood':loglik,'AIC':aic,'BIC':bic,
                 'Histogram_RMSE':rmse,'CDF_Bin_KS':ks5,
                 'Parameters':'; '.join(f'{key}={val:.6g}' for key,val in params.items()),
                 **params}
        entries.append(rowspec)
        fitted[name]={'distribution':dist,'params':params,'aic':aic,'bic':bic,'rmse':rmse,
                      'interval_mass':conditional_interval_mass,'fit_bin_prob':fit_bin}

    table=pd.DataFrame(entries)
    table.attrs['fitted_candidates']=fitted
    valid=table.loc[np.isfinite(table['AIC'])].sort_values(['AIC','Histogram_RMSE'])
    if valid.empty: return None,table

    if _training_only or train_table is None or heldout_years is None:
        winner=str(valid.iloc[0]['Distribution'])
        fitted[winner]['name']=winner
        return fitted[winner],table

    # 训练集同时决定分布选择及参数；验证集仅做拟合优度检验，不参与选择。
    train_objs=train_table.attrs.get('fitted_candidates',{})
    train_valid=train_table.loc[np.isfinite(train_table['AIC'])]
    if train_valid.empty:
        fallback_name = str(valid.iloc[0]['Distribution'])
        fitted[fallback_name]['name'] = fallback_name
        return fitted[fallback_name], table
    train_selected=str(train_valid.sort_values(['AIC','Histogram_RMSE']).iloc[0]['Distribution'])
    val_counts,_=np.histogram(heldout_years,bins=edges)
    pvalue_map={}
    for name,model in fitted.items():
        model_from_train=train_objs.get(name)
        if model_from_train is None: continue
        test_dist=model_from_train['distribution']
        p_train=model_from_train['interval_mass'](test_dist,cuts[:-1],cuts[1:])
        if p_train is None or p_train.sum()<=0: continue
        chi2,pval,df,status=_heldout_chi_square_gof(
            val_counts,p_train,min_expected=FIT_GOF_MIN_EXPECTED)
        pvalue_map[name]=(chi2,pval,df,status)
    table['Selection_Train_AIC']=table['Distribution'].map(
        dict(zip(train_table['Distribution'],train_table['AIC'])))
    table['GOF_Pvalue']=table['Distribution'].map({n:v[1] for n,v in pvalue_map.items()})
    table['GOF_Chi2']=table['Distribution'].map({n:v[0] for n,v in pvalue_map.items()})
    table['GOF_DF']=table['Distribution'].map({n:v[2] for n,v in pvalue_map.items()})
    table['GOF_Status']=table['Distribution'].map({n:v[3] for n,v in pvalue_map.items()})
    table['GOF_Test']='Independent holdout Pearson chi-square, discrete 5-year bins'
    table['N_Fit_Train']=len(observed)-len(heldout_years)
    table['N_GOF_Holdout']=len(heldout_years)
    table['Alpha']=FIT_GOF_ALPHA
    table['GOF_NotRejected_at_0.05']=table['GOF_Pvalue']>=FIT_GOF_ALPHA
    table['Is_Selected']=table['Distribution']==train_selected
    if train_selected not in fitted:
        train_selected=str(valid.iloc[0]['Distribution'])
    winner=fitted[train_selected]
    winner['name']=train_selected
    winner['gof_pvalue']=pvalue_map.get(train_selected,(np.nan,np.nan,np.nan,'N/A'))[1]
    winner['gof_chi2']=pvalue_map.get(train_selected,(np.nan,np.nan,np.nan,'N/A'))[0]
    winner['gof_df']=pvalue_map.get(train_selected,(np.nan,np.nan,np.nan,'N/A'))[2]
    winner['selection_train_aic']=float(train_table.loc[train_table['Distribution']==train_selected,'AIC'].iloc[0])
    winner['n_gof_holdout']=len(heldout_years)
    return winner,table


TRANSITION_TYPE_FIG_HEIGHT = 4.5  # inches; only this histogram figure


def make_transition_type_plot(transition_df):
    """Relative frequencies among ALL lifecycle samples, including no transition."""
    type_fig_df = get_transition_type_figure_data(transition_df)
    labels = type_fig_df["Transition_Type"].tolist()
    probs = type_fig_df["Probability_AllSamples"].to_numpy(dtype=float)
    colors = ["#33A1BC" if " to FFF" in v else
              "#EE95A3" if " to CSF" in v else "#9A9A9A" for v in labels]
    fig, ax = plt.subplots(figsize=(6, TRANSITION_TYPE_FIG_HEIGHT), dpi=220)
    x = np.arange(len(labels))
    ax.bar(x, probs, color=colors, edgecolor="black", linewidth=0.5, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=32, ha="right",
                       fontsize=FIGURE_TEXT_FONT_SIZE)
    set_axis_labels(ax, "Transition type", "Probability")
    set_dynamic_y_axis(ax, float(np.max(probs)) if probs.size else 0.0,
                       probability=True)
    ax.grid(False)
    apply_academic_style(ax)
    fig.subplots_adjust(left=0.16, right=0.99, top=0.97, bottom=0.38)
    return fig


def fit_transition_density_binaverage_independent(selected_probability, observed_density,
                                                   bin_width=5.0, max_year=100.0):
    """对5年区间平均密度单独进行最小二乘参数估计，不复用概率图的MLE参数。

    为便于两图直接对比，分布族保持与概率图AIC选中的分布一致；只独立
    估计该分布族的参数。观测/模型密度都是：5年分组条件概率 / 5年。
    区间理论概率沿用原代码的整数年边界0/4.5/9.5/.../100，保证
    概率图、密度图的统计年份完全一致。不同拟合损失可能产生不同参数，
    但两组参数也有可能在数值上接近，不会人为强制差异。
    """
    from scipy.optimize import minimize

    if selected_probability is None:
        return None
    name = str(selected_probability['name'])
    old = selected_probability['params']
    observed_density = np.asarray(observed_density, dtype=float)
    edges, cuts = _gof_group_cutpoints(float(max_year), float(bin_width))
    if len(observed_density) != len(cuts)-1:
        raise ValueError('区间概率密度长度与5年分组不一致')
    if not np.all(np.isfinite(observed_density)) or np.any(observed_density < 0):
        raise ValueError('观测区间密度中存在非法数值')
    if not np.isclose(observed_density.sum() * bin_width, 1.0, atol=1e-8):
        raise ValueError('观测区间平均密度的积分面积不为1')

    if name == 'Normal':
        initial = np.array([float(old['mu']), np.log(float(old['sigma']))])
        bounds = [(-150.0, 250.0), (np.log(0.05), np.log(1000.0))]
        def make_dist(v):
            mu, sigma = float(v[0]), float(np.exp(v[1]))
            return stats.norm(loc=mu, scale=sigma), {'mu':mu, 'sigma':sigma}
    elif name == 'Lognormal':
        initial = np.log([float(old['sigma_ln']), float(old['scale'])])
        bounds = [(np.log(0.05), np.log(10.0)), (np.log(0.05), np.log(2000.0))]
        def make_dist(v):
            sigma_ln, scale = map(float, np.exp(v))
            return stats.lognorm(s=sigma_ln, loc=0, scale=scale), {'sigma_ln':sigma_ln, 'scale':scale}
    elif name == 'Gamma':
        initial = np.log([float(old['shape_k']), float(old['scale_theta'])])
        bounds = [(np.log(0.02), np.log(120.0)), (np.log(0.02), np.log(10000.0))]
        def make_dist(v):
            k, theta = map(float, np.exp(v))
            return stats.gamma(a=k, loc=0, scale=theta), {'shape_k':k, 'scale_theta':theta}
    elif name == 'Weibull':
        initial = np.log([float(old['shape_k']), float(old['scale_lambda'])])
        bounds = [(np.log(0.05), np.log(50.0)), (np.log(0.1), np.log(5000.0))]
        def make_dist(v):
            k, scale = map(float, np.exp(v))
            return stats.weibull_min(c=k, loc=0, scale=scale), {'shape_k':k, 'scale_lambda':scale}
    elif name == 'Beta_0_100':
        initial = np.log([float(old['a']), float(old['b'])])
        bounds = [(np.log(0.02), np.log(200.0)), (np.log(0.02), np.log(200.0))]
        def make_dist(v):
            a, b = map(float, np.exp(v))
            return stats.beta(a=a, b=b, loc=0, scale=max_year), {'a':a, 'b':b}
    else:
        raise ValueError(f'不支持的区间密度拟合分布: {name}')

    interval_mass = selected_probability['interval_mass']
    normalization = max(float(np.sum(observed_density**2)), 1e-12)

    def objective(v):
        try:
            dist, _ = make_dist(v)
            prob = interval_mass(dist, cuts[:-1], cuts[1:])
            if prob is None:
                return 1e12
            expected_density = np.asarray(prob, dtype=float) / bin_width
            if not np.all(np.isfinite(expected_density)):
                return 1e12
            return float(np.sum((expected_density - observed_density)**2) / normalization)
        except (ValueError, FloatingPointError, OverflowError):
            return 1e12

    lo = np.array([v[0] for v in bounds]); hi = np.array([v[1] for v in bounds])
    starts = [initial, initial + np.array([0.25, -0.25]), initial + np.array([-0.25, 0.25])]
    best = None
    for guess in starts:
        solution = minimize(objective, np.clip(guess, lo, hi), method='L-BFGS-B',
                            bounds=bounds, options={'maxiter':650, 'ftol':1e-13})
        if np.isfinite(solution.fun) and (best is None or solution.fun < best.fun):
            best = solution
    if best is None or best.fun >= 1e10:
        raise RuntimeError(f'{name}: 5年平均密度独立拟合未得到有效结果')

    dist, params = make_dist(best.x)
    fit_mass = interval_mass(dist, cuts[:-1], cuts[1:])
    fit_mass = np.asarray(fit_mass, dtype=float)
    fitted_density = fit_mass / bin_width
    rmse = float(np.sqrt(np.mean((fitted_density - observed_density)**2)))
    return {
        'name': name,
        'params': params,
        'distribution': dist,
        'interval_mass': interval_mass,
        'bin_probability': fit_mass,
        'bin_average_density': fitted_density,
        'density_rmse': rmse,
        'normalized_sse': float(best.fun),
        'fit_method': 'Independent least squares of 5-year conditional bin-average densities',
    }


def get_fitted_five_year_bin_density(selected):
    """5-year fitted densities from the *independent density-LS fit*.

    The independently fitted density parameters also supply every displayed
    legend statistic and every BestFit_* field in the Excel export.
    """
    if selected is None or selected.get("density_fit") is None:
        return None, None, None
    density_fit = selected["density_fit"]
    width = float(TRANSITION_HIST_BIN_WIDTH)
    edges = np.arange(0.0, 100.0 + width * 0.5, width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mass = np.asarray(density_fit["bin_probability"], dtype=float)
    density = np.asarray(density_fit["bin_average_density"], dtype=float)
    if (mass.size != centers.size or density.size != centers.size
            or not np.all(np.isfinite(density))):
        return None, None, None
    return centers, density, mass


def _five_year_average_fitted_curve(selected):
    """Reference implementation: PCHIP through independently fitted 5-y densities.

    Center nodes equal the independent least-squares fitted average density;
    both endpoints are closed to zero, exactly as in the uploaded reference.
    """
    from scipy.interpolate import PchipInterpolator

    centers, fitted_density, _ = get_fitted_five_year_bin_density(selected)
    if centers is None:
        return None, None
    x_knots = np.r_[0.0, centers, 100.0]
    y_knots = np.r_[0.0, fitted_density, 0.0]
    x_curve = np.linspace(0.0, 100.0, 1001)
    y_curve = np.maximum(0.0, PchipInterpolator(
        x_knots, y_knots, extrapolate=False)(x_curve))
    y_curve[0] = 0.0
    y_curve[-1] = 0.0
    return x_curve, y_curve


def get_fitted_transition_moments(selected):
    """Mean and COV of the fitted 0–100 year conditional integer-year model.

    The reference fits observed *integer* transition years and conditions on
    a first transition occurring by 100 years; use precisely the same bins
    when deriving the fitted distribution's statistics.
    """
    if selected is None:
        return np.nan, np.nan, np.nan
    yr = np.arange(101, dtype=float)
    lo = np.maximum(yr - 0.5, 0.0)
    hi = np.minimum(yr + 0.5, 100.0)
    mass = selected["interval_mass"](selected["distribution"], lo, hi)
    if mass is None:
        return np.nan, np.nan, np.nan
    mass = np.asarray(mass, dtype=float)
    if not np.all(np.isfinite(mass)) or mass.sum() <= 0:
        return np.nan, np.nan, np.nan
    mass /= mass.sum()
    avg = float(np.dot(yr, mass))
    std = float(np.sqrt(np.dot((yr - avg) ** 2, mass)))
    return avg, std, float(std / avg) if avg > 1e-12 else np.nan


# The two histogram panels include ONLY observed, reliable FIRST transitions.
# Use the exact same underlying samples for displayed mean/COV, not the
# histogram-bin centers, the PCHIP curve, or the fitted parametric PDF.
TRANSITION_TIME_TARGET_TYPES = {
    "To_FFF": ("PFF->FFF", "PSF->FFF"),
    "To_CSF": ("PFF->PSF", "FFF->PSF"),
}


def get_target_observed_transition_years(transition_df, target):
    """Observed first-transition years for the given plotted target group."""
    if target not in TRANSITION_TIME_TARGET_TYPES:
        raise ValueError(f"Unknown transition target: {target}")
    mask = (
        transition_df["Transition_Flag"].astype(bool)
        & transition_df["Transition_Type"].isin(TRANSITION_TIME_TARGET_TYPES[target])
    )
    years = transition_df.loc[mask, "Transition_Year"].to_numpy(dtype=float)
    return years[np.isfinite(years) & (years >= 0) & (years <= 100)]


def get_observed_transition_sample_moments(years):
    """Empirical mean, population Std and COV among observed transition samples.

    ddof=0 matches the Monte Carlo population statistics used in other plots.
    The non-transitioned samples are right-censored and must not be assigned
    a transition time of 100 years when calculating these sample statistics.
    """
    years = np.asarray(years, dtype=float)
    years = years[np.isfinite(years) & (years >= 0) & (years <= 100)]
    if years.size == 0:
        return {"count": 0, "mean": np.nan, "std": np.nan, "cov": np.nan}
    mean = float(np.mean(years))
    std = float(np.std(years, ddof=0))
    return {
        "count": int(years.size),
        "mean": mean,
        "std": std,
        "cov": float(std / mean) if mean > 1e-12 else np.nan,
    }


def format_fitted_legend(selected):
    """Three-line chart legend: fitted distribution name and OBSERVED sample moments.

    The second line of fitted distribution parameters is intentionally hidden
    from the PLOT LEGEND only. Its numerical values are kept in the fitted
    model and in the Excel BestFit_Parameters/BestFit_Param_* fields.
    """
    density_fit = selected.get("density_fit") if selected is not None else None
    if density_fit is None:
        raise ValueError("The transition-time legend requires a fitted 5-year density model.")

    titles = {
        "Normal": "Fitted normal PDF",
        "Lognormal": "Fitted lognormal PDF",
        "Gamma": "Fitted gamma PDF",
        "Weibull": "Fitted Weibull PDF",
        "Beta_0_100": "Fitted Beta PDF",
    }
    dist_name = str(density_fit["name"])
    if dist_name not in titles:
        raise ValueError(f"Unknown fitted distribution for legend: {dist_name}")

    sample_stats = selected.get("sample_moments")
    if sample_stats is None:
        raise ValueError("Observed transition sample statistics were not supplied.")

    return (f"{titles[dist_name]}\n"
            f"Mean = {sample_stats['mean']:.1f} years\n"
            f"COV = {sample_stats['cov']:.2f}")


def compute_transition_fits(transition_df, hist_df):
    """Reference five candidate MLE fits and independent held-out GOF tests."""
    fit_results = {}
    tables = []
    summary = []
    for target in TRANSITION_TIME_TARGET_TYPES:
        yr = get_target_observed_transition_years(transition_df, target)
        sample_stats = get_observed_transition_sample_moments(yr)
        selected, fits = _fit_transition_times_best_distribution(
            yr, bin_width=float(TRANSITION_HIST_BIN_WIDTH), max_year=100.0)
        if selected is not None:
            selected["sample_moments"] = sample_stats
            g = hist_df.loc[hist_df["Target_Group"] == target].sort_values("Bin_Start")
            if not g.empty:
                width = (g["Bin_End"] - g["Bin_Start"]).to_numpy(dtype=float)
                observed_density = g["Probability_within_group"].to_numpy(dtype=float) / width
                # AIC selects the distribution FAMILY. The independent 5-year
                # density LS fit supplies the plotted curve and fitted
                # parameters. Legend Mean/COV and Excel moments are empirical.
                selected["density_fit"] = fit_transition_density_binaverage_independent(
                    selected, observed_density,
                    bin_width=float(TRANSITION_HIST_BIN_WIDTH), max_year=100.0)
        fit_results[target] = selected
        if not fits.empty:
            fits = fits.copy()
            fits.insert(0, "Target_Group", target)
            tables.append(fits)
        row = {"Target_Group": target, "N_Observed": int(yr.size),
               "Distribution": selected["name"] if selected else "Not fitted",
               "Mean_Sample_Years": sample_stats["mean"],
               "Std_Sample_Years": sample_stats["std"],
               "COV_Sample": sample_stats["cov"]}
        if selected:
            row.update({"AIC": selected["aic"], "BIC": selected["bic"],
                        "GOF_Pvalue": selected.get("gof_pvalue", np.nan)})
            dfit = selected.get("density_fit")
            if dfit is None:
                raise ValueError(f"{target}: missing independent 5-year density fit")
            row.update(dfit["params"])
            if dfit["name"] == "Lognormal":
                row["lambda"] = float(np.log(dfit["params"]["scale"]))
                row["zeta"] = float(dfit["params"]["sigma_ln"])
            row["Fit_Method"] = dfit["fit_method"]
            row["Density_Fit_RMSE"] = dfit["density_rmse"]
            row["Density_Fit_Parameters"] = "; ".join(
                f"{k}={v:.9g}" for k, v in dfit["params"].items())
        summary.append(row)
    return fit_results, (pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()), pd.DataFrame(summary)


# ============================================================
# THE DISTRIBUTIONS OF SFM TRANSITION TIMES — independent controls
# All Matplotlib font sizes/line widths are in points (pt).
# The Streamlit section heading uses CSS pixels (px).
# Changing these affects ONLY this 1x2 transition-time figure.
# ============================================================
TRANSITION_TIME_AXIS_LABEL_FONT = 18          # x/y axis label font size (pt)
TRANSITION_TIME_TICK_FONT = 17                # tick label font size (pt)
TRANSITION_TIME_LEGEND_FONT = 17              # fitted-distribution legend (pt)
TRANSITION_TIME_SUBPLOT_TITLE_FONT = 17       # "Transition to FFF/CSF" (pt)
TRANSITION_TIME_NOTE_FONT = 13                # insufficient-fit annotation (pt)
TRANSITION_TIME_SECTION_TITLE_PX = 17        # full-section heading (px)

TRANSITION_TIME_FRAME_LINEWIDTH = 1.0        # 4-sided axes frame (pt)
TRANSITION_TIME_TICK_LINEWIDTH = 1.0         # tick strokes (pt)
TRANSITION_TIME_TICK_LENGTH = 3.5            # tick length (pt)
TRANSITION_TIME_FIT_LINEWIDTH = 2.0          # fitted probability curve (pt)
TRANSITION_TIME_BAR_EDGE_LINEWIDTH = 0.5    # histogram bar outlines (pt)

# Transition-time figure legends. All dimensions are Matplotlib points unless noted.
# Only the appearance of the two transition-time legends is changed.
TRANSITION_LEGEND_LINE_LENGTH_PT = 18.0      # Colored line before "Fitted ... PDF"
TRANSITION_LEGEND_LINE_TEXT_GAP_PT = 4.0     # Line-to-text spacing
TRANSITION_LEGEND_ROW_GAP_PT = -0.6          # Tight vertical gap between Mean and COV (pt)
TRANSITION_LEGEND_TITLE_TO_MEAN_GAP_PT = 6.0  # Extra gap after Fitted ... PDF, before Mean (pt)
TRANSITION_LEGEND_CORNER_INSET = 0.04     # Distance from the top/corner in axes fraction


def apply_transition_time_font_sizes(ax):
    """Override global academic typography only on the transition-time panel."""
    ax.xaxis.label.set_fontsize(TRANSITION_TIME_AXIS_LABEL_FONT)
    ax.yaxis.label.set_fontsize(TRANSITION_TIME_AXIS_LABEL_FONT)
    ax.tick_params(axis="both", labelsize=TRANSITION_TIME_TICK_FONT)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(TRANSITION_TIME_TICK_FONT)
    ax.title.set_fontsize(TRANSITION_TIME_SUBPLOT_TITLE_FONT)
    legend = ax.get_legend()
    if legend is not None:
        for legend_text in legend.get_texts():
            legend_text.set_fontsize(TRANSITION_TIME_LEGEND_FONT)


def get_transition_fitted_curve(histogram_rows, selected_fit):
    """Independent 5-year density LS + reference PCHIP, shared by plot/Excel."""
    if selected_fit is None or histogram_rows.empty:
        return None, None
    return _five_year_average_fitted_curve(selected_fit)


def draw_compact_transition_legend(ax, fit_label, line_color, target, *, dashboard=False):
    """Three-line legend with separate title-to-Mean and Mean-to-COV spacing.

    The short colored line aligns with the fitted distribution name.
    FFF is upper-right; CSF is upper-left.
    """
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker

    rows = str(fit_label).split("\n")
    if len(rows) != 3:
        raise ValueError("Transition legend must contain title, sample Mean and sample COV.")
    font_size = (DASHBOARD_PLOT_LEGEND_FONT_PT if dashboard
                 else TRANSITION_TIME_LEGEND_FONT)
    handle_length = TRANSITION_LEGEND_LINE_LENGTH_PT
    handle_gap = TRANSITION_LEGEND_LINE_TEXT_GAP_PT
    row_height = font_size * 1.20
    packed_rows = []

    for i, text in enumerate(rows):
        # Identical left-hand space on all three lines aligns the text baselines.
        swatch = DrawingArea(handle_length, row_height, 0, 0)
        if i == 0:
            swatch.add_artist(Line2D([0, handle_length], [row_height * 0.50] * 2,
                                     color=line_color,
                                     linewidth=TRANSITION_TIME_FIT_LINEWIDTH))
        text_area = TextArea(text, textprops={
            "size": font_size, "color": "black", "fontproperties": GLOBAL_FONT_PROP,
        })
        packed_rows.append(HPacker(children=[swatch, text_area], align="center",
                                   pad=0.0, sep=handle_gap))

    # Nest two statistic rows separately so the title-to-Mean spacing can be
    # increased without also increasing the tighter Mean-to-COV spacing.
    sample_stat_rows = VPacker(children=packed_rows[1:], align="left", pad=0.0,
                               sep=TRANSITION_LEGEND_ROW_GAP_PT)
    legend_contents = VPacker(children=[packed_rows[0], sample_stat_rows],
                              align="left", pad=0.0,
                              sep=TRANSITION_LEGEND_TITLE_TO_MEAN_GAP_PT)
    inset = TRANSITION_LEGEND_CORNER_INSET
    on_right = target == "To_FFF"
    anchored = AnchoredOffsetbox(
        loc="upper right" if on_right else "upper left", child=legend_contents,
        pad=0.0, borderpad=0.0, frameon=False,
        bbox_to_anchor=(1.0 - inset if on_right else inset, 1.0 - inset),
        bbox_transform=ax.transAxes,
    )
    anchored.set_zorder(12)
    ax.add_artist(anchored)
    return anchored


def make_transition_histogram_panel(hist_df, selected_fits, target_only=None):
    """1x2 conditional 5-year average densities and grouped fitted curves."""
    fig, axes = (plt.subplots(1, 2, figsize=(10.8, 4.2), dpi=220)
                 if target_only is None
                 else plt.subplots(figsize=(4.3, 3.15), dpi=190))
    plot_cfg = [
        ("To_FFF", "Transition to FFF", "#33A1BC", "#24798E"),
        ("To_CSF", "Transition to CSF", "#EE95A3", "#C96F80"),
    ]
    if target_only is not None:
        if target_only not in ("To_FFF", "To_CSF"):
            raise ValueError(f"Unknown transition target: {target_only}")
        plot_cfg = [item for item in plot_cfg if item[0] == target_only]
        axes = [axes]
    for ax, (target, title, color, line_color) in zip(axes, plot_cfg):
        g = hist_df.loc[hist_df["Target_Group"] == target].sort_values("Bin_Start")
        # A 5-year bin probability is dimensionless; divide by its bin width
        # to obtain the displayed density (1/year). This yields total
        # histogram area = 1 for observed first transitions to this target.
        bin_density = (
            g["Probability_within_group"].to_numpy(dtype=float)
            / (g["Bin_End"].to_numpy(dtype=float)
               - g["Bin_Start"].to_numpy(dtype=float))
        ) if not g.empty else np.empty(0, dtype=float)
        if not g.empty:
            ax.bar(g["Bin_Center"], bin_density,
                   width=(g["Bin_End"] - g["Bin_Start"]).to_numpy(dtype=float),
                   color=color,
                   edgecolor="black", linewidth=TRANSITION_TIME_BAR_EDGE_LINEWIDTH,
                   alpha=0.82, zorder=2)
        selected = selected_fits.get(target)
        max_fit = 0.0
        fit_label = "Best fit"
        if selected is not None and not g.empty:
            curve_x, curve_y = get_transition_fitted_curve(g, selected)
            if curve_x is not None:
                ax.plot(curve_x, curve_y, color=line_color,
                        linewidth=TRANSITION_TIME_FIT_LINEWIDTH, zorder=4)
                max_fit = float(curve_y.max())
            fit_label = format_fitted_legend(selected)
        else:
            ax.text(.5, .86, "Insufficient data for fitting", ha="center",
                    va="center", transform=ax.transAxes,
                    fontproperties=GLOBAL_FONT_PROP,
                    fontsize=TRANSITION_TIME_NOTE_FONT)
        if selected is not None:
            draw_compact_transition_legend(ax, fit_label, line_color, target,
                                           dashboard=(target_only is not None))
        ymax = max(float(np.max(bin_density)) if bin_density.size else 0., max_fit)
        # Densities are not probabilities: keep the original upward-rounded
        # y-axis ticks, but do not impose an artificial maximum of 1.0.
        set_dynamic_y_axis(ax, ymax, probability=False)
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 20))
        set_axis_labels(ax, r"Transition time, $t_{tran}$ (years)", "Probability density")
        ax.set_title(title, fontproperties=GLOBAL_FONT_PROP,
                     fontsize=FIGURE_TITLE_FONT_SIZE, pad=6)
        ax.grid(False)
        apply_academic_style(ax)
        style_thin_black_axes(
            ax,
            frame_linewidth=TRANSITION_TIME_FRAME_LINEWIDTH,
            tick_linewidth=TRANSITION_TIME_TICK_LINEWIDTH,
            tick_length=TRANSITION_TIME_TICK_LENGTH,
        )
        apply_transition_time_font_sizes(ax)
    if target_only is None:
        fig.subplots_adjust(left=.10, right=.985, bottom=.19, top=.89, wspace=.32)
    else:
        style_dashboard_plot(fig, kind="histogram")
    return fig


def compute_all_sample_transition_cdf(transition_df, years=YEARS_FULL,
                                      threshold_fff=0.20, threshold_csf=0.10):
    """Cumulative probability of a first reliable transition / ALL samples."""
    n_total = len(transition_df)
    if n_total < 1:
        raise ValueError("No transition samples")
    years = np.asarray(years, dtype=int)
    flagged = transition_df.loc[transition_df["Transition_Flag"]]
    result = pd.DataFrame({"Year": years})
    summaries = []
    for target, internal, threshold in [
        ("FFF", "FFF", float(threshold_fff)),
        ("CSF", "PSF", float(threshold_csf)),
    ]:
        mask = flagged["Transition_Mode"] == internal
        vals = flagged.loc[mask, "Transition_Year"].to_numpy(dtype=int)
        counts = np.bincount(vals, minlength=len(years))[:len(years)]
        cum_counts = np.cumsum(counts)
        probs = cum_counts / n_total
        result[f"{target}_New_Transition_Count"] = counts
        result[f"{target}_Cumulative_Count"] = cum_counts
        result[f"{target}_Cumulative_Probability_AllSamples"] = probs
        first = np.flatnonzero(probs >= threshold - 1e-12)
        first_yr = int(years[first[0]]) if first.size else np.nan
        summaries.append({"Target_Mode": target, "Threshold_Probability": threshold,
                          "Threshold_Percent": threshold * 100,
                          "First_Reaching_Year": first_yr,
                          "Probability_At_First_Reaching_Year":
                            float(probs[first[0]]) if first.size else np.nan,
                          "Cumulative_Probability_At_100yr": float(probs[-1]),
                          "Reached_Within_100yr": bool(first.size)})
    return result, pd.DataFrame(summaries)


# ============================================================
# CUMULATIVE SFM TRANSITION PROBABILITY CURVES — independent controls
# All Matplotlib font sizes/line widths are in points (pt).
# The Streamlit section heading uses CSS pixels (px).
# Changing these affects ONLY this 1x2 cumulative-probability figure.
# ============================================================
CUMULATIVE_CDF_AXIS_LABEL_FONT = 16          # x/y axis label font size (pt)
CUMULATIVE_CDF_TICK_FONT = 16                # tick label font size (pt)
CUMULATIVE_CDF_SUBPLOT_TITLE_FONT = 17       # "Transition to FFF/CSF" (pt)
CUMULATIVE_CDF_ANNOTATION_FONT = 16          # threshold-crossing year (pt)
CUMULATIVE_CDF_SECTION_TITLE_PX = 17        # full-section heading (px)

CUMULATIVE_CDF_FRAME_LINEWIDTH = 1.0         # 4-sided axes frame (pt)
CUMULATIVE_CDF_TICK_LINEWIDTH = 1.0          # tick strokes (pt)
CUMULATIVE_CDF_TICK_LENGTH = 3.5             # tick length (pt)
CUMULATIVE_CDF_CURVE_LINEWIDTH = 2.0         # cumulative step curve (pt)
CUMULATIVE_CDF_GUIDE_LINEWIDTH = 0.8         # threshold/vertical guides (pt)
CUMULATIVE_CDF_MARKER_EDGE_LINEWIDTH = 1.0   # threshold point ring (pt)
CUMULATIVE_CDF_MARKER_SIZE = 4.8             # threshold point diameter (pt)


def apply_cumulative_cdf_font_sizes(ax):
    """Override fonts exclusively on the cumulative-transition CDF panels."""
    ax.xaxis.label.set_fontsize(CUMULATIVE_CDF_AXIS_LABEL_FONT)
    ax.yaxis.label.set_fontsize(CUMULATIVE_CDF_AXIS_LABEL_FONT)
    ax.tick_params(axis="both", labelsize=CUMULATIVE_CDF_TICK_FONT)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(CUMULATIVE_CDF_TICK_FONT)
    ax.title.set_fontsize(CUMULATIVE_CDF_SUBPLOT_TITLE_FONT)


def make_all_sample_cdf_panel(cdf_df, threshold_df, target_only=None):
    """All-sample CDFs: no legend, consistent dashed lines, year at threshold."""
    fig, axes = (plt.subplots(1, 2, figsize=(9.3, 4.0), dpi=220)
                 if target_only is None
                 else plt.subplots(figsize=(4.3, 3.15), dpi=190))
    years = cdf_df["Year"].to_numpy(dtype=int)
    dashed = dict(color="#6F6F6F", linestyles="--",
                  linewidth=CUMULATIVE_CDF_GUIDE_LINEWIDTH, zorder=2)

    target_config = [("FFF", "#33A1BC"), ("CSF", "#EE95A3")]
    if target_only is not None:
        if target_only not in ("FFF", "CSF"):
            raise ValueError(f"Unknown CDF target: {target_only}")
        target_config = [item for item in target_config if item[0] == target_only]
        axes = [axes]
    for ax, (target, color) in zip(axes, target_config):
        tr = threshold_df.loc[threshold_df["Target_Mode"] == target].iloc[0]
        threshold = float(tr["Threshold_Probability"])
        prob = cdf_df[f"{target}_Cumulative_Probability_AllSamples"].to_numpy()
        ax.step(years, prob, where="post", color=color,
                linewidth=CUMULATIVE_CDF_CURVE_LINEWIDTH, zorder=3)
        ax.axhline(threshold, color=dashed["color"], linestyle="--",
                   linewidth=dashed["linewidth"], zorder=2)
        upper_data = max(float(np.max(prob)), threshold)
        set_dynamic_y_axis(ax, upper_data, probability=True)
        # CDF panels only: original "nice" upper limit, plus 0.10, capped at 1.
        previous_upper = float(ax.get_ylim()[1])
        cdf_upper = min(1.0, round(previous_upper + 0.10, 10))
        ax.set_ylim(0.0, cdf_upper)
        prior_ticks = np.asarray(ax.get_yticks(), dtype=float)
        step = float(prior_ticks[1] - prior_ticks[0]) if prior_ticks.size > 1 else 0.1
        if step > 0:
            added_ticks = np.arange(previous_upper + step, cdf_upper + step * 0.25, step)
            ticks = np.r_[prior_ticks[prior_ticks <= cdf_upper + 1e-9], added_ticks]
            if not np.isclose(ticks, cdf_upper, atol=1e-8).any():
                ticks = np.r_[ticks, cdf_upper]
            ax.set_yticks(np.unique(np.round(ticks[ticks <= cdf_upper + 1e-9], 10)))

        if bool(tr["Reached_Within_100yr"]):
            first_yr = int(tr["First_Reaching_Year"])
            # For a post-step CDF, the first reaching year is the intersection
            # x-coordinate, and threshold (not the step top) is its y-coordinate.
            ax.plot(first_yr, threshold, marker="o",
                    markersize=CUMULATIVE_CDF_MARKER_SIZE,
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=CUMULATIVE_CDF_MARKER_EDGE_LINEWIDTH,
                    zorder=5)
            ax.vlines(first_yr, 0.0, threshold, **dashed)
            ax.annotate(f"{first_yr} years", xy=(first_yr, threshold),
                        xytext=(8, -9), textcoords="offset points",
                        ha="left", va="top",
                        fontsize=CUMULATIVE_CDF_ANNOTATION_FONT,
                        fontproperties=GLOBAL_FONT_PROP,
                        color="black", zorder=6,
                        annotation_clip=False)
            # Keep the lower-right annotation inside the plot for late crossings.
            ax.set_xlim(0, max(100, first_yr + 22) if first_yr >= 87 else 100)
        else:
            # There is no mathematical intersection; the message is printed
            # below this panel by the GUI rather than placed on the chart.
            ax.set_xlim(0, 100)

        set_axis_labels(ax, r"Service time, $t$ (years)",
                        "Cumulative probability")
        ax.set_title(f"Transition to {target}",
                     fontproperties=GLOBAL_FONT_PROP,
                     fontsize=FIGURE_TITLE_FONT_SIZE, pad=6)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.grid(False)
        apply_academic_style(ax)
        style_thin_black_axes(
            ax,
            frame_linewidth=CUMULATIVE_CDF_FRAME_LINEWIDTH,
            tick_linewidth=CUMULATIVE_CDF_TICK_LINEWIDTH,
            tick_length=CUMULATIVE_CDF_TICK_LENGTH,
        )
        apply_cumulative_cdf_font_sizes(ax)
    if target_only is None:
        fig.subplots_adjust(left=.12, right=.985, bottom=.21, top=.88, wspace=.34)
    else:
        style_dashboard_plot(fig, kind="cdf")
    return fig


# ============================================================
# 9c. 4x2 dashboard plot presentation, independent from model calculations
# ============================================================
DASHBOARD_PLOT_WIDTH_IN = 4.25
DASHBOARD_PLOT_HEIGHT_IN = 3.10
DASHBOARD_PLOT_AXIS_FONT_PT = 11.0
DASHBOARD_PLOT_TICK_FONT_PT = 9.0
DASHBOARD_PLOT_TITLE_FONT_PT = 11.0
DASHBOARD_PLOT_LEGEND_FONT_PT = 8.7
DASHBOARD_PLOT_DPI = 170


def style_dashboard_plot(fig, kind="standard"):
    """Fit an existing Matplotlib figure legibly into one of four equal cells."""
    fig.set_size_inches(DASHBOARD_PLOT_WIDTH_IN, DASHBOARD_PLOT_HEIGHT_IN)
    for ax in fig.axes:
        ax.xaxis.label.set_fontsize(DASHBOARD_PLOT_AXIS_FONT_PT)
        ax.yaxis.label.set_fontsize(DASHBOARD_PLOT_AXIS_FONT_PT)
        ax.title.set_fontsize(DASHBOARD_PLOT_TITLE_FONT_PT)
        ax.tick_params(axis="both", labelsize=DASHBOARD_PLOT_TICK_FONT_PT)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontsize(DASHBOARD_PLOT_TICK_FONT_PT)
        legend = ax.get_legend()
        if legend is not None:
            for t in legend.get_texts():
                t.set_fontsize(DASHBOARD_PLOT_LEGEND_FONT_PT)
        # Threshold year labels / insufficient-fit notes are Text artists,
        # rather than tick labels or ordinary legends.
        for note in ax.texts:
            note.set_fontsize(DASHBOARD_PLOT_TICK_FONT_PT)
    if kind == "types":
        # The previous bottom=.43 reserved 43% of the PNG beneath the bars,
        # producing excessive blank space. Keep the full x-axis title and all
        # rotated category labels, but use most of that space for the bar chart.
        fig.subplots_adjust(left=.19, right=.955, top=.96,
                            bottom=TRANSITION_TYPES_BOTTOM_FRAC)
    elif kind == "histogram":
        fig.subplots_adjust(left=.19, right=.955, top=.90, bottom=.20)
    elif kind == "cdf":
        fig.subplots_adjust(left=.19, right=.955, top=.90, bottom=.20)
    else:
        fig.subplots_adjust(left=.19, right=.955, top=.96, bottom=.20)
    return fig


def dashboard_figure_png(fig):
    """Store only compact plot bytes, not the large MC state, across UI reruns."""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DASHBOARD_PLOT_DPI,
                    facecolor="white", edgecolor="white")
        return buf.getvalue()
    finally:
        plt.close(fig)


DASHBOARD_CHART_TITLES = {
    "corrosion": "Corrosion level evolution",
    "failure": "Life-cycle SFM probability curves",
    "hist_fff": "The distributions of SFM transition times",
    "hist_csf": "The distributions of SFM transition times",
    "scour": "Scour depth evolution",
    "types": "The distribution of SFM transition types",
    "cdf_fff": "Cumulative SFM transition probability curves",
    "cdf_csf": "Cumulative SFM transition probability curves",
}


def render_dashboard_results(results):
    """Eight charts in precisely two aligned rows of four, with top captions."""
    with st.container(key="dashboard_plots"):
        top = st.columns(4, gap="small")
        for col, key in zip(top, ("corrosion", "failure", "hist_fff", "hist_csf")):
            with col:
                st.markdown(
                    f"<div class='dashboard-chart-title'>{DASHBOARD_CHART_TITLES[key]}</div>",
                    unsafe_allow_html=True,
                )
                st.image(results["plots"][key], width="stretch")

        bottom = st.columns(4, gap="small")
        for col, key in zip(bottom, ("scour", "types", "cdf_fff", "cdf_csf")):
            with col:
                st.markdown(
                    f"<div class='dashboard-chart-title'>{DASHBOARD_CHART_TITLES[key]}</div>",
                    unsafe_allow_html=True,
                )
                st.image(results["plots"][key], width="stretch")

    # Original output texts, preserved below all eight charts.
    st.markdown("<div class='section-header'>Simulation results</div>",
                unsafe_allow_html=True)
    note_columns = st.columns([1.15, 1, 1.2], gap="medium")
    with note_columns[0]:
        st.markdown("<div class='dashboard-notes'>" +
                    results["crossover_html"] + "</div>",
                    unsafe_allow_html=True)
    with note_columns[1]:
        st.markdown("<div class='dashboard-notes'>" +
                    results["transition_summary_html"] + "</div>",
                    unsafe_allow_html=True)
    with note_columns[2]:
        st.markdown("<div class='dashboard-notes'>" +
                    results["cdf_years_html"] + "</div>",
                    unsafe_allow_html=True)



# ============================================================
# 9b. Excel export: EIGHT independent worksheets, ONE per visible chart.
#     Worksheet order strictly matches the Streamlit 2 x 4 chart arrangement.
#     Plotted values are exported directly, with no sample-level MC database.
# ============================================================
def get_scour_figure_data(years, samples_dict, sd_cov):
    """Theoretical mean and mean +/- Std actually shown in the scour chart."""
    x = np.asarray(years, dtype=float)
    def first(key):
        return float(np.asarray(samples_dict[key], dtype=float)[0])
    mean = first("lambda_sd") * first("B_val") * (
        first("p_val") * (1.0 - np.exp(-first("q_val") * x))
        + first("r_val") * (1.0 - np.exp(-first("s_val") * x)))
    std = float(sd_cov) * mean
    return pd.DataFrame({
        "Year": x.astype(int),
        "Scour_Mean_m": mean,
        "Scour_Std_m": std,
        "Scour_MeanMinus1Std_m": np.clip(mean - std, 0.0, 8.0),
        "Scour_MeanPlus1Std_m": np.clip(mean + std, 0.0, 8.0),
    })


def get_corrosion_figure_data(years, corr_long, corr_stir):
    """The plotted mean and population +/-1 Std curves for both bar classes."""
    data = {"Year": np.asarray(years, dtype=int)}
    for name, matrix in [("Transverse", corr_stir), ("Longitudinal", corr_long)]:
        mu = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0, ddof=0)
        data[f"{name}_Mean"] = mu
        data[f"{name}_Std"] = std
        data[f"{name}_MeanMinus1Std"] = np.clip(mu - std, 0.0, 1.0)
        data[f"{name}_MeanPlus1Std"] = np.clip(mu + std, 0.0, 1.0)
    return pd.DataFrame(data)


def get_transition_type_figure_data(transition_df):
    """Bar chart categories in exactly the chart's displayed order."""
    vc = transition_df["Transition_Type"].value_counts()
    labels = [str(k).replace("PFF", "CFF").replace("PSF", "CSF")
              .replace("->", " to ").replace("No transition within 100 years", "No transition")
              for k in vc.index]
    return pd.DataFrame({
        "Transition_Type": labels,
        "Count": vc.to_numpy(dtype=int),
        "Probability_AllSamples": vc.to_numpy(dtype=float) / max(1, len(transition_df)),
    })


def get_transition_time_target_figure_data(hist_df, selected_fits, target,
                                           transition_df=None):
    """Export ONE histogram panel to its own sheet (FFF or CSF).

    Curve_Time_Year/Fitted_5Year_Average_Density_PerYear: the exact smooth
    fitted 5-year bin-average curve shown in the plot (1001 rows).
    Bin_*: the 20 five-year histogram bins actually used for the plotted bars.
    These occupy the first 20 rows, with remaining bin cells blank.
    Empirical sample statistics and fitted parameters appear in row 1 metadata.
    Bin_Probability_WithinTarget remains the original 5-year-bin probability;
    Bin_Probability_Density_PerYear is the ACTUAL bar height shown in the GUI.
    Keeping both allows users to verify bin probability = density * bin width.
    Fit_5Year_Bin_Probability and Fit_5Year_Average_Density_PerYear are
    derived from a SEPARATE fit to 5-y bin-average observed densities,
    following the provided independent-density reference. The sample moments
    are computed from the original observed transition years, not that fit.
    """
    if target not in ("To_FFF", "To_CSF"):
        raise ValueError(f"Unexpected transition target: {target}")
    g = hist_df.loc[hist_df["Target_Group"] == target].sort_values("Bin_Start")
    selected = selected_fits.get(target)
    if transition_df is not None:
        # Always use the exact observed first-transition years, including when
        # the sample is too small to fit a parametric probability density.
        sample_stats = get_observed_transition_sample_moments(
            get_target_observed_transition_years(transition_df, target))
    elif selected is not None and "sample_moments" in selected:
        sample_stats = selected["sample_moments"]
    else:
        sample_stats = {"count": 0, "mean": np.nan, "std": np.nan, "cov": np.nan}
    curve_x, curve_y = get_transition_fitted_curve(g, selected)
    # Retain the full horizontal coordinates even if no curve can be fitted,
    # in which case the figure shows only the histogram and an annotation.
    if curve_x is None:
        curve_x = np.linspace(0.0, 100.0, 1001)
        curve_y = np.full(curve_x.shape, np.nan, dtype=float)

    output = pd.DataFrame({
        "Curve_Time_Year": np.asarray(curve_x, dtype=float),
        "Fitted_5Year_Average_Density_PerYear": np.asarray(curve_y, dtype=float),
    })
    for exported_name, input_name in [
        ("Bin_Start_Year", "Bin_Start"),
        ("Bin_End_Year", "Bin_End"),
        ("Bin_Center_Year", "Bin_Center"),
        ("Bin_Count", "Count"),
        ("Bin_Probability_WithinTarget", "Probability_within_group"),
        ("Bin_Probability_AllSamples", "Probability_in_all_samples"),
        ("Observed_Target_Count", "N_group"),
    ]:
        arr = np.full(len(output), np.nan, dtype=float)
        vals = g[input_name].to_numpy(dtype=float)
        arr[:len(vals)] = vals
        output[exported_name] = arr

    output["Bin_Probability_Density_PerYear"] = (
        output["Bin_Probability_WithinTarget"]
        / (output["Bin_End_Year"] - output["Bin_Start_Year"])
    )
    fit_centers, fit_densities, fit_masses = get_fitted_five_year_bin_density(selected)
    output["Fit_5Year_Bin_Probability"] = np.nan
    output["Fit_5Year_Average_Density_PerYear"] = np.nan
    if fit_centers is not None:
        n_bins = len(fit_centers)
        output.loc[:n_bins - 1, "Fit_5Year_Bin_Probability"] = fit_masses
        output.loc[:n_bins - 1, "Fit_5Year_Average_Density_PerYear"] = fit_densities
    output.loc[0, "Target_Mode"] = target.removeprefix("To_")
    output.loc[0, "Histogram_Bin_Width_Year"] = float(TRANSITION_HIST_BIN_WIDTH)
    output.loc[0, "Fit_Method"] = "Independent least squares of 5-year mean densities"
    output.loc[0, "Curve_Method"] = "Reference PCHIP: fitted bin centers, endpoints (0,0) and (100,0)"
    output.loc[0, "Legend_Statistics_Method"] = (
        "Observed first-transition time samples for this target within 0-100 years; "
        "Mean=sample average, Std=np.std(ddof=0), COV=Std/Mean. "
        "Fitted distribution parameters still describe the fitted density curve."
    )
    output.loc[0, "Sample_Count"] = int(sample_stats["count"])
    output.loc[0, "Sample_Mean_Year"] = sample_stats["mean"]
    output.loc[0, "Sample_Std_Year"] = sample_stats["std"]
    output.loc[0, "Sample_COV"] = sample_stats["cov"]
    if selected is None:
        output.loc[0, "BestFit_Distribution"] = "Not fitted (insufficient observations)"
    else:
        dfit = selected.get("density_fit")
        if dfit is None:
            raise ValueError(f"{target}: missing density fit for Excel export")
        output.loc[0, "BestFit_Distribution"] = dfit["name"]
        output.loc[0, "BestFit_Parameters"] = "; ".join(
            f"{name}={value:.9g}" for name, value in dfit["params"].items()
        )
        # Use only the density-LS parameters, including the exact lambda/zeta
        # parameterization shown in the GUI for a fitted lognormal model.
        for key, value in dfit["params"].items():
            output.loc[0, f"BestFit_Param_{key}"] = float(value)
        if dfit["name"] == "Lognormal":
            output.loc[0, "BestFit_Param_lambda"] = float(np.log(dfit["params"]["scale"]))
            output.loc[0, "BestFit_Param_zeta"] = float(dfit["params"]["sigma_ln"])
        output.loc[0, "BestFit_Legend_Text"] = format_fitted_legend(selected)
        output.loc[0, "BestFit_Fit_Method"] = dfit["fit_method"]
        output.loc[0, "Independent_DensityFit_Parameters"] = output.loc[0, "BestFit_Parameters"]
        output.loc[0, "Independent_DensityFit_RMSE_PerYear"] = float(dfit["density_rmse"])
        output.loc[0, "Independent_DensityFit_NormalizedSSE"] = float(dfit["normalized_sse"])
    return output


def get_cdf_target_figure_data(cumulative_df, crossing_df, target):
    """Export ONE cumulative transition probability panel, FFF or CSF.

    Cumulative probability is relative to all MC samples (not conditioned
    on observed transitions). Threshold_Probability is constant for drawing
    the dashed horizontal line. Reaching-year metadata and marker coordinates
    are in the first row only; if unreached they remain blank.
    """
    if target not in ("FFF", "CSF"):
        raise ValueError(f"Unexpected cumulative target: {target}")
    tr = crossing_df.loc[crossing_df["Target_Mode"] == target].iloc[0]
    threshold = float(tr["Threshold_Probability"])
    output = pd.DataFrame({
        "Year": cumulative_df["Year"].to_numpy(dtype=int),
        "New_Transition_Count": cumulative_df[f"{target}_New_Transition_Count"].to_numpy(dtype=int),
        "Cumulative_Transition_Count": cumulative_df[f"{target}_Cumulative_Count"].to_numpy(dtype=int),
        "Cumulative_Probability_AllSamples": cumulative_df[
            f"{target}_Cumulative_Probability_AllSamples"].to_numpy(dtype=float),
        "Threshold_Probability": threshold,
    })
    output.loc[0, "Target_Mode"] = target
    output.loc[0, "Reached_Within_100yr"] = bool(tr["Reached_Within_100yr"])
    output.loc[0, "Cumulative_Probability_At_100yr"] = float(
        tr["Cumulative_Probability_At_100yr"])
    if bool(tr["Reached_Within_100yr"]):
        yr = int(tr["First_Reaching_Year"])
        output.loc[0, "First_Reaching_Year"] = yr
        output.loc[0, "Probability_At_First_Reaching_Year"] = float(
            tr["Probability_At_First_Reaching_Year"])
        # The visible circle is drawn at (year, threshold), not at step top.
        output.loc[0, "Plot_Marker_Year"] = yr
        output.loc[0, "Plot_Marker_Probability"] = threshold
    return output


def build_eight_figure_export_tables(result, sd_cov, selected_fits,
                                     cumulative_df, crossing_df):
    """Eight Excel sheets in top-row then bottom-row chart order."""
    annual = pd.DataFrame({"Year": YEARS_FULL})
    annual_curves = {
        display_label(name): result["annual_probs"][:, idx]
        for idx, name in enumerate(result["label_names"])
    }
    for label in ("CFF", "FFF", "CSF"):
        if label in annual_curves:
            annual[f"Probability_{label}"] = annual_curves[label]

    # Ordered insertion is intentional and matches render_dashboard_results:
    # row 1: corrosion | SFM probs | transition to FFF | transition to CSF
    # row 2: scour     | transition types | CDF to FFF | CDF to CSF
    return {
        "01_Corrosion_Level": get_corrosion_figure_data(
            YEARS_FULL, result["corr_long"], result["corr_stir"]),
        "02_SFM_Probabilities": annual,
        "03_TimeDist_to_FFF": get_transition_time_target_figure_data(
            result["hist_df"], selected_fits, "To_FFF", result["transition_df"]),
        "04_TimeDist_to_CSF": get_transition_time_target_figure_data(
            result["hist_df"], selected_fits, "To_CSF", result["transition_df"]),
        "05_Scour_Depth": get_scour_figure_data(
            YEARS_FULL, result["samples_dict"], sd_cov),
        "06_Transition_Types": get_transition_type_figure_data(result["transition_df"]),
        "07_Cumulative_to_FFF": get_cdf_target_figure_data(
            cumulative_df, crossing_df, "FFF"),
        "08_Cumulative_to_CSF": get_cdf_target_figure_data(
            cumulative_df, crossing_df, "CSF"),
    }


# ============================================================
# 9d. Supplementary formulas, schematic, and input definitions
# ============================================================
def render_calculation_formulas():
    """ONE aligned KaTeX expression: four captions centered over four formulas.

    Adjustable horizontal equation spacing, header padding, and caption-to-math
    distance are available in GUI presentation controls at the top of the file.
    """
    st.markdown(
        "<div class='reference-section-title'>Related Calculation Formulas</div>",
        unsafe_allow_html=True,
    )
    formulas = [
        (
            "Corrosion initiation time",
            r"t_{corr}=X_I\left[\frac{d_c^2}{4k_ek_tk_cD_0(t_0)^n}"
            r"\left[{erf}^{-1}\left(1-\frac{C_{cr}}{C_s}\right)\right]^{-2}"
            r"\right]^{\tfrac{1}{1-n}}",
        ),
        (
            "Surface chloride concentration",
            r"C_s=A_{cs}(w/c)+\varepsilon_{cs}",
        ),
        (
            "Corrosion rate",
            r"i_{corr,0}=\frac{37.8\,\lambda_{corr}(1-w)^{-1.64}}{d_c}",
        ),
        (
            "Mean scour depth",
            r"SD_{mean}(t)=\lambda_{sd}B"
            r"\left\{p\left[1-\exp(-qt)\right]+r\left[1-\exp(-st)\right]\right\}",
        ),
    ]
    # Use ONLY standard KaTeX array column codes "cccc".  The former
    # c@{\hspace{...}}c syntax is not supported consistently by Streamlit's
    # KaTeX renderer and resulted in the full LaTeX being printed in red.
    # Put equivalent adjustable spacing *inside* both heading and formula
    # cells, maintaining exactly the same four column centers for both rows.
    cell_gap = rf"\hspace{{{FORMULA_SPACING_EM}em}}"
    heading_padding = rf"\hspace{{{FORMULA_TITLE_PADDING_EM}em}}"
    caption_cells = []
    equation_cells = []
    for i, (caption, equation) in enumerate(formulas):
        suffix = cell_gap if i < len(formulas) - 1 else ""
        caption_cells.append(
            rf"{heading_padding}\textbf{{{caption}}}{heading_padding}{suffix}"
        )
        equation_cells.append(rf"\displaystyle {equation}{suffix}")

    combined_latex = (
        r"\begin{array}{cccc}"
        + " & ".join(caption_cells)
        + rf"\\[{FORMULA_HEADING_ROW_GAP_EM}em]"
        + " & ".join(equation_cells)
        + r"\end{array}"
    )
    with st.container(key="equation_toolbar"):
        st.latex(combined_latex)


def render_glossary_group(title, configs):
    """Individual rows with a separator, using the INPUT TABLE math symbols."""
    from html import escape

    st.markdown(
        f"<div class='glossary-group-title'>{escape(title)}</div>",
        unsafe_allow_html=True,
    )
    for cfg in configs:
        param_id, description = cfg[0], cfg[2]
        # No artificial line breaks or manually aligned Markdown tables:
        # the entire symbol + meaning is inside one row container.
        with st.container(key=f"glossary_row_{param_id}"):
            symbol_column, description_column = st.columns([1.25, 4.35], gap="small")
            with symbol_column:
                st.markdown("$" + format_parameter_math(param_id) + "$")
            with description_column:
                st.markdown(
                    f"<div class='glossary-meaning'>{description if param_id == 'SD_val' else escape(description)}</div>",
                    unsafe_allow_html=True,
                )


def render_schematic_and_descriptions(structure_cfg, scour_cfg, corrosion_cfg):
    """Left: two diagrams. Right: responsive 35-parameter glossary."""
    # Parent st.container gives stable scope for height matching CSS.
    with st.container(key="reference_materials"):
        _render_reference_materials(structure_cfg, scour_cfg, corrosion_cfg)


def _render_reference_materials(structure_cfg, scour_cfg, corrosion_cfg):
    schematic_col, descriptions_col = st.columns([1.05, 2.95], gap="medium")
    with schematic_col:
        with st.container(border=True, key="schematic_reference"):
            st.markdown(
                "<div class='reference-card-title'>Structural Schematic</div>",
                unsafe_allow_html=True,
            )
            structure_path = Path(__file__).parent / "GUI_structure.png"
            if structure_path.exists():
                st.image(str(structure_path), width="stretch")
            else:
                st.caption(
                    "Place GUI_structure.png beside this Python file "
                    "to display the structural schematic."
                )
        with st.container(border=True, key="sfm_reference"):
            st.markdown(
                "<div class='reference-card-title'>Seismic failure modes</div>",
                unsafe_allow_html=True,
            )
            sfm_image_path = Path(__file__).parent / "GUI_SFM.png"
            if sfm_image_path.exists():
                st.image(str(sfm_image_path), width="stretch")
            else:
                st.caption(
                    "Place GUI_SFM.png beside this Python file "
                    "to display the seismic failure modes."
                )
    with descriptions_col:
        with st.container(border=True, key="parameter_glossary"):
            st.markdown(
                "<div class='reference-card-title'>Parameter Descriptions</div>",
                unsafe_allow_html=True,
            )
            # One native KaTeX symbol for each of the 35 input definitions.
            # Balance the 17 structural rows against 7 + 11 right-side rows.
            structural_defs, physical_defs = st.columns([1.0, 1.0], gap="medium")
            with structural_defs:
                render_glossary_group("Structural and soil parameters", structure_cfg)
            with physical_defs:
                render_glossary_group("Scour model parameters", scour_cfg)
                render_glossary_group("Corrosion model parameters", corrosion_cfg)


# ============================================================
# 10. Main lifecycle page
# ============================================================
def render_lifecycle_app(assets=None):
    """Top input dashboard and 2x4 plot matrix, matching GUI.png reference."""
    inject_css()
    if assets is None:
        assets = load_numpy_assets()

    # Explicit, full-height block plus toolbar clearance avoids the cropped
    # native H1 observed at the very top of the user's Streamlit screenshot.
    with st.container(key="dashboard_header"):
        st.markdown(
            "<div class='dashboard-main-title'>"
            "Life-cycle probabilistic seismic failure mode assessment "
            "of coastal bridge bents</div>"
            "<hr class='dashboard-title-divider'>",
            unsafe_allow_html=True,
        )

    # Upper input area: left = structure + other; right = scour + corrosion.
    left_panel, right_panel = st.columns([1, 1], gap="small")
    with left_panel:
        with st.container(border=True, key="structure_card"):
            struct_opts = [
                "Normal",
                "Lognormal",
                "Uniform",
                "Deterministic",
            ]

            part1_config = [
                ("Dc_Dp", "D<sub>c</sub>/D<sub>p</sub>", "Column-to-pile diameter ratio", "1.5~3.0",
                 1.5, 3.0, 2.0, "Normal", 0.10, 0.1, "%.2f", struct_opts),
                ("Hp_Dc", "H<sub>p</sub>/D<sub>c</sub>", "Column aspect ratio", "1~5",
                 1.0, 5.0, 3.0, "Normal", 0.26, 0.1, "%.2f", struct_opts),
                ("alpha", "α", "Column axial load ratio", "0.05~0.25",
                 0.05, 0.25, 0.15, "Normal", 0.12, 0.01, "%.2f", struct_opts),
                ("d_l", "d<sub>l</sub> (m)", "Column longitudinal reinforcement diameter", "0.018~0.032",
                 0.018, 0.032, 0.025, "Normal", 0.10, 0.001, "%.3f", struct_opts),
                ("d_t", "d<sub>s</sub> (m)", "Transverse reinforcement diameter", "0.01~0.02",
                 0.01, 0.02, 0.016, "Normal", 0.10, 0.001, "%.3f", struct_opts),
                ("t", "t<sub>c</sub> (m)", "Column cover concrete thickness", "0.04~0.08",
                 0.04, 0.08, 0.05, "Normal", 0.20, 0.01, "%.2f", struct_opts),
                ("rho_cl", "ρ<sub>c,l</sub>", "Column longitudinal reinforcement ratio", "0.005~0.015",
                 0.005, 0.015, 0.010, "Normal", 0.27, 0.001, "%.3f", struct_opts),
                ("rho_cs", "ρ<sub>c,s</sub>", "Column transverse reinforcement ratio", "0.003~0.013",
                 0.003, 0.013, 0.008, "Normal", 0.42, 0.001, "%.3f", struct_opts),
                ("N", "N", "Number of pile rows along loading direction", "2~4",
                 2.0, 4.0, 3.0, "Uniform", 0.0, 1.0, "%.0f", struct_opts),
                ("Dp", "D<sub>p</sub> (m)", "Pile diameter", "0.6~1.8",
                 0.6, 1.8, 1.2, "Normal", 0.10, 0.1, "%.2f", struct_opts),
                ("rho_pl", "ρ<sub>p,l</sub>", "Pile longitudinal reinforcement ratio", "0.005~0.015",
                 0.005, 0.015, 0.010, "Normal", 0.27, 0.001, "%.3f", struct_opts),
                ("rho_ps", "ρ<sub>p,s</sub>", "Pile transverse reinforcement ratio", "0.003~0.013",
                 0.003, 0.013, 0.008, "Normal", 0.42, 0.001, "%.3f", struct_opts),
                ("S_Dp", "S<sub>p/D<sub>p</sub>", "Pile spacing-to-diameter ratio", "2.5~3.5",
                 2.5, 3.5, 3.0, "Normal", 0.15, 0.1, "%.2f", struct_opts),
                ("fyl", "f<sub>yl</sub> (MPa)", "Longitudinal reinforcement yield strength", "300~500",
                 300.0, 500.0, 400.0, "Lognormal", 0.106, 10.0, "%.0f", struct_opts),
                ("fyt", "f<sub>ys</sub> (MPa)", "Transverse reinforcement yield strength", "250~450",
                 250.0, 450.0, 350.0, "Lognormal", 0.106, 10.0, "%.0f", struct_opts),
                ("fc", "f<sub>c</sub> (MPa)", "Concrete compressive strength", "20~60",
                 20.0, 60.0, 40.0, "Lognormal", 0.20, 1.0, "%.1f", struct_opts),
                ("Dr", "D<sub>r</sub>", "Relative density of sand", "0.35~0.75",
                 0.35, 0.75, 0.55, "Uniform", 0.0, 0.05, "%.2f", struct_opts),
            ]
            user_struct = render_param_section(
                "1. Structural and soil parameters", part1_config,
                use_std=False,
            )
        with st.container(border=True, key="other_card"):
            st.markdown("<div class='section-header'>4. Other parameters</div>",
                        unsafe_allow_html=True)
            o1, o2, o3 = st.columns(3, gap="small")
            with o1:
                mc_samples = st.number_input(
                    "Monte Carlo sample count",
                    min_value=100, max_value=200000,
                    value=N_SAMPLES, step=1000, format="%d",
                    help="Independent MC samples; shared fixed sample quantiles over 0–100 years.",
                )
            with o2:
                threshold_fff_pct = st.number_input(
                    "Transition to FFF threshold (%)",
                    min_value=0.01, max_value=100.0,
                    value=DEFAULT_CDF_THRESHOLD_FFF, step=1.0, format="%.2f",
                )
            with o3:
                threshold_csf_pct = st.number_input(
                    "Transition to CSF threshold (%)",
                    min_value=0.01, max_value=100.0,
                    value=DEFAULT_CDF_THRESHOLD_CSF, step=1.0, format="%.2f",
                )

    with right_panel:
        with st.container(border=True, key="scour_card"):
            scour_opts = [
                "Normal",
                "Deterministic",
            ]
            part3_config = [
                ("SD_val", "SD (m)",
                 "Scour depth",
                 "0~8", 0.0, 8.0, None, "Normal", 0.27, 0.5, "%.3f", ["Normal"]),
                ("lambda_sd", "λ<sub>sd</sub>",
                 "Scour depth adjustment coefficient", "-",
                 None, None, 2.000, "Deterministic", 0.0, 0.1, "%.2f", scour_opts),
                ("B_val", "B (m)", "Base width of the pile foundation", "-",
                 None, None, 2.260, "Deterministic", 0.0, 0.01, "%.3f", scour_opts),
                ("p_val", "p", "Empirical scour parameter p", "-",
                 None, None, 1.093, "Deterministic", 0.0, 0.01, "%.3f", scour_opts),
                ("q_val", "q", "Empirical scour parameter q", "-",
                 None, None, 0.021, "Deterministic", 0.0, 0.01, "%.3f", scour_opts),
                ("r_val", "r", "Empirical scour parameter r", "-",
                 None, None, 0.269, "Deterministic", 0.0, 0.01, "%.3f", scour_opts),
                ("s_val", "s", "Empirical scour parameter s", "-",
                 None, None, 2.135, "Deterministic", 0.0, 0.01, "%.3f", scour_opts),
            ]
            user_scour = render_param_section(
                "2. Scour model parameters", part3_config,
                use_std=False,
            )

        with st.container(border=True, key="corrosion_card"):
            st.markdown("<div class='section-header'>3. Corrosion model parameters</div>",
                        unsafe_allow_html=True)
            # Environmental Zone selector removed by request.
            # Preserve original default (Tidal and Splash) parameters.
            acs_def = (7.758, 1.360)
            ecs_def = (0.000, 1.105)
            ccr_def = (0.900, 0.150)
            corr_opts = [
                "Normal",
                "Lognormal",
                "Beta",
                "Gumbel",
                "Deterministic",
            ]

            part2_config = [
                ("Acs", "A<sub>cs</sub>", "Regression variable for corrosion initiation", ">0",
                 1e-6, None, acs_def[0], "Normal", acs_def[1], 0.1, "%.3f", corr_opts),
                ("ecs", "ε<sub>cs</sub>", "Regression error term for corrosion initiation", "-",
                 None, None, ecs_def[0], "Normal", ecs_def[1], 0.1, "%.3f", corr_opts),
                ("Ccr", "C<sub>cr</sub>", "Critical chloride concentration", ">0",
                 1e-6, None, ccr_def[0], "Normal", ccr_def[1], 0.1, "%.3f", corr_opts),
                ("D0", "D<sub>0</sub>", "Reference chloride diffusion coefficient", ">0",
                 1e-6, None, 473.0, "Normal", 43.20, 1.0, "%.1f", corr_opts),
                ("kc", "k<sub>c</sub>", "Curing factor for chloride diffusion", ">0",
                 1e-6, None, 0.800, "Normal", 0.100, 0.1, "%.3f", corr_opts),
                ("kt", "k<sub>t</sub>", "Test method factor for chloride diffusion", ">0",
                 1e-6, None, 0.850, "Normal", 0.024, 0.01, "%.3f", corr_opts),
                ("ke", "k<sub>e</sub>", "Environmental factor for chloride diffusion", ">0",
                 1e-6, None, 1.000, "Normal", 0.300, 0.1, "%.3f", corr_opts),
                ("n_val", "n", "Age factor for chloride diffusion", "0~1",
                 None, None, 0.250, "Beta", 0.050, 0.01, "%.3f", corr_opts),
                ("X1", "X<sub>I</sub>", "Model uncertainty factor", ">0",
                 None, None, 1.000, "Lognormal", 0.050, 0.01, "%.3f", corr_opts),

                ("R", "R", "Pitting corrosion factor", ">0",
                 1e-6, None, 5.560, "Gumbel", 1.160, 0.1, "%.3f", corr_opts),

                ("lam_corr", "λ<sub>corr</sub>", "Corrosion rate adjustment coefficient", "-",
                 None, None, 2.000, "Deterministic", 0.000, 0.1, "%.2f", corr_opts),
            ]
            user_corr = render_param_section("", part2_config, use_std=True)

    # Main actions: equally positioned between the parameter editor and 2x4 plots.
    action_left, action_right = st.columns([1.0, 1.0], gap="small")
    with action_left:
        _, simulate_center, _ = st.columns([0.9, 1.7, 0.9], gap="small")
        with simulate_center:
            with st.container(key="simulate_action"):
                predict_clicked = st.button(
                    "Simulate", type="primary", width="stretch")
    with action_right:
        _, download_center, _ = st.columns([0.9, 1.7, 0.9], gap="small")
        with download_center:
            with st.container(key="download_action"):
                # Populated after MC finishes, without requiring a rerun.
                download_placeholder = st.empty()

    # All eight graphs are BELOW both input columns, never nested in either one.
    if predict_clicked:
        st.session_state.pop("lifecycle_dashboard_display", None)
        st.session_state.pop("lifecycle_excel_data", None)
        if assets is None:
            st.error("model_assets_numpy.pkl was not found. "
                     "Place it in the same directory as this Python file.")
        else:
            with st.spinner("Running Monte Carlo lifecycle analysis and generating plots..."):
                all_inputs = user_struct + user_corr + user_scour
                sd_cov = float(next(p for p in all_inputs
                                    if p["id"] == "SD_val")["raw_disp"])
                result = run_lifecycle_prediction(
                    assets=assets,
                    all_inputs=all_inputs,
                    sd_cov=sd_cov,
                    n_samples=int(mc_samples),
                    seed=RANDOM_SEED,
                )
                tdf = result["transition_df"]
                selected_fits, fit_table, fit_summary = compute_transition_fits(
                    tdf, result["hist_df"])
                cumulative_df, crossing_df = compute_all_sample_transition_cdf(
                    tdf, YEARS_FULL,
                    threshold_fff=float(threshold_fff_pct) / 100.0,
                    threshold_csf=float(threshold_csf_pct) / 100.0,
                )

                # Preserve numeric output & all eight existing plot contents.
                plot_builders = {
                    "corrosion": lambda: make_corrosion_plot(
                        YEARS_FULL, result["corr_long"], result["corr_stir"]),
                    "failure": lambda: make_failure_probability_plot(
                        YEARS_FULL, result["annual_probs"], result["label_names"]),
                    "hist_fff": lambda: make_transition_histogram_panel(
                        result["hist_df"], selected_fits, target_only="To_FFF"),
                    "hist_csf": lambda: make_transition_histogram_panel(
                        result["hist_df"], selected_fits, target_only="To_CSF"),
                    "scour": lambda: make_scour_plot(
                        YEARS_FULL, result["samples_dict"], sd_cov),
                    "types": lambda: make_transition_type_plot(tdf),
                    "cdf_fff": lambda: make_all_sample_cdf_panel(
                        cumulative_df, crossing_df, target_only="FFF"),
                    "cdf_csf": lambda: make_all_sample_cdf_panel(
                        cumulative_df, crossing_df, target_only="CSF"),
                }
                plots = {}
                for name, builder in plot_builders.items():
                    fig = builder()
                    if name in ("corrosion", "failure", "scour", "types"):
                        style_dashboard_plot(fig,
                            kind="types" if name == "types" else "standard")
                    plots[name] = dashboard_figure_png(fig)

                # Existing annual-mode crossover text (same logic and order).
                curve_by_display = {
                    display_label(name): result["annual_probs"][:, i]
                    for i, name in enumerate(result["label_names"])
                }
                crossing_items = []
                for label_a, label_b in (("FFF", "CFF"), ("FFF", "CSF"),
                                         ("CFF", "CSF")):
                    if label_a in curve_by_display and label_b in curve_by_display:
                        crossing_items.extend(find_all_crossovers(
                            YEARS_FULL, curve_by_display[label_a],
                            curve_by_display[label_b], label_a, label_b))
                crossing_items.sort(key=lambda row: row[0])
                if crossing_items:
                    crossover_html = (
                        "<b>Life-cycle SFM probability curves crossover time (years):</b>"
                        "<ul style='margin:4px 0;padding-left:18px'>" +
                        "".join(f"<li>{yr:.2f} ({kind})</li>"
                                for yr, kind in crossing_items) + "</ul>"
                    )
                else:
                    crossover_html = (
                        "<b>Life-cycle SFM probability curves crossover time "
                        "(years):</b> None"
                    )
                observed_ratio = float(tdf["Transition_Flag"].mean())
                transition_summary_html = (
                    f"<b>Observed transition by 100 years:</b> {observed_ratio:.1%}"
                    f"<div><b>No reliable transition by 100 years:</b> "
                    f"{1.0 - observed_ratio:.1%}</div>"
                )
                crossing_rows = crossing_df.set_index("Target_Mode")
                def threshold_year_text(target):
                    row = crossing_rows.loc[target]
                    return (f"{int(row['First_Reaching_Year'])} years"
                            if bool(row["Reached_Within_100yr"])
                            else "Threshold not reached")
                cdf_years_html = (
                    "<b>Cumulative probabilities of transition to FFF:</b> "
                    f"{threshold_year_text('FFF')}"
                    "<div><b>Cumulative probabilities of transition to CSF:</b> "
                    f"{threshold_year_text('CSF')}</div>"
                )

                # Eight worksheets: one-to-one with the eight displayed plots.
                figure_tables = build_eight_figure_export_tables(
                    result, sd_cov, selected_fits, cumulative_df, crossing_df)
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    for sheet_name, table in figure_tables.items():
                        table.to_excel(writer, sheet_name=sheet_name, index=False)
                st.session_state.lifecycle_excel_data = output.getvalue()
                st.session_state.lifecycle_dashboard_display = {
                    "plots": plots,
                    "crossover_html": crossover_html,
                    "transition_summary_html": transition_summary_html,
                    "cdf_years_html": cdf_years_html,
                }
    # Fill the top-right download position without a second rerun.
    with download_placeholder.container():
        excel_bytes = st.session_state.get("lifecycle_excel_data")
        st.download_button(
            "Download Results (Excel)",
            data=excel_bytes if excel_bytes is not None else b"",
            file_name="Lifecycle_Assessment_Results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            disabled=excel_bytes is None,
            width="stretch",
        )

    if "lifecycle_dashboard_display" in st.session_state:
        render_dashboard_results(st.session_state.lifecycle_dashboard_display)

    # Supplementary reference content follows the eight result charts.
    render_calculation_formulas()
    render_schematic_and_descriptions(part1_config, part3_config, part2_config)
    st.markdown(
        "<div class='dashboard-author-footer'>"
        "Created by Jingcheng Wang, Fuzhou University."
        "<span class='author-contact-gap'></span>"
        "Contact: jingchengwang@fzu.edu.cn"
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# 11. Standalone entry
# ============================================================
def main():
    st.set_page_config(
        page_title=(
            "Lifecycle probabilistic seismic failure mode "
            "assessment of coastal bridge bents"
        ),
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    render_lifecycle_app()


if __name__ == "__main__":
    main()
