"""
DerivX 4D - Universal Symbolic Derivative Engine & Spatial Manifold Visualizer
Full symbolic differentiation (Single & Multivariable Partials, Gradient, Hessian, Curvature)
and interactive 3D/4D spatial visualization.
"""

import streamlit as st
import sympy as sp
import calculus_engine as ce
import visualizer as viz

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="DerivX 4D | Universal Derivative Engine",
    page_icon="https://cdn-icons-png.flaticon.com/512/3771/3771278.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End SaaS Typography & Dual Theme Adaptation (Dark & Light Mode Friendly)
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">

<style>
    /* Universal Typography Enforcement */
    *, html, body, p, span, div, h1, h2, h3, h4, h5, h6, label, input, button, select,
    .stMarkdown, .stMarkdownContainer, .stButton button, [data-testid="stExpander"] summary {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    code, kbd, samp, pre, .stCodeBlock, .font-mono, [data-testid="stCodeBlock"] {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Streamlit Tabs Global Typography & Appearance */
    div[data-testid="stTabs"] {
        margin-top: 14px !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 6px !important;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2) !important;
        padding-bottom: 0px !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        color: var(--text-color, currentColor) !important;
        opacity: 0.72 !important;
        padding: 8px 18px !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab"]:hover {
        opacity: 1 !important;
        color: #10B981 !important;
        background: rgba(16, 185, 129, 0.08) !important;
        border-radius: 6px 6px 0 0 !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
        color: #10B981 !important;
        opacity: 1 !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #10B981 !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: #10B981 !important;
    }

    /* Streamlit Expander Typography */
    div[data-testid="stExpander"] {
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        border-radius: 8px !important;
        background: var(--secondary-background-color, rgba(128, 128, 128, 0.04)) !important;
    }
    div[data-testid="stExpander"] summary {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        color: var(--text-color, currentColor) !important;
        padding: 10px 14px !important;
    }

    /* Fix Streamlit Material Icons font inheritance */
    [data-testid="stIconMaterial"], .material-icons, [class*="material-symbols"] {
        font-family: 'Material Symbols Rounded', 'Material Icons' !important;
        display: inline-block !important;
        letter-spacing: normal !important;
        white-space: nowrap !important;
    }

    /* Page & Sidebar Padding */
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1440px;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 2rem !important;
    }

    /* Sidebar Top Brand Header */
    .sidebar-brand-box {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 4px 2px 14px 2px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 14px;
    }
    .sidebar-brand-icon {
        width: 36px;
        height: 36px;
        min-width: 36px;
        border-radius: 9px;
        background: var(--secondary-background-color, rgba(128, 128, 128, 0.08));
        border: 1px solid rgba(128, 128, 128, 0.25);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    .sidebar-brand-title {
        font-weight: 800;
        font-size: 1.05rem;
        letter-spacing: -0.025em;
        color: var(--text-color, currentColor) !important;
        line-height: 1.15;
    }
    .sidebar-brand-edition {
        font-size: 0.65rem;
        font-weight: 700;
        color: var(--text-color, currentColor);
        opacity: 0.8;
        background: rgba(128, 128, 128, 0.12);
        border: 1px solid rgba(128, 128, 128, 0.28);
        padding: 1px 5px;
        border-radius: 4px;
        margin-left: 4px;
        vertical-align: middle;
        font-family: 'JetBrains Mono', monospace;
    }
    .sidebar-brand-desc {
        font-size: 0.72rem;
        opacity: 0.68;
        font-weight: 500;
        color: var(--text-color, currentColor) !important;
        margin-top: 1px;
    }

    /* Centered, Balanced Hero Header */
    .brand-hero-centered {
        text-align: center;
        padding: 6px 16px 16px 16px;
        margin-bottom: 20px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.16);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .brand-hero-top {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 14px;
    }
    .brand-logo-box {
        width: 48px;
        height: 48px;
        min-width: 48px;
        border-radius: 12px;
        background: var(--secondary-background-color, rgba(128, 128, 128, 0.08));
        border: 1px solid rgba(128, 128, 128, 0.25);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
    }
    .brand-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 2.35rem;
        font-weight: 900;
        letter-spacing: -0.035em;
        color: var(--text-color, currentColor) !important;
        line-height: 1.15 !important;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .brand-edition-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        padding: 3px 9px;
        border-radius: 5px;
        background: rgba(128, 128, 128, 0.12);
        color: var(--text-color, currentColor) !important;
        border: 1.2px solid rgba(128, 128, 128, 0.3);
        text-transform: uppercase;
        vertical-align: middle;
    }
    .brand-sub {
        font-size: 0.92rem;
        color: var(--text-color, currentColor) !important;
        opacity: 0.78;
        margin-top: 6px;
        font-weight: 500;
        letter-spacing: 0.01em;
        max-width: 720px;
        line-height: 1.35;
    }

    /* Sidebar Headings */
    .sidebar-section-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-color, currentColor) !important;
        opacity: 0.65;
        margin-top: 14px;
        margin-bottom: 6px;
    }

    /* Container Cards (Adaptive to Theme) */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background: var(--secondary-background-color, #161920) !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.1) !important;
        padding: 16px 20px !important;
        color: var(--text-color, currentColor) !important;
    }

    /* Reusable Typography Classes */
    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-color, currentColor) !important;
        letter-spacing: -0.01em;
    }
    .card-subtitle {
        font-size: 0.82rem;
        color: var(--text-color, currentColor) !important;
        opacity: 0.72;
        margin-top: 2px;
    }
    .section-label {
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: var(--text-color, currentColor) !important;
        opacity: 0.68;
    }
    .preview-box {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 8px 14px;
        margin: 8px 0 10px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: var(--text-color, currentColor) !important;
    }
    .derivative-card {
        background: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .derivative-tag {
        font-size: 0.78rem;
        font-weight: 800;
        color: #34D399;
        margin-bottom: 4px;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.02em;
    }

    /* Standard Button Styling */
    .stButton > button {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        border-radius: 6px !important;
        transition: all 0.15s ease !important;
        padding: 0.45rem 0.9rem !important;
        background: rgba(128, 128, 128, 0.1) !important;
        color: var(--text-color, currentColor) !important;
        border: 1px solid rgba(128, 128, 128, 0.25) !important;
    }
    .stButton > button:hover {
        background: rgba(16, 185, 129, 0.15) !important;
        border-color: #10B981 !important;
        color: #10B981 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
    }

    /* Primary Action Buttons (Calculate / Solve) */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"],
    div[data-testid="stHorizontalBlock"] button[kind="primary"],
    div[data-testid="stHorizontalBlock"] button[data-testid="stBaseButton-primary"] {
        background: #059669 !important;
        border: 1px solid #10B981 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 12px -2px rgba(16, 185, 129, 0.35) !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background: #047857 !important;
        border-color: #34D399 !important;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.45) !important;
    }

    /* Segmented Control Categories */
    div[data-testid="stSegmentedControl"] button {
        background: rgba(128, 128, 128, 0.08) !important;
        color: var(--text-color, currentColor) !important;
        opacity: 0.8 !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        padding: 5px 12px !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
        background: rgba(16, 185, 129, 0.15) !important;
        color: #10B981 !important;
        opacity: 1 !important;
        border-color: #10B981 !important;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.2) !important;
        font-weight: 700 !important;
    }

    /* Keypad Symbol Buttons */
    div[data-testid="stHorizontalBlock"] button {
        background: rgba(128, 128, 128, 0.09) !important;
        color: var(--text-color, currentColor) !important;
        border: 1px solid rgba(128, 128, 128, 0.22) !important;
        font-family: 'JetBrains Mono', 'Plus Jakarta Sans', monospace !important;
        font-size: 0.94rem !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        padding: 4px 4px !important;
        min-height: 40px !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stHorizontalBlock"] button:hover {
        background: rgba(16, 185, 129, 0.15) !important;
        border-color: #10B981 !important;
        color: #10B981 !important;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.2) !important;
        transform: translateY(-1px);
    }

    /* Curvature & Optimization Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.80rem;
        letter-spacing: 0.02em;
        line-height: 1.4;
        transition: all 0.2s ease;
    }
    .status-badge-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }
    .status-min {
        background: rgba(16, 185, 129, 0.16);
        color: #34D399;
        border: 1.5px solid #10B981;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.2);
    }
    .status-min .status-badge-dot {
        background: #34D399;
        box-shadow: 0 0 6px #10B981;
    }
    .status-max {
        background: rgba(239, 68, 68, 0.16);
        color: #F87171;
        border: 1.5px solid #EF4444;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.2);
    }
    .status-max .status-badge-dot {
        background: #F87171;
        box-shadow: 0 0 6px #EF4444;
    }
    .status-saddle {
        background: rgba(99, 102, 241, 0.18);
        color: #C7D2FE;
        border: 1.5px solid #6366F1;
        box-shadow: 0 2px 10px rgba(99, 102, 241, 0.25);
    }
    .status-saddle .status-badge-dot {
        background: #818CF8;
        box-shadow: 0 0 7px #6366F1;
    }
    .status-inconclusive {
        background: rgba(148, 163, 184, 0.14);
        color: #E2E8F0;
        border: 1.5px solid #94A3B8;
    }
    .status-inconclusive .status-badge-dot {
        background: #94A3B8;
    }

    /* High-Precision Engineering Telemetry Badges */
    .telemetry-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 10px;
        border-radius: 5px;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        line-height: 1.4;
        transition: all 0.2s ease;
    }
    .telemetry-green {
        background: rgba(16, 185, 129, 0.15) !important;
        border: 1.5px solid #10B981 !important;
        color: #34D399 !important;
        box-shadow: 0 1px 6px rgba(16, 185, 129, 0.2);
    }
    .telemetry-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #34D399;
        box-shadow: 0 0 6px #10B981;
        display: inline-block;
        flex-shrink: 0;
    }

    /* Documentation Cards */
    .doc-card-valid {
        background: rgba(16, 185, 129, 0.05);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .doc-card-invalid {
        background: rgba(239, 68, 68, 0.05);
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Centered, Balanced Hero Header
st.markdown("""
<div class="brand-hero-centered">
    <div class="brand-hero-top">
        <div class="brand-logo-box">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
                <!-- Smooth calculus manifold curve -->
                <path d="M3 18C7 18 8 6 13 6C17 6 18 12 21 12" stroke="currentColor" stroke-width="1.8" opacity="0.55"/>
                <!-- Partial derivative ∂ stylized glyph -->
                <path d="M14.5 5.5C13.5 4.5 11.5 4.5 10 5.8C8 7.5 8 11.5 8 14.5C8 17.5 9.8 19.5 12.5 19.5C15.2 19.5 17 17.2 17 14.5C17 12 15.2 10.5 12.5 10.5C10.5 10.5 9 11.8 8.5 13" stroke="#10B981" stroke-width="2.3"/>
                <!-- Tangent slope line & point -->
                <circle cx="13" cy="6" r="2" fill="#10B981"/>
                <path d="M9.5 9.5L16.5 2.5" stroke="#10B981" stroke-width="1.8" stroke-dasharray="1.5 1.5"/>
            </svg>
        </div>
        <div class="brand-title">
            Deriv<span style="color:#10B981;">X</span>
            <span class="brand-edition-badge">4D</span>
        </div>
    </div>
    <div class="brand-sub">
        Universal Symbolic Derivative Engine & Spatial Curvature Visualizer
    </div>
</div>
""", unsafe_allow_html=True)

# Presets Library
preset_options = {
    "Custom Expression": "",
    "Paraboloid (2-Var)": "x^2 + y^2",
    "Hyperbolic Saddle (2-Var)": "x^2 - y^2",
    "Gaussian Bell (2-Var)": "exp(-(x^2 + y^2))",
    "Monkey Saddle (2-Var)": "x^3 - 3*x*y^2",
    "Trig Wave (2-Var)": "sin(x) * cos(y)",
    "Ellipsoid Potential (3-Var)": "x^2 + y^2 + z^2",
    "Multivariable Mix (3-Var)": "x^2 * y - cos(z) + exp(x)",
    "Complex Rational & Exp (3-Var)": "4*x^3*y^2 - e^z * y^4 + z^3/(x^2) + 4*y - x^16",
    "4D Quadratic Form (4-Var)": "x^2 + y^2 - z^2 + w^2",
    "4D Multivariable Potential (4-Var)": "2*x^2 + 3*y - sin(z) + exp(w)"
}

if "expr_input" not in st.session_state:
    st.session_state.expr_input = "x^2 - y^2"

# Math Palette Categories & Symbols
MATH_CATEGORIES = {
    "Quick Access": [
        ("■/■", "( ) / ( )", "Fraction: (numerator) / (denominator)"),
        ("■^□", "^( )", "Power: base^(exponent)"),
        ("■²", "^2", "Square: ^2"),
        ("√■", "sqrt( )", "Square root: sqrt(x)"),
        ("∛■", "( )^(1/3)", "Cube root: (x)^(1/3)"),
        ("e^■", "e^( )", "Exponential: e^(x)"),
        ("ln(■)", "ln( )", "Natural Logarithm: ln(x)"),
        ("sin(■)", "sin( )", "Sine: sin(x)"),
        ("cos(■)", "cos( )", "Cosine: cos(x)"),
        ("|■|", "abs( )", "Absolute Value: abs(x)"),
        ("( )", "( )", "Parentheses"),
        ("+", " + ", "Addition"),
        ("-", " - ", "Subtraction"),
        ("×", " * ", "Multiplication"),
        ("÷", " / ", "Division"),
        ("x", "x", "Variable x"),
        ("y", "y", "Variable y"),
        ("z", "z", "Variable z"),
        ("w", "w", "Variable w"),
    ],
    "Algebra & Radicals": [
        ("■/■", "( ) / ( )", "Fraction: (A) / (B)"),
        ("■^□", "^( )", "Power: ^(n)"),
        ("■²", "^2", "Square: ^2"),
        ("■³", "^3", "Cube: ^3"),
        ("√■", "sqrt( )", "Square Root: sqrt(x)"),
        ("∛■", "( )^(1/3)", "Cube Root: (x)^(1/3)"),
        ("ⁿ√■", "( )^(1/n)", "N-th Root: (x)^(1/n)"),
        ("|■|", "abs( )", "Absolute Value: abs(x)"),
        ("( )", "( )", "Parentheses"),
        ("+", " + ", "Addition"),
        ("-", " - ", "Subtraction"),
        ("×", " * ", "Multiplication"),
        ("÷", " / ", "Division"),
        ("±", " + ", "Plus / Minus"),
        ("π", "pi", "Constant Pi"),
        ("e", "e", "Euler Constant e"),
        ("∞", "oo", "Infinity"),
    ],
    "Exponents & Logs": [
        ("e^■", "e^( )", "Exponential: e^(x)"),
        ("e^z · yⁿ", "e^z * y^2", "Product: e^z * y^n"),
        ("ln(■)", "ln( )", "Natural Logarithm: ln(x)"),
        ("log₁₀(■)", "log10( )", "Logarithm Base 10: log10(x)"),
        ("10^■", "10^( )", "Power of 10: 10^(x)"),
        ("2^■", "2^( )", "Power of 2: 2^(x)"),
        ("■^□", "^( )", "Arbitrary Power: ^(n)"),
        ("exp(■)", "exp( )", "Natural Exponential: exp(x)"),
        ("log(■)", "log( )", "Logarithm: log(x)"),
        ("e", "e", "Euler constant e"),
        ("π", "pi", "Constant pi"),
        ("x", "x", "Variable x"),
        ("y", "y", "Variable y"),
        ("z", "z", "Variable z"),
        ("w", "w", "Variable w"),
    ],
    "Trigonometric": [
        ("sin(■)", "sin( )", "Sine: sin(x)"),
        ("cos(■)", "cos( )", "Cosine: cos(x)"),
        ("tan(■)", "tan( )", "Tangent: tan(x)"),
        ("sec(■)", "sec( )", "Secant: sec(x)"),
        ("csc(■)", "csc( )", "Cosecant: csc(x)"),
        ("cot(■)", "cot( )", "Cotangent: cot(x)"),
        ("sin⁻¹(■)", "asin( )", "Inverse Sine / Arcsin"),
        ("cos⁻¹(■)", "acos( )", "Inverse Cosine / Arccos"),
        ("tan⁻¹(■)", "atan( )", "Inverse Tangent / Arctan"),
        ("π", "pi", "Constant Pi"),
        ("x", "x", "Variable x"),
        ("y", "y", "Variable y"),
    ],
    "Hyperbolic": [
        ("sinh(■)", "sinh( )", "Hyperbolic Sine: sinh(x)"),
        ("cosh(■)", "cosh( )", "Hyperbolic Cosine: cosh(x)"),
        ("tanh(■)", "tanh( )", "Hyperbolic Tangent: tanh(x)"),
        ("sech(■)", "sech( )", "Hyperbolic Secant: sech(x)"),
        ("csch(■)", "csch( )", "Hyperbolic Cosecant: csch(x)"),
        ("coth(■)", "coth( )", "Hyperbolic Cotangent: coth(x)"),
        ("sinh⁻¹(■)", "asinh( )", "Inverse Hyperbolic Sine: asinh(x)"),
        ("cosh⁻¹(■)", "acosh( )", "Inverse Hyperbolic Cosine: acosh(x)"),
        ("tanh⁻¹(■)", "atanh( )", "Inverse Hyperbolic Tangent: atanh(x)"),
    ],
    "Variables & Arithmetic": [
        ("x", "x", "Variable x"),
        ("y", "y", "Variable y"),
        ("z", "z", "Variable z"),
        ("w", "w", "Variable w"),
        ("+", " + ", "Addition"),
        ("-", " - ", "Subtraction"),
        ("*", " * ", "Multiplication"),
        ("/", " / ", "Division"),
        ("^", "^", "Power Caret"),
        ("(", "(", "Left Parenthesis"),
        (")", ")", "Right Parenthesis"),
        ("0", "0", "Digit 0"),
        ("1", "1", "Digit 1"),
        ("2", "2", "Digit 2"),
        ("3", "3", "Digit 3"),
        ("4", "4", "Digit 4"),
        ("5", "5", "Digit 5"),
        ("6", "6", "Digit 6"),
        ("7", "7", "Digit 7"),
        ("8", "8", "Digit 8"),
        ("9", "9", "Digit 9"),
        (".", ".", "Decimal Point"),
    ]
}

def smart_insert(curr: str, token: str) -> str:
    """Intelligently inserts mathematical tokens into the active formula string."""
    if token == '( ) / ( )':
        if curr and not curr.endswith(('+', '-', '*', '/', '^', '(')):
            return f"({curr}) / ( )"
        return curr + "( ) / ( )"

    if curr.endswith('( ) / ( )'):
        return curr[:-9] + f"({token}) / ( )"

    if curr.endswith('/ ( )'):
        return curr[:-5] + f"/ ({token})"

    if curr.endswith('( )'):
        if token.endswith('( )'):
            return curr[:-3] + token
        return curr[:-2] + f"{token})"

    if curr and (curr[-1].isalnum() or curr.endswith(')')) and token in ['x', 'y', 'z', 'w']:
        return curr + " * " + token

    return curr + token

def insert_math_token(token: str):
    curr = st.session_state.get("expr_input", "")
    st.session_state["expr_input"] = smart_insert(curr, token)
    st.session_state["preset_choice"] = "Custom Expression"

def backspace_token():
    curr = st.session_state.get("expr_input", "")
    if not curr:
        return
    multi_funcs = [
        "sqrt( )", "( ) / ( )", "( )^(1/3)", "( )^(1/n)",
        "sinh( )", "cosh( )", "tanh( )", "sech( )", "csch( )", "coth( )",
        "asinh( )", "acosh( )", "atanh( )", "log10( )",
        "asin( )", "acos( )", "atan( )", "sin( )", "cos( )", "tan( )", "sec( )", "csc( )", "cot( )",
        "exp( )", "log( )", "abs( )", "e^( )", "ln( )",
        " * ", " + ", " - ", " / ", "^2", "^3"
    ]
    for mf in multi_funcs:
        if curr.endswith(mf):
            st.session_state["expr_input"] = curr[:-len(mf)]
            return
    st.session_state["expr_input"] = curr[:-1]

def clear_token():
    st.session_state["expr_input"] = ""
    st.session_state["preset_choice"] = "Custom Expression"

def update_preset():
    choice = st.session_state.preset_choice
    if choice != "Custom Expression":
        st.session_state.expr_input = preset_options[choice]

def set_formula(new_formula):
    st.session_state.expr_input = new_formula
    st.session_state.preset_choice = "Custom Expression"

# ----------------- MIDDLE WORKSPACE: CLEAN FUNCTION DEFINITION -----------------
with st.container(border=True):
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
        <div>
            <div class="card-title">Function Input & Derivative Formulation</div>
            <div class="card-subtitle">
                Enter single or multivariable scalar fields f(x, y, z, w) using standard syntax or the mathematical palette.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_calc, col_bksp, col_clear = st.columns([6, 1.4, 0.7, 0.7])
    with col_input:
        expr_input = st.text_input(
            "Mathematical Function:",
            key="expr_input",
            placeholder="e.g. 4*x^3*y^2 - e^z * y^4 + sqrt(x^2 + y^2)",
            label_visibility="collapsed"
        )
    with col_calc:
        calc_action = st.button("Calculate", type="primary", use_container_width=True, help="Compute symbolic derivatives, Hessian matrix, and 3D manifold visualizer")
    with col_bksp:
        st.button("⌫", on_click=backspace_token, use_container_width=True, help="Delete last symbol")
    with col_clear:
        st.button("Clear", on_click=clear_token, use_container_width=True, help="Clear formula")

    if calc_action:
        st.toast("Derivatives and visuals updated.")

    # Real-Time Mathematical Structure Preview (KaTeX)
    curr_expr_text = st.session_state.get("expr_input", "").strip()
    if curr_expr_text:
        p_expr, p_syms, p_err = ce.parse_function(curr_expr_text)
        if p_expr is not None:
            p_varnames = [s.name for s in p_syms]
            f_sig = f"f({', '.join(p_varnames)})" if p_varnames else "f"
            var_chips_html = "".join([f'<span class="telemetry-badge telemetry-green" style="margin-left:6px; font-family:\'JetBrains Mono\', monospace; padding:2px 8px; font-size:0.75rem;">{v}</span>' for v in p_varnames])
            st.markdown(f"""
            <div class="preview-box">
                <span class="section-label">Mathematical Structure Preview:</span>
                <div>{var_chips_html}</div>
            </div>
            """, unsafe_allow_html=True)
            st.latex(f"{f_sig} = {sp.latex(p_expr)}")
        else:
            st.markdown("""
            <div style="background:rgba(99, 102, 241, 0.1); border:1px solid rgba(99, 102, 241, 0.35); border-radius:8px; padding:8px 14px; margin:8px 0 10px 0; color:#C7D2FE; font-size:0.82rem; font-weight:600;">
                <b>Drafting Expression:</b> Use the keypad below or type variables inside the template (e.g. <code>sqrt(x)</code>).
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="preview-box">
            <span class="section-label">Mathematical Structure Preview:</span>
            <span style="font-size:0.75rem; color:var(--text-color, currentColor); opacity:0.6; font-style:italic;">Awaiting expression...</span>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"f(x, y) = —")

    # Natural Language Assistant & Query Bar
    with st.expander("Natural Language Query Assistant", expanded=False):
        c_nl1, c_nl2 = st.columns([5, 1.2])
        with c_nl1:
            nl_query = st.text_input("Natural Language Prompt:", placeholder="e.g. critical points of x^3 - 3*x*y + y^3 or derivative of sin(x)*cos(y)", label_visibility="collapsed", key="nl_query_box")
        with c_nl2:
            if st.button("Parse Query", use_container_width=True):
                if nl_query:
                    extracted, intent = ce.parse_natural_language_query(nl_query)
                    if extracted:
                        set_formula(extracted)
                        st.toast(f"Formula applied: {extracted}")
                        st.rerun()

    # Keypad Palette Category Selector
    st.markdown('<div class="section-label" style="margin-top:12px; margin-bottom:6px;">Math Palette Categories:</div>', unsafe_allow_html=True)
    
    cat_keys = list(MATH_CATEGORIES.keys())
    selected_cat = st.segmented_control(
        "Math Palette Categories",
        cat_keys,
        default=cat_keys[0],
        key="keypad_category",
        label_visibility="collapsed"
    )
    if not selected_cat:
        selected_cat = cat_keys[0]

    # Render Category Keypad Buttons Grid
    buttons_list = MATH_CATEGORIES[selected_cat]
    cols_per_row = 10 if len(buttons_list) >= 15 else 8
    
    for row_start in range(0, len(buttons_list), cols_per_row):
        row_items = buttons_list[row_start:row_start + cols_per_row]
        cols = st.columns(len(row_items))
        for col, (label, token, tooltip) in zip(cols, row_items):
            with col:
                st.button(
                    label,
                    key=f"btn_{selected_cat}_{label}_{row_start}",
                    on_click=insert_math_token,
                    args=(token,),
                    help=tooltip,
                    use_container_width=True
                )

# ----------------- SIDEBAR CONTROLS & PRESETS -----------------
st.sidebar.markdown("""
<div class="sidebar-brand-box">
    <div class="sidebar-brand-icon">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14.5 5.5C13.5 4.5 11.5 4.5 10 5.8C8 7.5 8 11.5 8 14.5C8 17.5 9.8 19.5 12.5 19.5C15.2 19.5 17 17.2 17 14.5C17 12 15.2 10.5 12.5 10.5C10.5 10.5 9 11.8 8.5 13" stroke="#10B981" stroke-width="2.3"/>
            <circle cx="13" cy="6" r="2" fill="#10B981"/>
        </svg>
    </div>
    <div>
        <div class="sidebar-brand-title">DerivX <span class="sidebar-brand-edition">4D</span></div>
        <div class="sidebar-brand-desc">Calculus Studio Workspace</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-section-label">Configuration & Presets</div>', unsafe_allow_html=True)

st.sidebar.selectbox(
    "Preset Models:",
    list(preset_options.keys()),
    key="preset_choice",
    on_change=update_preset
)

# Active Function Chip in Sidebar
st.sidebar.markdown(f"""
<div style="background:var(--secondary-background-color, rgba(128, 128, 128, 0.08)); border:1px solid rgba(128, 128, 128, 0.22); border-radius:8px; padding:10px 12px; margin-bottom:12px;">
    <div style="font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:var(--text-color, currentColor); opacity:0.7;">Active Function:</div>
    <div style="font-family:'JetBrains Mono', monospace; font-size:0.86rem; color:#10B981; font-weight:700; word-break:break-all; margin-top:3px;">
        f = {st.session_state.expr_input if st.session_state.expr_input else 'None'}
    </div>
</div>
""", unsafe_allow_html=True)

sidebar_calc = st.sidebar.button("Execute Calculation", type="primary", use_container_width=True)
if sidebar_calc:
    st.toast("Calculation complete.")

# Quick Reference in Sidebar
with st.sidebar.expander("Syntax Reference Card", expanded=False):
    st.markdown("""
    - **Multiplication:** Use `*` (e.g., `4*x^3*y^2`)
    - **Division:** Wrap denominator `1 / (x^2 + y^2)`
    - **Powers:** `x^2`, `y^3`
    - **Exponentials:** `exp(x)`, `e^x`, `e^z * y^4`
    - **Logarithms:** `ln(x)`, `log(y)`
    - **Trigonometry:** `sin(x)`, `cos(y)`, `tan(z)`
    """)

# Parse Function for Full Multivariable Analysis
has_input = bool(st.session_state.expr_input and st.session_state.expr_input.strip())
expr, symbols, error = ce.parse_function(st.session_state.expr_input)

# Only display warning if the user entered something that is malformed
if has_input and not expr and error:
    st.warning(error)
elif has_input and expr and error:
    st.warning(error)

var_count = len(symbols) if symbols else 0
var_names = [s.name for s in symbols] if symbols else []
active_sidebar_symbols = symbols if symbols else [sp.Symbol('x'), sp.Symbol('y')]

# Sidebar: Evaluation Point Controls
st.sidebar.markdown('<div class="sidebar-section-label">Point Coordinates (P₀)</div>', unsafe_allow_html=True)
point_dict = {}
direction_dict = {}

col_pt = st.sidebar.columns(min(len(active_sidebar_symbols), 4))
for i, s in enumerate(active_sidebar_symbols):
    with col_pt[i % len(col_pt)]:
        val = st.number_input(f"{s.name}₀", value=1.0 if i == 0 else 0.0, step=0.5, key=f"pt_{s.name}")
        point_dict[s.name] = val

# Direction vector for directional derivative
st.sidebar.markdown('<div class="sidebar-section-label">Direction Vector (v)</div>', unsafe_allow_html=True)
col_dir = st.sidebar.columns(min(len(active_sidebar_symbols), 4))
for i, s in enumerate(active_sidebar_symbols):
    with col_dir[i % len(col_dir)]:
        dir_val = st.number_input(f"v_{s.name}", value=1.0 if i == 0 else 0.0, step=0.5, key=f"dir_{s.name}")
        direction_dict[s.name] = dir_val

# Sidebar: Plot Range Settings
st.sidebar.markdown('<div class="sidebar-section-label">Spatial Domain Bounds</div>', unsafe_allow_html=True)
ranges = {}
for s in active_sidebar_symbols:
    ranges[s.name] = st.sidebar.slider(f"Range for {s.name}", -10.0, 10.0, (-4.0, 4.0), step=0.5, key=f"rng_{s.name}")


# =========================================================================
# DUAL WORKSPACE: SOLUTION OUTPUT (LEFT) & SPATIAL GRAPH (RIGHT)
# =========================================================================
col_sol, col_graph = st.columns([1.05, 1.15], gap="large")

# ----------------- LEFT COLUMN: SOLUTION & SYMBOLIC DERIVATIVES -----------------
with col_sol:
    # 1. Function Specification Card
    with st.container(border=True):
        st.markdown('<div class="section-label" style="margin-bottom:8px;">Target Function Specification</div>', unsafe_allow_html=True)
        if expr is not None:
            if symbols:
                st.latex(f"f({', '.join(var_names)}) = {sp.latex(expr)}")
                st.caption(f"Defined on $\\mathbb{{R}}^{{{var_count}}} \\to \\mathbb{{R}}$ with orthogonal coordinate system $({', '.join(var_names)})$.")
            else:
                st.latex(f"f = {sp.latex(expr)}")
                st.caption("Scalar invariant constant.")
        else:
            st.latex(r"f(x, y) = —")
            st.caption("Defined on $\\mathbb{R}^2 \\to \\mathbb{R}$. Awaiting expression input.")

    # 2. First-Order Partial Derivatives Card
    with st.container(border=True):
        st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <div class="section-label">First-Order Partial Derivatives</div>
            <span class="telemetry-badge telemetry-green">∂f / ∂xᵢ</span>
        </div>
        """, unsafe_allow_html=True)

        if expr is not None:
            try:
                partials = ce.compute_partial_derivatives(expr, symbols)
                if partials:
                    grid_cols = st.columns(2 if len(partials) > 1 else 1)
                    for idx, p in enumerate(partials):
                        with grid_cols[idx % len(grid_cols)]:
                            st.markdown(f"""
                            <div class="derivative-card">
                                <div class="derivative-tag">∂f / ∂{p['var_name']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.latex(f"{p['diff_latex']} = {p['latex']}")

                    # Step-by-Step Derivation Breakdown
                    with st.expander("Step-by-Step Analytical Derivation", expanded=False):
                        for p in partials:
                            st.markdown(f"#### Derivation of $\\frac{{\\partial f}}{{\\partial {p['var_name']}}}$")
                            steps = ce.explain_derivative_steps(expr, p['var'])
                            for s in steps:
                                st.markdown(f"**{s['title']}**")
                                st.markdown(s['desc'], unsafe_allow_html=True)
                                if "latex" in s:
                                    st.latex(s["latex"])
                                if "breakdowns" in s:
                                    for b in s["breakdowns"]:
                                        st.markdown(f"""
                                        <div style="background:var(--secondary-background-color, rgba(128,128,128,0.06)); border-left:3px solid #10B981; padding:6px 12px; margin:4px 0; border-radius:4px; font-size:0.83rem;">
                                            <b>Term:</b> ${b['term_latex']}$ &nbsp;→&nbsp; <b>{b['rule_name']}:</b> ${b['diff_latex']}$<br>
                                            <span style="opacity:0.8; font-size:0.78rem;">{b['rule_note']}</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                            st.markdown("---")
                else:
                    st.info("The constant expression has no variable dependency (all derivatives equal zero).")
            except Exception as e:
                st.error(f"Error computing partial derivatives: {str(e)}")
        else:
            grid_cols = st.columns(2)
            with grid_cols[0]:
                st.markdown("""
                <div class="derivative-card">
                    <div class="derivative-tag">∂f / ∂x</div>
                </div>
                """, unsafe_allow_html=True)
                st.latex(r"\frac{\partial f}{\partial x} = —")
            with grid_cols[1]:
                st.markdown("""
                <div class="derivative-card">
                    <div class="derivative-tag">∂f / ∂y</div>
                </div>
                """, unsafe_allow_html=True)
                st.latex(r"\frac{\partial f}{\partial y} = —")
            st.caption("Rate of change with respect to each variable.")

    # 3. Gradient Vector Field Card
    with st.container(border=True):
        st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <div class="section-label">Gradient Vector Field (∇f)</div>
            <span class="telemetry-badge telemetry-green">Steepest Ascent</span>
        </div>
        """, unsafe_allow_html=True)
        if expr is not None:
            try:
                grad_list, grad_latex = ce.compute_gradient(expr, symbols)
                st.latex(grad_latex)
                st.caption("Vector field of first-order partials pointing in the direction of greatest spatial rate of increase.")
            except Exception as e:
                st.error(f"Error computing gradient: {str(e)}")
        else:
            st.latex(r"\nabla f = \begin{bmatrix} — \\ — \end{bmatrix}")
            st.caption("Vector field of first-order partials pointing in the direction of greatest spatial rate of increase.")

    # 4. Second-Order Hessian Matrix & Curvature Card
    with st.container(border=True):
        st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <div class="section-label">Second-Order Hessian Matrix H(f)</div>
            <span class="telemetry-badge telemetry-green">Curvature Tensor</span>
        </div>
        """, unsafe_allow_html=True)
        if expr is not None:
            try:
                hessian_data = ce.compute_hessian(expr, symbols)
                st.latex(f"H(f) = {hessian_data['matrix_latex']}")
                st.latex(f"\\det(H) = {hessian_data['determinant_latex']}")

                if var_count == 2:
                    eval_res_tab2 = ce.evaluate_at_point(expr, symbols, point_dict)
                    h_eval = eval_res_tab2.get("hessian_evaluated")
                    if h_eval and h_eval.get("det") is not None:
                        d_val = h_eval["det"]
                        fxx_val = float(h_eval["matrix"][0, 0].evalf()) if h_eval["matrix"][0, 0].is_real else None
                        st.markdown("**Local Curvature Classification at $P_0$:**")
                        if d_val > 0 and fxx_val is not None:
                            if fxx_val > 0:
                                st.markdown(f'<span class="status-badge status-min"><span class="status-badge-dot"></span> Local Minimum ($D = {d_val:.4f} > 0, f_{{xx}} = {fxx_val:.4f} > 0$)</span>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<span class="status-badge status-max"><span class="status-badge-dot"></span> Local Maximum ($D = {d_val:.4f} > 0, f_{{xx}} = {fxx_val:.4f} < 0$)</span>', unsafe_allow_html=True)
                        elif d_val < 0:
                            st.markdown(f'<span class="status-badge status-saddle"><span class="status-badge-dot"></span> Saddle Point Manifold ($D = {d_val:.4f} < 0$)</span>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<span class="status-badge status-inconclusive"><span class="status-badge-dot"></span> Inconclusive Curvature ($D = {d_val:.4f}$)</span>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error computing Hessian matrix: {str(e)}")
        else:
            st.latex(r"H(f) = \begin{bmatrix} — & — \\ — & — \end{bmatrix}")
            st.latex(r"\det(H) = —")
            st.caption("Curvature tensor and local extrema classification.")

    # 5. Point Evaluation & Directional Derivative Card
    with st.container(border=True):
        st.markdown('<div class="section-label" style="margin-bottom:8px;">Coordinate Evaluation at P₀</div>', unsafe_allow_html=True)
        if expr is not None:
            try:
                eval_res = ce.evaluate_at_point(expr, symbols, point_dict)
                pt_str = ", ".join([f"{k}={v}" for k, v in point_dict.items()])
                st.caption(f"Evaluated Point: $P_0({pt_str})$")

                c_e1, c_e2 = st.columns(2)
                with c_e1:
                    st.latex(f"f(P_0) \\approx {eval_res['f_float']:.4f}" if eval_res['f_float'] is not None else "f(P_0) = \\text{Undefined}")
                with c_e2:
                    grad_vals_str = ", ".join([f"{g['float']:.3f}" if g['float'] is not None else "N/A" for g in eval_res['gradient_evaluated']])
                    st.latex(f"\\nabla f(P_0) = \\left[ {grad_vals_str} \\right]")

                if var_count > 0:
                    dir_res, dir_err = ce.compute_directional_derivative(expr, symbols, point_dict, direction_dict)
                    if not dir_err:
                        st.markdown(f"**Directional Derivative along $\\vec{{u}}$:** $D_{{\\vec{{u}}}} f(P_0) = {dir_res['directional_derivative']:.5f}$")

                if var_count == 2:
                    t_plane = ce.compute_tangent_plane_2d(expr, symbols, point_dict)
                    if t_plane:
                        st.markdown(f"**Tangent Plane Equation:** ${t_plane['plane_latex']}$")
            except Exception as e:
                st.error(f"Error evaluating coordinates: {str(e)}")
        else:
            pt_str = ", ".join([f"{k}={v}" for k, v in point_dict.items()])
            st.caption(f"Evaluated Point: $P_0({pt_str})$")
            c_e1, c_e2 = st.columns(2)
            with c_e1:
                st.latex(r"f(P_0) = —")
            with c_e2:
                st.latex(r"\nabla f(P_0) = \left[ —, — \right]")
            st.caption("Point evaluation, tangent plane, and directional derivative calculated at P₀.")


# ----------------- RIGHT COLUMN: DEDICATED SPATIAL GRAPH -----------------
with col_graph:
    with st.container(border=True):
        st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <div class="card-title">Spatial Manifold Visualizer</div>
            <span class="telemetry-badge telemetry-green"><span class="telemetry-dot"></span> Interactive 3D Engine</span>
        </div>
        """, unsafe_allow_html=True)

        if expr is None:
            fig_empty = viz.plot_empty_3d_grid()
            st.plotly_chart(fig_empty, use_container_width=True)
            st.caption("3D Coordinate space is active. Enter or click any formula above to render its surface manifold.")

        elif var_count == 0:
            st.info("Constant scalar function: graph is an invariant horizontal hyper-plane.")
        
        elif var_count == 1:
            st.caption(f"Visualizing curve $y = f({symbols[0].name})$ with derivative tangent line overlay.")
            x_pt = point_dict.get(symbols[0].name, None)
            try:
                fig1d = viz.plot_1d_function(
                    expr=expr,
                    symbol=symbols[0],
                    x_range=ranges.get(symbols[0].name, (-5, 5)),
                    point_val=x_pt
                )
                st.plotly_chart(fig1d, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating 1D plot: {str(e)}")

        elif var_count == 2:
            col_v1, col_v2 = st.columns([1.1, 1.3])
            with col_v1:
                plot_mode = st.radio("Display View:", ["3D Surface", "2D Contour Map"], horizontal=True, label_visibility="collapsed")
            with col_v2:
                c_chk1, c_chk2 = st.columns(2)
                with c_chk1:
                    show_tangent = st.checkbox("Tangent Plane", value=True)
                with c_chk2:
                    show_extrema = st.checkbox("Critical Points", value=True)
                show_taylor_3d = st.checkbox("Overlay Taylor Paraboloid", value=False)

            crit_pts_for_plot = ce.find_critical_points(expr, symbols) if (show_extrema and expr) else None
            taylor_for_plot = ce.compute_taylor_polynomial(expr, symbols, point_dict) if (show_taylor_3d and expr) else None

            try:
                if plot_mode == "3D Surface":
                    fig3d = viz.plot_2d_surface(
                        expr=expr,
                        symbols=symbols,
                        ranges=ranges,
                        point_dict=point_dict,
                        show_tangent=show_tangent,
                        critical_points=crit_pts_for_plot,
                        show_taylor=show_taylor_3d,
                        taylor_expr=taylor_for_plot['T2'] if taylor_for_plot else None
                    )
                    st.plotly_chart(fig3d, use_container_width=True)
                else:
                    fig_cont = viz.plot_2d_contour(expr, symbols, ranges)
                    st.plotly_chart(fig_cont, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating surface visual: {str(e)}")

        else:
            # 3 or 4 Variables: Dynamic Slicing
            st.caption(f"Projecting {var_count}D hyper-surface into 3D Euclidean space:")
            col_ax1, col_ax2 = st.columns(2)
            with col_ax1:
                active_x = st.selectbox("Axis 1 (X):", var_names, index=0)
            with col_ax2:
                remaining_y = [v for v in var_names if v != active_x]
                active_y = st.selectbox("Axis 2 (Y):", remaining_y, index=0)

            sym_map = {s.name: s for s in symbols}
            active_symbols = [sym_map[active_x], sym_map[active_y]]
            inactive_symbols = [s for s in symbols if s.name not in [active_x, active_y]]

            slice_fixed_vals = {}
            sl_cols = st.columns(len(inactive_symbols))
            for c, s in zip(sl_cols, inactive_symbols):
                with c:
                    s_rng = ranges.get(s.name, (-4.0, 4.0))
                    slice_fixed_vals[s.name] = st.slider(
                        f"Fixed {s.name}:",
                        float(s_rng[0]), float(s_rng[1]), float(point_dict.get(s.name, 0.0)),
                        step=0.25,
                        key=f"slice_{s.name}"
                    )

            try:
                fig_slice, slice_expr = viz.plot_multivariable_slice(
                    expr=expr,
                    symbols=symbols,
                    active_vars=active_symbols,
                    fixed_values=slice_fixed_vals,
                    ranges=ranges
                )
                st.plotly_chart(fig_slice, use_container_width=True)
                st.caption(f"Active Slice: $f_{{slice}}({active_x}, {active_y}) = {sp.latex(slice_expr)}$")
            except Exception as e:
                st.error(f"Error generating dynamic slice: {str(e)}")


# =========================================================================
# ADVANCED MULTIVARIABLE ANALYSIS SUITE (TABS)
# =========================================================================
st.markdown("""
<div style="margin-top:28px; margin-bottom:14px; padding-bottom:6px; border-bottom:1px solid rgba(128,128,128,0.2);">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <h3 style="margin:0; font-size:1.18rem; font-weight:700; color:var(--text-color, currentColor);">Advanced Multivariable Analysis Suite</h3>
            <div style="opacity:0.75; font-size:0.83rem; margin-top:2px;">
                Extrema optimization, Taylor series approximations, vector calculus, and export tools.
            </div>
        </div>
        <span class="telemetry-badge telemetry-green"><span class="telemetry-dot"></span> Research & Engineering Tools</span>
    </div>
</div>
""", unsafe_allow_html=True)

tab_opt, tab_taylor, tab_vector, tab_lagrange, tab_export = st.tabs([
    "Critical Points & Extrema",
    "Taylor Polynomials",
    "Vector Calculus & Laplacian",
    "Lagrange Multipliers",
    "Export & Code Generation"
])

# ----------------- TAB 1: CRITICAL POINTS & OPTIMIZATION -----------------
with tab_opt:
    with st.container(border=True):
        st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <div class="section-label">Stationary & Critical Points (∇f = 0)</div>
            <span class="telemetry-badge telemetry-green">Extrema & Saddle</span>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Critical points occur where all first-order partial derivatives vanish simultaneously ($\\nabla f = \\vec{0}$).")

        if expr is not None and symbols:
            try:
                cp_list = ce.find_critical_points(expr, symbols)
                if cp_list:
                    st.success(f"Detected **{len(cp_list)}** real critical point(s):")
                    c_cols = st.columns(min(len(cp_list), 3) if len(cp_list) > 0 else 1)
                    for idx, cp in enumerate(cp_list):
                        with c_cols[idx % len(c_cols)]:
                            badge_cls = cp['badge']
                            f_flt_str = f"{cp['f_float']:.4f}" if cp['f_float'] is not None else "N/A"
                            st.markdown(f"""
                            <div style="background:var(--secondary-background-color, rgba(128,128,128,0.06)); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:12px; margin-bottom:10px;">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                                    <span style="font-weight:700; font-size:0.85rem;">Point P*_{idx+1}</span>
                                    <span class="status-badge {badge_cls}"><span class="status-badge-dot"></span> {cp['classification']}</span>
                                </div>
                                <div style="font-family:'JetBrains Mono', monospace; font-size:0.82rem; margin:6px 0;">
                                    Coordinates: <b>({cp['coords_latex']})</b>
                                </div>
                                <div style="font-size:0.82rem;">
                                    Value: <b>f(P*) = {cp['f_latex']}</b> &nbsp;≈ <code>{f_flt_str}</code>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No algebraic real critical points detected for this domain. The function may be strictly monotonic or unbounded.")
            except Exception as e:
                st.error(f"Error computing critical points: {str(e)}")
        else:
            st.info("Enter a mathematical function above to analyze critical points.")

# ----------------- TAB 2: TAYLOR POLYNOMIAL APPROXIMATION -----------------
with tab_taylor:
    with st.container(border=True):
        st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <div class="section-label">Multivariable Taylor Polynomial Expansion</div>
            <span class="telemetry-badge telemetry-green">Expansion at P₀</span>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Approximates the curved hyper-surface locally around $P_0$ using first (linear) and second (quadratic) order differential tensors.")

        if expr is not None and symbols:
            try:
                t_res = ce.compute_taylor_polynomial(expr, symbols, point_dict)
                if t_res:
                    t_col1, t_col2 = st.columns(2)
                    with t_col1:
                        st.markdown("**1st-Order Taylor Polynomial (Linear / Tangent Plane):**")
                        st.latex(f"T_1(\\vec{{x}}) = {t_res['T1_latex']}")
                        st.caption("Matches function value and first-order directional gradients at $P_0$.")
                    with t_col2:
                        st.markdown("**2nd-Order Taylor Polynomial (Osculating Paraboloid):**")
                        st.latex(f"T_2(\\vec{{x}}) = {t_res['T2_latex']}")
                        st.caption("Incorporates Hessian curvature tensor to model surface bending around $P_0$.")
            except Exception as e:
                st.error(f"Error computing Taylor polynomial: {str(e)}")
        else:
            st.info("Enter a mathematical function above to compute Taylor series expansions.")

# ----------------- TAB 3: VECTOR CALCULUS & LAPLACIAN -----------------
with tab_vector:
    with st.container(border=True):
        st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <div class="section-label">Scalar Field Laplacian & Vector Field Properties</div>
            <span class="telemetry-badge telemetry-green">∇²f & Operators</span>
        </div>
        """, unsafe_allow_html=True)

        if expr is not None and symbols:
            try:
                lap_res = ce.compute_laplacian(expr, symbols)
                if lap_res:
                    st.markdown(r"**1. Scalar Laplacian ($\Delta f = \nabla^2 f$):**")
                    v_terms = " + ".join([f"\\frac{{\\partial^2 f}}{{\\partial {s.name}^2}}" for s in symbols])
                    st.latex(f"\\nabla^2 f = {v_terms} = {lap_res['latex']}")
                    if lap_res['is_harmonic']:
                        st.markdown(r'<span class="status-badge status-min"><span class="status-badge-dot"></span> Harmonic Function ($\nabla^2 f = 0$)</span> &nbsp; Satisfies Laplace equation $\Delta f = 0$.', unsafe_allow_html=True)
                    else:
                        st.caption("Non-zero Laplacian; function exhibits spatial curvature and flux.")

                st.markdown("---")
                st.markdown(r"**2. Vector Field Operators (on Gradient Field $\vec{F} = \nabla f$):**")
                vf_c1, vf_c2 = st.columns(2)
                with vf_c1:
                    div_val = lap_res['latex'] if lap_res else "0"
                    st.markdown(r"**Divergence $\text{div}(\nabla f) = \nabla \cdot \nabla f$:**")
                    st.latex(f"\\nabla \\cdot (\\nabla f) = {div_val}")
                    st.caption("Divergence of gradient field equals the scalar Laplacian.")
                with vf_c2:
                    st.markdown(r"**Curl $\text{curl}(\nabla f) = \nabla \times \nabla f$:**")
                    st.latex(r"\nabla \times (\nabla f) = \vec{0}")
                    st.markdown('<span class="status-badge status-min"><span class="status-badge-dot"></span> Irrotational & Conservative</span> &nbsp; Gradient fields always have zero curl identically.', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error computing vector calculus properties: {str(e)}")
        else:
            st.info("Enter a mathematical function above to compute Laplacian and vector operators.")

# ----------------- TAB 4: LAGRANGE MULTIPLIERS -----------------
with tab_lagrange:
    with st.container(border=True):
        st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <div class="section-label">Constrained Optimization (Lagrange Multipliers)</div>
            <span class="telemetry-badge telemetry-green">∇f = λ ∇g</span>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Optimize target function $f(\\vec{x})$ subject to equality constraint $g(\\vec{x}) = c$.")

        if expr is not None and symbols:
            default_constr = "x^2 + y^2" if len(symbols) >= 2 else f"{symbols[0].name}^2"
            lag_c1, lag_c2, lag_c3 = st.columns([3, 1, 1.2])
            with lag_c1:
                constr_input = st.text_input("Constraint Function g:", value=default_constr, key="lag_constr_input")
            with lag_c2:
                constr_c = st.number_input("Constant (c):", value=4.0, step=1.0, key="lag_c_val")
            with lag_c3:
                st.write("")
                st.write("")
                run_lag = st.button("Solve System", type="secondary", use_container_width=True)

            c_expr, c_syms, c_err = ce.parse_function(constr_input)
            if c_expr is not None:
                try:
                    lag_sols = ce.solve_lagrange_multipliers(expr, c_expr, constr_c, symbols)
                    if lag_sols:
                        st.markdown(f"**Constrained Critical Points on $g = {constr_c}$:**")
                        lag_grid = st.columns(min(len(lag_sols), 3))
                        for idx, sol in enumerate(lag_sols):
                            with lag_grid[idx % len(lag_grid)]:
                                st.markdown(f"""
                                <div style="background:var(--secondary-background-color, rgba(128,128,128,0.06)); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:10px; margin-bottom:8px;">
                                    <div style="font-weight:700; font-size:0.85rem; color:#10B981;">Candidate #{idx+1}</div>
                                    <div style="font-family:'JetBrains Mono', monospace; font-size:0.8rem; margin:4px 0;">({sol['point_latex']})</div>
                                    <div style="font-size:0.8rem;">Target Value: <b>f = {sol['f_latex']}</b> ({sol['f_val']:.4f})</div>
                                    <div style="font-size:0.75rem; opacity:0.75;">Multiplier λ = {sol['lambda']:.4f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("No algebraic solutions found for this constraint system on the active domain.")
                except Exception as e:
                    st.error(f"Lagrange solver error: {str(e)}")
        else:
            st.info("Enter an objective function above to set up Lagrange constraints.")

# ----------------- TAB 5: EXPORT SUITE -----------------
with tab_export:
    with st.container(border=True):
        st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <div class="section-label">Code & Mathematical Report Export</div>
            <span class="telemetry-badge telemetry-green">LaTeX & NumPy</span>
        </div>
        """, unsafe_allow_html=True)

        if expr is not None and symbols:
            try:
                partials_exp = ce.compute_partial_derivatives(expr, symbols)
                grad_l, grad_latex_exp = ce.compute_gradient(expr, symbols)
                hess_exp = ce.compute_hessian(expr, symbols)
                eval_exp = ce.evaluate_at_point(expr, symbols, point_dict)
                crit_exp = ce.find_critical_points(expr, symbols)
                lap_exp = ce.compute_laplacian(expr, symbols)

                latex_doc = ce.generate_latex_report(
                    expr=expr,
                    symbols=symbols,
                    partials=partials_exp,
                    grad_latex=grad_latex_exp,
                    hessian_data=hess_exp,
                    point_dict=point_dict,
                    eval_res=eval_exp,
                    critical_points=crit_exp,
                    laplacian_data=lap_exp
                )

                python_script = ce.generate_python_code(expr, symbols)

                ex_col1, ex_col2 = st.columns(2)
                with ex_col1:
                    st.markdown("**Publication LaTeX Code:**")
                    st.code(latex_doc, language="latex")
                    st.download_button(
                        label="Download LaTeX (.tex)",
                        data=latex_doc,
                        file_name="multivariable_calculus_report.tex",
                        mime="text/plain",
                        use_container_width=True
                    )
                with ex_col2:
                    st.markdown("**Vectorized Python (NumPy) Implementation:**")
                    st.code(python_script, language="python")
                    st.download_button(
                        label="Download Python (.py)",
                        data=python_script,
                        file_name="vectorized_function.py",
                        mime="text/x-python",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Error generating export reports: {str(e)}")
        else:
            st.info("Enter a mathematical function above to generate exportable reports.")


# ----------------- COLLAPSIBLE REFERENCE & SYNTAX DOCUMENTATION -----------------
with st.expander("Mathematical Reference & Syntax Specification", expanded=False):
    st.markdown("#### Sample Models (Load Directly into the Engine)")
    ex_cols = st.columns(3)
    with ex_cols[0]:
        st.markdown("**Rational Function:**")
        st.code("(x^2 - y^2) / (x^2 + y^2 + 1)")
        st.button("Load Rational Model", on_click=set_formula, args=("(x^2 - y^2) / (x^2 + y^2 + 1)",))

    with ex_cols[1]:
        st.markdown("**Exponential Mixed Polynomial:**")
        st.code("4*x^3*y^2 - e^z * y^4 + z^3/(x^2) + 4*y - x^16")
        st.button("Load Polynomial Model", on_click=set_formula, args=("4*x^3*y^2 - e^z * y^4 + z^3/(x^2) + 4*y - x^16",))

    with ex_cols[2]:
        st.markdown("**4-Variable Coordinate System:**")
        st.code("2*x^2 + 3*y - sin(z) + e^w")
        st.button("Load 4-Variable Model", on_click=set_formula, args=("2*x^2 + 3*y - sin(z) + e^w",))

    st.markdown("---")
    st.markdown("#### Standard Syntax Specification")
    st.markdown(r"""
| Mathematical Operator | Standard Mathematical Form | Engine Input Standard | Working Example |
| :--- | :--- | :--- | :--- |
| **Multiplication** | $4 x^3 y^2$ | Explicit `*` operator between factors | `4*x^3*y^2` |
| **Rational Division** | $\frac{A}{B}$ | Denominator encapsulated in `(...)` | `1 / (x^2 + y^2)` or `(x+y)/(x-y)` |
| **Exponents / Powers** | $x^2, y^3$ | Caret `^` or Python `**` | `x^2` or `y**3` |
| **Natural Exponential** | $e^x, e^{2y}$ | `exp(...)` or `e^(...)` | `exp(x)` or `e^(2*y)` |
| **Exponential Products** | $e^z \cdot y^4$ | Explicit `*` between $e^z$ and variable factor | `e^z * y^4` or `exp(z) * y^4` |
| **Natural Logarithm** | $\ln(x)$ | `ln(...)` or `log(...)` | `ln(x^2 + 1)` or `log(x)` |
| **Trigonometric Functions** | $\sin(x), \cos(y), \tan(z)$ | `sin(...)`, `cos(...)`, `tan(...)` | `sin(x) * cos(y)` |
| **Square Root** | $\sqrt{x^2 + y^2}$ | `sqrt(...)` | `sqrt(x^2 + y^2)` |
| **Absolute Value** | $\|x - y\|$ | `abs(...)` | `abs(x - y)` |
| **Hyperbolic Functions** | $\sinh(x), \cosh(y)$ | `sinh(...)`, `cosh(...)` | `sinh(x) + cosh(y)` |
| **Inverse Trigonometric** | $\arcsin(x), \arctan(y)$ | `asin(...)`, `atan(...)` | `atan(y / x)` |
| **Mathematical Constants** | $\pi, e$ | `pi`, `e` or `E` | `pi * x^2` |
""")

    st.markdown("---")
    st.markdown("#### Input Validation: Syntax Comparison")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown('<div class="doc-card-invalid">', unsafe_allow_html=True)
        st.markdown("**Ambiguous or Malformed Input:**")
        st.markdown("""
        - `4x^3y^2` (Missing `*` multiplication between adjacent powers)
        - `-e^zy^4` (Ambiguous exponent: unclear if $y^4$ belongs in exponent or product)
        - `1 / x^2 + y^2` (Parsed as $\\frac{1}{x^2} + y^2$ due to order of operations)
        - `x + y / 2` (Parsed as $x + \\frac{y}{2}$ instead of $\\frac{x+y}{2}$)
        - `–x^2` (Unicode En-dash from document editors instead of standard minus)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with p_col2:
        st.markdown('<div class="doc-card-valid">', unsafe_allow_html=True)
        st.markdown("**Standard Compliant Input:**")
        st.markdown("""
        - `4*x^3*y^2` (Explicit multiplicative factors)
        - `-e^z * y^4` or `-e^(z*y^4)` (Unambiguous exponentiation)
        - `1 / (x^2 + y^2)` (Enclosed denominator preserving rational term)
        - `(x + y) / 2` (Enclosed numerator ensuring proper quotient)
        - `-x^2` (Standard ASCII minus sign)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
