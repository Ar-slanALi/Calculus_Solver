# DerivX 4D — Universal Symbolic Derivative Engine & Spatial Curvature Visualizer

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![SymPy](https://img.shields.io/badge/SymPy-1.13%2B-3B5526?style=for-the-badge&logo=python&logoColor=white)](https://www.sympy.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.24%2B-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**DerivX 4D** is an advanced, high-performance symbolic computer algebra system (CAS) and spatial manifold visualizer designed for single-variable and multivariable calculus (up to 4 dimensions: $x, y, z, w$). 

Engineered with academic rigor and an executive telemetry interface, DerivX 4D computes exact analytical derivatives, step-by-step rule breakdowns, stationary extrema, curvature tensors, vector calculus operators, constrained Lagrange optimizations, and renders interactive 3D/4D spatial manifolds in real time.

---

## 📑 Table of Contents
1. [System Architecture](#-system-architecture)
2. [Key Features & Modules](#-key-features--modules)
   - [Symbolic Multivariable Differentiation Engine](#1-symbolic-multivariable-differentiation-engine)
   - [Advanced Multivariable Analysis Suite](#2-advanced-multivariable-analysis-suite)
   - [Spatial Manifold Visualizer](#3-spatial-manifold-visualizer-plotly-engine)
   - [Natural Language Assistant & Workspace](#4-natural-language-assistant--mathematical-workspace)
3. [Repository Structure](#-repository-structure)
4. [Quickstart & Installation](#-quickstart--installation)
5. [Mathematical Syntax Reference](#-mathematical-syntax-reference)
6. [Automated Verification Suite](#-automated-verification-suite)
7. [Tech Stack](#-tech-stack)
8. [License](#-license)

---

## 🏗️ System Architecture

```text
  ┌─────────────────────────────────────────────────────────────┐
  │                 User Input & Interaction                    │
  │   - Text Formulation Box     - 6-Category Math Palette      │
  │   - Natural Language Prompt  - Preset Mathematical Models   │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │         Mathematical Preprocessing & Parser (AST)           │
  │   - Implicit Multiplication Resolution (e.g. 4x^3 -> 4*x^3) │
  │   - Unicode / Typography Normalization (–, —, ×, ÷)         │
  │   - Safe Symbol Extraction (x, y, z, w)                     │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │             Core Calculus Engine (SymPy CAS)                │
  │  ├── First-Order Partial Derivatives & Step-by-Step Proofs  │
  │  ├── Gradient Vector Field (∇f) & Steepest Ascent           │
  │  ├── Second-Order Hessian Matrix (H) & Clairaut Symmetry    │
  │  ├── Stationary Critical Points & Extrema Classification    │
  │  ├── 1st & 2nd Order Multivariable Taylor Polynomials       │
  │  ├── Scalar Laplacian (∇²f), Divergence & Curl Operators     │
  │  └── Constrained Optimization (Lagrange Multipliers)        │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
  ┌─────────────────────────────┐ ┌─────────────────────────────┐
  │ Spatial Manifold Visualizer │ │     Export & Code Suite     │
  │ - 1D Curve & Tangent Line   │ │ - Publication LaTeX (.tex)  │
  │ - 2D Surface 3D Mesh        │ │ - Vectorized NumPy (.py)    │
  │ - 2D Contour Map (Heatmap)  │ └─────────────────────────────┘
  │ - 3D Critical Point Markers │
  │ - 4D Dynamic Axis Slicing   │
  └──────────────┬──────────────┘
                 │
                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 Streamlit Reactive Frontend                 │
  │  - Non-Collapsing Layout Architecture                       │
  │  - High-Contrast Telemetry Badges & Status Indicators       │
  │  - Real-Time KaTeX Mathematical Formatting                  │
  │  - Dual-Theme Adaptive CSS (Dark Slate & Crisp Light Mode)  │
  └─────────────────────────────────────────────────────────────┘
```

---

## ⚡ Key Features & Modules

### 1. Symbolic Multivariable Differentiation Engine
- **Universal Variable Support:** Automatically parses and differentiates functions across 1 to 4 independent variables ($x, y, z, w$).
- **First-Order Partial Derivatives ($\frac{\partial f}{\partial x_i}$):** Computes exact symbolic partial derivatives with respect to each coordinate axis while treating other variables as constants.
- **Step-by-Step Analytical Derivation Breakdown:** Expandable step-by-step proofs decomposing terms using core calculus rules (Power Rule, Product Rule, Quotient Rule, Chain Rule, Exponential, and Trigonometric rules).
- **Gradient Vector Field ($\nabla f$):** Algebraic gradient vector pointing in the direction of steepest spatial ascent with real-time coordinate evaluation.
- **Second-Order Hessian Curvature Matrix ($H(f)$):** Complete $n \times n$ symmetric Hessian matrix containing all mixed second partials $\frac{\partial^2 f}{\partial x_i \partial x_j}$, determinant calculation ($\det H$), and verification of Clairaut's theorem ($\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x}$).
- **Local Extrema & Curvature Classification:** Classifies critical points and local surface curvature into **Local Minima**, **Local Maxima**, and **Saddle Point Manifolds** using leading principal minors and second derivative tests.
- **Directional Derivative ($D_{\vec{u}} f$):** Exact spatial rate of change along any user-defined direction vector $\vec{v}$, normalized automatically to a unit vector $\vec{u} = \frac{\vec{v}}{\|\vec{v}\|}$.
- **Tangent Planes & Linear Approximations:** Analytical and numerical tangent plane generation at point $P_0$:
  $$z - z_0 = f_x(x_0, y_0)(x - x_0) + f_y(x_0, y_0)(y - y_0)$$

---

### 2. Advanced Multivariable Analysis Suite
The application includes a dedicated 5-tab research and engineering suite:
1. **Critical Points & Extrema ($\nabla f = \vec{0}$):** Solves the nonlinear system of first-order partials to identify all real stationary points, evaluate critical function values $f(P^*)$, and classify each point with dedicated telemetry indicators.
2. **Multivariable Taylor Polynomials:**
   - 1st-Order Taylor Polynomial ($T_1(\vec{x})$): Local linear tangent hyper-plane.
   - 2nd-Order Taylor Polynomial ($T_2(\vec{x})$): Quadratic osculating paraboloid incorporating full Hessian curvature around evaluation point $P_0$.
3. **Vector Calculus & Laplacian ($\nabla^2 f$):**
   - Exact scalar Laplacian $\Delta f = \nabla^2 f = \sum \frac{\partial^2 f}{\partial x_i^2}$.
   - Automatic harmonic function verification ($\nabla^2 f = 0$).
   - Vector field operators on the gradient field $\vec{F} = \nabla f$: Divergence ($\text{div}(\nabla f) = \nabla \cdot \nabla f$) and Curl ($\text{curl}(\nabla f) = \nabla \times \nabla f \equiv \vec{0}$), confirming conservative and irrotational properties.
4. **Constrained Optimization (Lagrange Multipliers):**
   - Solves constrained extrema problems subject to equality constraints $g(\vec{x}) = c$ using the Lagrange condition:
     $$\nabla f = \lambda \nabla g, \quad g(\vec{x}) = c$$
5. **Code & Mathematical Report Export:**
   - **Publication LaTeX Report (.tex):** One-click downloadable LaTeX document containing formal equations, derivative tables, Hessian matrices, and critical point analyses.
   - **Vectorized Python NumPy Implementation (.py):** Clean, vectorized Python code for numerical simulation and machine learning pipelines.

---

### 3. Spatial Manifold Visualizer (Plotly Engine)
- **1-Variable Visualizer:** 2D interactive function curve with dynamic derivative tangent line overlay and point tracking.
- **2-Variable 3D Surface:** Full 3D rotatable and zoomable Euclidean surface mesh with Viridis colormap, contour projection lines, active tangent plane overlay, and 3D critical point markers (Minima in Emerald, Maxima in Coral, Saddle Points in Royal Indigo).
- **2D Contour Map:** High-precision contour level plot with heatmaps for topological gradient analysis.
- **3D Taylor Paraboloid Overlay:** Simultaneously visualizes original surface $f(x, y)$ alongside its second-order Taylor polynomial $T_2(x, y)$ to observe local approximation curvature.
- **3- and 4-Variable Dynamic Slicing:** Projects higher-dimensional hyper-surfaces into 3D Euclidean space by allowing users to select any 2 active axes while dynamically slicing through remaining dimensions via real-time interactive sliders.

---

### 4. Natural Language Assistant & Mathematical Workspace
- **Natural Language Query Bar:** Allows users to input plain-English prompts (e.g., *"critical points of x^3 - 3*x*y + y^3"* or *"derivative of sin(x)*cos(y)"*) to automatically parse, extract, and load targets.
- **Interactive On-Screen Math Palette:** 6 categorized keypad drawers for tablet, touch, and rapid input:
  - *Quick Operators:* Fraction templates, powers, square roots, logs, trig, parenthesis.
  - *Algebra & Roots:* Radicals, absolute values, common polynomials.
  - *Powers & Logs:* Natural exponential ($e^x$), natural logarithm ($\ln$), base-10 logarithm.
  - *Trigonometry:* Standard circular trig functions ($\sin, \cos, \tan, \sec, \csc, \cot$).
  - *Hyperbolic Functions:* Hyperbolic formulations ($\sinh, \cosh, \tanh$).
  - *Variables & Numbers:* Independent variables ($x, y, z, w$), arithmetic operators, and numbers.
- **Real-Time KaTeX Structure Preview:** Live mathematical preview with variable detection chips before execution.
- **Persistent Non-Collapsing UI:** Clean placeholders that maintain viewport stability without screen layout jumping.
- **Adaptive Theme Engine:** High-contrast Dark Slate theme and clean Crisp Light Mode with dedicated typography.

---

## 📁 Repository Structure

```
Calculus_Solver/
│
├── app.py                 # Streamlit web application & UI orchestrator
├── calculus_engine.py     # Symbolic mathematics core (SymPy, differentiation, solvers)
├── visualizer.py          # 2D/3D Plotly visualization & dynamic slicing engine
├── test_engine.py         # Unit test suite for symbolic engine & edge cases
├── requirements.txt       # Project dependencies & versions
├── run_app.bat            # 1-Click Windows launch script
├── .gitattributes         # Git line ending normalization
├── .gitignore             # Git exclusion rules (venv, cache, logs)
├── LICENSE                # MIT Open Source License
└── README.md              # Comprehensive manual & technical documentation
```

---

## 🚀 Quickstart & Installation

### Prerequisites
- **Python 3.10 or higher** installed.
- **Git** installed on your system.

### Step 1: Clone the Repository
```bash
git clone https://github.com/Ar-slanALi/Calculus_Solver.git
cd Calculus_Solver
```

### Step 2: Set Up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
**Option A (Windows 1-Click Launcher):**
Simply double-click `run_app.bat`.

**Option B (Command Line):**
```bash
streamlit run app.py
```
The application will launch in your default web browser at `http://localhost:8501`.

---

## 📐 Mathematical Syntax Reference

| Operation | Mathematical Form | Input Syntax | Working Example |
| :--- | :--- | :--- | :--- |
| **Multiplication** | $4 x^3 y^2$ | Explicit `*` operator | `4*x^3*y^2` |
| **Rational Division** | $\frac{x+y}{x-y}$ | Denominator in `(...)` | `(x + y) / (x - y)` |
| **Powers & Exponents** | $x^2, y^4$ | Caret `^` or `**` | `x^2 - y^4` |
| **Natural Exponential** | $e^{2x}, e^z y^4$ | `exp(...)` or `e^(...)` | `exp(2*x)` or `e^z * y^4` |
| **Natural Logarithm** | $\ln(x^2 + 1)$ | `ln(...)` or `log(...)` | `ln(x^2 + 1)` |
| **Trigonometry** | $\sin(x) \cos(y)$ | `sin(...) * cos(...)` | `sin(x) * cos(y)` |
| **Square Root** | $\sqrt{x^2 + y^2}$ | `sqrt(...)` | `sqrt(x^2 + y^2)` |
| **Absolute Value** | $\|x - y\|$ | `abs(...)` | `abs(x - y)` |
| **Hyperbolic** | $\sinh(x) + \cosh(y)$ | `sinh(...) + cosh(...)` | `sinh(x) + cosh(y)` |
| **Constants** | $\pi, e$ | `pi`, `e` | `pi * x^2 + e` |

---

## 🧪 Automated Verification Suite
To verify all symbolic routines, Taylor expansions, critical point solvers, and Plotly visualization pipelines:
```bash
python test_engine.py
```

---

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Symbolic Computation:** [SymPy](https://www.sympy.org/) (Computer Algebra System)
- **Interactive Visualization:** [Plotly Graphing Libraries](https://plotly.com/python/)
- **Application Framework:** [Streamlit](https://streamlit.io/)
- **Numerical Computations:** [NumPy](https://numpy.org/) & [Pandas](https://pandas.pydata.org/)
- **Mathematical Typography:** KaTeX & LaTeX

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
