"""
Calculus Engine for Multivariable Calculus Functions
Supports up to 4 variables (e.g., x, y, z, w)
Computes:
- Robust input expression parsing
- First-order partial derivatives
- Gradient vector
- Higher-order mixed partial derivatives
- Hessian matrix and its determinant
- Directional derivatives
- Tangent planes & linear approximations
- Point evaluation (Exact symbolic and float)
"""

import re
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def preprocess_expression(expr_str: str) -> str:
    """
    Cleans and pre-processes common user math notations:
    - Normalizes Unicode dashes, en-dash, em-dash, minus to standard '-'
    - Replaces non-breaking spaces
    - Supports Unicode math symbols: square root √, cube root ∛, π, ×, ÷
    - Converts superscripts (² -> **2, ³ -> **3, etc.)
    - Supports inverse trig / hyperbolic: sin⁻¹, cos⁻¹, tan⁻¹, sinh⁻¹, cosh⁻¹, tanh⁻¹
    - Converts absolute value |x| -> abs(x)
    - Normalizes log10, ln(...) -> log(...)
    - Normalizes e^(...) -> exp(...) and e^x -> exp(x)
    - Caret ^ -> **
    """
    s = expr_str.strip()
    if not s:
        return ""

    # Replace any unicode dashes / minus signs with standard ASCII minus '-'
    s = re.sub(r"[\u2010-\u2015\u2212\ufe63\uff0d–—−]", "-", s)

    # Replace non-breaking spaces or zero-width whitespace
    s = re.sub(r"[\u00a0\u200b\u200c\u200d\ufeff\u202f\u2009\u2008]", " ", s)

    # Inverse trigonometric & hyperbolic notations (do before superscripts)
    s = re.sub(r"\bsin\^(-1|⁻¹)\b", "asin", s)
    s = re.sub(r"\bcos\^(-1|⁻¹)\b", "acos", s)
    s = re.sub(r"\btan\^(-1|⁻¹)\b", "atan", s)
    s = re.sub(r"\bsinh\^(-1|⁻¹)\b", "asinh", s)
    s = re.sub(r"\bcosh\^(-1|⁻¹)\b", "acosh", s)
    s = re.sub(r"\btanh\^(-1|⁻¹)\b", "atanh", s)
    s = s.replace("sin⁻¹", "asin").replace("cos⁻¹", "acos").replace("tan⁻¹", "atan")
    s = s.replace("sinh⁻¹", "asinh").replace("cosh⁻¹", "acosh").replace("tanh⁻¹", "atanh")

    # Superscript numbers
    superscripts = {
        '⁰': '**0', '¹': '**1', '²': '**2', '³': '**3', '⁴': '**4',
        '⁵': '**5', '⁶': '**6', '⁷': '**7', '⁸': '**8', '⁹': '**9',
        'ⁿ': '**n'
    }
    for sup_char, replacement in superscripts.items():
        s = s.replace(sup_char, replacement)

    # Arithmetic operators and math constants
    s = s.replace("×", "*").replace("·", "*").replace("÷", "/")
    s = s.replace("π", "pi")

    # Square roots: √(x) -> sqrt(x), √x -> sqrt(x)
    s = re.sub(r"√\(([^)]+)\)", r"sqrt(\1)", s)
    s = re.sub(r"√([a-zA-Z0-9]+)", r"sqrt(\1)", s)

    # Cube roots: ∛(x) -> ((x)**(1/3)), ∛x -> ((x)**(1/3))
    s = re.sub(r"∛\(([^)]+)\)", r"((\1)**(1/3))", s)
    s = re.sub(r"∛([a-zA-Z0-9]+)", r"((\1)**(1/3))", s)

    # Absolute value notation: |expr| -> abs(expr)
    s = re.sub(r"\|([^|]+)\|", r"abs(\1)", s)

    # Replace ln with log (SymPy uses log for natural log)
    s = re.sub(r"\bln\b", "log", s)

    # Replace e^(...) with exp(...)
    s = re.sub(r"\be\^\(([^)]+)\)", r"exp(\1)", s)
    # Replace e^x or e^[a-zA-Z0-9_]
    s = re.sub(r"\be\^([a-zA-Z0-9_]+)", r"exp(\1)", s)
    # Replace standalone e**(...) with exp(...)
    s = re.sub(r"\be\*\*([a-zA-Z0-9_]+)", r"exp(\1)", s)
    s = re.sub(r"\be\*\*\(([^)]+)\)", r"exp(\1)", s)

    # Convert ^ to **
    s = s.replace("^", "**")

    return s


def parse_function(expr_str: str, custom_vars=None):
    """
    Safely parses mathematical expression string into a SymPy expression.
    Detects variables automatically or allows custom variable specification.
    Returns: (sympy_expr, list_of_symbols, error_message)
    """
    cleaned = preprocess_expression(expr_str)
    if not cleaned:
        return None, [], "Please enter a function expression."

    # Define standard allowed functions and constants
    local_dict = {
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "cot": sp.cot,
        "sec": sp.sec,
        "csc": sp.csc,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "sech": sp.sech,
        "csch": sp.csch,
        "coth": sp.coth,
        "asinh": sp.asinh,
        "acosh": sp.acosh,
        "atanh": sp.atanh,
        "exp": sp.exp,
        "log": sp.log,
        "ln": sp.log,
        "log10": lambda x: sp.log(x, 10),
        "sqrt": sp.sqrt,
        "abs": sp.Abs,
        "Abs": sp.Abs,
        "pi": sp.pi,
        "oo": sp.oo,
        "E": sp.E,
        "e": sp.E,
    }

    try:
        expr = parse_expr(cleaned, local_dict=local_dict, transformations=TRANSFORMATIONS)
    except Exception as e:
        return None, [], f"Syntax Error in formula: {str(e)}"

    # Get free symbols (variables)
    free_syms = list(expr.free_symbols)
    # Remove math constants if any got parsed as symbol
    free_syms = [s for s in free_syms if s.name not in ["pi", "E", "e"]]

    # Sort variables nicely: x, y, z, w preferred
    def sort_key(s):
        name = s.name
        order = {"x": 1, "y": 2, "z": 3, "w": 4, "t": 5, "u": 6, "v": 7}
        return (order.get(name, 99), name)

    symbols = sorted(free_syms, key=sort_key)

    if custom_vars:
        sym_map = {s.name: s for s in symbols}
        symbols = [sym_map[name] for name in custom_vars if name in sym_map]

    if len(symbols) > 4:
        return expr, symbols[:4], f"Found {len(symbols)} variables. The system will use the first 4: {[s.name for s in symbols[:4]]}."

    return expr, symbols, None


def compute_partial_derivatives(expr, symbols):
    """
    Computes first-order partial derivatives for each variable.
    Returns list of dicts: [{'var': symbol, 'derivative': expr, 'latex': str}, ...]
    """
    partials = []
    for var in symbols:
        try:
            d = sp.diff(expr, var)
            d_simp = sp.simplify(d)
        except Exception:
            d_simp = sp.diff(expr, var)

        partials.append({
            "var": var,
            "var_name": var.name,
            "derivative": d_simp,
            "latex": sp.latex(d_simp),
            "diff_latex": f"\\frac{{\\partial f}}{{\\partial {var.name}}}",
        })
    return partials


def compute_gradient(expr, symbols):
    """
    Computes the gradient vector ∇f.
    Returns (gradient_vector, latex_representation)
    """
    grad_list = []
    for var in symbols:
        try:
            d = sp.simplify(sp.diff(expr, var))
        except Exception:
            d = sp.diff(expr, var)
        grad_list.append(d)

    if symbols:
        op_rows = [f"\\dfrac{{\\partial f}}{{\\partial {v.name}}}" for v in symbols]
        val_rows = [f"{sp.latex(g)}" for g in grad_list]
        ops_str = " \\\\[6pt] ".join(op_rows)
        vals_str = " \\\\[6pt] ".join(val_rows)
        grad_latex = f"\\nabla f = \\left[ \\begin{{array}}{{c}} {ops_str} \\end{{array}} \\right] = \\left[ \\begin{{array}}{{c}} {vals_str} \\end{{array}} \\right]"
    else:
        grad_latex = "\\nabla f = \\left[ 0 \\right]"

    return grad_list, grad_latex


def compute_hessian(expr, symbols):
    """
    Computes the Hessian matrix of second-order partial derivatives:
    H_ij = d^2 f / (dx_i dx_j)
    """
    n = len(symbols)
    hessian_matrix = sp.zeros(n, n)
    detailed_elements = []

    for i, vi in enumerate(symbols):
        row_elements = []
        for j, vj in enumerate(symbols):
            try:
                d2 = sp.simplify(sp.diff(expr, vi, vj))
            except Exception:
                d2 = sp.diff(expr, vi, vj)
            hessian_matrix[i, j] = d2
            row_elements.append({
                "var_i": vi.name,
                "var_j": vj.name,
                "expr": d2,
                "latex": sp.latex(d2)
            })
        detailed_elements.append(row_elements)

    try:
        det_hessian = sp.simplify(hessian_matrix.det()) if n > 0 else sp.Integer(0)
    except Exception:
        det_hessian = hessian_matrix.det() if n > 0 else sp.Integer(0)

    return {
        "matrix": hessian_matrix,
        "matrix_latex": sp.latex(hessian_matrix),
        "determinant": det_hessian,
        "determinant_latex": sp.latex(det_hessian),
        "elements": detailed_elements,
    }


def evaluate_at_point(expr, symbols, point_dict):
    """
    Evaluates function, gradient, and Hessian at a specific point.
    Safely handles complex numbers, singularities, and non-real evaluations.
    """
    subs_dict = {}
    for sym in symbols:
        val = point_dict.get(sym) or point_dict.get(sym.name)
        if val is not None:
            subs_dict[sym] = sp.sympify(val)

    # Function value
    try:
        f_val_exact = expr.subs(subs_dict)
        f_evalf = f_val_exact.evalf()
        if f_evalf.is_real and not (f_evalf in [sp.nan, sp.zoo, sp.oo, -sp.oo]):
            f_val_float = float(f_evalf)
        else:
            f_val_float = None
    except Exception:
        f_val_exact = None
        f_val_float = None

    # Gradient evaluated
    grad_list, _ = compute_gradient(expr, symbols)
    grad_eval = []
    for g in grad_list:
        try:
            val_ex = g.subs(subs_dict)
            val_ev = val_ex.evalf()
            if val_ev.is_real and not (val_ev in [sp.nan, sp.zoo, sp.oo, -sp.oo]):
                val_fl = float(val_ev)
            else:
                val_fl = None
            grad_eval.append({"exact": val_ex, "float": val_fl, "latex": sp.latex(val_ex)})
        except Exception:
            grad_eval.append({"exact": None, "float": None, "latex": "N/A"})

    # Hessian evaluated
    hessian_data = compute_hessian(expr, symbols)
    h_mat = hessian_data["matrix"]
    h_eval = None
    try:
        h_eval_mat = h_mat.subs(subs_dict)
        det_ev = h_eval_mat.det().evalf()
        det_fl = float(det_ev) if (det_ev.is_real and det_ev not in [sp.nan, sp.zoo]) else None
        h_eval = {
            "matrix": h_eval_mat,
            "latex": sp.latex(h_eval_mat),
            "det": det_fl
        }
    except Exception:
        h_eval = None

    return {
        "f_exact": f_val_exact,
        "f_float": f_val_float,
        "f_latex": sp.latex(f_val_exact) if f_val_exact is not None else "N/A",
        "gradient_evaluated": grad_eval,
        "hessian_evaluated": h_eval,
    }


def compute_directional_derivative(expr, symbols, point_dict, direction_dict):
    """
    Computes the directional derivative D_u f at a point along direction v.
    u = v / ||v||
    """
    subs_dict = {sym: sp.sympify(point_dict.get(sym.name, 0)) for sym in symbols}
    dir_vector = [float(direction_dict.get(sym.name, 0)) for sym in symbols]

    # Calculate magnitude
    magnitude = sum(d**2 for d in dir_vector) ** 0.5
    if magnitude == 0:
        return None, "Direction vector cannot be zero vector."

    unit_vector = [d / magnitude for d in dir_vector]

    # Grad evaluated
    grad_list, _ = compute_gradient(expr, symbols)
    d_u_val = 0.0
    exact_terms = []
    for g, u_comp in zip(grad_list, unit_vector):
        try:
            g_eval = g.subs(subs_dict).evalf()
            if not g_eval.is_real or g_eval in [sp.nan, sp.zoo]:
                return None, "Gradient contains non-real or undefined values at this point."
            g_val = float(g_eval)
            d_u_val += g_val * u_comp
            exact_terms.append(f"({g_val:.4f} \\times {u_comp:.4f})")
        except Exception as e:
            return None, f"Cannot evaluate derivative at this point: {str(e)}"

    calc_explanation = " + ".join(exact_terms) + f" = {d_u_val:.4f}"
    return {
        "directional_derivative": d_u_val,
        "unit_vector": unit_vector,
        "magnitude": magnitude,
        "explanation": calc_explanation,
    }, None


def compute_tangent_plane_2d(expr, symbols, point_dict):
    """
    For f(x, y), computes the tangent plane at (x0, y0):
    z = z0 + f_x(x0, y0)*(x - x0) + f_y(x0, y0)*(y - y0)
    """
    if len(symbols) != 2:
        return None

    try:
        x, y = symbols[0], symbols[1]
        x0 = point_dict.get(x.name, 0)
        y0 = point_dict.get(y.name, 0)

        subs = {x: sp.sympify(x0), y: sp.sympify(y0)}
        z0 = expr.subs(subs)
        fx = sp.diff(expr, x).subs(subs)
        fy = sp.diff(expr, y).subs(subs)

        # Check for singularities or complex values
        if any(v in [sp.nan, sp.zoo, sp.oo, -sp.oo] for v in [z0, fx, fy]):
            return None

        # Tangent plane expression
        z_plane = z0 + fx * (x - x0) + fy * (y - y0)
        z_plane = sp.simplify(z_plane)

        return {
            "z0": z0,
            "fx": fx,
            "fy": fy,
            "plane_expr": z_plane,
            "plane_latex": f"z = {sp.latex(z_plane)}",
            "standard_form_latex": f"z - {sp.latex(z0)} = ({sp.latex(fx)})(x - {x0}) + ({sp.latex(fy)})(y - {y0})"
        }
    except Exception:
        return None


def explain_derivative_steps(expr, var):
    """
    Generates a structured, pedagogical step-by-step breakdown of how
    the partial derivative with respect to `var` is derived.
    """
    steps = []
    # Identify variables
    free_vars = [s.name for s in expr.free_symbols if s.name != var.name]
    other_vars_str = ', '.join(free_vars) if free_vars else 'all orthogonal parameters'

    steps.append({
        "title": "1. Orthogonal Variable Treatment",
        "desc": f"Treat <b>{var.name}</b> as the independent differentiation variable and hold <b>{other_vars_str}</b> strictly constant.",
        "latex": f"\\frac{{\\partial}}{{\\partial {var.name}}} \\left[ f \\right]"
    })

    terms = expr.as_ordered_terms() if expr.is_Add else [expr]

    if len(terms) > 1:
        steps.append({
            "title": "2. Sum & Linearity Rule",
            "desc": f"Decompose the expression into {len(terms)} separate algebraic terms and differentiate each independently:",
            "latex": f"\\frac{{\\partial}}{{\\partial {var.name}}} \\sum_{{k=1}}^{{{len(terms)}}} u_k = \\sum_{{k=1}}^{{{len(terms)}}} \\frac{{\\partial u_k}}{{\\partial {var.name}}}"
        })

        term_breakdowns = []
        for term in terms:
            d_term = sp.diff(term, var)
            if not term.has(var):
                rule_name = "Constant Rule"
                rule_note = f"Independent of ${var.name}$; derivative evaluates to $0$."
            elif term.is_Pow and term.base == var:
                rule_name = "Power Rule"
                rule_note = f"$\\frac{{d}}{{d{var.name}}}[{var.name}^n] = n {var.name}^{{n-1}}$."
            elif term.is_Mul:
                rule_name = "Constant Multiple & Product Rule"
                rule_note = "Constant factors are factored out; variable factors differentiated."
            elif any(term.has(fn) for fn in [sp.sin, sp.cos, sp.tan, sp.sinh, sp.cosh, sp.tanh]):
                rule_name = "Trigonometric & Chain Rule"
                rule_note = "Trigonometric derivative combined with inner derivative."
            elif any(term.has(fn) for fn in [sp.exp, sp.log]):
                rule_name = "Transcendental Exponential/Log Rule"
                rule_note = "Exponential/logarithmic derivative rule applied."
            elif term.has(sp.sqrt):
                rule_name = "Radical & Chain Rule"
                rule_note = "Square root power derivative $\\frac{1}{2\\sqrt{u}} u'$."
            else:
                rule_name = "Standard Differentiation Rule"
                rule_note = "Evaluated via symbolic chain rule."

            term_breakdowns.append({
                "term_latex": sp.latex(term),
                "diff_latex": sp.latex(d_term),
                "rule_name": rule_name,
                "rule_note": rule_note
            })

        steps.append({
            "title": "3. Term-by-Term Differentiation",
            "desc": "Individual derivatives computed for each summand:",
            "breakdowns": term_breakdowns
        })
    else:
        d_term = sp.diff(expr, var)
        steps.append({
            "title": "2. Direct Differentiation Rule",
            "desc": f"Applying standard differentiation rule to single monomial/term:",
            "latex": f"\\frac{{\\partial}}{{\\partial {var.name}}} \\left( {sp.latex(expr)} \\right) = {sp.latex(d_term)}"
        })

    final_diff = sp.simplify(sp.diff(expr, var))
    steps.append({
        "title": "4. Algebraic Consolidation & Final Form",
        "desc": "Combine all differentiated summands and algebraically simplify:",
        "latex": f"\\frac{{\\partial f}}{{\\partial {var.name}}} = {sp.latex(final_diff)}"
    })

    return steps


def find_critical_points(expr, symbols):
    """
    Finds stationary/critical points where grad(f) = 0.
    Classifies each point using the Hessian matrix eigenvalues or 2D determinant.
    """
    if not symbols:
        return []

    grad = [sp.diff(expr, s) for s in symbols]
    try:
        raw_sols = sp.solve(grad, symbols, dict=True)
    except Exception:
        return []

    crit_points = []
    hessian = sp.Matrix([[sp.diff(expr, vi, vj) for vj in symbols] for vi in symbols])

    for sol in raw_sols:
        # Check if coordinates are real
        is_real = True
        pt_eval = {}
        for s in symbols:
            val = sol.get(s)
            if val is None or not val.is_real:
                is_real = False
                break
            pt_eval[s] = val

        if not is_real:
            continue

        try:
            f_val = sp.simplify(expr.subs(pt_eval))
            f_float = float(f_val.evalf()) if f_val.is_real else None
            H_eval = hessian.subs(pt_eval)

            classification = "Inconclusive"
            badge_type = "status-inconclusive"

            if len(symbols) == 2:
                d_val = float(H_eval.det().evalf()) if H_eval.det().is_real else 0.0
                fxx = float(H_eval[0, 0].evalf()) if H_eval[0, 0].is_real else 0.0
                if d_val > 1e-6:
                    if fxx > 0:
                        classification = "Local Minimum (Valley)"
                        badge_type = "status-min"
                    else:
                        classification = "Local Maximum (Peak)"
                        badge_type = "status-max"
                elif d_val < -1e-6:
                    classification = "Saddle Point"
                    badge_type = "status-saddle"
                else:
                    classification = "Inconclusive (D = 0)"
            else:
                try:
                    eigs = [float(e.evalf()) for e in H_eval.eigenvals().keys() if e.is_real]
                    if len(eigs) == len(symbols):
                        if all(e > 1e-6 for e in eigs):
                            classification = "Local Minimum (Valley)"
                            badge_type = "status-min"
                        elif all(e < -1e-6 for e in eigs):
                            classification = "Local Maximum (Peak)"
                            badge_type = "status-max"
                        elif any(e > 1e-6 for e in eigs) and any(e < -1e-6 for e in eigs):
                            classification = "Saddle Point"
                            badge_type = "status-saddle"
                except Exception:
                    pass

            coords_float = {s.name: float(v.evalf()) for s, v in pt_eval.items()}
            coords_latex = ", ".join([f"{s.name} = {sp.latex(v)}" for s, v in pt_eval.items()])

            crit_points.append({
                "coords_exact": pt_eval,
                "coords_float": coords_float,
                "coords_latex": coords_latex,
                "f_exact": f_val,
                "f_latex": sp.latex(f_val),
                "f_float": f_float,
                "classification": classification,
                "badge": badge_type
            })
        except Exception:
            continue

    return crit_points


def compute_taylor_polynomial(expr, symbols, point_dict, order=2):
    """
    Computes multivariable Taylor series approximation around P0 up to order 2.
    T1 = Linear approximation (tangent plane)
    T2 = Quadratic osculating paraboloid approximation
    """
    sub_dict = {s: sp.sympify(point_dict.get(s.name, 0.0)) for s in symbols}
    try:
        f0 = expr.subs(sub_dict)

        # 1st Order (Linear)
        t1 = f0
        for s in symbols:
            d = sp.diff(expr, s).subs(sub_dict)
            t1 += d * (s - sub_dict[s])

        # 2nd Order (Quadratic)
        t2 = t1
        if order >= 2:
            for i, s1 in enumerate(symbols):
                for j, s2 in enumerate(symbols):
                    d2 = sp.diff(expr, s1, s2).subs(sub_dict)
                    t2 += sp.Rational(1, 2) * d2 * (s1 - sub_dict[s1]) * (s2 - sub_dict[s2])

        t1_simp = sp.simplify(t1)
        t2_simp = sp.simplify(t2)

        return {
            "f0": f0,
            "T1": t1_simp,
            "T1_latex": sp.latex(t1_simp),
            "T2": t2_simp,
            "T2_latex": sp.latex(t2_simp)
        }
    except Exception:
        return None


def compute_laplacian(expr, symbols):
    """
    Computes the scalar Laplacian: Delta(f) = sum(d^2 f / dx_i^2).
    Classifies harmonicity.
    """
    if not symbols:
        return None

    try:
        lap = sum(sp.diff(expr, s, 2) for s in symbols)
        lap_simp = sp.simplify(lap)
        is_harmonic = (lap_simp == 0)

        return {
            "laplacian": lap_simp,
            "latex": sp.latex(lap_simp),
            "is_harmonic": is_harmonic
        }
    except Exception:
        return None


def compute_vector_field_properties(component_exprs, symbols):
    """
    For a vector field F = <F_x, F_y, F_z>, computes:
    - Divergence: div(F) = dF_x/dx + dF_y/dy + dF_z/dz
    - Curl: curl(F) = det([i, j, k; d/dx, d/dy, d/dz; F_x, F_y, F_z])
    """
    if len(symbols) < 2 or len(component_exprs) < 2:
        return None

    try:
        div = 0
        for i, s in enumerate(symbols[:len(component_exprs)]):
            div += sp.diff(component_exprs[i], s)
        div = sp.simplify(div)

        curl_components = []
        if len(symbols) >= 3 and len(component_exprs) >= 3:
            x, y, z = symbols[0], symbols[1], symbols[2]
            fx, fy, fz = component_exprs[0], component_exprs[1], component_exprs[2]
            curl_x = sp.simplify(sp.diff(fz, y) - sp.diff(fy, z))
            curl_y = sp.simplify(sp.diff(fx, z) - sp.diff(fz, x))
            curl_z = sp.simplify(sp.diff(fy, x) - sp.diff(fx, y))
            curl_components = [curl_x, curl_y, curl_z]
        elif len(symbols) == 2 and len(component_exprs) == 2:
            x, y = symbols[0], symbols[1]
            fx, fy = component_exprs[0], component_exprs[1]
            curl_z = sp.simplify(sp.diff(fy, x) - sp.diff(fx, y))
            curl_components = [sp.Integer(0), sp.Integer(0), curl_z]

        is_solenoidal = (div == 0)
        is_irrotational = all(c == 0 for c in curl_components) if curl_components else False

        return {
            "divergence": div,
            "div_latex": sp.latex(div),
            "curl": curl_components,
            "curl_latex": [sp.latex(c) for c in curl_components],
            "is_solenoidal": is_solenoidal,
            "is_irrotational": is_irrotational
        }
    except Exception:
        return None


def solve_lagrange_multipliers(obj_expr, constr_expr, constr_val, symbols):
    """
    Solves constrained optimization:
    Maximize / Minimize f(x1, ...) subject to g(x1, ...) = constr_val
    Sets up grad(f) = lambda * grad(g) and g = constr_val
    """
    if not symbols:
        return []

    try:
        lam = sp.Symbol("lambda")
        g = constr_expr - constr_val
        eqs = [sp.diff(obj_expr, s) - lam * sp.diff(g, s) for s in symbols] + [g]
        sols = sp.solve(eqs, symbols + [lam], dict=True)

        results = []
        for sol in sols:
            pt = {s: sol[s] for s in symbols if s in sol}
            if len(pt) == len(symbols) and all(v.is_real for v in pt.values()):
                f_val = sp.simplify(obj_expr.subs(pt))
                results.append({
                    "point": {s.name: float(v.evalf()) for s, v in pt.items()},
                    "point_latex": ", ".join([f"{s.name} = {sp.latex(v)}" for s, v in pt.items()]),
                    "lambda": float(sol[lam].evalf()) if sol.get(lam) and sol[lam].is_real else None,
                    "f_val": float(f_val.evalf()) if f_val.is_real else None,
                    "f_latex": sp.latex(f_val)
                })
        # Sort by function value
        results.sort(key=lambda r: r["f_val"] if r["f_val"] is not None else 0.0)
        return results
    except Exception:
        return []


def generate_latex_report(expr, symbols, partials, grad_latex, hessian_data, point_dict, eval_res, critical_points=None, laplacian_data=None):
    """
    Generates a publication-ready LaTeX snippet documenting all calculus derivations.
    """
    var_str = ", ".join([s.name for s in symbols])
    pt_str = ", ".join([f"{k}={v}" for k, v in point_dict.items()])

    lines = [
        r"% ========================================================================",
        r"% DerivX 4D - Multivariable Symbolic Calculus Derivation Report",
        r"% ========================================================================",
        r"\section*{Multivariable Calculus Analysis}",
        r"\subsection*{1. Target Function}",
        f"\\[ f({var_str}) = {sp.latex(expr)} \\]",
        r"\subsection*{2. First-Order Partial Derivatives}",
        r"\begin{align*}"
    ]

    for p in partials:
        lines.append(f"  {p['diff_latex']} &= {p['latex']} \\\\")
    lines.append(r"\end{align*}")

    lines.extend([
        r"\subsection*{3. Gradient Vector Field}",
        f"\\[ {grad_latex} \\]",
        r"\subsection*{4. Second-Order Hessian Matrix}",
        f"\\[ H(f) = {hessian_data['matrix_latex']} \\]",
        f"\\[ \\det(H) = {hessian_data['determinant_latex']} \\]"
    ])

    if laplacian_data:
        lines.extend([
            r"\subsection*{5. Scalar Laplacian ($\Delta f$)}",
            f"\\[ \\nabla^2 f = \\Delta f = {laplacian_data['latex']} \\]"
        ])

    lines.extend([
        r"\subsection*{6. Coordinate Evaluation at $P_0$}",
        f"Evaluated at point $P_0({pt_str})$:",
        r"\begin{itemize}",
        f"  \\item Value: $f(P_0) \\approx {eval_res['f_float']:.4f}$",
        r"\end{itemize}"
    ])

    if critical_points:
        lines.extend([
            r"\subsection*{7. Critical Points & Local Extrema}",
            r"\begin{itemize}"
        ])
        for cp in critical_points:
            lines.append(f"  \\item Point $({cp['coords_latex']})$: \\textbf{{{cp['classification']}}}, $f = {cp['f_latex']}$")
        lines.append(r"\end{itemize}")

    return "\n".join(lines)


def generate_python_code(expr, symbols):
    """
    Generates a self-contained, vectorized Python script using NumPy / SymPy.
    """
    var_names = [s.name for s in symbols]
    args_str = ", ".join(var_names)

    code = f'''"""
Generated by DerivX 4D Engine
Vectorized Multivariable Function & Gradient Implementation
"""
import numpy as np
import sympy as sp

# Symbolic Definition
{args_str} = sp.symbols('{args_str}')
f_expr = {repr(str(expr))}
f_sym = sp.sympify(f_expr)

# Symbolic Partial Derivatives
partials = [sp.diff(f_sym, var) for var in [{args_str}]]
print("Symbolic Partials:", partials)

# Vectorized Fast Numerical Function (NumPy)
f_numpy = sp.lambdify([{args_str}], f_sym, modules=["numpy"])
grad_numpy = [sp.lambdify([{args_str}], p, modules=["numpy"]) for p in partials]

def evaluate_at(coords):
    """
    Evaluates function and gradient at given coordinates tuple.
    """
    val = f_numpy(*coords)
    grad = [g(*coords) for g in grad_numpy]
    return val, grad

# Sample Evaluation
test_pt = tuple([1.0] * len([{args_str}]))
val, grad = evaluate_at(test_pt)
print(f"Evaluation at {{test_pt}}:")
print(f"  f({{test_pt}}) = {{val}}")
print(f"  grad({{test_pt}}) = {{grad}}")
'''
    return code


def parse_natural_language_query(query_str):
    """
    Extracts mathematical expression and intent from natural language prompts:
    e.g. "derivative of x^2 - y^2", "critical points of x^3 - 3*x*y", "gradient of sin(x)*cos(y)"
    """
    q = query_str.strip()
    if not q:
        return None, "empty"

    lower_q = q.lower()
    intent = "derivative"
    if "critical" in lower_q or "extrema" in lower_q or "optimize" in lower_q or "min" in lower_q or "max" in lower_q:
        intent = "critical_points"
    elif "taylor" in lower_q or "approx" in lower_q:
        intent = "taylor"
    elif "laplac" in lower_q or "curl" in lower_q or "div" in lower_q:
        intent = "vector"
    elif "lagrange" in lower_q or "constrain" in lower_q:
        intent = "lagrange"

    # Remove common conversational prefixes
    patterns = [
        r"^(?:calculate|find|solve|compute|get|show|give me)?\s*(?:the)?\s*(?:partial)?\s*(?:derivative|derivatives|gradient|hessian|critical points|extrema|laplacian)?\s*(?:of|for)?\s*",
        r"^(?:differentiate)?\s*",
        r"^(?:what is the derivative of)?\s*"
    ]
    cleaned_formula = q
    for pat in patterns:
        cleaned_formula = re.sub(pat, "", cleaned_formula, flags=re.IGNORECASE)

    # Remove "with respect to ..." trailer
    cleaned_formula = re.sub(r"\s+with\s+respect\s+to\s+.*$", "", cleaned_formula, flags=re.IGNORECASE)
    cleaned_formula = re.sub(r"\s+at\s+point\s+.*$", "", cleaned_formula, flags=re.IGNORECASE)

    return cleaned_formula.strip(), intent

