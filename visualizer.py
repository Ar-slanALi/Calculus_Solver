"""
Visualization Engine for Multivariable Calculus
Supports 1D, 2D, 3D, and 4D functions using Plotly for high-interactivity.
"""

import numpy as np
import plotly.graph_objects as go
import sympy as sp


def get_lambda_func(expr, symbols):
    """
    Safely converts a SymPy expression into a fast vectorized NumPy function.
    """
    try:
        f = sp.lambdify(symbols, expr, modules=["numpy"])
        return f
    except Exception:
        # Fallback to standard math
        return sp.lambdify(symbols, expr)


def safe_eval(f, *grid_args):
    """
    Evaluates vectorized function and handles singularities (NaN, Inf, ZeroDivision).
    """
    try:
        z = f(*grid_args)
        if np.isscalar(z):
            z = np.full(grid_args[0].shape, float(z), dtype=float)
        else:
            z = np.array(z, dtype=float)
            z = np.where(np.isinf(z) | np.isnan(z), np.nan, z)
            # Clip extreme values for visual stability
            valid = z[~np.isnan(z)]
            if len(valid) > 0:
                median_abs = np.median(np.abs(valid))
                if median_abs > 0:
                    limit = 50 * median_abs
                    z = np.clip(z, -limit, limit)
        return z
    except Exception:
        return np.full(grid_args[0].shape, np.nan)


def plot_1d_function(expr, symbol, x_range=(-5, 5), num_points=200, point_val=None):
    """
    Generates interactive 2D line plot for f(x), its derivative f'(x), and tangent line.
    """
    f_num = get_lambda_func(expr, [symbol])
    d_expr = sp.simplify(sp.diff(expr, symbol))
    df_num = get_lambda_func(d_expr, [symbol])

    x_vals = np.linspace(x_range[0], x_range[1], num_points)
    y_vals = safe_eval(f_num, x_vals)
    dy_vals = safe_eval(df_num, x_vals)

    fig = go.Figure()

    # Function plot
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="lines",
        name=f"f({symbol.name})",
        line=dict(color="#10B981", width=3.5)
    ))

    # Derivative plot
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=dy_vals,
        mode="lines",
        name=f"f'({symbol.name})",
        line=dict(color="#DC2626", width=2, dash="dash")
    ))

    # Tangent line at given point
    if point_val is not None:
        try:
            x0 = float(point_val)
            y0_raw = f_num(x0)
            m_raw = df_num(x0)
            if np.isreal(y0_raw) and np.isreal(m_raw):
                y0 = float(y0_raw)
                m = float(m_raw)

                # Tangent segment around x0
                delta = (x_range[1] - x_range[0]) * 0.25
                xt = np.linspace(x0 - delta, x0 + delta, 50)
                yt = y0 + m * (xt - x0)

                fig.add_trace(go.Scatter(
                    x=xt,
                    y=yt,
                    mode="lines",
                    name=f"Tangent at {symbol.name}={x0:.2f} (slope={m:.2f})",
                    line=dict(color="#10B981", width=2.5, dash="dot")
                ))

                fig.add_trace(go.Scatter(
                    x=[x0],
                    y=[y0],
                    mode="markers",
                    name=f"Point ({x0:.2f}, {y0:.2f})",
                    marker=dict(size=10, color="#10B981", symbol="circle")
                ))
        except Exception:
            pass

    fig.update_layout(
        title=f"Plot of f({symbol.name}) & its Derivative",
        xaxis_title=f"{symbol.name}",
        yaxis_title="Output Value",
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def plot_2d_surface(expr, symbols, ranges, num_points=60, point_dict=None, show_tangent=True, critical_points=None, show_taylor=False, taylor_expr=None):
    """
    Generates interactive 3D Surface Plot for f(x, y).
    Includes optional Tangent Plane, Evaluation Point, Critical Points, and 2nd-Order Taylor Paraboloid.
    """
    x_sym, y_sym = symbols[0], symbols[1]
    f_num = get_lambda_func(expr, [x_sym, y_sym])

    x_min, x_max = ranges.get(x_sym.name, (-5, 5))
    y_min, y_max = ranges.get(y_sym.name, (-5, 5))

    x_vals = np.linspace(x_min, x_max, num_points)
    y_vals = np.linspace(y_min, y_max, num_points)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = safe_eval(f_num, X, Y)

    fig = go.Figure()

    # Main 3D Surface
    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale="Viridis",
        name=f"f({x_sym.name}, {y_sym.name})",
        contours={
            "z": {
                "show": True,
                "usecolormap": True,
                "project": {"z": True}
            }
        },
        colorbar=dict(title=f"f({x_sym.name},{y_sym.name})", len=0.75)
    ))

    # Optional 2nd-Order Taylor Surface Overlay
    if show_taylor and taylor_expr is not None:
        try:
            taylor_num = get_lambda_func(taylor_expr, [x_sym, y_sym])
            Z_taylor = safe_eval(taylor_num, X, Y)
            fig.add_trace(go.Surface(
                x=X,
                y=Y,
                z=Z_taylor,
                colorscale=[[0, "rgba(59, 130, 246, 0.45)"], [1, "rgba(59, 130, 246, 0.45)"]],
                showscale=False,
                name="2nd-Order Taylor Paraboloid",
                opacity=0.65
            ))
        except Exception:
            pass

    # Point & Tangent plane
    if point_dict and x_sym.name in point_dict and y_sym.name in point_dict:
        try:
            x0 = float(point_dict[x_sym.name])
            y0 = float(point_dict[y_sym.name])
            z0_val = expr.subs({x_sym: x0, y_sym: y0}).evalf()

            if z0_val.is_real:
                z0 = float(z0_val)

                # Marker at point
                fig.add_trace(go.Scatter3d(
                    x=[x0],
                    y=[y0],
                    z=[z0],
                    mode="markers",
                    marker=dict(size=8, color="#EF4444", symbol="diamond"),
                    name=f"Point ({x0:.2f}, {y0:.2f}, {z0:.2f})"
                ))

                # Tangent plane
                if show_tangent:
                    fx_eval = sp.diff(expr, x_sym).subs({x_sym: x0, y_sym: y0}).evalf()
                    fy_eval = sp.diff(expr, y_sym).subs({x_sym: x0, y_sym: y0}).evalf()

                    if fx_eval.is_real and fy_eval.is_real:
                        fx0 = float(fx_eval)
                        fy0 = float(fy_eval)

                        # Plane span
                        dx = (x_max - x_min) * 0.25
                        dy = (y_max - y_min) * 0.25
                        xp = np.linspace(x0 - dx, x0 + dx, 15)
                        yp = np.linspace(y0 - dy, y0 + dy, 15)
                        XP, YP = np.meshgrid(xp, yp)
                        ZP = z0 + fx0 * (XP - x0) + fy0 * (YP - y0)

                        fig.add_trace(go.Surface(
                            x=XP,
                            y=YP,
                            z=ZP,
                            colorscale=[[0, "rgba(239, 68, 68, 0.6)"], [1, "rgba(239, 68, 68, 0.6)"]],
                            showscale=False,
                            name="Tangent Plane"
                        ))
        except Exception:
            pass

    # Critical Points Markers
    if critical_points:
        for cp in critical_points:
            coords = cp.get("coords_float", {})
            if x_sym.name in coords and y_sym.name in coords and cp.get("f_float") is not None:
                c_x = coords[x_sym.name]
                c_y = coords[y_sym.name]
                c_z = cp["f_float"]

                # Only plot if within visible or reasonable boundary
                if (x_min - 2 <= c_x <= x_max + 2) and (y_min - 2 <= c_y <= y_max + 2):
                    classification = cp.get("classification", "Critical Point")
                    if "Min" in classification:
                        cp_color = "#10B981"  # Emerald
                        cp_sym = "circle"
                    elif "Max" in classification:
                        cp_color = "#EF4444"  # Red
                        cp_sym = "diamond"
                    else:
                        cp_color = "#818CF8"  # Royal Electric Indigo (Saddle Point)
                        cp_sym = "cross"

                    fig.add_trace(go.Scatter3d(
                        x=[c_x],
                        y=[c_y],
                        z=[c_z],
                        mode="markers",
                        marker=dict(size=10, color=cp_color, symbol=cp_sym, line=dict(color="#ffffff", width=2)),
                        name=f"{classification} ({c_x:.2f}, {c_y:.2f})",
                        hovertext=f"<b>{classification}</b><br>({c_x:.3f}, {c_y:.3f}, z={c_z:.3f})"
                    ))

    fig.update_layout(
        title=f"3D Surface Plot: f({x_sym.name}, {y_sym.name})",
        scene=dict(
            xaxis_title=f"{x_sym.name}",
            yaxis_title=f"{y_sym.name}",
            zaxis_title=f"f({x_sym.name}, {y_sym.name})",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2))
        ),
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


def plot_2d_contour(expr, symbols, ranges, num_points=80):
    """
    Generates 2D Contour Plot with Level Curves.
    """
    x_sym, y_sym = symbols[0], symbols[1]
    f_num = get_lambda_func(expr, [x_sym, y_sym])

    x_min, x_max = ranges.get(x_sym.name, (-5, 5))
    y_min, y_max = ranges.get(y_sym.name, (-5, 5))

    x_vals = np.linspace(x_min, x_max, num_points)
    y_vals = np.linspace(y_min, y_max, num_points)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = safe_eval(f_num, X, Y)

    fig = go.Figure(data=go.Contour(
        z=Z,
        x=x_vals,
        y=y_vals,
        colorscale="Viridis",
        contours=dict(
            coloring="heatmap",
            showlabels=True,
            labelfont=dict(size=11, color="white")
        ),
        colorbar=dict(title="Value")
    ))

    fig.update_layout(
        title=f"2D Contour Map (Level Curves): f({x_sym.name}, {y_sym.name})",
        xaxis_title=f"{x_sym.name}",
        yaxis_title=f"{y_sym.name}",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


def plot_multivariable_slice(expr, symbols, active_vars, fixed_values, ranges, num_points=50):
    """
    Visualizes 3-variable or 4-variable functions by dynamic slicing:
    User chooses 2 active variables for the 3D surface, while remaining 1 or 2 variables
    are held fixed at slider values.
    """
    substitutions = {}
    for sym in symbols:
        if sym.name not in [v.name for v in active_vars]:
            val = fixed_values.get(sym.name, 0.0)
            substitutions[sym] = sp.sympify(val)

    # Partially substitute fixed variables to obtain a 2-variable slice expression
    slice_expr = sp.simplify(expr.subs(substitutions))

    fig = plot_2d_surface(
        expr=slice_expr,
        symbols=active_vars,
        ranges=ranges,
        num_points=num_points,
        point_dict=None,
        show_tangent=False
    )

    fixed_info = ", ".join([f"{k}={v:.2f}" for k, v in fixed_values.items() if k not in [v.name for v in active_vars]])
    fig.update_layout(
        title=f"3D Slice: f({active_vars[0].name}, {active_vars[1].name}) at [{fixed_info}]"
    )

    return fig, slice_expr


def plot_empty_3d_grid():
    """
    Renders an elegant interactive 3D coordinate frame placeholder
    when no mathematical function is entered yet.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(size=4, color="#10B981"),
        name="Origin (0,0,0)",
        hoverinfo="name"
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-4, 4], title="X", gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(16,185,129,0.4)"),
            yaxis=dict(range=[-4, 4], title="Y", gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(16,185,129,0.4)"),
            zaxis=dict(range=[-4, 4], title="Z", gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(16,185,129,0.4)"),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2))
        ),
        template="plotly_dark",
        margin=dict(l=10, r=10, t=30, b=10),
        annotations=[
            dict(
                text="Awaiting function input to render 3D manifold",
                xref="paper", yref="paper",
                x=0.5, y=0.05,
                showarrow=False,
                font=dict(size=13, color="rgba(148, 163, 184, 0.8)")
            )
        ]
    )
    return fig

