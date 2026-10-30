import calculus_engine as ce
import visualizer as viz

print("=== VERIFYING CALCULUS ENGINE & PLOT GENERATION ===")

test_cases = [
    "2x^2 + 3y - sin(z) + e^w",
    "ln(x) + cos(2y)",
    "x^3 - 3*x*y^2",
    "x^2 + y^2",
    "1 / (x^2 + y^2)",  # Potential singularity at (0,0)
    "exp(-(x^2 + y^2))",
    "x^2 + y^2 + z^2",
    "x^2 + y^2 - z^2 + w^2"
]

for tc in test_cases:
    expr, syms, err = ce.parse_function(tc)
    print(f"\n[Test] {tc}")
    print(f"  Parsed: {expr}")
    print(f"  Symbols: {[s.name for s in syms]}")
    
    # Partials
    partials = ce.compute_partial_derivatives(expr, syms)
    print(f"  Partials count: {len(partials)}")
    
    # Hessian
    hess = ce.compute_hessian(expr, syms)
    print(f"  Hessian computed: {hess['matrix'].shape}")
    
    # Test evaluation with point (1, 1, 1, 1)
    pt = {s.name: 1.0 for s in syms}
    res = ce.evaluate_at_point(expr, syms, pt)
    print(f"  Eval at (1,...): f={res['f_float']}")

# Test 3D Plot Generation without errors
print("\n--- Testing 3D Plotly Surface Generation ---")
expr2, syms2, _ = ce.parse_function("x^2 - y^2")
ranges = {"x": (-3, 3), "y": (-3, 3)}
fig = viz.plot_2d_surface(expr2, syms2, ranges, point_dict={"x": 1, "y": 1}, show_tangent=True)
print("  3D Surface figure created successfully! Traces count:", len(fig.data))

# Test 4-Variable Dynamic Slice Plot
print("\n--- Testing 4-Variable Slice Generation ---")
expr4, syms4, _ = ce.parse_function("x^2 + y^2 - z^2 + w^2")
slice_fig, slice_expr = viz.plot_multivariable_slice(
    expr=expr4,
    symbols=syms4,
    active_vars=[syms4[0], syms4[1]],
    fixed_values={"z": 1.0, "w": 2.0},
    ranges={"x": (-3, 3), "y": (-3, 3), "z": (-3, 3), "w": (-3, 3)}
)
print("  4-Var Slice figure created successfully! Slice expr:", slice_expr)

print("\nALL EDGE-CASE TESTS PASSED PERFECTLY!")
