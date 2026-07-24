from numpy import (
    array, ascontiguousarray, pi, empty, zeros, 
    sin, cos, isnan, isinf, bool_, float64, 
    int64, complex128, any, ndim
)
from numba import njit, prange, void

# ====================================================================
# 1 CODE SINGULARITY PATCHING ENGINE (STRICT INT64 JIT TYPING)
# ====================================================================
@njit(
    void(
        complex128[:], int64[:], float64[:], complex128[:],
        float64, float64, float64, float64, float64
    ),
    fastmath=True, 
    cache=True,
    locals={
        'num_elements': int64,
        'i': int64,
        'ni': int64,
        'wi': float64,
        'abs_wi': float64,
        'gp': complex128,
        'dw': float64,
        'sinc_dw': float64,
        'half_phase': complex128,
        'I0': complex128,
        'I1': complex128,
        'I2': complex128
    }
)
def _numba_patch_singularities(
    out, 
    n_arr, 
    w, 
    global_phase, 
    L, 
    c0, 
    c1, 
    c2, 
    abs_L
):
    """
    Out-of-line patch router for rare mathematical singularities
    Keeps branches out of the hot loop to maintain SIMD throughput
    """
    num_elements = out.size
    for i in range(num_elements):
        ni = n_arr[i]
        wi = w[i]
        abs_wi = abs(wi)
        gp = global_phase[i]

        # --- CASE 1: Near-Zero Frequency or Micro-Interval Limit ---
        if abs_wi < 1e-8 or abs_L < 1e-7:
            I0 = (1.0 - (wi**2) / 6.0) - 0.5j * wi
            I1 = -1j * wi / (pi**2)
            I2 = complex(2.0 / pi, 0.0)
            out[i] = L * gp * ((c0 * I0) - (c1 * I1) + (c2 * I2))

        # --- CASE 2: Positive Resonance Pole (w is near pi) ---
        elif abs(wi - pi) < 1e-7:
            dw = wi - pi
            if dw == 0.0:
                I1 = complex(0.5, 0.0)
                I2 = complex(0.0, -0.5)
                I0 = complex(0.0, 0.0)
            else:
                sinc_dw = sin(dw) / dw
                half_phase = cos(-0.5 * dw) + 1j * sin(-0.5 * dw)
                I1 = 0.5 * sinc_dw * half_phase
                I2 = -0.5j * sinc_dw * half_phase
                I0 = (sin(wi) / wi) * (
                    cos(-0.5 * wi) + 1j * sin(-0.5 * wi)
                )
            out[i] = L * gp * ((c0 * I0) - (c1 * I1) + (c2 * I2))

        # --- CASE 3: Negative Resonance Pole (w is near -pi) ---
        elif abs(wi + pi) < 1e-7:
            dw = wi + pi
            if dw == 0.0:
                I1 = complex(0.5, 0.0)
                I2 = complex(0.0, 0.5)
                I0 = complex(0.0, 0.0)
            else:
                sinc_dw = sin(dw) / dw
                half_phase = cos(-0.5 * dw) + 1j * sin(-0.5 * dw)
                I1 = 0.5 * sinc_dw * half_phase
                I2 = 0.5j * sinc_dw * half_phase
                I0 = (sin(wi) / wi) * (
                    cos(-0.5 * wi) + 1j * sin(-0.5 * wi)
                )
            out[i] = L * gp * ((c0 * I0) - (c1 * I1) + (c2 * I2))


# ====================================================================
# 2 PARALLEL VECTOR HOT LOOP ENGINE (STRICT INT64 JIT TYPING)
# ====================================================================
@njit(
    complex128[:](int64[:], float64, float64, float64, float64, float64),
    parallel=True, 
    fastmath=True, 
    cache=True,
    locals={
        'num_elements': int64,
        'L': float64,
        'abs_L': float64,
        'c0': float64,
        'c1': float64,
        'c2': float64,
        'two_pi_L': float64,
        'two_pi_x0': float64,
        'i': int64,
        'ni': int64,
        'wi': float64,
        'phase_angle': float64,
        'gp': complex128,
        'exp_w': complex128,
        'denom': float64,
        'I0': complex128,
        'I1': complex128,
        'I2': complex128
    }
)
def _numba_integrate_cpu_engine(
    n_arr, 
    x0, 
    x1, 
    y0, 
    y1, 
    k
):
    """
    Multi-threaded SIMD CPU engine running under strict compile-time
    int64 type constraints Exclusively processes 1D arrays
    """
    num_elements = n_arr.size
    out = empty(num_elements, dtype=complex128)

    # --- SCALAR ACCURACY GUARDS ---
    if (isnan(x0) or isnan(x1) or isnan(y0) or isnan(y1) or
        isnan(k) or isinf(x0) or isinf(x1) or isinf(y0) or
        isinf(y1) or isinf(k)):
        out.fill(complex(float('nan'), float('nan')))
        return out

    L = x1 - x0
    abs_L = abs(L)

    if L == 0.0:
        out.fill(complex(0.0, 0.0))
        return out

    c0 = 0.5 * (y0 + y1)
    c1 = 0.5 * (y1 - y0)
    c2 = k * (y1 - y0)

    two_pi_L = 2.0 * pi * L
    two_pi_x0 = 2.0 * pi * x0

    # Statically typed temporary execution arrays
    w = empty(num_elements, dtype=float64)
    global_phase = empty(num_elements, dtype=complex128)
    needs_patching = zeros(num_elements, dtype=bool_)

    # --- STRIDE 1: Unrolled Parallel SIMD Loop ---
    for i in prange(num_elements):
        ni = n_arr[i]
        
        wi = two_pi_L * float(ni)
        w[i] = wi
        
        phase_angle = -two_pi_x0 * float(ni)
        gp = cos(phase_angle) + 1j * sin(phase_angle)
        global_phase[i] = gp

        # Identify structural instabilities to handle out-of-line
        if (abs(wi) < 1e-8 or abs_L < 1e-7 or 
            abs(abs(wi) - pi) < 1e-7):
            needs_patching[i] = True
            out[i] = complex(0.0, 0.0)
        else:
            # Branch-free continuous hardware execution path
            exp_w = cos(-wi) + 1j * sin(-wi)
            denom = wi**2 - pi**2
            
            I0 = (sin(wi) / wi) * (
                cos(-0.5 * wi) + 1j * sin(-0.5 * wi)
            )
            I1 = -1j * wi * (exp_w + 1.0) / denom
            I2 = -pi * (exp_w + 1.0) / denom

            out[i] = L * gp * ((c0 * I0) - (c1 * I1) + (c2 * I2))

    # --- STRIDE 2: Singular Matrix Patches ---
    if any(needs_patching):
        _numba_patch_singularities(
            out, n_arr, w, global_phase, L, c0, c1, c2, abs_L
        )

    return out


# ====================================================================
# 3 PYTHON WRAPPER INTERFACE & SPECIALIZATIONS
# ====================================================================
def integrate_h_exponential_final(
    n, 
    x0, 
    x1, 
    y0, 
    y1, 
    k
):
    """
    Python array wrapper interface Exclusively accepts array-like inputs 
    and guarantees a 1D contiguous complex128 NumPy array response
    """
    # Enforce C-contiguous integer memory layout
    n_arr = ascontiguousarray(n, dtype=int64)
    
    # Direct handover to vector engine
    return _numba_integrate_cpu_engine(
        n_arr, float(x0), float(x1), float(y0), float(y1), float(k)
    )


def integrate_h_exponential_n0(
    x0, x1, y0, y1, k
):
    """Computes exact analytical integral for n = 0 as a real float"""
    n_arr = array([0], dtype=int64)
    res = _numba_integrate_cpu_engine(
        n_arr, float(x0), float(x1), float(y0), float(y1), float(k)
    )
    scalar_res = res[0]
    return float(scalar_res.real)


def integrate_h_exponential_n1(
    x0, x1, y0, y1, k
):
    """Computes exact analytical integral for n = 1 as a scalar"""
    n_arr = array([1], dtype=int64)
    res = _numba_integrate_cpu_engine(
        n_arr, float(x0), float(x1), float(y0), float(y1), float(k)
    )
    return complex(res[0])
