import numpy as np
from numpy import isfinite, empty, exp, sign, float64, complex128, pi
from numba import njit, prange
from typing import Final

# --- STRUCTURAL AND GEOMETRIC CONSTANTS ---
TWO_PI: Final[float] = 6.283185307179586
PI_SQUARED: Final[float] = 9.869604401089358
CORE_SINE_AREA_FACTOR: Final[float] = 0.20264236728467558
RESONANCE_SCALE_FACTOR: Final[complex] = -0.15915494309189535j
TAYLOR_TRIG_BASE: Final[float] = 0.20264236728467558
TAYLOR_TRIG_ORDER2: Final[float] = 0.02053229
UNDERFLOW_THRESHOLD: Final[float] = 1e-15
RESONANCE_NEARNESS_THRESHOLD: Final[float] = 1e-4


@njit(float64(float64, float64, float64), inline="always")
def _clamp_njit(val: float, low: float, high: float) -> float:
    """Fast inline primitive clamp."""
    if val < low:
        return low
    if val > high:
        return high
    return val


@njit(float64(float64, float64, float64, float64, float64, float64))
def compute_h_integral_n0(
    x0: float, x1: float, y0: float, y1: float, u: float, v: float
) -> float:
    """
    Highly optimized scalar calculator for n=0 (Total Area).
    """
    if not (
        isfinite(x0)
        and isfinite(x1)
        and isfinite(y0)
        and isfinite(y1)
        and isfinite(u)
        and isfinite(v)
    ):
        return 0.0

    dx: float = x1 - x0
    if abs(dx) < UNDERFLOW_THRESHOLD:
        return 0.0

    u_c: float = _clamp_njit(u, 0.0, 1.0)
    v_c: float = _clamp_njit(v, -1.0, 1.0)

    return dx * (
        y0 + (y1 - y0) * (0.5 + CORE_SINE_AREA_FACTOR * u_c * v_c)
    )


@njit(complex128(float64, float64, float64, float64, float64, float64))
def compute_h_integral_n1(
    x0: float, x1: float, y0: float, y1: float, u: float, v: float
) -> complex:
    """
    Highly optimized scalar calculator for n=1 (Fundamental Harmonic).
    """
    if not (
        isfinite(x0)
        and isfinite(x1)
        and isfinite(y0)
        and isfinite(y1)
        and isfinite(u)
        and isfinite(v)
    ):
        return 0.0j

    dx: float = x1 - x0
    if abs(dx) < UNDERFLOW_THRESHOLD:
        return 0.0j

    dy: float = y1 - y0
    u_c: float = _clamp_njit(u, 0.0, 1.0)
    v_c: float = _clamp_njit(v, -1.0, 1.0)

    const_weight: float = y0 + 0.5 * (1.0 - u_c) * dy
    linear_weight: float = u_c * dy
    uv: float = u_c * v_c

    omega: float = TWO_PI * dx
    abs_omega: float = abs(omega)
    phase_shift: complex = exp(-1j * TWO_PI * x0)
    exp_w: complex = exp(-1j * omega)

    if abs_omega < RESONANCE_NEARNESS_THRESHOLD:
        om2: float = omega * omega
        int_c = 1.0 - 0.5j * omega - (1.0 / 6.0) * om2
        int_l = 0.5 - (1.0j / 3.0) * omega - (1.0 / 8.0) * om2
        int_t = (TAYLOR_TRIG_BASE + TAYLOR_TRIG_ORDER2 * om2) * (
            uv - 1j * omega * (0.5 - 0.5 * u_c)
        )
        return dx * phase_shift * (
            const_weight * int_c + linear_weight * int_l + dy * int_t
        )

    if abs(pi - abs_omega) < RESONANCE_NEARNESS_THRESHOLD:
        inv_om: float = 1.0 / omega
        int_c = (exp_w - 1.0) * (1j * inv_om)
        int_l = (exp_w * (-1j * omega - 1.0) + 1.0) * (-(inv_om * inv_om))
        epsilon: float = omega - sign(omega) * pi
        factor_trig = (RESONANCE_SCALE_FACTOR * sign(omega)) * (
            1.0 - 0.5j * epsilon - (1.0 / 6.0) * (epsilon * epsilon)
        )
        int_t = factor_trig * (uv - 1j * omega * (0.5 - 0.5 * u_c))
        return dx * phase_shift * (
            const_weight * int_c + linear_weight * int_l + dy * int_t
        )

    om2 = omega * omega
    inv_om = 1.0 / omega
    int_c = (exp_w - 1.0) * (1j * inv_om)
    int_l = (exp_w * (-1j * omega - 1.0) + 1.0) * (-(inv_om * inv_om))
    int_t = ((1.0 + exp_w) / (PI_SQUARED - om2)) * (
        uv - 1j * omega * (0.5 - 0.5 * u_c)
    )

    return dx * phase_shift * (
        const_weight * int_c + linear_weight * int_l + dy * int_t
    )


@njit(
    "complex128[:](float64[:], float64, float64, "
    "float64, float64, float64, float64)",
    parallel=True,
    fastmath=True,
)
def compute_h_integral_numba(
    n_array: np.ndarray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    u: float,
    v: float,
) -> np.ndarray:
    """
    Fastest Parallel SIMD Vectorized Fourier Integration Array Engine.
    Uses uniform vector divergence branching to bypass dead math pipelines.
    """
    n: np.ndarray = np.asarray(n_array, dtype=float64)
    size: int = n.size
    result: np.ndarray = empty(size, dtype=complex128)

    if not (
        isfinite(x0)
        and isfinite(x1)
        and isfinite(y0)
        and isfinite(y1)
        and isfinite(u)
        and isfinite(v)
    ):
        result.fill(0.0j)
        return result

    dx: float = x1 - x0
    if abs(dx) < UNDERFLOW_THRESHOLD:
        result.fill(0.0j)
        return result

    dy: float = y1 - y0
    u_c: float = _clamp_njit(u, 0.0, 1.0)
    v_c: float = _clamp_njit(v, -1.0, 1.0)

    const_weight: float = y0 + 0.5 * (1.0 - u_c) * dy
    linear_weight: float = u_c * dy
    uv: float = u_c * v_c

    two_pi_dx: float = TWO_PI * dx
    two_pi_x0: float = TWO_PI * x0

    for i in prange(size):
        n_val: float = n[i]

        if n_val == 0.0:
            result[i] = dx * (y0 + dy * (0.5 + CORE_SINE_AREA_FACTOR * uv))
            continue

        omega: float = n_val * two_pi_dx
        abs_omega: float = abs(omega)

        phase_shift: complex = exp(-1j * two_pi_x0 * n_val)
        exp_neg_i_omega: complex = exp(-1j * omega)

        if (
            abs_omega >= RESONANCE_NEARNESS_THRESHOLD
            and abs(pi - abs_omega) >= RESONANCE_NEARNESS_THRESHOLD
        ):
            om2: float = omega * omega
            inv_om: float = 1.0 / omega
            int_c: complex = (exp_neg_i_omega - 1.0) * (1j * inv_om)
            int_l: complex = (
                exp_neg_i_omega * (-1j * omega - 1.0) + 1.0
            ) * (-(inv_om * inv_om))
            int_t: complex = (
                (1.0 + exp_neg_i_omega) / (PI_SQUARED - om2)
            ) * (uv - 1j * omega * (0.5 - 0.5 * u_c))
        else:
            om2_a: float = omega * omega
            int_c_a: complex = 1.0 - 0.5j * omega - (1.0 / 6.0) * om2_a
            int_l_a: complex = 0.5 - (1.0j / 3.0) * omega - (1.0 / 8.0) * om2_a
            int_t_a: complex = (
                TAYLOR_TRIG_BASE + TAYLOR_TRIG_ORDER2 * om2_a
            ) * (uv - 1j * omega * (0.5 - 0.5 * u_c))

            int_c_b: complex = (exp_neg_i_omega - 1.0) / (omega + 1e-20) * 1j
            int_l_b: complex = (
                exp_neg_i_omega * (-1j * omega - 1.0) + 1.0
            ) / (-(omega * omega + 1e-20))
            epsilon: float = omega - sign(omega) * pi
            factor_trig: complex = (
                RESONANCE_SCALE_FACTOR * sign(omega)
            ) * (1.0 - 0.5j * epsilon - (1.0 / 6.0) * (epsilon * epsilon))
            int_t_b: complex = factor_trig * (
                uv - 1j * omega * (0.5 - 0.5 * u_c)
            )

            om2_c: float = omega * omega
            inv_om_c: float = 1.0 / (omega + 1e-20)
            int_c_c: complex = (exp_neg_i_omega - 1.0) * (1j * inv_om_c)
            int_l_c: complex = (
                exp_neg_i_omega * (-1j * omega - 1.0) + 1.0
            ) * (-(inv_om_c * inv_om_c))
            int_t_c: complex = (
                (1.0 + exp_neg_i_omega) / (PI_SQUARED - om2_c + 1e-20)
            ) * (uv - 1j * omega * (0.5 - 0.5 * u_c))

            is_near_zero: float = (
                1.0 if abs_omega < RESONANCE_NEARNESS_THRESHOLD else 0.0
            )
            is_near_pi: float = (
                1.0
                if abs(pi - abs_omega) < RESONANCE_NEARNESS_THRESHOLD
                else 0.0
            )
            is_standard: float = (
                1.0
                if (
                    abs_omega >= RESONANCE_NEARNESS_THRESHOLD
                    and abs(pi - abs_omega) >= RESONANCE_NEARNESS_THRESHOLD
                )
                else 0.0
            )

            int_c = (
                is_near_zero * int_c_a
                + is_near_pi * int_c_b
                + is_standard * int_c_c
            )
            int_l = (
                is_near_zero * int_l_a
                + is_near_pi * int_l_b
                + is_standard * int_l_c
            )
            int_t = (
                is_near_zero * int_t_a
                + is_near_pi * int_t_b
                + is_standard * int_t_c
            )

        core_integral: complex = (
            const_weight * int_c + linear_weight * int_l + dy * int_t
        )
        result[i] = dx * phase_shift * core_integral

    return result
