from math import pi
from numba import (
    complex128 as nb_c128,
    njit,
    prange,
)
from numpy import (
    cos,
    empty,
    ndarray,
    sin,
    zeros,
)


@njit(inline="always")
def _sign(val):
    return 1.0 if val >= 0.0 else -1.0


@njit(inline="always")
def _calc_element(
    x0, L, y0, dy, c_u, u, c_uv, current_n
):
    omega = 2.0 * pi * current_n * L
    p_om = 2.0 * pi * current_n * x0

    phase_shift = cos(p_om) - 1j * sin(p_om)
    abs_om = abs(omega)

    exp_w = cos(omega) - 1j * sin(omega)

    # 1. Standard AC High-Frequency Lane
    if (
        abs_om >= 1e-3
        and abs(abs_om - pi) >= 1e-4
    ):
        I_1 = (1.0 - exp_w) / (1j * omega)
        w2 = (
            omega**2
            if omega**2 > 1e-30
            else 1e-30
        )
        I_X = (
            (1.0 + 1j * omega) * exp_w - 1.0
        ) / w2
        denom = omega**2 - pi**2
        denom_safe = (
            denom
            if abs(denom) > 1e-30
            else 1e-30
        )
        term_p = 1.0 + exp_w
        I_cos = (
            -1j * omega * term_p
        ) / denom_safe
        I_sin = (-pi * term_p) / denom_safe

    # 2. Exact DC Singularity Node
    elif current_n == 0:
        I_1 = 1.0 + 0j
        I_X = 0.5 + 0j
        I_cos = 0.0 + 0j
        I_sin = (2.0 / pi) + 0j

    # 3. Shape Resonance Node (omega -> pi)
    # Fixed: Removed walrus operator allocation
    elif abs(abs_om - pi) < 1e-4:
        I_1 = (1.0 - exp_w) / (1j * omega)
        w2 = (
            omega**2
            if omega**2 > 1e-30
            else 1e-30
        )
        I_X = (
            (1.0 + 1j * omega) * exp_w - 1.0
        ) / w2
        I_cos = 0.5 + 0j
        I_sin = -0.5j * _sign(omega)

    # 4. Near-DC / Micro-Interval Window
    else:
        om2 = omega**2
        om3 = omega**3
        I_1 = (1.0 - om2 / 6.0) - 1j * (
            omega / 2.0 - om3 / 24.0
        )
        I_X = (0.5 - om2 / 8.0) + 1j * (
            omega / 3.0 - om3 / 30.0
        )
        I_cos = 0.0 + 0j
        I_sin = (2.0 / pi) + 0j

    I_g = (
        c_u * (I_1 - I_cos)
        + u * I_X
        + c_uv * I_sin
    )
    return L * phase_shift * (
        y0 * I_1 + dy * I_g
    )


@njit(
    parallel=True,
    fastmath=True,
    cache=True,
)
def _parallel_orchestrator(
    x0, L, y0, dy, c_u, u, c_uv, n
) -> ndarray:
    out = empty(n, dtype=nb_c128)
    for i in prange(n):
        out[i] = _calc_element(
            x0, L, y0, dy, c_u, u, c_uv, i
        )
    return out


def compute_fourier_from_zero(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    u: float,
    v: float,
    n: int,
) -> ndarray:
    if n <= 0:
        return zeros(0, dtype="complex128")

    L: float = x1 - x0
    if abs(L) < 1e-15:
        return zeros(n, dtype="complex128")

    dy: float = y1 - y0
    c_u: float = 0.5 * (1.0 - u)
    c_uv: float = (u * v) / pi

    return _parallel_orchestrator(
        x0, L, y0, dy, c_u, u, c_uv, n
    )
