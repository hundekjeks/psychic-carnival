import numpy as np

def integrate_h_dc(x0, x1, y0, y1, k):
    """Computes the exact analytical integral for n = 0 (DC)."""
    L = x1 - x0
    return L * ((y0 + y1) / 2.0 + k * (y1 - y0) / np.pi)


def integrate_h_n1(x0, x1, y0, y1, k):
    """Computes the exact analytical Fourier integral for n = 1."""
    L = x1 - x0
    if L < 1e-15:
        return 0.0 + 0.0j

    omega = 2.0 * np.pi * L
    omega_sq = omega * omega
    pi_sq = np.pi * np.pi

    exp_minus_i_omega = np.exp(-1j * omega)
    exp_prefix = np.exp(-2j * np.pi * x0)

    # Resonance guard when L == 0.5 (omega == pi)
    if np.abs(pi_sq - omega_sq) < 1e-12:
        t1 = (y0 + y1) * (1.0 - exp_minus_i_omega) / (2j * omega)
        t2_res = (y1 - y0) * (k * np.pi - 1j * omega) * (1j / (2.0 * np.pi))
        result = L * exp_prefix * (t1 + t2_res)
    else:
        t1 = (y0 + y1) * (1.0 - exp_minus_i_omega) / (2j * omega)
        t2 = (
            (y1 - y0)
            * (k * np.pi - 1j * omega)
            * (1.0 + exp_minus_i_omega)
            / (2.0 * (pi_sq - omega_sq))
        )
        result = L * exp_prefix * (t1 + t2)

    if np.isnan(result) or np.isinf(result):
        return 0.0 + 0.0j

    return result


def integrate_h_vectorized_positive(x0, x1, y0, y1, k, n_array):
    """Computes exact Fourier integrals over positive harmonics (n > 0)."""
    n = np.asarray(n_array, dtype=float)
    L = x1 - x0
    if L < 1e-15:
        return np.zeros_like(n, dtype=complex)

    omega = 2.0 * np.pi * n * L
    omega_sq = omega * omega
    pi_sq = np.pi * np.pi

    # Track structural resonance conditions (where L = 1 / 2n)
    res_mask = np.abs(pi_sq - omega_sq) < 1e-12
    denom_safe = np.where(res_mask, 1.0, 2.0 * (pi_sq - omega_sq))

    exp_minus_i_omega = np.exp(-1j * omega)
    exp_prefix = np.exp(-2j * np.pi * n * x0)

    # Core evaluation with safe denominators
    t1 = (y0 + y1) * (1.0 - exp_minus_i_omega) / (2j * omega)
    t2 = (
        (y1 - y0)
        * (k * np.pi - 1j * omega)
        * (1.0 + exp_minus_i_omega)
        / denom_safe
    )

    if np.any(res_mask):
        t2_res = (y1 - y0) * (k * np.pi - 1j * omega) * (1j / (2.0 * np.pi))
        t2 = np.where(res_mask, t2_res, t2)

    final_output = L * exp_prefix * (t1 + t2)
    return np.nan_to_num(final_output, nan=0.0, posinf=0.0, neginf=0.0)
