#
# Various frequency / delay estimators.
#
# Ludwig Schwardt
# 16 March 2016
#

import numpy as np


def mean_phase_diff(x):
    """Circular mean of phase difference."""
    dangle = np.diff(np.angle(x))
    return np.arctan2(np.sin(dangle).mean(axis=-1),
                      np.cos(dangle).mean(axis=-1))


def lank_reed_pollon(x):
    """Lank-Reed-Pollon estimator."""
    xx_lag1 = x[..., :-1].conj() * x[..., 1:]
    return np.angle(xx_lag1.sum(axis=-1))


def _fft_abs_peak(x, NFFT):
    absX = np.abs(np.fft.fft(x, NFFT))
    fft_peak = absX.argmax(axis=-1)
    return absX, fft_peak


def _index_to_freq(k, NFFT):
    deshift = np.where(k < NFFT // 2, k, k - NFFT)
    return 2. * np.pi * deshift / NFFT


def fft_coarse(x, NFFT=None):
    """FFT (no interpolation, nearest bin)."""
    if NFFT is None:
        NFFT = np.shape(x)[-1]
    _, fft_peak = _fft_abs_peak(x, NFFT)
    return _index_to_freq(fft_peak, NFFT)


def fft_quadratic(x, NFFT=None):
    """FFT (quadratic interpolation)."""
    if NFFT is None:
        NFFT = np.shape(x)[-1]
    absX, fft_peak = _fft_abs_peak(x, NFFT)
    assert absX.ndim > 1
    K = absX.shape[0]
    cols, rows = np.meshgrid(range(-1, 2), range(K))
    peak_ind = cols + fft_peak[:, np.newaxis]
    peak_val = absX[rows, peak_ind % NFFT]
    vander = np.dstack((peak_ind ** 2, peak_ind, peak_ind ** 0))
    poly = np.einsum('...jk,...k', np.linalg.inv(vander), peak_val)
    return _index_to_freq(-0.5 * poly[..., 1] / poly[..., 0], NFFT)


def fft_leastsq(x, NFFT=None):
    """FFT (linear regression)."""
    assert np.ndim(x) > 1
    N = x.shape[-1]
    if NFFT is None:
        NFFT = N
    fft = fft_coarse(x, NFFT)
    n = np.arange(N, dtype=float)
    angle_post_fft = np.angle(x * np.exp(-1j * np.outer(fft, n)))
    centred_n = n - n.mean()
    norm_n = centred_n / np.dot(centred_n, centred_n)
    leastsq_slope = np.zeros_like(fft)
    for k in range(x.shape[0]):
        valid = np.abs(x[k]) > 0
        y = np.unwrap(angle_post_fft[k, valid])
        leastsq_slope[k] = np.dot(norm_n[valid], y.T - y.mean(axis=-1))
    return fft + leastsq_slope


def _deriv(f, x):
    N = np.shape(x)[-1]
    n = np.arange(N, dtype=float)
    dx = -1j * x * n
    basis = np.exp(-1j * np.outer(f, n)) / N
    X = (x * basis).sum(axis=-1)
    dX = (dx * basis).sum(axis=-1)
    return 2 * (X * dX.conj()).real


def _deriv_fast(f, x, n, temp):
    temp.real = 0.0
    np.outer(-f, n, temp.imag)
    np.exp(temp, temp)
    temp *= x
    X = temp.mean(axis=-1)
    temp *= -1j * n
    dX = temp.mean(axis=-1)
    return 2 * (X * dX.conj()).real


def _secant(x, left, right, epsilon):
    delta = np.ones_like(left)
    done = delta < epsilon
    f_old, f_new = left, right
    d_old, d_new = _deriv(f_old, x), _deriv(f_new, x)
    iter = 0
    while not all(done):
        iter += 1
        with np.errstate(divide='ignore', invalid='ignore'):
            delta = d_new * (f_new - f_old) / (d_new - d_old)
        delta[~np.isfinite(delta) | done] = 0.
        done = np.abs(delta) < epsilon
        f_old, d_old = f_new, d_new
        f_new = f_old - delta
        d_new = _deriv(f_new, x)
        # print iter, (delta == 0.).sum(), np.abs(delta).max()
    return f_new


def _secant_fast(x, left, right, epsilon, max_iters, discard_unconverged):
    delta = np.ones_like(left)
    active = delta >= epsilon
    N = np.shape(x)[-1]
    n = np.arange(N, dtype=float)
    temp = np.empty(x.shape, dtype=np.complex128)
    f_old = left.copy()
    f_new = right.copy()
    d_old = _deriv_fast(f_old, x, n, temp)
    d_new = _deriv_fast(f_new, x, n, temp)
    iteration = 0
    while np.any(active) and iteration < max_iters:
        iteration += 1
        with np.errstate(divide='ignore', invalid='ignore'):
            delta = d_new * (f_new - f_old) / (d_new - d_old)
        delta[np.isnan(delta)] = 0.0
        active = ~np.isinf(delta) & (np.abs(delta) >= epsilon)
        f_old[:] = f_new
        d_old[:] = d_new
        f_new[active] = f_old[active] - delta[active]
        d_new[active] = _deriv_fast(f_new[active], x[active], n, temp[:sum(active)])
        # print(iteration, (delta == 0.).sum(), np.abs(delta).max())
    if discard_unconverged:
        unconverged = np.isinf(delta) | active
        f_new[unconverged] = np.nan
    return f_new


def fft_secant(x, NFFT=None, epsilon=1e-10, max_iters=100,
               discard_unconverged=False):
    front_shape = x.shape[:x.ndim - 1]
    if front_shape != ():
        x = x.reshape(-1, x.shape[-1])
    if NFFT is None:
        NFFT = np.shape(x)[-1]
    initial_guess = fft_coarse(x, NFFT)
    coarse_bin_width = 2 * np.pi / NFFT
    left = initial_guess - 0.5 * coarse_bin_width
    right = initial_guess + 0.5 * coarse_bin_width
    freq = _secant_fast(x, left, right, epsilon, max_iters, discard_unconverged)
    if front_shape != ():
        freq = freq.reshape(front_shape)
    return freq
