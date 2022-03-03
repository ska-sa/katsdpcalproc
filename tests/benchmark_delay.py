
# Delay estimator comparison for lightning talk.
#
# Ludwig Schwardt
# 16 March 2016
#

import numpy as np
import matplotlib.pyplot as plt

from katsdpcalproc.delay import (mean_phase_diff, fft_coarse, fft_quadratic,
                              fft_leastsq, fft_secant)
from katsdpcalproc.delay_mattieu import mattieu


FLUX = 10.
SEFD = 400.
DUMP_PERIOD = 2.0
N_CHANS = 1024
SAMPLE_RATE = 1712e6
METHODS = ('Ludwig', 'Laura', 'Lindsay', 'SKA', 'Secant')  # , 'Mattieu')


def _wrap_angle(th):
    return (th + np.pi) % (2. * np.pi) - np.pi


def _calculate_params(sample_rate, n_chans, dump_period, ampl, sefd):
    bandwidth = sample_rate / 2
    channel_width = bandwidth / n_chans
    n = np.arange(n_chans, dtype=float)
    channel_freqs = bandwidth + n * channel_width
    delay_alias = 1 / channel_width
    samples = dump_period / delay_alias
    noise_var = 2 * sefd * sefd / samples
    snr = ampl * ampl / noise_var
    return channel_freqs, delay_alias, noise_var, snr


def _ant_vs_baseline(n_ants):
    baselines = [(a1, a2) for a1 in range(n_ants) for a2 in range(n_ants) if a1 < a2]
    to_baseline = np.zeros((len(baselines), n_ants))
    for n, (ant1, ant2) in enumerate(baselines):
        to_baseline[n, ant1] = -1.0
        to_baseline[n, ant2] = +1.0
    U, s, Vrt = np.linalg.svd(to_baseline[:, 1:], full_matrices=False)
    to_ant = Vrt.T @ np.diag(1. / s) @ U.T
    return baselines, to_baseline, to_ant


def _generate_data(slopes, channel_freqs, ampl, noise_var, window=None, tec=0):
    n_slopes = len(slopes)
    n_chans = len(channel_freqs)
    phase = 2. * np.pi * np.random.rand(n_slopes)
    noise = np.random.randn(n_slopes, n_chans) + 1j * np.random.randn(n_slopes, n_chans)
    n = np.arange(n_chans, dtype=float)

    v = channel_freqs[:, np.newaxis] / 1e9
    # This is roughly the worst differential slant TEC at 15 degrees elevation
    baseline_km = 8
    iono = np.radians(0.26 * tec * baseline_km / v)

    angle = np.outer(n, slopes) + phase + iono
    x = ampl * np.exp(1j * angle.T) + np.sqrt(noise_var / 2) * noise
    if window is not None:
        x *= np.atleast_2d(window)
    return x


def _estimate_slopes(x, fft_factor, chan_range, window, snr):
    n_fft = fft_factor * x.shape[-1]
    estimates = []
    for method in METHODS:
        if method == 'Ludwig':
            # Circular mean of phase difference
            estimates.append(mean_phase_diff(x[:, chan_range]))
        elif method == 'Laura':
            # FFT (linear regression)
            estimates.append(fft_leastsq(x[:, chan_range]))
        elif method == 'Lindsay':
            # FFT (no interpolation)
            estimates.append(fft_coarse(x, n_fft))
        elif method == 'SKA':
            # FFT (quadratic interpolation)
            estimates.append(fft_quadratic(x, n_fft))
        elif method == 'Secant':
            # FFT (secant)
            estimates.append(fft_secant(x, n_fft, discard_unconverged=True))
        elif method == 'Mattieu':
            # Mattieu's phase slope method (with Bill's phase error estimate)
            estimates.append(mattieu(x, gain=window, phase_std=1 / np.sqrt(snr)))
    return np.array(estimates)


def _measure_error(slope_estimates, slopes, snr, n_chans, window, n_ants=None):
    # Calculate Cramer-Rao lower bound
    if window is None:
        snr_sum = snr * n_chans
        curvature = (n_chans * n_chans - 1.0) / 12.0
    else:
        gate = window.nonzero()[0]
        weights = window / np.sum(window)
        snr_sum = snr @ weights * len(gate)
        n = np.arange(n_chans, dtype=float)
        centroid = weights @ n
        curvature = weights @ (n - centroid) ** 2
    crlb = 0.5 / (snr_sum * curvature)
    if n_ants is not None:
        # Going from baseline-based to antenna-based, you score an extra factor
        # determined by np.sum((Vrt.T / s[np.newaxis, :]) ** 2, axis=1).mean()
        crlb *= 2.0 / n_ants
    # Collect standard deviations
    stdevs = np.nanstd(_wrap_angle(slope_estimates - slopes), axis=-1)
    # Optionally use RMS instead of standard deviation in case of bias
    # residual = _wrap_angle(slope_estimates - slopes)
    # stdevs = np.sqrt(np.nanmean(residual * residual, axis=-1))
    return np.r_[np.sqrt(crlb), stdevs]


def bl_experiment(ampl=FLUX, sefd=SEFD, dump_period=DUMP_PERIOD, n_chans=N_CHANS,
                  sample_rate=SAMPLE_RATE, n_repeats=1000, fft_factor=2, window=None,
                  chan_range=slice(None), delay_limit=None, tec=0):
    channel_freqs, delay_alias, noise_var, snr = _calculate_params(
        sample_rate, n_chans, dump_period, ampl, sefd)
    slopes = 2. * np.pi * (np.random.rand(n_repeats) - 0.5)
    # FFT+secant method does not like phase slopes around +- pi / channel
    slopes *= 0.99 if delay_limit is None else delay_limit / delay_alias
    x = _generate_data(slopes, channel_freqs, ampl, noise_var, window, tec)
    slope_estimates = _estimate_slopes(x, fft_factor, chan_range, window, snr)
    stdevs = _measure_error(slope_estimates, slopes, snr, n_chans, window)
    # Convert from phase slope in radians/channel to delay in seconds
    return stdevs * delay_alias / (2 * np.pi)


def ant_experiment(ampl=FLUX, sefd=SEFD, dump_period=DUMP_PERIOD, n_chans=N_CHANS,
                   sample_rate=SAMPLE_RATE, n_repeats=70, n_ants=15,
                   fft_factor=2, window=None, chan_range=slice(None),
                   delay_limit=None, tec=0):
    channel_freqs, delay_alias, noise_var, snr = _calculate_params(
        sample_rate, n_chans, dump_period, ampl, sefd)
    slopes_per_ant = 2. * np.pi * (np.random.rand(n_repeats, n_ants) - 0.5)
    # Technically we need non-linear solver for angle ambiguities unless we constrain slopes
    slopes_per_ant *= 0.25 if delay_limit is None else delay_limit / delay_alias
    # First antenna is reference antenna
    slopes_per_ant[:, 0] = 0.
    # Convert to per-baseline slopes
    baselines, to_baseline, to_ant = _ant_vs_baseline(n_ants)
    slopes_per_bl = (slopes_per_ant @ to_baseline.T).ravel()
    x = _generate_data(slopes_per_bl, channel_freqs, ampl, noise_var, window, tec)
    # Fit per-baseline slopes
    slope_estm_per_bl = _estimate_slopes(x, fft_factor, chan_range, window, snr)
    # Go back to per-antenna estimates
    slope_estm_per_bl = slope_estm_per_bl.reshape(-1, n_repeats, len(baselines))
    slope_estm_per_ant = slope_estm_per_bl @ to_ant.T
    slope_estm_per_ant = slope_estm_per_ant.reshape(-1, n_repeats * (n_ants - 1))
    stdevs = _measure_error(slope_estm_per_ant, slopes_per_ant[:, 1:].ravel(),
                            snr, n_chans, window, n_ants)
    # Convert from phase slope in radians/channel to delay in seconds
    return stdevs * delay_alias / (2 * np.pi)


def plot_loglog(x, y):
    fig, ax = plt.subplots(figsize=(8, 6))
    log_x = np.log10(x)
    crline = ax.semilogy(log_x, y[:, 0], 'k--', marker='o')
    lines = ax.semilogy(log_x, y[:, 1:], marker='.')
    ax.xaxis.set_ticks(log_x)
    ax.xaxis.set_ticklabels(['{:g}'.format(fl) for fl in x])
    ax.set_xlim(log_x[0], log_x[-1])
    ax.grid(axis='y')
    ax.legend(lines + crline, METHODS + ('Best (CRB)',))
    ax.set_ylabel('Delay standard deviation [s]')
    return fig, ax


fluxes = np.array([0.1, 0.2, 0.5, 1., 2., 5., 10., 20., 50., 100.])
delay_std = []
for flux in fluxes:
    delay_std.append(bl_experiment(ampl=flux))
fig, ax = plot_loglog(fluxes, np.array(delay_std))
ax.set_xlabel('Calibrator flux [Jy]')
ax.set_title(f'Delay estimator performance vs flux (N={N_CHANS})')
fig.savefig('delay_per_bl_vs_flux.png')

log_sizes = np.arange(7, 14)
delay_std = []
for log_size in log_sizes:
    n_chans = 2 ** log_size
    delay_std.append(bl_experiment(dump_period=DUMP_PERIOD * n_chans / N_CHANS, n_chans=n_chans))
fig, ax = plot_loglog(2 ** log_sizes, np.array(delay_std))
ax.set_xlabel('Number of samples (N)')
ax.set_title(f'Delay estimator performance vs N (flux={FLUX})')
fig.savefig('delay_per_bl_vs_N.png')

t = np.arange(N_CHANS) / N_CHANS
flux_shape = 1.6 * np.exp(-0.65 * np.log(t + 1))
sefd = 500 * np.exp(-0.3 * np.log(t + 1))
gain = np.full_like(t, 0.01)
for harmonic in range(1, 11, 2):
    gain += np.sin(2 * np.pi * t * harmonic / 2) / harmonic
gain /= np.median(gain)
gate = np.zeros_like(t)
gate_transitions = [0, 1, 49, 50, 81, 86, 93, 135, 190, 203, 279, 282, 338, 536,
                    657, 681, 793, 901, 910, 926, 955, 968, 972, 973, 1023, 1024]
gate_scale = int(N_CHANS / gate_transitions[-1])
segm_start = gate_transitions[:-1]
segm_end = gate_transitions[1:]
for n, (b, e) in enumerate(zip(segm_start, segm_end)):
    segment = slice(gate_scale * b, gate_scale * e)
    gate[segment] = float(n % 2 == 1)
gain *= gate
# chan_range = slice(681 * gate_scale, 792 * gate_scale)  # last run of old baseline cal
chan_range = slice(563 * gate_scale, 613 * gate_scale)  # cal pipeline k_bfreq..k_efreq

fluxes = np.array([0.1, 0.2, 0.5, 1., 2., 5., 10., 20., 50., 100.])
delay_std = []
for flux in fluxes:
    delay_std.append(bl_experiment(ampl=flux * flux_shape, sefd=sefd, window=gain,
                                   chan_range=chan_range, delay_limit=10 / SAMPLE_RATE))
fig, ax = plot_loglog(fluxes, np.array(delay_std))
ax.set_xlabel('Calibrator flux [Jy]')
ax.set_title(f'Realistic delay estimator performance (N={N_CHANS})')
fig.savefig('delay_per_bl_realistic.png')

fluxes = np.array([0.1, 0.2, 0.5, 1., 2., 5., 10., 20., 50., 100.])
delay_std = []
for flux in fluxes:
    delay_std.append(ant_experiment(ampl=flux))
fig, ax = plot_loglog(fluxes, np.array(delay_std))
ax.set_xlabel('Calibrator flux [Jy]')
ax.set_title(f'Delay estimator performance vs flux (N={N_CHANS})')
fig.savefig('delay_per_ant_vs_flux.png')

fluxes = np.array([0.1, 0.2, 0.5, 1., 2., 5., 10., 20., 50., 100.])
delay_std = []
for flux in fluxes:
    delay_std.append(ant_experiment(ampl=flux * flux_shape, sefd=sefd, window=gain,
                                    chan_range=chan_range, delay_limit=10 / SAMPLE_RATE))
fig, ax = plot_loglog(fluxes, np.array(delay_std))
ax.set_xlabel('Calibrator flux [Jy]')
ax.set_title(f'Realistic delay estimator performance (N={N_CHANS})')
fig.savefig('delay_per_ant_realistic.png')

plt.show()
