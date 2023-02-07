
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
from katsdpcal.calprocs import k_fit


FLUX = 10.
SEFD = 400.
DUMP_PERIOD = 120.0
N_CHANS = 1024
SAMPLE_RATE = 1712e6
N_ANTS = 15
METHODS = ('Ludwig', 'Laura', 'Lindsay', 'SKA', 'Secant')  # , 'Mattieu')
RESULTS = 'perant'


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
    baselines = np.c_[np.triu_indices(n_ants, 1)]
    to_baseline = np.zeros((len(baselines), n_ants))
    for n, (ant1, ant2) in enumerate(baselines):
        to_baseline[n, ant1] = -1.0
        to_baseline[n, ant2] = +1.0
    to_ant = np.linalg.pinv(to_baseline[:, 1:])
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
    # Add a nasty spike in unflagged region
    x[:, n_chans // 4] += 10 * n_chans
    if window is not None:
        x *= np.atleast_2d(window)
    return x


def _kill_spikes(x, median_factor=10.0):
    amplitude = np.abs(x)
    amplitude_threshold = median_factor * np.median(amplitude, axis=-1)
    spikes = amplitude > amplitude_threshold[:, np.newaxis]
    y = x.copy()
    y[spikes] *= 0.0
    return y


def _estimate_slopes(x, fft_factor, chan_range, window, snr):
    n_fft = fft_factor * x.shape[-1]
    x = _kill_spikes(x)
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


def _estimate_slopes_k_fit(x, baselines, n_ants, channel_freqs, delay_alias):
    n_repeats, n_bls, n_chans = x.shape
    cal_vis = np.empty((n_chans, 1, n_bls), dtype=np.complex64)
    cal_weights = np.ones_like(cal_vis, dtype=np.float32)
    slope_estimates = np.zeros((n_repeats, n_ants - 1))
    for m in range(n_repeats):
        cal_vis[:] = x[m].T[:, np.newaxis, :]
        cal_weights[:] = 1.0
        cal_weights[np.abs(cal_vis) == 0.0] = 0.0
        # The pipeline delays have the opposite sign to the definition
        k_delays = -k_fit(cal_vis, cal_weights, baselines, channel_freqs)
        slope_estimates[m] = k_delays[0, 1:] * 2 * np.pi / delay_alias
    return slope_estimates


def _cramer_rao_bound(snr, n_chans, window=None, n_ants=None):
    """Cramer-Rao bound as a standard deviation [radians/channel]."""
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
    crb = 0.5 / (snr_sum * curvature)
    if n_ants is not None:
        # Going from baseline-based to antenna-based, you score an extra factor
        # determined by np.sum((Vrt.T / s[np.newaxis, :]) ** 2, axis=1).mean()
        crb *= 2.0 / n_ants
    return np.sqrt(crb)


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
    # Convert from phase slope in radians/channel to delay in seconds
    crb = _cramer_rao_bound(snr, n_chans, window) * delay_alias / (2 * np.pi)
    residuals = _wrap_angle(slope_estimates - slopes) * delay_alias / (2 * np.pi)
    stdevs = np.nanstd(residuals, axis=-1)
    return np.r_[crb, stdevs]


def ant_experiment(ampl=FLUX, sefd=SEFD, dump_period=DUMP_PERIOD, n_chans=N_CHANS,
                   sample_rate=SAMPLE_RATE, n_repeats=10, n_ants=N_ANTS,
                   fft_factor=2, window=None, chan_range=slice(None),
                   delay_limit=None, tec=0, results_per_antenna=True):
    channel_freqs, delay_alias, noise_var, snr = _calculate_params(
        sample_rate, n_chans, dump_period, ampl, sefd)
    slopes_per_ant = 2. * np.pi * (np.random.rand(n_repeats, n_ants) - 0.5)
    # Technically we need non-linear solver for angle ambiguities unless we constrain slopes
    slopes_per_ant *= 0.25 if delay_limit is None else delay_limit / delay_alias
    # First antenna is reference antenna
    slopes_per_ant[:, 0] = 0.
    # Convert to per-baseline slopes
    baselines, to_baseline, to_ant = _ant_vs_baseline(n_ants)
    n_bls = len(baselines)
    slopes_per_bl = (slopes_per_ant @ to_baseline.T).ravel()
    x = _generate_data(slopes_per_bl, channel_freqs, ampl, noise_var, window, tec)
    # Fit per-baseline slopes
    slope_estm_per_bl = _estimate_slopes(x, fft_factor, chan_range, window, snr)
    # Go back to per-antenna estimates
    slope_estm_per_bl = slope_estm_per_bl.reshape(-1, n_repeats, n_bls)
    slope_estm_per_ant = slope_estm_per_bl @ to_ant.T
    slope_estm_per_ant = slope_estm_per_ant.reshape(-1, n_repeats * (n_ants - 1))
    # Add the cal pipeline solver
    x = x.reshape(n_repeats, n_bls, n_chans)
    slope_estm_k_fit_per_ant = _estimate_slopes_k_fit(
        x[..., chan_range], baselines, n_ants, channel_freqs[chan_range], delay_alias
    )
    if results_per_antenna:
        crb = _cramer_rao_bound(snr, n_chans, window, n_ants)
        slopes = slopes_per_ant[:, 1:].ravel()
        slope_estimates = np.vstack((slope_estm_per_ant,
                                     slope_estm_k_fit_per_ant.ravel()))
    else:
        crb = _cramer_rao_bound(snr, n_chans, window)
        slopes = slopes_per_bl
        slope_estm_k_fit_per_bl = slope_estm_k_fit_per_ant @ to_baseline.T[1:]
        slope_estimates = np.vstack((slope_estm_per_bl.reshape(-1, n_repeats * n_bls),
                                     slope_estm_k_fit_per_bl.ravel()))
    # Convert from phase slope in radians/channel to delay in seconds
    residuals = _wrap_angle(slope_estimates - slopes) * delay_alias / (2 * np.pi)
    return crb * delay_alias / (2 * np.pi), residuals


def plot_loglog(x, y, k_fit=False):
    fig, ax = plt.subplots(figsize=(8, 6))
    log_x = np.log10(x)
    crline = ax.semilogy(log_x, y[:, 0], 'k--', marker='o')
    lines = ax.semilogy(log_x, y[:, 1:], marker='.')
    ax.xaxis.set_ticks(log_x)
    ax.xaxis.set_ticklabels(['{:g}'.format(fl) for fl in x])
    ax.set_xlim(log_x[0], log_x[-1])
    ax.grid(axis='y')
    methods = METHODS + ('k_fit',) if k_fit else METHODS
    ax.legend(lines + crline, methods + ('Best (CRB)',))
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
delay_residuals = []
for flux in fluxes:
    crb, residuals = ant_experiment(ampl=flux, results_per_antenna=(RESULTS == 'perant'))
    # delay_std.append(np.r_[crb, np.nanstd(residuals, axis=-1)])
    perc25 = np.nanpercentile(residuals, 25, axis=-1)
    perc75 = np.nanpercentile(residuals, 75, axis=-1)
    iqr_to_std = 0.741301109253
    delay_std.append(np.r_[crb, iqr_to_std * (perc75 - perc25)])
    delay_residuals.append(residuals)
fig, ax = plot_loglog(fluxes, np.array(delay_std), k_fit=True)
ax.set_xlabel('Calibrator flux [Jy]')
ax.set_title(f'Delay errors ({RESULTS}, chans={N_CHANS}, ants={N_ANTS})')
fig.savefig(f'delay_basic_ant{N_ANTS}_chan{N_CHANS}_{RESULTS}.png')

fluxes = np.array([0.1, 0.2, 0.5, 1., 2., 5., 10., 20., 50., 100.])
delay_std = []
delay_residuals = []
for flux in fluxes:
    crb, residuals = ant_experiment(ampl=flux * flux_shape, sefd=sefd, window=gain,
                                    chan_range=chan_range, delay_limit=10 / SAMPLE_RATE,
                                    results_per_antenna=(RESULTS == 'perant'))
    # delay_std.append(np.r_[crb, np.nanstd(residuals, axis=-1)])
    perc25 = np.nanpercentile(residuals, 25, axis=-1)
    perc75 = np.nanpercentile(residuals, 75, axis=-1)
    iqr_to_std = 0.741301109253
    delay_std.append(np.r_[crb, iqr_to_std * (perc75 - perc25)])
    delay_residuals.append(residuals)
fig, ax = plot_loglog(fluxes, np.array(delay_std), k_fit=True)
ax.set_xlabel('Calibrator flux [Jy]')
ax.set_title(f'Delay errors ({RESULTS}, chans={N_CHANS}, ants={N_ANTS})')
fig.savefig(f'delay_realistic_ant{N_ANTS}_chan{N_CHANS}_{RESULTS}.png')

plt.show()
