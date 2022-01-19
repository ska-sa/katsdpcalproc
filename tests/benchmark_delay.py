
# Delay estimator comparison for lightning talk.
#
# Ludwig Schwardt
# 16 March 2016
#

import numpy as np
import matplotlib.pyplot as plt

from katsdpcalproc.delay import (mean_phase_diff, fft_coarse, fft_quadratic,
                              fft_leastsq, fft_secant)


DUMP_PERIOD = 2.0
CHANNELS = 4096


def _wrap_angle(th):
    return (th + np.pi) % (2. * np.pi) - np.pi


def experiment(flux=10., SEFD=400., dump_period=DUMP_PERIOD, channels=CHANNELS,
               sample_rate=1712e6, repeats=1000, fft_factor=2, window=None):
    N, K, ampl = channels, repeats, flux

    NFFT = fft_factor * N
    samples = dump_period * sample_rate / (2 * N)
    noise_var = 2 * SEFD * SEFD / samples
    SNR = ampl * ampl / noise_var

    freq = 2. * np.pi * (np.random.rand(K) - 0.5)
    freq = 0.99 * freq
    phase = 2. * np.pi * np.random.rand(K)
    noise = np.random.randn(K, N) + 1j * np.random.randn(K, N)
    if window is None:
        SNR_sum = SNR * N
        N_range = N
    else:
        gate = np.nonzero(window)[0]
        SNR_eff = np.zeros_like(SNR)
        SNR_eff[gate] = SNR[gate]
        SNR_sum = np.sum(SNR_eff)
        N_range = gate[-1] - gate[0] + 1
    crlb = 6.0 / (SNR_sum * (N_range * N_range - 1.0))

    n = np.arange(N, dtype=float)
    angle = np.outer(n, freq) + phase
    x = ampl * np.exp(1j * angle.T) + np.sqrt(noise_var / 2) * noise
    if window is not None:
        x *= np.atleast_2d(window)

    # Circular mean of phase difference
    cmpd = mean_phase_diff(x)
    # FFT (no interpolation)
    fft = fft_coarse(x, NFFT)
    # FFT (quadratic interpolation)
    fft_quad = fft_quadratic(x, NFFT)
    # FFT (linear regression)
    fft_lsq = fft_leastsq(x, NFFT)
    # FFT (secant)
    fft_sec = fft_secant(x, NFFT)
    # Collect standard deviations
    stdevs = [np.sqrt(crlb)]
    for freq_estm in [cmpd, fft_lsq, fft, fft_quad, fft_sec]:
        stdevs.append(_wrap_angle(freq_estm - freq).std())
    res = 2 * np.pi / N
    return np.array(stdevs) / res


fluxes = np.array([0.1, 0.2, 0.5, 1., 2., 5., 10., 20., 50., 100.])
scaled_std = np.empty((len(fluxes), 6))
for n, flux in enumerate(fluxes):
    scaled_std[n] = experiment(flux=flux)

fig, ax = plt.subplots(figsize=(8, 6))
log_fluxes = np.log10(fluxes)
crline = ax.semilogy(log_fluxes, scaled_std[:, 0], 'k--', marker='o')
lines = ax.semilogy(log_fluxes, scaled_std[:, 1:], marker='.')
ax.xaxis.set_ticks(log_fluxes)
ax.xaxis.set_ticklabels(['{:g}'.format(fl) for fl in fluxes])
ax.set_xlim(log_fluxes[0], log_fluxes[-1])
ax.legend(lines + crline,
          ('Ludwig', 'Laura', 'Lindsay', 'SKA', 'Secant', 'Best (CRLB)'))
ax.set_xlabel('Calibrator flux (Jy)')
ax.set_ylabel('Frequency standard deviation relative to 1/N')
ax.set_title('Frequency (delay) estimator performance')
fig.savefig('delay_estm_vs_flux.png')

log_sizes = np.arange(7, 14)
scaled_std = np.empty((len(log_sizes), 6))
for n, log_size in enumerate(log_sizes):
    N = 2 ** log_size
    scaled_std[n] = experiment(dump_period=DUMP_PERIOD * N / CHANNELS, channels=N)

fig, ax = plt.subplots(figsize=(8, 6))
crline = ax.semilogy(log_sizes, scaled_std[:, 0], 'k--', marker='o')
lines = ax.semilogy(log_sizes, scaled_std[:, 1:], marker='.')
ax.xaxis.set_ticks(log_sizes)
ax.xaxis.set_ticklabels(['{:g}'.format(2 ** ls) for ls in log_sizes])
ax.set_xlim(log_sizes[0], log_sizes[-1])
ax.legend(lines + crline,
          ('Ludwig', 'Laura', 'Lindsay', 'SKA', 'Secant', 'Best (CRLB)'))
ax.set_xlabel('Number of samples (N)')
ax.set_ylabel('Frequency standard deviation relative to 1/N')
ax.set_title('Frequency (delay) estimator performance')
fig.savefig('delay_estm_vs_N.png')

t = np.arange(CHANNELS) / CHANNELS
flux = 16 * np.exp(-0.65 * np.log(t + 1))
sefd = 500 * np.exp(-0.3 * np.log(t + 1))
gain = np.full_like(t, 0.01)
for harmonic in range(1, 11, 2):
    gain += np.sin(2 * np.pi * t * harmonic / 2) / harmonic
gain /= np.median(gain)
gate = np.zeros_like(t)
gate_transitions = [0, 50, 81, 86, 93, 135, 190, 203, 279, 282, 338,
                    536, 657, 681, 793, 901, 910, 926, 955, 968, 972, 1024]
gate_scale = int(CHANNELS / gate_transitions[-1])
segm_start = gate_transitions[:-1]
segm_end = gate_transitions[1:]
for n, (b, e) in enumerate(zip(segm_start, segm_end)):
    gate[gate_scale * b : gate_scale * e] = float(n % 2 == 1)
gain *= gate

fluxes = np.array([0.1, 0.2, 0.5, 1., 2., 5., 10., 20., 50., 100.])
scaled_std = np.empty((len(fluxes), 6))
for n, flux in enumerate(fluxes):
    scaled_std[n] = experiment(flux=flux, SEFD=sefd, window=gain)

fig, ax = plt.subplots(figsize=(8, 6))
log_fluxes = np.log10(fluxes)
crline = ax.semilogy(log_fluxes, scaled_std[:, 0], 'k--', marker='o')
lines = ax.semilogy(log_fluxes, scaled_std[:, 1:], marker='.')
ax.xaxis.set_ticks(log_fluxes)
ax.xaxis.set_ticklabels(['{:g}'.format(fl) for fl in fluxes])
ax.set_xlim(log_fluxes[0], log_fluxes[-1])
ax.legend(lines + crline,
          ('Ludwig', 'Laura', 'Lindsay', 'SKA', 'Secant', 'Best (CRLB)'))
ax.set_xlabel('Calibrator flux (Jy)')
ax.set_ylabel('Frequency standard deviation relative to 1/N')
ax.set_title('Frequency (delay) estimator performance')
fig.savefig('delay_estm_realistic.png')

plt.show()
